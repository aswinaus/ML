# Databricks notebook source
# ================================================================
# 00 - Install Dependencies (if needed)
# ================================================================
%pip install peft==0.10.0 transformers accelerate datasets ray[tune] > /dev/null


# ================================================================
# 01 - Imports & Config
# ================================================================
import os, json, time, traceback
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ray.air import session

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, f1_score
from peft import LoraConfig, get_peft_model
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.air import session

print("[INFO] Starting PBT + LoRA training pipeline")

BASE_MODEL_PATH = "/dbfs/tmp/hf_models/bge_m3"
DATA_FILE       = "synthetic_tax_memos_reference_100.jsonl"
MODEL_OUT       = "/dbfs/tmp/BAAI/pbt_lora_bge_m3"

label2id = {
    "tax_problem": 0,
    "tax_solution": 1,
    "tax_type": 2,
    "tax_topic": 3,
    "year": 4,
}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)


# ================================================================
# 02 - Dataset Loader (Arrow-free)
# ================================================================
raw_records = [json.loads(l) for l in open(DATA_FILE)]
split = int(len(raw_records) * 0.85)
train_records = raw_records[:split]
val_records   = raw_records[split:]


class TaxDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        text = item.get("text", "")
        labels = item.get("label", [])
        if not isinstance(labels, list):
            labels = [labels]

        vec = [0]*num_labels
        for l in labels:
            if l in label2id:
                vec[label2id[l]] = 1

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(vec, dtype=torch.float)
        }


# ================================================================
# 03 - Metrics
# ================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


# ================================================================
# 04 - SemanticTrainer (with LoRA support)
# ================================================================
class SemanticTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs,):
        labels = inputs["labels"]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits

        bce = nn.BCEWithLogitsLoss()(logits, labels)

        # semantic loss
        logits_n = torch.nn.functional.normalize(logits, dim=1)
        labels_n = torch.nn.functional.normalize(labels, dim=1)
        cosine_sim = torch.sum(logits_n * labels_n, dim=1).mean()
        sem_loss = 1 - cosine_sim

        loss = bce + 0.1 * sem_loss
        return (loss, outputs) if return_outputs else loss


# ================================================================
# 05 - Ray Tune Training Function
# ================================================================
def train_with_pbt(config):

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification"
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=16,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    train_ds = TaxDataset(train_records, tokenizer)
    val_ds   = TaxDataset(val_records, tokenizer)

    args = TrainingArguments(
        output_dir=session.get_trial_dir(),
        num_train_epochs=1,                     # only 1 epoch per iteration
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=config["lr"],
        warmup_ratio=config["warmup"],
        weight_decay=config["wd"],
        lr_scheduler_type="linear",
        logging_steps=20,
        evaluation_strategy="epoch",            # <-- FIXED
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = SemanticTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # ------------------------------------------------------
    # 🔥 IMPORTANT: MULTI-ITER TRAINING LOOP FOR PBT
    # ------------------------------------------------------
    MAX_ITERS = 5   # or more

    for iteration in range(MAX_ITERS):

        trainer.train()                       # trains for 1 epoch
        metrics = trainer.evaluate()

        session.report(
            {
                "training_iteration": iteration + 1,
                "f1_macro": metrics["eval_f1_macro"],
            }
        )

   


# ================================================================
# 06 - Ray Tune PBT Setup
# ================================================================
ray.init(ignore_reinit_error=True)

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="f1_macro",
    mode="max",
    perturbation_interval=2,
    hyperparam_mutations={
        "lr": tune.loguniform(1e-6, 5e-5),
        "wd": tune.uniform(0.0, 0.1),
        "warmup": tune.uniform(0.0, 0.2),
        "lora_rank": [4, 8, 16, 32],
    }
)

analysis = tune.run(
    train_with_pbt,
    name="pbt_lora_bge_m3_exp",
    scheduler=pbt,
    #metric="f1_macro",
    #mode="max",
    num_samples=8,
    resources_per_trial={"cpu": 4, "gpu": 1},
    config={
        "lr": 2e-5,
        "wd": 0.01,
        "warmup": 0.06,
        "lora_rank": 8,
    }
)


# ================================================================
# 07 - Save Best Model
# ================================================================
best_trial = analysis.get_best_trial("f1_macro", "max")
#best_checkpoint = analysis.get_best_checkpoint(best_trial)
best_checkpoint = analysis.get_best_checkpoint(
    trial=best_trial,
    metric="f1_macro",
    mode="max"
)

print("[INFO] Best checkpoint:", best_checkpoint)

# Load + save final model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

# Reapply best LoRA settings
best_config = analysis.get_best_config("f1_macro", "max")
print("\n[INFO] Best hyperparameters found:", best_config)
lora_cfg = LoraConfig(
    r=best_config["lora_rank"],
    lora_alpha=16,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_cfg)

model.save_pretrained(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)

print("[INFO] Final PBT + LoRA model saved to:", MODEL_OUT)
