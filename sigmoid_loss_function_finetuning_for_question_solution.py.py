# Databricks notebook source
# Install PyTorch and Transformers
%pip install "transformers==4.42.0"
%pip install "torch>=2.1.0,<2.4.0"
%pip install "mlflow>=2.14.1,<3.0.0"
%pip install "accelerate>=0.26.0"
%pip install "cloudpickle==3.0.0"
%pip install -U bitsandbytes
dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

# COMMAND ----------
# ================================================================
# 01 - Imports & Config (Arrow-free training pipeline)
# ================================================================

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import os
import time
import traceback

print("""
Tips:
- If val loss keeps dropping and F1_macro climbs past ~0.70+, keep training.
- If val loss plateaus for ~3 evals, early stopping will kick in.
- If train loss ↓ but val loss ↑, you're overfitting -> reduce epochs or LR.
""")

print("[INFO] Starting Arrow-free notebook execution...")
print("[INFO] Working directory:", os.getcwd())

# Paths
BASE_MODEL_PATH = "/dbfs/tmp/hf_models/bge_m3"          # your pretrained BGE-M3
DATA_FILE       = "synthetic_tax_memos_reference_100.jsonl"  # <-- update
MODEL_OUT       = "/dbfs/tmp/BAAI/bge_m3_multilabel"   # fine-tuned output

# Labels
label2id = {
    "tax_problem": 0,
    "tax_solution": 1,
    "tax_type": 2,
    "tax_topic": 3,
    "year": 4,
}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

print("[INFO] Configuration loaded.")
print("[DEBUG] label2id =", label2id)


# COMMAND ----------
# ================================================================
# 02 - Load Tokenizer & Model (PURE PYTORCH / NO ARROW)
# ================================================================

print("[INFO] Loading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    print("[INFO] Tokenizer loaded.")
except Exception:
    print("[ERROR] Tokenizer loading failed.")
    traceback.print_exc()
    raise

print("[INFO] Loading model...")

try:
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification"
    )
    print("[INFO] Model loaded from:", BASE_MODEL_PATH)
except Exception:
    print("[ERROR] Model loading failed.")
    traceback.print_exc()
    raise


# COMMAND ----------
# ================================================================
# 03 - Arrow-Free Dataset Loader
# ================================================================
# No HuggingFace load_dataset, no Arrow, no Dataset.map()

print("[INFO] Loading dataset from raw JSONL (Arrow-free)...")
print("[DEBUG] Dataset file:", DATA_FILE)

try:
    raw_records = []
    with open(DATA_FILE) as f:
        for line in f:
            raw_records.append(json.loads(line))
    print("[INFO] Loaded", len(raw_records), "records.")
except Exception:
    print("[ERROR] Failed to read JSONL.")
    traceback.print_exc()
    raise

if len(raw_records) == 0:
    raise ValueError("[ERROR] No records found in dataset.")

# Manual train/val split
split = int(len(raw_records) * 0.85)
train_records = raw_records[:split]
val_records   = raw_records[split:]

print("[INFO] Train size:", len(train_records))
print("[INFO] Val size:", len(val_records))
print("[DEBUG] First training sample:", train_records[0])


# COMMAND ----------
# ================================================================
# 04 - Custom PyTorch Dataset (NO ARROW / NO HF MAP)
# ================================================================
# maps the labels to their corresponding indices in the label2id dictionary. This is a direct matching approach,
# where the label is matched exactly to its corresponding index.

class TaxDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        text = item.get("text", "")
        lbls = item.get("label", [])

        if not isinstance(lbls, list):
            lbls = [lbls]

        # build multi-hot vector
        vec = [0] * num_labels
        for l in lbls:
            if l in label2id:
                vec[label2id[l]] = 1
            else:
                print("[WARN] Unknown label encountered:", l)

        # tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(vec, dtype=torch.float)
        }


print("[INFO] Building PyTorch datasets...")
train_dataset = TaxDataset(train_records, tokenizer)
val_dataset   = TaxDataset(val_records, tokenizer)

print("[INFO] Dataset objects created successfully.")
print("[DEBUG] Example encoded sample:", train_dataset[0])


# COMMAND ----------
# ================================================================
# 05 - Metrics (accuracy + F1_macro)
# ================================================================
# sigmoid function is used to calculate the probabilities from the logits

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

print("[INFO] Metrics function defined.")


# COMMAND ----------
# ================================================================
# 06 - TrainingArguments
# ================================================================

print("[INFO] Configuring training arguments...")

training_args = TrainingArguments(
    output_dir=MODEL_OUT,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    ddp_find_unused_parameters=False,# method for multi-GPU training, Distributed Data Parallel 
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to=["stdout"],
)

print("[INFO] Training arguments configured. Output dir:", MODEL_OUT)


# COMMAND ----------
# ================================================================
# 07 - Semantic Regularization Trainer
# ================================================================
# uses a semantic cosine regularization term in addition to the BCEWithLogitsLoss.
# This regularization term encourages the model to produce logits that are semantically 
# similar to the labels. The semantic similarity is measured using the cosine similarity 
# between the normalized logits and labels.

class SemanticTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits

        bce_loss = nn.BCEWithLogitsLoss()(logits, labels)

        # semantic cosine regularization
        logits_norm = torch.nn.functional.normalize(logits, dim=1)
        labels_norm = torch.nn.functional.normalize(labels, dim=1)

        cosine_sim = torch.sum(logits_norm * labels_norm, dim=1).mean()
        semantic_loss = 1 - cosine_sim

        loss = bce_loss + 0.1 * semantic_loss
        return (loss, outputs) if return_outputs else loss

print("[INFO] SemanticTrainer class defined.")


# COMMAND ----------
# ================================================================
# 08 - Trainer Initialization + Training
# ================================================================

print("[INFO] Initializing trainer...")

trainer = SemanticTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,   # pure PyTorch dataset
    eval_dataset=val_dataset,      # pure PyTorch dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("[INFO] Trainer initialized. Starting training...")

train_start = time.time()
trainer.train()
train_end = time.time()

print(f"[INFO] Training completed in {round(train_end - train_start, 2)} seconds.")


# COMMAND ----------
# ================================================================
# 09 - Save Model
# ================================================================

print("[INFO] Saving model...")

trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)

print(f"[INFO] Model saved successfully to {MODEL_OUT}")
