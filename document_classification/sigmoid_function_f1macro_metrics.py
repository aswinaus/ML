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
MODEL_OUT       = "/dbfs/tmp/BAAI/finetuned_bge_m3_multilabel"   # fine-tuned output

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
    report_to="none",
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


# COMMAND ----------

# Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Print training/validation monitoring tips
print("""
If val loss keeps dropping and F1_macro climbs past ~0.70+, keep training.

If val loss stops improving for ~3 evals, stop (early stopping will do it).

If training loss ↓ but val loss ↑, you’re overfitting → reduce epochs or lower LR (e.g., 1e-5) and add weight_decay.
""")

# Load base E5 model and tokenizer
model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Updated label mapping
label2id = {
    "tax_problem": 0,
    "tax_solution": 1,
    "tax_type": 2,
    "tax_topic": 3,
    "year": 4
}
id2label = {v: k for k, v in label2id.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)
# Get the current weights before training
raw_weights = model.state_dict()

# Load training data from JSONL file
dataset_raw = load_dataset("json", data_files="tax_classifier_train_data_v2.jsonl")["train"]

# Convert to list of dicts for Dataset.from_list
train_data = [dict(row) for row in dataset_raw]

# Split into train/validation for evaluation and early stopping (small validation set)
split_dataset = Dataset.from_list(train_data).train_test_split(test_size=0.15, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

def preprocess_function(examples):
    # Tokenize and pad/truncate to max_length
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Tokenize train and validation sets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    # Compute accuracy and macro F1 for evaluation
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }
# Training arguments with evaluation and early stopping
training_args = TrainingArguments(
    output_dir="/dbfs/tmp/e5_finetuned_tax_classifier",
    num_train_epochs=5,                  # Train for more epochs to allow early stopping
    per_device_train_batch_size=8,
    learning_rate=2e-5,# The learning rate controls how quickly the model's weights are updated during training.
    weight_decay=0.01, # This is the weight decay to use for training. Weight decay is a regularization technique that helps prevent overfitting by adding a penalty term to the loss function for large weights.
    warmup_ratio=0.06, # This is the ratio of the total number of training steps to use for the warmup phase. The warmup phase is a period at the beginning of training where the learning rate is increased from a small initial value to the final value.
    logging_steps=10, # This is the number of steps to log training metrics. Logging metrics helps track the model's performance during training.
    evaluation_strategy="steps",         # Evaluate every eval_steps
    eval_steps=50, # This is the number of steps to evaluate the model. The model will be evaluated every eval_steps steps.
    save_steps=50,
    save_total_limit=2, # This is the total number of models to save. If the model is saved more than save_total_limit times, the oldest saved model will be deleted.
    load_best_model_at_end=True,         # Restore best model (lowest val loss/highest metric)
    metric_for_best_model="f1_macro",    # Use macro F1 to select best model
    greater_is_better=True,              # Higher F1 is better
    report_to="none" # This is the platform to report training metrics to. In this case, no platform is specified, so metrics will not be reported.
)
# Trainer for supervised fine-tuning with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,          # Validation set for evaluation and early stopping
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,     # Compute accuracy and F1 on validation set
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if val metric doesn't improve for 3 evals
)
# Fine-tune the model; training will stop early if validation F1 does not improve for 3 evals
# update the weights after training it is done implicit in the trainer.train() call, and the updated weights are saved to the output directory after training is complete.
trainer.train()
# Get the updated weights
updated_weights = model.state_dict()
# Save the fine-tuned model and tokenizer
model.save_pretrained("/dbfs/tmp/e5_finetuned_tax_classifier")
tokenizer.save_pretrained("/dbfs/tmp/e5_finetuned_tax_classifier") 

# COMMAND ----------

