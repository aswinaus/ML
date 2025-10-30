# Databricks notebook source
# MAGIC %pip install transformers
# MAGIC # Install PyTorch and Transformers
# MAGIC %pip install torch
# MAGIC %pip install 'accelerate>=0.26.0'
# MAGIC %pip install -U bitsandbytes
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

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

# Path to the previously fine‑tuned model
prev_model_path = "/dbfs/tmp/BAAI/bge-m3"

# Load tokenizer and model from the saved checkpoint
tokenizer = AutoTokenizer.from_pretrained(prev_model_path)

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
    prev_model_path,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# Load training data from JSONL file
dataset_raw = load_dataset("json", data_files="synthetic_tax_memos_2000.jsonl")["train"]

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
    output_dir="/dbfs/tmp/BAAI/bge-m3-v2",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    report_to="none"
)

# Trainer for supervised fine-tuning with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Continue fine‑tuning
trainer.train()

# Save the further fine‑tuned model and tokenizer
model.save_pretrained("/dbfs/tmp/BAAI/bge-m3-v2")
tokenizer.save_pretrained("/dbfs/tmp/BAAI/bge-m3-v2")

# COMMAND ----------

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