# Databricks notebook source
# MAGIC %pip install ray
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import ray
ray.shutdown()

# Get the number of GPUs available on the cluster
num_gpus = 4  # Replace with the actual number of GPUs available on the cluster

# Initialize Ray with the required configuration
ray.init(
    address=None,
    num_gpus=num_gpus,
    resources={"accelerator_type:A10": num_gpus},
    dashboard_host="0.0.0.0",
    dashboard_port=8266,
    _system_config={"worker_register_timeout_seconds": 120}
)

# Print the cluster resources
print(ray.cluster_resources())

# To verify parallel execution
ray.available_resources()

# Remote function that explicitly requests 1 GPU
# @ray.remote(num_gpus=1, resources={"accelerator_type:A10": 1})
# def use_gpu():
#     import torch
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         return torch.cuda.get_device_name(device)
#     else:
#         return "No GPU available"

# # Launch tasks
# futures = [use_gpu.remote() for _ in range(4)]
# results = ray.get(futures)
# print("Task results:", results)

# COMMAND ----------

ray.cluster_resources()


# COMMAND ----------

# ==========================================================
# Dual-Encoder + LoRA + InfoNCE + Ray Tune PBT (CLEAN FINAL VERSION)
# ==========================================================
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

import ray
from ray import tune, air
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining

from peft import LoraConfig, get_peft_model


# ==========================================================
# PATHS
# ==========================================================
#DATA_DIR = "/dbfs/.../pairs"

DATA_DIR = "/dbfs/Workspace/Users/a1172300-msp01@ey.net/synthetic_tax_dual_encoder_v10_5k/pairs"
MODEL_NAME = "/dbfs/tmp/hf_models/bge_m3"
OUTPUT_ROOT = "/local_disk0/tmp/pbt_dual_encoder"
FINAL_MODEL_DIR = "/dbfs/tmp/pbt_dual_encoder_final"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================================
# Dataset
# ==========================================================
class DualEncoderDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        with open(jsonl_path, "r") as f:
            self.data = [json.loads(x) for x in f]

        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]
        doc = rec["doc_text"]
        label = rec["label_text"]

        t_doc = self.tok(
            doc,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        t_lab = self.tok(
            label,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "doc_input_ids": t_doc["input_ids"].squeeze(0),
            "doc_attention_mask": t_doc["attention_mask"].squeeze(0),
            "lab_input_ids": t_lab["input_ids"].squeeze(0),
            "lab_attention_mask": t_lab["attention_mask"].squeeze(0),
        }


# ==========================================================
# Collator
# ==========================================================
@dataclass
class DualEncoderCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        return {
            "doc_input_ids": torch.stack([b["doc_input_ids"] for b in batch]),
            "doc_attention_mask": torch.stack([b["doc_attention_mask"] for b in batch]),
            "lab_input_ids": torch.stack([b["lab_input_ids"] for b in batch]),
            "lab_attention_mask": torch.stack([b["lab_attention_mask"] for b in batch]),
        }


# ==========================================================
# Model — mean-pooling dual-encoder
# ==========================================================
class DualEncoderModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def gradient_checkpointing_enable(self, **kwargs):
        # if you don't implement it, just pass
        pass
    def mean_pool(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def forward(self, doc_input_ids, doc_attention_mask,
                lab_input_ids, lab_attention_mask):

        d_out = self.encoder(input_ids=doc_input_ids,
                             attention_mask=doc_attention_mask)
        l_out = self.encoder(input_ids=lab_input_ids,
                             attention_mask=lab_attention_mask)

        d_emb = self.mean_pool(d_out.last_hidden_state, doc_attention_mask)
        l_emb = self.mean_pool(l_out.last_hidden_state, lab_attention_mask)

        return {
            "doc_emb": F.normalize(d_emb, dim=-1),
            "lab_emb": F.normalize(l_emb, dim=-1),
        }


# ==========================================================
# Custom Trainer (InfoNCE, in-batch negatives)
# ==========================================================
class InfoNCEDualEncoderTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(
            inputs["doc_input_ids"],
            inputs["doc_attention_mask"],
            inputs["lab_input_ids"],
            inputs["lab_attention_mask"],
        )

        d = out["doc_emb"]     # (B, D)
        l = out["lab_emb"]     # (B, D)

        # batchwise similarity matrix
        sim = d @ l.T          # (B, B)
        sim = sim / 0.07       # temperature

        labels = torch.arange(sim.size(0), device=sim.device)

        loss = F.cross_entropy(sim, labels)

        if return_outputs:
            return loss, out
        return loss


# ==========================================================
# Report eval_loss to Ray Tune
# ==========================================================
class RayCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            session.report({"eval_loss": metrics["eval_loss"]})
        return control


# ==========================================================
# Training function (Ray Trial)
# ==========================================================
def train_with_pbt(config):
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = DualEncoderDataset(os.path.join(DATA_DIR, "train.jsonl"),
                                  tokenizer, max_len=config["max_len"])
    val_ds = DualEncoderDataset(os.path.join(DATA_DIR, "val.jsonl"),
                                tokenizer, max_len=config["max_len"])
    collator = DualEncoderCollator(tokenizer)

    # -------------------------
    # Base model + LoRA
    # -------------------------
    base = AutoModel.from_pretrained(MODEL_NAME)

    lora_cfg = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
        target_modules=["query", "key", "value", "dense"],  # <- correct for BGE
    )

    encoder = get_peft_model(base, lora_cfg)
    encoder.print_trainable_parameters()

    model = DualEncoderModel(encoder)

    # -------------------------
    # Mixed precision
    # -------------------------
    bf16 = torch.cuda.is_bf16_supported()

    trial_id = os.environ.get("TUNE_TRIAL_ID", "trial")

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_ROOT, trial_id),
        per_device_train_batch_size=config["batch"],
        per_device_eval_batch_size=config["batch"],
        num_train_epochs=config["epochs"],
        learning_rate=config["lr"],
        warmup_ratio=config["warmup"],
        weight_decay=config["wd"],
        gradient_accumulation_steps=config["accum"],

        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=50,

        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=False,

        remove_unused_columns=False,
        save_strategy="epoch",       # <- ensures checkpoints exist
        load_best_model_at_end=False,
        report_to=[],
    )

    trainer = InfoNCEDualEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[RayCallback()],
    )

    trainer.train()

    # final evaluation
    metrics = trainer.evaluate()

    # Ray checkpoint = HF output directory
    session.report(
        {"eval_loss": metrics["eval_loss"]},
        checkpoint=air.Checkpoint.from_directory(args.output_dir)
    )


# ==========================================================
# PBT Launcher
# ==========================================================
if __name__ == "__main__":

    ray.init(ignore_reinit_error=True)

    search_space = {
        "batch": tune.choice([4, 8]),
        "accum": tune.choice([1, 2, 4]),
        "lora_rank": tune.choice([4, 8]),
        "lr": tune.choice([1e-5, 2e-5, 5e-5]),
        "warmup": tune.choice([0.03, 0.1]),
        "wd": tune.choice([0.0, 0.01]),
        "max_len": 256,
        "epochs": 1.0,
        "eval_steps": 300,
        "grad_ckpt": tune.choice([True, False]),
    }

    trainable = tune.with_resources(train_with_pbt, {"gpu": 1, "cpu": 4})

    pbt = PopulationBasedTraining(
        metric="eval_loss",
        mode="min",
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": [1e-5, 2e-5, 5e-5],
            "batch": [4, 8],
            "warmup": [0.03, 0.1],
            "wd": [0.0, 0.01],
            "lora_rank": [4, 8],
        },
    )

    tuner = tune.Tuner(
        trainable,
        run_config=air.RunConfig(
            name="dual_encoder_pbt_final",
            storage_path="/root/ray_results",
        ),
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=4,
            max_concurrent_trials=4,
            reuse_actors=False,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best = results.get_best_result("eval_loss", "min")

    print("Best trial:", best.metrics)

    # ======================================================
    # Final export (load LoRA → merge → save)
    # ======================================================
    print("Loading best checkpoint:", best.checkpoint.path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)

    final_cfg = LoraConfig(
        r=best.config["lora_rank"],
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
        target_modules=["query", "key", "value", "dense"],
    )
    encoder = get_peft_model(base_model, final_cfg)

    # load HF checkpoint stored by trainer
    encoder.load_state_dict(
        torch.load(
            os.path.join(best.checkpoint.path, "pytorch_model.bin"),
            map_location="cpu"
        ),
        strict=False
    )

    # merge LoRA
    try:
        encoder = encoder.merge_and_unload()
    except Exception as e:
        print("WARNING: merge failed:", e)

    model = DualEncoderModel(encoder)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("Saved final merged model to", FINAL_MODEL_DIR)


# COMMAND ----------

