# Databricks notebook source
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import mlflow.pyfunc
import pandas as pd

# ---------- Your Dual Encoder Wrapper ----------
class SharedDualEncoderInference(torch.nn.Module):
    def __init__(self, base_model, num_pii_types=11):
        super().__init__()
        self.encoder = base_model
        hidden_size = base_model.config.hidden_size

        self.pii_flag_head = torch.nn.Linear(hidden_size, 1)
        self.pii_score_head = torch.nn.Linear(hidden_size, 1)
        self.pii_types_head = torch.nn.Linear(hidden_size, num_pii_types)

    def forward(self, doc_input_ids, doc_attention_mask,
                label_input_ids, label_attention_mask):

        # Encode document
        doc_emb = self.encoder(doc_input_ids, attention_mask=doc_attention_mask).pooler_output
        doc_emb = F.normalize(doc_emb, dim=-1)

        # Encode label
        label_emb = self.encoder(label_input_ids, attention_mask=label_attention_mask).pooler_output
        label_emb = F.normalize(label_emb, dim=-1)

        return {
            "doc_emb": doc_emb,
            "label_emb": label_emb,
            "pii_flag": torch.sigmoid(self.pii_flag_head(doc_emb)),
            "pii_score": torch.sigmoid(self.pii_score_head(doc_emb)),
            "pii_types": torch.sigmoid(self.pii_types_head(doc_emb)),
        }


# ---------- MLflow Model Serving Wrapper ----------
class DualEncoderServingModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        MODEL_PATH = context.artifacts["model_dir"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load base encoder
        base_model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        # Build inference model
        self.model = SharedDualEncoderInference(base_model)
        self.model.eval()

    def predict(self, context, model_input):

        # Expect DataFrame with columns: doc_text, label_text
        docs = model_input["doc_text"].tolist()
        labels = model_input["label_text"].tolist()

        # Tokenize
        doc_enc = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt", max_length=256)
        label_enc = self.tokenizer(labels, padding=True, truncation=True, return_tensors="pt", max_length=256)

        # Inference
        with torch.no_grad():
            outputs = self.model(
                doc_input_ids=doc_enc["input_ids"],
                doc_attention_mask=doc_enc["attention_mask"],
                label_input_ids=label_enc["input_ids"],
                label_attention_mask=label_enc["attention_mask"]
            )

        # Compute cosine similarity
        cos_sim = torch.sum(outputs["doc_emb"] * outputs["label_emb"], dim=1).cpu().numpy()

        return pd.DataFrame({
            "cosine_similarity": cos_sim,
            "pii_flag": outputs["pii_flag"].cpu().numpy().flatten(),
            "pii_score": outputs["pii_score"].cpu().numpy().flatten(),
            "pii_types": outputs["pii_types"].cpu().numpy().tolist()
        })


# COMMAND ----------


account="mymydocsmladls"
container="model"
# abfss = lambda container, path="": f"abfss://{container}@{account}.dfs.core.windows.net/{path}"

dbutils.fs.cp(
    "dbfs:/tmp/pbt_pii_semantic_dual_encoder_merged",
    "abfss://model@mymydocsmladls.dfs.core.windows.net/classifier_model/",
    recurse=True
)


# COMMAND ----------

import mlflow
import shutil

custom_env = {
    "name": "dual_encoder_env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.12",
        "pip",
        {
            "pip": [
                "torch",
                "transformers",
                "accelerate",
                "sentencepiece",
                "datasets",
                "numpy",
                "cloudpickle==2.2.1",
                "peft==0.10.0",
                "mlflow==2.14.1"   # match your cluster version
            ]
        }
    ]
}

# MODEL_PATH = "dbfs:/tmp/pbt_pii_semantic_dual_encoder_merged"
# MODEL_PATH="abfss://model@mymydocsmladls.dfs.core.windows.net/classifier_model"

# src = "/dbfs/tmp/pbt_pii_semantic_dual_encoder_merged"
# dst = "model_dir"   # local folder inside the MLflow run
# shutil.copytree(src, dst, dirs_exist_ok=True)


MODEL_PATH = "dbfs:/tmp/pbt_pii_semantic_dual_encoder_merged"
local_tmp = "/tmp/model_dir"
dbutils.fs.cp(MODEL_PATH, f"file:{local_tmp}", recurse=True)



model_card_md = """
# Dual Encoder Semantic + PII Model

## **Purpose**
This model performs:
- Document-label semantic similarity scoring (dual encoder)
- PII detection:
  - `pii_flag`: Whether PII is present
  - `pii_score`: Strength of PII presence
  - `pii_types`: Multi-label PII classification

## **Training Data**
- Synthetic and real tax-themed documents  
- Mixed PII annotations  
- Balanced tax/non-tax negative samples  
- Dual-objective training:
  - Contrastive similarity loss  
  - Multi-task PII loss  

## **Intended Use**
- Document routing  
- Tax category relevance scoring  
- PII detection before indexing  
- Pre-processing pipeline for downstream LLM systems  

## **Limitations**
- PII scores are **probabilistic**, not deterministic  
- Tax similarity depends on embedding strength, not kmyword match  
- Not designed to detect image-based PII  

## **Contacts**
- Owner: Aswin Bhaskaran  
- Created by: aswin.bhaskaran@my.com
"""

import mlflow
# Simple register without python model or loader module
mlflow.set_registry_uri("databricks")  # legacy workspace registry

mlflow.transformers.log_model(
    transformers_model=local_tmp,        # path to local Qwen2.5_base model directory
    artifact_path="dual_encoder_serving", # name for the logged artifact
    registered_model_name="DualEncoderSemanticPIIModel", # registry name
    task="feature-extraction"             # REQUIRED when using a local checkpoint
)


# with mlflow.start_run():
#     mlflow.set_tag("model_card", model_card_md)

#     mlflow.pyfunc.log_model(
#         artifact_path="dual_encoder_serving",
#         # python_model=DualEncoderServingModel(),
#         artifacts={"model_dir": local_tmp},
#         registered_model_name="DualEncoderSemanticPIIModel",
#         # conda_env=custom_env,
#         metadata={"model_card_text": model_card_md}
#     )


# COMMAND ----------

# Optional
import mlflow

MODEL_PATH = "/dbfs/tmp/pbt_pii_semantic_dual_encoder_merged"

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="mydocs_dual_encoder_serving",
        python_model=DualEncoderServingModel(),
        artifacts={"model_dir": MODEL_PATH},
        registered_model_name="DualEncoderSemanticPIIModel"
    )
