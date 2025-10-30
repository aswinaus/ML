# Databricks notebook source
# MAGIC %pip install transformers accelerate bitsandbytes mlflow
# MAGIC # Install PyTorch and Transformers
# MAGIC %pip install torch
# MAGIC %pip install torchvision
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import mlflow
import mlflow.transformers


# COMMAND ----------

# Workspace MLflow registered model name
registered_model_name = "multilingual_e5_base_workspace"


# COMMAND ----------

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, pipeline
import mlflow
import mlflow.transformers

# Load tokenizer and 8-bit model in memory
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModel.from_pretrained(
    "intfloat/multilingual-e5-base",
    quantization_config=bnb_config
)

# Wrap in a minimal pipeline (required for MLflow logging)
pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

# Log model to workspace MLflow
mlflow.transformers.log_model(
    transformers_model=pipe,
    artifact_path="multilingual_e5_base",
    registered_model_name="multilingual_e5_base_workspace"
)
