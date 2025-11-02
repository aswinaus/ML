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

from transformers import AutoTokenizer, AutoModel

#model_name = "intfloat/e5_finetuned_tax_classifier"
local_path = "/dbfs/tmp/e5_finetuned_tax_classifier"
if not os.path.exists(os.path.dirname(local_path)):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save locally
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)

# COMMAND ----------

mlflow.set_registry_uri("databricks")  # legacy workspace registry

mlflow.transformers.log_model(
    transformers_model=local_path,        # path to local e5 finetuned model directory
    artifact_path="multilingual_e5_base", # name for the logged artifact
    registered_model_name="multilingual_e5_base_workspace", # registry name
    task="feature-extraction"             # REQUIRED when using a local checkpoint
)