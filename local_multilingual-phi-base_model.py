# Databricks notebook source
# MAGIC %pip install transformers
# MAGIC # Install PyTorch and Transformers
# MAGIC %pip install torch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install 'accelerate>=0.26.0'
# MAGIC %pip install -U bitsandbytes
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig

# Set the local DBFS path
local_dbfs_path = "/dbfs/llm/phi/microsoft/Phi-3-mini-4k-instruct"

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
tokenizer.save_pretrained(local_dbfs_path)

# Quantize model to 8-bit and save
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModel.from_pretrained(
    'microsoft/Phi-3-mini-4k-instruct',
    quantization_config=bnb_config
)
model.save_pretrained(local_dbfs_path)

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/llm/phi/microsoft/Phi-3-mini-4k-instruct"))