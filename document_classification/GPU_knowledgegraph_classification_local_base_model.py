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

# Set the local DBFS path
local_dbfs_path = "/tmp/BAAI/bge_m3"

# Download and save the tokenizer
#tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
#tokenizer.save_pretrained(local_dbfs_path)

# Download and save the model
#model = AutoModel.from_pretrained('BAAI/bge-m3')
#model.save_pretrained(local_dbfs_path)

# List the contents of the local DBFS path
display(dbutils.fs.ls(local_dbfs_path))

# COMMAND ----------

# Install torchvision
%pip install torch torchvision


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col
from pyspark.sql.functions import regexp_extract
from pyspark.sql import functions as F, types as T
import io, json, csv, email
import base64, json, os

# Initialize SparkSession for Databricks
spark = (
    SparkSession.builder
    .appName("Document-Classification")
    .getOrCreate()
)

# Azure Storage Configuration (same as PDF pipeline)
configs = {
  "fs.azure.account.auth.type": "OAuth",
  "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
  "fs.azure.account.oauth2.client.id": "",
  "fs.azure.account.oauth2.client.secret": "",
  "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com//oauth2/token"
}

DOCUMENTINTEL_ENDPOINT = "https://.cognitiveservices.azure.com/" #dbutils.secrets.get("kv-scope","di_endpoint")  # e.g., https://<res>.cognitiveservices.azure.com
DOCUMENTINTEL_KEY      = ""# dbutils.secrets.get("kv-scope","di_key")
DOCUMENTINTEL_MODEL_ID = "prebuilt-document"  # or your custom model

# --- Azure Configurations ---
AZURE_OPENAI_EMBEDDING_ENDPOINT = "https://.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
AZURE_OPENAI_API_KEY = ""
AZURE_SEARCH_ENDPOINT = "https://.search.windows.net"
AZURE_SEARCH_KEY = ""
AZURE_SEARCH_INDEX = "mydocs-knowledgeharvester-index"

AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = "https://-openai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"

spark.conf.set("fs.azure.account.auth.type.adls.dfs.core.windows.net", "OAuth")
for key, val in configs.items():
    spark.conf.set(f"{key}.adls.dfs.core.windows.net", val)

# Spark optimization settings
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

account="adls"
container_raw   = "raw"
container_stage = "stage"
container_redacted = "redacted"
container_classified="classified"

abfss = lambda container, path="": f"abfss://{container}@{account}.dfs.core.windows.net/{path}"

# system folders
PATH_DOCS_IN   = abfss(container_raw,   "incoming/docs/")     # .pdf/.docx/.xlsx will be stored here
PATH_IMAGES_OUT= abfss(container_stage, "images/")            # extracted images go here
PATH_JSON_OUT  = abfss(container_stage, "extracted_json/")    # DI/Vision JSON here
PATH_CLEANED   = abfss(container_redacted, "cleaned/text")           # PII-clean text
PATH_EMBED     = abfss(container_stage, "embeddings/")        # parquet with embeddings
PATH_TEXT_OUT = abfss(container_stage, "text/")
PATH_CLASSIFIED  = abfss(container_classified, "problem_solution/")
PATH_OTHER_TAX  = abfss(container_classified, "other_tax/")




# COMMAND ----------

# ==========================================================
# GPU Embedding Task - Creates df_emb.parquet
# ==========================================================
from pyspark.sql import functions as F, types as T
import torch
from transformers import AutoTokenizer, AutoModel
from pyspark import StorageLevel

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16
MAX_LEN = 512

account = "eymydocsmladls"
container_redacted = "redacted"
container_stage = "stage"
container_classified = "classified"
abfss = lambda c, p="": f"abfss://{c}@{account}.dfs.core.windows.net/{p}"

PATH_TEXT_IN = abfss(container_redacted, "cleaned/text/")
PATH_EMB_OUT = abfss(container_stage, "embeddings/")

df = (spark.read.parquet(PATH_TEXT_IN)
      .filter(F.col("redacted_text").isNotNull() & (F.length("redacted_text") > 0))
      .select("path", "redacted_text"))
df.persist(StorageLevel.MEMORY_AND_DISK)

print("Docs:", df.count())

# --- Broadcast anchors & params ---
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME).eval()
bc_model = spark.sparkContext.broadcast(MODEL_NAME)
bc_params = spark.sparkContext.broadcast({"BATCH_SIZE": BATCH_SIZE, "MAX_LEN": MAX_LEN})

# --- Partition embedding function ---
def embed_partition(rows):
    import torch
    from transformers import AutoTokenizer, AutoModel
    import numpy as np

    model_name = bc_model.value
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    mdl.eval()

    B = bc_params.value["BATCH_SIZE"]
    MAXLEN = bc_params.value["MAX_LEN"]

    for r in rows:
        text = r.redacted_text
        if not text:
            continue
        chunks = [text[i:i+1500] for i in range(0, len(text), 1500)] or [text]
        embs = []
        with torch.no_grad():
            for i in range(0, len(chunks), B):
                batch = [f"passage: {t}" for t in chunks[i:i+B]]
                enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAXLEN).to(mdl.device)
                out = mdl(**enc).last_hidden_state.mean(dim=1)
                out = out / out.norm(dim=1, keepdim=True)
                embs.append(out)
        doc_emb = torch.cat(embs, dim=0).mean(dim=0)
        doc_emb = doc_emb / doc_emb.norm()
        yield (r.path, doc_emb.cpu().numpy().astype(np.float32).tolist())

schema = T.StructType([
    T.StructField("path", T.StringType()),
    T.StructField("embedding_arr", T.ArrayType(T.FloatType()))
])

df_emb = spark.createDataFrame(df.rdd.mapPartitions(embed_partition), schema)

df_emb = df_emb.repartition(512)  # create more, smaller files

df_emb.write \
    .mode("overwrite") \
    .option("maxRecordsPerFile", 2000) \
    .parquet(PATH_EMB_OUT)


#df_emb.write.mode("overwrite").parquet(PATH_EMB_OUT)
print(f"✅ Embeddings written to {PATH_EMB_OUT}")
