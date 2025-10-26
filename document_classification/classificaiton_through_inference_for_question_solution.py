# Databricks notebook source
# MAGIC %pip install transformers
# MAGIC # Install PyTorch and Transformers
# MAGIC %pip install torch
# MAGIC %pip install 'accelerate>=0.26.0'
# MAGIC %pip install -U bitsandbytes
# MAGIC dbutils.library.restartPython()
# MAGIC

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
# In Databricks, the SparkSession is typically pre-configured,
# so you don't need to explicitly set Hadoop Azure dependencies.
spark = (
    SparkSession.builder
    .appName("Document-Classification_Model")
    .getOrCreate()
)

configs = {
  "fs.azure.account.auth.type": "OAuth",
  "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
  "fs.azure.account.oauth2.client.id": "640059f9-cebf-47e5-a3f1-7bbe06e7e4a3",
  "fs.azure.account.oauth2.client.secret": "",
  "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/115ee48c-5146-4054-9f33-83e2bfe089fd/oauth2/token"
}

spark.conf.set("fs.azure.account.auth.type.docclassifierstoragecct.dfs.core.windows.net", "OAuth")
for key, val in configs.items():
    spark.conf.set(f"{key}.docclassifierstoragecct.dfs.core.windows.net", val)

#configuring spark parallelism
#**Parallelism**: Instead of single-node Python loops for clarity. For more scale, convert the work to Spark parallelism: read the list of file paths to a DataFrame and `mapPartitions` / `foreachPartition` or use UDFs to parallelize extraction and calls.

spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

#todo
#Large files / memory**: For very large PDFs or huge images you may want streaming approaches. The above copies files locally before processing; adapt if you have memory concerns.
#todo
#**Idempotency**: The code stores SHA256 for each saved image. You can use that to skip reprocessing duplicates.

#account = dbutils.secrets.get("kv-scope","storage_account_name")
account="docclassifierstoragecct"
container_raw   = "raw"
container_model   = "docintclassificationmodel"
container_stage = "stage"
container_redacted = "redacted"
container_processed="docclassifiercontainer"
container_classified="classified"
abfss = lambda container, path="": f"abfss://{container}@{account}.dfs.core.windows.net/{path}"


# system folders
PATH_DOCS_IN   = abfss(container_raw,   "incoming/docs/")     # .pdf/.docx/.xlsx will be stored here
PATH_IMAGES_OUT= abfss(container_stage, "images/")            # extracted images go here
PATH_JSON_OUT  = abfss(container_stage, "extracted_json/")    # DI/Vision JSON here
PATH_CLEANED   = abfss(container_redacted, "cleaned/")           # PII-clean text
PATH_EMBED     = abfss(container_stage, "embeddings/")        # parquet with embeddings
PATH_TEXT_OUT = abfss(container_stage, "text/")
#PATH_Mistral = abfss(container_model, "Mistral 24B Instruct/")

# Mount if not already mounted
#mount_point = f"/mnt/{container_model}/Mistral 24B Instruct"
#if not any(m.mountPoint == mount_point for m in dbutils.fs.mounts()):
#    dbutils.fs.mount(
#        source=f"abfss://{container_model}@{account}.dfs.core.windows.net/",
#        mount_point=mount_point,
#        extra_configs=configs
#    )

# COMMAND ----------

display(dbutils.fs.ls("abfss://classified@docclassifierstoragecct.dfs.core.windows.net/problem_solution/_temporary/"))


# COMMAND ----------

# ==========================================================
# 03_semantic_problem_solution_classifier_native.py
# ==========================================================
# Purpose:
#   - Read multilingual text documents from ADLS (Parquet)
#   - Compute semantic "problem + solution" scores (no pandas)
#   - Preserve all original columns
#   - Write results back to ADLS as Parquet, partitioned by source folder
# ==========================================================

# COMMAND ----------
import requests
import mlflow

# Databricks Model Serving endpoint
SERVING_ENDPOINT = "https://adb-3249086852123311.11.azuredatabricks.net/serving-endpoints/multilingual_e5_base_service/invocations"
DATABRICKS_TOKEN = ""
bc_token = spark.sparkContext.broadcast(DATABRICKS_TOKEN)
#dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def get_embeddings_serving(texts):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": [f"passage: {t}" for t in texts]
    }
    response = requests.post(SERVING_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    # Adjust this line based on the actual response structure
    # For MLflow serving, embeddings are usually under 'predictions' or 'embeddings'
    if isinstance(result, dict):
        # Try common keys
        for key in ["predictions", "embeddings"]:
            if key in result:
                return result[key]
        # If only one key, return its value
        if len(result) == 1:
            return list(result.values())[0]
        raise ValueError(f"Unknown response structure: {result}")
    return result

def _embed_text_driver_serving(text: str):
    return get_embeddings_serving([text])[0]
    
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark import StorageLevel
from pyspark.sql import Row

# ---------- Your provided config ----------
account = "docclassifierstoragecct"
container_raw         = "raw"
container_model       = "docintclassificationmodel"
container_stage       = "stage"
container_redacted    = "redacted"
container_processed   = "docclassifiercontainer"
container_classified  = "classified"

abfss = lambda container, path="": f"abfss://{container}@{account}.dfs.core.windows.net/{path}"

PATH_DOCS_IN     = abfss(container_raw,        "incoming/docs/")      # raw docs (binary) - FYI
PATH_IMAGES_OUT  = abfss(container_stage,      "images/")
PATH_JSON_OUT    = abfss(container_stage,      "extracted_json/")
PATH_CLEANED     = abfss(container_redacted,   "cleaned/")            # PII-clean text (if used)
PATH_EMBED       = abfss(container_stage,      "embeddings/")
PATH_TEXT_OUT    = abfss(container_stage,      "text/")               # <-- input parquet (expects columns incl. 'path','text')
PATH_Mistral     = abfss(container_model,      "Mistral 24B Instruct/")

# Output for classified docs
PATH_CLASSIFIED  = abfss(container_classified, "problem_solution/")

# ---------- Model / classification settings ----------
LOCAL_MODEL_PATH = "/tmp/hf_models/intfloat_multilingual_e5_base"  # quantized model path
MODEL_NAME = LOCAL_MODEL_PATH
SCORE_THRESHOLD = 0.70                          # tune as needed
BATCH_SIZE = 8                                   # per-partition batch for speed
MAX_LENGTH = 512                                 # tokenizer max length per chunk
CHUNK_CHARS = 1500                               # text chunk size in characters

# COMMAND ----------
# ---------- Load reference embeddings ONCE on driver ----------
import numpy as np

def _embed_text_driver_serving(text: str):
    return get_embeddings_serving([text])[0]

emb_problem_drv  = _embed_text_driver_serving("This document describes a problem or issue.")
emb_solution_drv = _embed_text_driver_serving("This document provides a solution or resolution.")

# Broadcast references & settings
bc_problem   = spark.sparkContext.broadcast(np.array(emb_problem_drv))
bc_solution  = spark.sparkContext.broadcast(np.array(emb_solution_drv))
bc_params    = spark.sparkContext.broadcast({
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "chunk_chars": CHUNK_CHARS,
    "threshold": SCORE_THRESHOLD
})

# COMMAND ----------
# ---------- Read input parquet (expects at least 'path' and 'text') ----------
df_in = spark.read.parquet(PATH_TEXT_OUT)

required_cols = {"path", "chunk_text"}

df_in = df_in.filter(F.col("chunk_text").isNotNull())
df_in.persist(StorageLevel.MEMORY_AND_DISK)

# COMMAND ----------
# ---------- Partition-level inference using serving endpoint ----------
def process_partition_serving(rows_iter):
    import numpy as np
    import requests
    import os

        # Use the broadcasted token
    DATABRICKS_TOKEN = bc_token.value
    if not DATABRICKS_TOKEN:
        from pyspark.dbutils import DBUtils
        import IPython
        dbutils = DBUtils(IPython.get_ipython())
        DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

    SERVING_ENDPOINT = "https://adb-3249086852123311.11.azuredatabricks.net/serving-endpoints/multilingual_e5_base_service/invocations"
    params = bc_params.value
    B = int(params["batch_size"])
    CHUNK = int(params["chunk_chars"])
    TH = float(params["threshold"])

    emb_problem = np.array(bc_problem.value)
    emb_solution = np.array(bc_solution.value)

    def chunk_text(t: str, n: int):
        t = t or ""
        if len(t) <= n:
            return [t]
        return [t[i:i+n] for i in range(0, len(t), n)]

    def embed_texts_serving(text_list):
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        embs = []
        for i in range(0, len(text_list), B):
            batch = [f"passage: {t}" for t in text_list[i:i+B]]
            data = {"inputs": batch}
            response = requests.post(SERVING_ENDPOINT, headers=headers, json=data)
            response.raise_for_status()
            embs.extend(response.json())
        return [np.array(e) for e in embs]

    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    results = []
    for row in rows_iter:
        rd = row.asDict(recursive=True)
        text = rd.get("chunk_text")
        if not text or not isinstance(text, str):
            rd["semantic_score"] = 0.0
            rd["classification"] = "Other"
            results.append(Row(**rd))
            continue

        try:
            chunks = chunk_text(text, CHUNK)
            emb_chunks = embed_texts_serving(chunks)

            best_score = 0.0
            for emb in emb_chunks:
                p = cosine_similarity(emb, emb_problem)
                s = cosine_similarity(emb, emb_solution)
                score = (p + s) / 2.0
                if score > best_score:
                    best_score = score

            rd["semantic_score"] = float(round(best_score, 4))
            rd["classification"] = "Problem+Solution" if best_score >= TH else "Other"

        except Exception:
            rd["semantic_score"] = 0.0
            rd["classification"] = "Other"

        results.append(Row(**rd))

    return iter(results)

df_scored = spark.createDataFrame(
    df_in.rdd.mapPartitions(process_partition_serving),
    schema=df_in.schema.add("semantic_score", "double").add("classification", "string")
)

# COMMAND ----------
# ---------- Derive relative folder to preserve source hierarchy ----------
def _rel_folder_from_path(p: str) -> str:
    if not p:
        return ""
    s = p.replace("\\", "/")
    key = "/incoming/docs/"
    if key in s:
        tail = s.split(key, 1)[-1]
        return "/".join(tail.split("/")[:-1])
    return "/".join(s.split("/")[:-1])

extract_rel_udf = F.udf(_rel_folder_from_path, StringType())

df_final = (
    df_scored
    .withColumn("relative_folder", extract_rel_udf(F.col("path")))
)

# COMMAND ----------
# ---------- Write results back to ADLS as Parquet, partitioned by relative_folder ----------
(
    df_final
    .coalesce(200)
    .write
    .mode("overwrite")
    .partitionBy("relative_folder")
    .parquet(PATH_CLASSIFIED)
)

# COMMAND ----------
# ---------- Quick verification ----------
print("Wrote classified outputs to:", PATH_CLASSIFIED)
display(dbutils.fs.ls(PATH_CLASSIFIED))

# ===========================
# WHY IS THIS SUPERSLOW?
# ===========================
# 1. HuggingFace models (AutoModel) are loaded ONCE PER PARTITION, PER EXECUTOR, from disk.
#    - This is very slow if the model is large or if the model path is on DBFS or remote storage.
#    - Model loading is not parallelized and is repeated for every partition.
# 2. No GPU acceleration: Unless your cluster has GPUs and your Spark workers are configured to use them,
#    inference will run on CPU, which is much slower for transformer models.
# 3. Each row (document chunk) is processed sequentially within the partition, and embedding computation is expensive.
# 4. Large batch sizes or large text chunks can cause memory pressure and slow down processing.
# 5. If the number of partitions is high, model load overhead is multiplied.
# 6. If the model is not cached locally on each worker node, it is reloaded from remote storage every time.
# 7. Spark overhead: Spark is not optimized for deep learning inference workloads.
# 8. Writing with .coalesce(200) can cause shuffling and slow down the write phase if the data is not large enough.

# ===========================
# HOW TO MAKE IT FASTER?
# ===========================
# - Use GPU clusters and ensure torch uses CUDA.
#Set .coalesce() to a smaller number (e.g. 50 or 100) → starts writing sooner.

#Increase your Model Serving endpoint replicas (in the UI) → more parallel inference.

#Use async batching to send multiple texts per request.

#Consider writing intermediate checkpoints (e.g. per batch) instead of one giant .write() #at the end.
# - Reduce number of partitions to match number of executors.
# - Pre-cache the model on local disk of each worker.
# - Use a lighter model or quantized model.
# - Increase batch size if memory allows.
# - Consider using Databricks Model Serving or a distributed inference service.
# - For large scale, run inference outside Spark (e.g., batch process with Ray, Dask, or TorchServe).