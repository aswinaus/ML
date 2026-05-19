# Databricks notebook source
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
PATH_CLASSIFIED = abfss(container_classified, "problem_solution/")

# Mount if not already mounted
#mount_point = f"/mnt/{container_model}/Mistral 24B Instruct"
#if not any(m.mountPoint == mount_point for m in dbutils.fs.mounts()):
#    dbutils.fs.mount(
#        source=f"abfss://{container_model}@{account}.dfs.core.windows.net/",
#        mount_point=mount_point,
#        extra_configs=configs
#    )

# COMMAND ----------

# ==========================================================
# 05_threshold_sweep_tax_classifier_spark.py
# ==========================================================
# Purpose:
#   - Use Databricks Model Serving endpoint (fine-tuned model)
#   - Run distributed inference across Spark DataFrame (df_in)
#   - Incrementally classify and write results to destination JSON
# ==========================================================

import requests, numpy as np, torch
import torch.nn.functional as F
from pyspark.sql import Row, functions as Fp, types as T

# -----------------------------
# CONFIGURATION
# -----------------------------
SERVING_ENDPOINT = "https://adb-3249086852123311.11.azuredatabricks.net/serving-endpoints/finetuned_model_inference/invocations"
DATABRICKS_TOKEN = ""
LABELS = ["problem", "solution", "topic", "year"]
DEST_PATH = PATH_CLASSIFIED + "classified_results.json"

# -----------------------------
# Load sample dataset
# -----------------------------
df_in = spark.read.parquet(PATH_TEXT_OUT).filter(Fp.col("chunk_text").isNotNull())
df_sample = df_in.limit(200)  # small subset for validation sweep

print(f"Loaded {df_sample.count()} text chunks for threshold sweep.")

# -----------------------------
# Function for distributed inference
# -----------------------------
def call_model_serving(batch_texts):
    """Call Databricks Model Serving endpoint and return probabilities."""
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
    data = {"inputs": batch_texts}
    r = requests.post(SERVING_ENDPOINT, headers=headers, json=data)
    r.raise_for_status()
    result = r.json()

    if isinstance(result, dict):
        preds = result.get("predictions") or result.get("data") or result
    else:
        preds = result

    outputs = []
    for p in preds:
        # Normalize response format
        if isinstance(p, dict) and "logits" in p:
            logits = np.array(p["logits"])
        elif isinstance(p, list):
            logits = np.array(p)
        elif all(isinstance(x, (float, int)) for x in preds):
            logits = np.array(preds)
        else:
            raise ValueError(f"Unexpected format: {preds}")

        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        outputs.append(probs)
    return outputs

def process_partition_and_write(rows_iter):
    """Map Spark partition to model inference results and write incrementally to JSON."""
    import numpy as np, json
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
    rows = list(rows_iter)
    texts = [r["chunk_text"] for r in rows if r.get("chunk_text")]

    if not texts:
        return iter([])

    try:
        probs = call_model_serving(texts)
    except Exception as e:
        print(f"⚠️ Inference failed: {e}")
        return iter([])

    results = []
    for row, prob in zip(rows, probs):
        rdict = row.asDict()
        for lbl, val in zip(LABELS, prob):
            rdict[f"prob_{lbl}"] = float(val)
        rdict["pred_label"] = LABELS[int(np.argmax(prob))]
        rdict["pred_confidence"] = float(np.max(prob))
        results.append(rdict)

    # Write results incrementally to destination JSON
    # Use append mode for each partition
    json_lines = [json.dumps(r) for r in results]
    # Use a unique file per partition to avoid write conflicts
    import uuid
    part_file = f"{DEST_PATH.rstrip('.json')}_part_{uuid.uuid4().hex}.json"
    with open("/dbfs" + part_file, "a") as f:
        for line in json_lines:
            f.write(line + "\n")
    return iter([])

# -----------------------------
# Run distributed inference and incremental write
# -----------------------------
df_sample.rdd.foreachPartition(process_partition_and_write)

print(f"✅ Incremental inference and write completed. Results written to {DEST_PATH} (one file per partition).")