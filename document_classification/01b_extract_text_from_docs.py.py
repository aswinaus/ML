# Databricks notebook source
# MAGIC %pip install pypdf
# MAGIC %pip install extract-msg

# COMMAND ----------

import sys
sys.path.append("/Workspace/Users/aswin@eyaswin.onmicrosoft.com")

from utils_common import post_json
from map_partitions_api import map_partitions_api


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
from pypdf import PdfReader
import extract_msg   # pip install extract-msg
import base64, json, os

# Initialize SparkSession for Databricks
# In Databricks, the SparkSession is typically pre-configured,
# so you don't need to explicitly set Hadoop Azure dependencies.
spark = (
    SparkSession.builder
    .appName("Document-Classification")
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

spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

#todo
#Large files / memory**: For very large PDFs or huge images you may want streaming approaches. The above copies files locally before processing; adapt if you have memory concerns.
#todo
#**Idempotency**: The code stores SHA256 for each saved image. You can use that to skip reprocessing duplicates.

#account = dbutils.secrets.get("kv-scope","storage_account_name")
account="docclassifierstoragecct"
container_raw   = "raw"
container_stage = "stage"
container_redacted = "redacted"

abfss = lambda container, path="": f"abfss://{container}@{account}.dfs.core.windows.net/{path}"

# system folders
PATH_DOCS_IN   = abfss(container_raw,   "incoming/docs/")     # .pdf/.docx/.xlsx will be stored here
PATH_IMAGES_OUT= abfss(container_stage, "images/")            # extracted images go here
PATH_JSON_OUT  = abfss(container_stage, "extracted_json/")    # DI/Vision JSON here
PATH_CLEANED   = abfss(container_stage, "cleaned/")           # PII-clean text
PATH_EMBED     = abfss(container_stage, "embeddings/")        # parquet with embeddings
PATH_TEXT_OUT = abfss(container_stage, "text/")


from pyspark.sql import functions as F, types as T
import io, json, csv, email
import extract_msg   # pip install extract-msg

PATH_TEXT_OUT = abfss(container_stage, "text/")

# ===========================================================
# 1) Extractors for unsupported formats
# ===========================================================

def extract_text_from_csv(content: bytes) -> str:
    """Return structured JSON string with rows and columns preserved."""
    try:
        with io.StringIO(content.decode(errors="ignore")) as buf:
            reader = csv.DictReader(buf)
            rows = [row for row in reader]
            return json.dumps(rows, ensure_ascii=False)
    except Exception as e:
        return f"[ERROR extracting CSV: {e}]"

def extract_text_from_eml(content: bytes) -> str:
    try:
        msg = email.message_from_bytes(content)
        parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    parts.append(part.get_payload(decode=True).decode(errors="ignore"))
        else:
            parts.append(msg.get_payload(decode=True).decode(errors="ignore"))
        return "\n".join(parts)
    except Exception as e:
        return f"[ERROR extracting EML: {e}]"

def extract_text_from_msg(content: bytes) -> str:
    try:
        with io.BytesIO(content) as buf:
            msg = extract_msg.Message(buf)
            subject = msg.subject or ""
            body = msg.body or ""
            return "\n".join([subject, body])
    except Exception as e:
        return f"[ERROR extracting MSG: {e}]"

# ===========================================================
# 2) Dispatcher
# ===========================================================

def extract_text(row):
    #ensures to access Spark Row fields correctly using attribute access
    path = row.path.lower()
    content = row.content

    if path.endswith(".csv"):
        return extract_text_from_csv(content), "csv"
    if path.endswith(".eml"):
        return extract_text_from_eml(content), "eml"
    if path.endswith(".msg"):
        return extract_text_from_msg(content), "msg"

    return "[UNSUPPORTED FILE TYPE FOR TASK 1B]", "unsupported"

# ===========================================================
# 3) Spark Job — iterate over files and extract text
# ===========================================================

files = (spark.read.format("binaryFile")
         .load(PATH_DOCS_IN)   # same input as Task 1
         .select("path","content"))

def map_partition(it):
    for r in it:
        text, stype = extract_text(r)
        #text, stype = extract_text(bytes(r["content"]))
        yield {"path": r.path, "text": text, "source_type": stype}

schema = T.StructType([
    T.StructField("path", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("source_type", T.StringType())
])

rdd = files.rdd.mapPartitions(map_partition)
df_text = spark.createDataFrame(rdd, schema)

# ===========================================================
# 4) Write results to ADLS
# ===========================================================
#This ensures the folder_name column is available for partitioning when writing the output and writes the output to the folder by the file name.
df_text = df_text.withColumn(
    "output_",
    regexp_extract("path", r"/([^/]+)\.[^/.]+$", 1)
)

(df_text.write
 .mode("overwrite")
 .partitionBy("folder_name")
 .json(PATH_TEXT_OUT))
