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

spark.conf.set("spark.sql.shuffle.partitions", "400")
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
container_processed="docclassifiercontainer"
abfss = lambda container, path="": f"abfss://{container}@{account}.dfs.core.windows.net/{path}"


import base64, json, os

DI_ENDPOINT = "https://documentsclassifier.cognitiveservices.azure.com/" #dbutils.secrets.get("kv-scope","di_endpoint")  # e.g., https://<res>.cognitiveservices.azure.com
DI_KEY      = ""# dbutils.secrets.get("kv-scope","di_key")
DI_MODEL_ID = "prebuilt-document"  # or your custom model

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
    if path.endswith((".pdf", ".docx", ".xlsx", ".pptx")):
        try:
            text = analyze_doc(content)
            return text, "docintelligence"
        except Exception as e:
            return f"[ERROR calling Document Intelligence API: {e}]", "docintelligence_error"
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
#defining the schema for the dataframe
schema = T.StructType([
    T.StructField("path", T.StringType()),
    T.StructField("text", T.StringType()),
    T.StructField("source_type", T.StringType())
])
print("Number of Partitions ",files.rdd.getNumPartitions())
rdd = files.rdd.mapPartitions(map_partition)
# create dataframe with the schema
# The schema is used to ensure that the DataFrame(JSON) columns have the correct names and types.
df_text = spark.createDataFrame(rdd, schema)
# repartition DataFrame before writing (e.g., 200 partitions)
# if partition is too small right of task scheduling overhead increases meaning spark has to launch 200 tasks for very tiny chunks of data which will slow down the job. and if its too large then out-of-memory(OOM) may occur. Typiacally between 100 and 200 partitions is a good starting point.
# example 100GB data then 100GB/200=0.5GB per partition which is 512 MB risk of OOM.
df_text = df_text.repartition(16)
# ===========================================================
# 4) Write results to ADLS
# ===========================================================
#This ensures the folder_name column is available for partitioning when writing the output and writes the output to the folder by the file name.
df_text = df_text.withColumn(
    "folder_name",
    regexp_extract("path", r"/([^/]+)\.[^/.]+$", 1)
)

(df_text.write
 .mode("overwrite")
 .partitionBy("folder_name")
 .parquet(PATH_TEXT_OUT)) # For PII redaction and embedding. Later conver to JSON format as needed.

# Collect the paths to process on the driver
paths_to_move = df_text.select("path").distinct().collect()

# This order guarantees that only files which have been processed and written are moved to the processed container.
for row in paths_to_move:
    src_path = row["path"]
    # Convert abfss:// to dbfs:/mnt/ if you have mounted, or use abfss:// directly if not
    # Here, we replace the container and folder in the path string
    dst_path = src_path.replace(
        f"/{container_raw}/incoming/docs/",
        f"/{container_processed}/processed/docs/"
    )
    dbutils.fs.mv(src_path, dst_path)

# COMMAND ----------

headers = {"Ocp-Apim-Subscription-Key": DI_KEY, "Content-Type": "application/json"}

docs = (spark.read.format("binaryFile").load(PATH_DOCS_IN)
        .select("path","content","length"))

def analyze_doc(row):
    url = f"{DI_ENDPOINT}/formrecognizer/documentModels/{DI_MODEL_ID}:analyze?api-version=2024-02-29-preview"
    payload = {"base64Source": base64.b64encode(bytes(row["content"])).decode("utf-8")}
    result = post_json(url, headers, payload)
    return {"path": row["path"], "json": json.dumps(result)}

rdd = docs.rdd.mapPartitions(lambda it: map_partitions_api(
    ({"path": r["path"], "content": bytes(r["content"])} for r in it),
    analyze_doc
))

out = spark.createDataFrame(rdd)
(out.write.mode("overwrite").json(PATH_JSON_OUT + "docintelligence/"))