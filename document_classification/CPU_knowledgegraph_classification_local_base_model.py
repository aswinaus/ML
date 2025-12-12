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
# CPU Graph + Classification Task
# ==========================================================
from pyspark.sql import functions as F, types as T, Window
from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT
from graphframes import GraphFrame
from pyspark.storagelevel import StorageLevel
import numpy as np

account = "eymydocsmladls"
container_stage = "stage"
container_classified = "classified"
abfss = lambda c, p="": f"abfss://{c}@{account}.dfs.core.windows.net/{p}"

PATH_EMB_IN = abfss(container_stage, "embeddings/")
PATH_COMMUNITIES_OUT = abfss(container_classified, "graph_communities/")

# --- Load embeddings ---
df_emb = spark.read.parquet(PATH_EMB_IN)
to_vec = F.udf(lambda a: Vectors.dense(a), VectorUDT())
df_emb = df_emb.withColumn("features", to_vec("embedding_arr"))
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=2.0)
df_emb = normalizer.transform(df_emb).drop("features")

# --- Build KNN edges ---
lsh = BucketedRandomProjectionLSH(inputCol="features_norm", outputCol="hashes", bucketLength=1.8)
lsh_model = lsh.fit(df_emb)
ann = lsh_model.approxSimilarityJoin(df_emb, df_emb, 2.0, distCol="dist").select(
    F.col("datasetA.path").alias("src"),
    F.col("datasetB.path").alias("dst"),
    "dist"
).filter(F.col("src") < F.col("dst"))

@F.udf("double")
def dot_cos(u, v):
    return float(np.dot(u, v))
edges = ann.withColumn("cosine", 1 - (F.col("dist")**2)/2).filter(F.col("cosine") >= 0.65)

verts = df_emb.select(F.col("path").alias("id"))
g = GraphFrame(verts, edges.select("src", "dst", "cosine"))

spark.conf.set("spark.sql.shuffle.partitions", "128")       # match total CPU parallelism (2×64)
spark.conf.set("spark.sql.adaptive.enabled", "true")        # let Spark auto-optimize joins
spark.conf.set("spark.sql.shuffle.partitions", "256")
#spark.conf.set("spark.dynamicAllocation.enabled", "true")

spark.sparkContext.setCheckpointDir("/tmp/checkpoints")

verts.persist(StorageLevel.MEMORY_AND_DISK)
edges.persist(StorageLevel.MEMORY_AND_DISK)
lpa = g.labelPropagation(maxIter=3)
# Write with smaller files, parallelized
(lpa
 .repartition(16)
 .write
 .mode("overwrite")
 .format("json")
 .option("maxRecordsPerFile", 10000)
 .save(PATH_COMMUNITIES_OUT))

print(f"Graph + community results written to {PATH_COMMUNITIES_OUT}")
