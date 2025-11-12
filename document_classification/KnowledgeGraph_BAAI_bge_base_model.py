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
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
tokenizer.save_pretrained(local_dbfs_path)

# Download and save the model
model = AutoModel.from_pretrained('BAAI/bge-m3')
model.save_pretrained(local_dbfs_path)

# List the contents of the local DBFS path
display(dbutils.fs.ls(local_dbfs_path))

# COMMAND ----------

# Install torchvision
%pip install torch torchvision


# COMMAND ----------

display(dbutils.fs.ls("/tmp/hf_models/intfloat_multilingual_e5_base"))

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
  "fs.azure.account.oauth2.client.id": "",
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
PATH_GRAPH_OUT = abfss(container_classified, "graph/")

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
# 05_graph_communities_bge_m3.py
# ==========================================================
# Goal:
#  - Embed documents with BAAI/bge-m3 (Spark mapPartitions; no pandas)
#  - Score Problem/Solution semantics per doc (anchor-based cosine)
#  - Build KNN graph with MLlib LSH (approx neighbors) + exact cosine
#  - Run GraphFrames Label Propagation -> community ids
#  - Classify each community: Problem / Solution / Both / Other
#  - Write results back to ADLS (Delta/Parquet)
# ==========================================================

# COMMAND ----------
# Install deps once per cluster (if needed)
# %pip install transformers==4.44.2 torch==2.4.0 graphframes==0.8.3
# dbutils.library.restartPython()

# COMMAND ----------
from pyspark.sql import functions as F, types as T
from pyspark.sql import Window
from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT
from graphframes import GraphFrame
from pyspark import StorageLevel

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ----------------- CONFIG -----------------
account = "docclassifierstoragecct"
container_stage = "stage"
container_classified = "classified"

abfss = lambda c, p="": f"abfss://{c}@{account}.dfs.core.windows.net/{p}"

# Input: cleaned/parsed text parquet with at least: path, content_text (or text)
PATH_TEXT_IN   = abfss(container_stage, "text/")
# Output: community classification per doc
PATH_COMMUNITIES_OUT = abfss(container_classified, "graph_communities/")

MODEL_NAME = "BAAI/bge-m3"
MAX_LENGTH = 512                     # tokenizer max length
BATCH_SIZE = 16                      # per-partition infer batch
K_NEIGHBORS = 20                     # KNN per node (after prune)
COSINE_THR  = 0.60                   # edge kept if cosine >= threshold
LP_MAX_ITER = 20                     # label propagation iterations

# Anchor thresholds to mark doc-level signals
PROBLEM_THR  = 0.65
SOLUTION_THR = 0.65

# ---------- Load docs ----------
df = spark.read.parquet(PATH_TEXT_IN)
# Normalize to a 'text' column
if "text" not in df.columns:
    candidates = ["content_text", "content", "redacted_text", "body","chunk_text"]
    found = next((c for c in candidates if c in df.columns), None)
    if not found:
        raise ValueError(f"No text-like column found. Available: {df.columns}")
    df = df.withColumnRenamed(found, "text")

required = {"path", "text"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required cols: {missing}")

df = df.filter(F.col("text").isNotNull() & (F.length("text") > 0)).select("path", "text")
df.persist(StorageLevel.MEMORY_AND_DISK)
print("Docs:", df.count())

# ---------- Broadcast anchors ----------
# We’ll compute anchor embeddings on the driver then broadcast.
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME); _model.eval()

def _embed_one(s: str):
    with torch.no_grad():
        inp = _tokenizer([f"passage: {s}"], return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        out = _model(**inp)
        emb = out.last_hidden_state.mean(dim=1)[0]
        return emb / emb.norm()  # L2-normalize

emb_problem_drv  = _embed_one("This document describes a problem or issue to be solved.")
emb_solution_drv = _embed_one("This document provides a solution, resolution, or steps to solve a problem.")

bc_problem  = spark.sparkContext.broadcast(emb_problem_drv.numpy())
bc_solution = spark.sparkContext.broadcast(emb_solution_drv.numpy())
bc_model    = spark.sparkContext.broadcast(MODEL_NAME)
bc_params   = spark.sparkContext.broadcast({"BATCH_SIZE": BATCH_SIZE, "MAX_LENGTH": MAX_LENGTH})

# ---------- Partition inference (no pandas) ----------
@F.udf(returnType=T.ArrayType(T.FloatType()))
def to_array(v):
    return [float(x) for x in v]

def embed_partition(iter_rows):
    # lazy init per executor
    import torch
    from transformers import AutoTokenizer, AutoModel
    model_name = bc_model.value
    B = bc_params.value["BATCH_SIZE"]; MAXLEN = bc_params.value["MAX_LENGTH"]
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name); mdl.eval()

    problem = torch.tensor(bc_problem.value)
    solution = torch.tensor(bc_solution.value)
    cos = torch.nn.functional.cosine_similarity

    for row in iter_rows:
        r = row.asDict() if hasattr(row, "asDict") else row
        text = r["text"] or ""
        # chunk by ~1500 chars to avoid truncation loss
        chunks = [text[i:i+1500] for i in range(0, len(text), 1500)] or [text]
        embs = []
        with torch.no_grad():
            for i in range(0, len(chunks), B):
                batch = [f"passage: {t}" for t in chunks[i:i+B]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
                out = mdl(**enc).last_hidden_state.mean(dim=1)  # [b, d]
                # L2 normalize each
                out = out / out.norm(dim=1, keepdim=True)
                for e in out:
                    embs.append(e)

        # doc-level embedding = mean over chunk embeddings
        doc_emb = torch.stack(embs, dim=0).mean(dim=0)
        doc_emb = doc_emb / doc_emb.norm()

        # anchor-based signals
        prob = cos(doc_emb, problem, dim=0).item()
        sol  = cos(doc_emb, solution, dim=0).item()

        yield (r["path"], doc_emb.numpy().astype(np.float32).tolist(), float(prob), float(sol))

embed_schema = T.StructType([
    T.StructField("path", T.StringType()),
    T.StructField("embedding_arr", T.ArrayType(T.FloatType())),
    T.StructField("problem_score", T.FloatType()),
    T.StructField("solution_score", T.FloatType()),
])

df_emb = spark.createDataFrame(df.rdd.mapPartitions(embed_partition), embed_schema)

# Convert to ML vector & normalize (for LSH / distances)
to_vec = F.udf(lambda a: Vectors.dense(a), VectorUDT())
df_emb = df_emb.withColumn("features_raw", to_vec("embedding_arr"))
normalizer = Normalizer(inputCol="features_raw", outputCol="features", p=2.0)
df_emb = normalizer.transform(df_emb).drop("features_raw")

# ---------- Build KNN edges via LSH (approx) ----------
# Uses MLlib's LSH (Locality Sensitive Hashing) to find approximate nearest neighbors for each # document based on their embeddings.
# Use Euclidean on L2-normalized vectors -> monotonic with cosine.
lsh = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=1.8, numHashTables=4)
lsh_model = lsh.fit(df_emb)

# Approx neighbors per point (cap edges by K and cosine threshold)
# Note: approxNearestNeighbors returns nearest by Euclidean distance.
# We'll compute cosine from normalized vectors: cos = 1 - dist^2/2 (approx) or do exact dot product.
ann = lsh_model.approxSimilarityJoin(df_emb, df_emb, 2.0, distCol="euclid_dist") \
               .select(
                   F.col("datasetA.path").alias("src"),
                   F.col("datasetB.path").alias("dst"),
                   "euclid_dist",
                   F.col("datasetA.features").alias("src_vec"),
                   F.col("datasetB.features").alias("dst_vec")
               ) \
               .filter(F.col("src") < F.col("dst"))  # undirected, drop self & dup pairs

# exact cosine from normalized vectors: dot(src, dst)
@F.udf("double")
def dot_cos(u, v):
    # u, v are DenseVectors (already L2-normalized)
    return float(np.dot(u, v))

edges_scored = ann.withColumn("cosine", dot_cos("src_vec","dst_vec")) \
                  .filter(F.col("cosine") >= F.lit(COSINE_THR)) \
                  .select("src", "dst", "cosine")

# Keep top-K per src (and per dst) to sparsify graph
w_src = Window.partitionBy("src").orderBy(F.desc("cosine"))
w_dst = Window.partitionBy("dst").orderBy(F.desc("cosine"))
edges_topk = (edges_scored
              .withColumn("rk_s", F.row_number().over(w_src))
              .withColumn("rk_d", F.row_number().over(w_dst))
              .filter((F.col("rk_s") <= K_NEIGHBORS) | (F.col("rk_d") <= K_NEIGHBORS))
              .select("src","dst","cosine"))

# ---------- Build GraphFrame ----------
# Vertices hold document signals
verts = (df_emb
         .select(
             F.col("path").alias("id"),
             "problem_score","solution_score"
         ))

g = GraphFrame(verts, edges_topk)

# ---------- Community detection ----------
# Label Propagation (unsupervised)
lpa = g.labelPropagation(maxIter=LP_MAX_ITER)
verts_lpa = lpa.withColumnRenamed("label", "community_id")

# ---------- Classify each community ----------
# Community-level aggregate signals
comm_agg = (verts_lpa.groupBy("community_id")
           .agg(
               F.avg("problem_score").alias("comm_problem_avg"),
               F.avg("solution_score").alias("comm_solution_avg"),
               F.count("*").alias("community_size"))
           )

# Heuristic classification per community
def comm_label(p, s):
    if p >= PROBLEM_THR and s >= SOLUTION_THR: return "Both"
    if p >= PROBLEM_THR: return "Problem"
    if s >= SOLUTION_THR: return "Solution"
    return "Other"

comm_label_udf = F.udf(comm_label, T.StringType())
comm_final = comm_agg.withColumn("community_label", comm_label_udf("comm_problem_avg","comm_solution_avg"))

# Join back to docs
df_result = (verts_lpa.join(comm_final, on="community_id", how="left")
             .select("id","problem_score","solution_score","community_id","community_label","community_size")
             .withColumnRenamed("id","path"))

# ---------- (Optional) Per-doc label (independent) ----------
def doc_label(p, s):
    if p >= PROBLEM_THR and s >= SOLUTION_THR: return "Both"
    if p >= PROBLEM_THR: return "Problem"
    if s >= SOLUTION_THR: return "Solution"
    return "Other"
doc_label_udf = F.udf(doc_label, T.StringType())
df_result = df_result.withColumn("doc_label", doc_label_udf("problem_score","solution_score"))

# ---------- Write results ----------
(df_result.write
   .mode("overwrite")
   .format("delta")
   .save(PATH_COMMUNITIES_OUT))


# Convert Delta to JSON for Azure AI Search ingestion
import json

df_json_ready = (
    df_result
    .withColumnRenamed("path", "id")   # Azure Search key field
    .select(
        F.col("id"),
        F.col("community_id"),
        F.col("community_label"),
        F.col("doc_label"),
        F.col("problem_score"),
        F.col("solution_score"),
        F.col("community_size")
    )
)

# Each record becomes a JSON object
PATH_GRAPH_OUT = f"{PATH_COMMUNITIES_OUT}/json_ready"
(df_json_ready
    .coalesce(1)   # optional: single JSON file for upload
    .write
    .mode("overwrite")
    .format("json")
    .save(PATH_GRAPH_OUT)
)

print(f"JSON export ready at: {PATH_GRAPH_OUT}")


print("Wrote community classifications to:", PATH_COMMUNITIES_OUT)
display(df_result.limit(10))
