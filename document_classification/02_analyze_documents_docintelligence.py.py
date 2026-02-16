# Databricks notebook source
import sys
sys.path.append("/Workspace/Users/aswin@eyaswin.onmicrosoft.com")

from utils_common import post_json
from map_partitions_api import map_partitions_api

# COMMAND ----------

import base64, json, os

DI_ENDPOINT = "https://documentsclassifier.cognitiveservices.azure.com/" #dbutils.secrets.get("kv-scope","di_endpoint")  # e.g., https://<res>.cognitiveservices.azure.com
DI_KEY      = ""# dbutils.secrets.get("kv-scope","di_key")
DI_MODEL_ID = "prebuilt-document"  # or your custom model

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
