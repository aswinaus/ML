# Databricks notebook source
# Databricks Resource Cleanup Script
# ---------------------------------
# Frees up memory, cached data, temp files, old checkpoints, and unused clusters.

import requests
import os
from pyspark.sql import SparkSession

# ─────────────────────────────────────────────
# CONFIGURATION — update these
# ─────────────────────────────────────────────
DOMAIN = "https://adb-3249086852123311.11.azuredatabricks.net"   # e.g. https://adb-1234567890.11.azuredatabricks.net
TOKEN = dbutils.secrets.get("databricks", "token")  # Or set via os.environ
CHECKPOINT_DIRS = ["dbfs:/tmp/", "dbfs:/mnt/checkpoints/"]
DELTA_TABLES = ["my_table_1", "my_table_2"]  # optional: list of Delta tables to vacuum
AUTO_TERMINATE_IDLE_MINUTES = 20
# ─────────────────────────────────────────────

spark = SparkSession.builder.getOrCreate()

print("🧹 Starting Databricks cleanup...")

# 1️⃣ Clear Spark cache
print("→ Clearing Spark cache...")
spark.catalog.clearCache()

# 2️⃣ Remove temporary and checkpoint files
for path in CHECKPOINT_DIRS:
    try:
        print(f"→ Removing temp/checkpoint directory: {path}")
        dbutils.fs.rm(path, recurse=True)
    except Exception as e:
        print(f"⚠️ Could not remove {path}: {e}")

# 3️⃣ Optimize and vacuum Delta tables
for table in DELTA_TABLES:
    try:
        print(f"→ Optimizing {table}...")
        spark.sql(f"OPTIMIZE {table}")
        print(f"→ Vacuuming {table} (retaining 7 days)...")
        spark.sql(f"VACUUM {table} RETAIN 168 HOURS")
    except Exception as e:
        print(f"⚠️ Could not optimize/vacuum {table}: {e}")

# 4️⃣ Terminate idle clusters
print("→ Checking for active clusters...")
headers = {"Authorization": f"Bearer {TOKEN}"}

try:
    clusters_resp = requests.get(f"{DOMAIN}/api/2.0/clusters/list", headers=headers)
    clusters_resp.raise_for_status()
    clusters = clusters_resp.json().get("clusters", [])

    for c in clusters:
        if c["state"] == "RUNNING":
            cid = c["cluster_id"]
            cname = c["cluster_name"]
            print(f"→ Terminating cluster: {cname} ({cid})")
            resp = requests.post(f"{DOMAIN}/api/2.0/clusters/delete", headers=headers, json={"cluster_id": cid})
            if resp.status_code == 200:
                print(f"✅ Terminated {cname}")
            else:
                print(f"⚠️ Failed to terminate {cname}: {resp.text}")
except Exception as e:
    print(f"⚠️ Error checking clusters: {e}")

print("✨ Cleanup complete.")
