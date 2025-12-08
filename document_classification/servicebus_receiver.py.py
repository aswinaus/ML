# Databricks notebook source
# Azure Service Bus → Spark Streaming (Python)

This package contains a simple polling-based streaming pipeline using:

- Spark (Databricks or any cluster)
- Azure Service Bus Queue (SBQ)
- Python 3.10

## Files

- servicebus_receiver.py — Polls Service Bus Queue
- spark_streaming_job.py — Spark loop that writes messages to Delta
- requirements.txt — Python dependencies

## Running on Databricks

1. Upload `servicebus_receiver.py` and `spark_streaming_job.py` to DBFS or workspace.
2. Install the dependency:

   `%pip install azure-servicebus==7.11.4`

3. Run spark_streaming_job.py as a notebook or job.



# COMMAND ----------

from azure.servicebus import ServiceBusClient
from typing import List

CONNECTION_STR = "<YOUR SERVICE BUS CONNECTION STRING>"
QUEUE_NAME = "<YOUR QUEUE NAME>"

def read_service_bus_batch(max_batch: int = 50) -> List[str]:
    """
    Reads messages from Azure Service Bus Queue.
    Returns a list of message bodies (decoded strings).
    """

    client = ServiceBusClient.from_connection_string(CONNECTION_STR)

    bodies = []
    with client:
        receiver = client.get_queue_receiver(queue_name=QUEUE_NAME, max_wait_time=3)
        with receiver:
            messages = receiver.receive_messages(max_message_count=max_batch)

            for msg in messages:
                try:
                    body = msg.body
                    if hasattr(body, "decode"):
                        body = body.decode("utf-8")
                    bodies.append(str(body))
                finally:
                    receiver.complete_message(msg)

    return bodies


# COMMAND ----------

# spark streaming

import time
from pyspark.sql import SparkSession, Row
from servicebus_receiver import read_service_bus_batch

spark = SparkSession.builder.appName("ServiceBusToSpark").getOrCreate()

def messages_to_df(messages):
    rows = [Row(body=m) for m in messages]
    return spark.createDataFrame(rows)

OUTPUT_PATH = "/mnt/output/servicebus_data"

while True:
    msgs = read_service_bus_batch()

    if msgs:
        df = messages_to_df(msgs)
        df.write.format("delta").mode("append").save(OUTPUT_PATH)
        print(f"Processed {len(msgs)} messages")
    else:
        print("No messages received")

    time.sleep(2)
