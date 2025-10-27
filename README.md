# Maching Learning
Machine Learning experiment tracking, model checkpointing


**Core conceptual difference between how embedding similarity models (like your original intfloat/multilingual-e5-base) and fine-tuned classification models (like finetuned_model_inference) work.**

Embedding + Cosine Similarity Logic (What we had)
** Goal: measure semantic closeness between any two pieces of text, without explicit labels.**

🔹 Mechanism

The model (e.g., intfloat/multilingual-e5-base) converts text into a high-dimensional vector → an embedding (e.g., 768-D float vector).


"tax problem about dividends" → [0.11, -0.02, 0.33, ..., 0.87]

We store or compute embeddings for reference prompts (“problem”, “solution”, etc.).

For a new text, compute its embedding, then take cosine similarity with each reference:
sim = dot(a,b) / (||a|| * ||b||)

Values range from -1 (opposite meaning) to 1 (same meaning).

The class with the highest similarity above a threshold (e.g., 0.8) is chosen.

🔹 Example
Label	Reference Text	Cosine Similarity

problem	"This describes a tax problem"	0.91

solution	"This gives a solution"	0.60

topic	"This discusses participation exemption"	0.55

year	"This refers to tax year"	0.12


→ Classified as Tax Problem
**Characteristics**
Works without supervision — no need for labeled data.

You can compare any text to any other text (universal).

But classification is approximate; it relies on semantic proximity, not learned decision boundaries.

Sensitive to the prompt wording of reference texts.

2️⃣ Fine-Tuned Classifier Logic (What we have now after supervised fine tuning on labelled dataset)


**Goal:** predict explicit class probabilities learned from labeled examples (problem, solution, topic, year).

**
🔹 Mechanism
**
Start from a pretrained model (like E5) and add a classification head (a small linear layer mapping embeddings → logits for 4 classes).


Fine-tune on labeled pairs:

"This describes a tax problem …" → label=problem

"This provides a tax solution …" → label=solution

After training, the model directly outputs class logits — one scalar per label:


logits = [-0.8, 1.3, 0.2, -0.7]


Apply softmax to convert logits → probabilities:


probs = [0.10, 0.68, 0.16, 0.06]

The class with the highest probability is the predicted label.

🔹 Example

Label	Probability

problem	0.10

solution	0.68

topic	0.16

year	0.06

→ Classified as Tax Solution

🔹 Characteristics

Supervised — learns from labeled examples.

Directly optimized to minimize misclassification.

Learns nonlinear decision boundaries between classes.

Doesn’t compute vector similarity — it outputs class scores.

**How They Differ Mathematically**
Concept	Embedding Similarity	Fine-Tuned Classifier
Output	Vector (e.g., 768-D)	Logits (4-D for 4 classes)
Metric	Cosine similarity	Softmax + argmax
Training	Unsupervised	Supervised (fine-tuned)
Interpretability	General similarity	Categorical probability
Thresholding	Manual (e.g., 0.8)	Confidence-based (prob > 0.5)
Use cases	Semantic search, clustering	Explicit classification
Speed	Needs multiple reference comparisons	One forward pass

**Intuitive Analogy**
Analogy	Embedding Model	Classifier Model
How it behaves	Measures “how close” two meanings are in general space	Decides “which bucket this text belongs to”
Example	“Are these two texts semantically alike?”	“Is this text a problem, solution, topic, or year?”
Mental model	Semantic map of the world	Decision boundary separating categories

**Practical Impact in our Case**
Aspect	Old (Cosine)	New (Classifier)
Endpoint	multilingual_e5_base_service	finetuned_model_inference
Output shape	768-dim embeddings	4-class logits
Evaluation	Similarity threshold	Softmax probability
Code	cosine_similarity()	softmax → argmax()
Use case	Clustering, retrieval	Direct labeling in pipeline

Notebook : https://github.com/aswinaus/ML/blob/main/ADLS_Databricks_ApacheSpark.ipynb
<img width="831" height="417" alt="image" src="https://github.com/user-attachments/assets/f3fa2972-b16e-45f7-990a-0b858a9bbda7" />

The Classification Model Training explicitly uses distributed XGBoost , leveraging multiple nodes in the cluster for scalable training.
Distributed XGBoost training in Databricks can performed using PySpark with parameters like num_workers to specify parallelism.
This enables efficient handling of large sharepoint data and faster model training times.

Notebook : ADLS_AzureSynapse_ApacheSpark.ipynb

<img width="929" height="704" alt="image" src="https://github.com/user-attachments/assets/b357d7e6-25df-45bd-a438-621f1be6ccf2" />



- Azure Blob Storage is the underlying object storage service.
- ADLS Gen2 extends Blob Storage with hierarchical namespace, fine-grained security, optimizing big data analytics.
- Azure Synapse Analytics provides a unified analytics platform combining big data (Spark Pools) and data warehousing (SQL Pools).
- Apache Spark running inside Synapse or Databricks uses Hadoop Azure filesystem connectors to read and write data from ADLS/Blob storage.
- Hadoop components (like YARN as resource manager in HDInsight) enable cluster resource management for Spark jobs.

**Key Challenges at Scale (Millions of Lines)**

| Challenge                      | What It Means                                           | How to Solve It                                             |
| ------------------------------ | ------------------------------------------------------- | ----------------------------------------------------------- |
| **LLM API Rate Limits**        | You can't call the API millions of times per minute     | Use batching, backoff, and parallelization                  |
| **Token Limits per Prompt**    | Each LLM (e.g., GPT-4) has a max token limit (\~128k)   | Limit number/size of docs per batch                         |
| **Memory & Collection Limits** | `.collect()` pulls all data to driver (can crash Spark) | Avoid `.collect()`; use `mapPartitions`, `foreachPartition` |
| **Long Job Runtime**           | Serial execution would be slow for millions of rows     | Use Spark’s distributed processing with UDFs or partitions  |
| **Error Handling**             | LLM API calls can fail (timeouts, overuse)              | Add retries, logging, and failover logic                    |

---------------------------------------------------------------------------------------------------------------------------------------------------------------

**Why Use Spark DataFrames vs Just Python.**

**Scalability**
| Feature                          | Spark DataFrame               | Python Only                      |
| -------------------------------- | ----------------------------- | -------------------------------- |
| Multi-core/multi-node processing | ✅ Yes (distributed computing) | ❌ No (limited to single machine) |
| Handles 10M+ documents?          | ✅ Easily                      | ⚠️ Risk of OOM / slowness        |
| Retry/fault tolerance            | ✅ Built-in                    | ❌ Must handle manually           |

**Data Integration and Pipelines**
| Feature                          | Spark DataFrame               | Python Only                      |
| -------------------------------- | ----------------------------- | -------------------------------- |
| Multi-core/multi-node processing | ✅ Yes (distributed computing) | ❌ No (limited to single machine) |
| Handles 10M+ documents?          | ✅ Easily                      | ⚠️ Risk of OOM / slowness        |
| Retry/fault tolerance            | ✅ Built-in                    | ❌ Must handle manually           |


**Core Advantages of Apache Spark (Beyond Just Distribution)**
| Feature                                 | Why It Matters                                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Distributed Computing**             | Yes, it's the biggest one. Enables processing of **gigabytes to petabytes** of data across a cluster.        |
| **Unified Data Processing Engine**   | Supports **batch**, **streaming**, **SQL**, **ML**, **graph**, and **structured data** — all in one engine.  |
| **In-memory Processing**             | Faster than MapReduce because it keeps intermediate data in memory (vs writing to disk).                     |
| **Optimized for Big Data Workflows** | Built-in fault tolerance, DAG optimization, task scheduling, and caching.                                    |
| **Rich SQL Support**                 | Spark SQL lets you run **SQL queries on big data**, with full ANSI compliance and integration with BI tools. |
| **Easy Integration**                 | Reads/writes from: Azure Data Lake Storage                                                                                           |


**With Spark:**
1) Load all file paths into a DataFrame
2) Distribute text extraction + cleaning
3) Run mapPartitions to batch + classify via LLM
4) Store structured output into Delta Lake or SQL
5) Entire pipeline is parallel, fault-tolerant, scalable

| Databricks                           |
| ------------------------------------ |
| ✅ Fully managed and autoscaling     |
| ✅ Integrated with ADLS, Key Vault    |
| ✅ Built-in job scheduler & alerts    |
| ✅ Built-in lineage, logs, dashboards |
| ✅ Collaborative notebooks + repos    |

**Strategy to handle documents with images**

Reads .docx, .pdf, .xlsx files from ADLS.
Extracts embedded images
Saves images to ADLS
Optionally preprocesses the images (resize / convert / compress)
Send each image to Azure AI Vision via REST
Store the JSON results into a Delta table.
or Azure SQL.

**Cost Optimization Tips**
Area	Suggestions
Azure Document Intelligence	Batch process large documents to reduce API calls; avoid unneeded models (tables, signatures, etc.)
Azure OpenAI summarization / classification	Limit token size by summarizing first; or use extractive techniques + LLM on demand
Embeddings	Use OpenAI for best quality, or fallback to Hugging Face multilingual embeddings (e.g., sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
Vector storage	Use hybrid indexes (keyword + vector) in Azure Search to reduce recall latency
GPU cost	Run summarization/classification in Databricks jobs with low-cost instances, and cache intermediate outputs (e.g., summaries)

**Based on summarizing a 200,000-word document via chunking**

| Model             | Max Input Tokens | Cost per 1K tokens (input + output)\*    | Approx Chunks for 200K words | Summarization Cost (est.) | Notes                              |
| ----------------- | ---------------- | ---------------------------------------- | ---------------------------- | ------------------------- | ---------------------------------- |
| **GPT-4o**        | 128,000          | \$5.00 / 1M input + \$15.00 / 1M output  | \~12–15 chunks               | ✅ **\$3–\$6**             | Best quality + efficiency          |
| **GPT-4 Turbo**   | 128,000          | Same as GPT-4o                           | \~12–15 chunks               | ✅ \~\$3–\$6               | Similar to GPT-4o, slightly slower |
| **GPT-4**         | 32,768           | \$30.00 / 1M input + \$60.00 / 1M output | \~40–50 chunks               | ❌ **\$20–\$40**           | High-quality but expensive         |
| **GPT-3.5 Turbo** | 16,384           | \$0.50 / 1M input + \$1.50 / 1M output   | \~80–100 chunks              | ✅ **\$2–\$4**             | Fast + cheap, lower quality        |
| **Davinci-003**   | 4,096            | \$0.02 / 1K tokens                       | \~350–400 chunks             | ❌ **\$10–\$20+**          | Legacy model, not efficient        |

**Assumptions:**
Each chunk is ~2,000–4,000 words (≈3,000 tokens)
You summarize each chunk to ~300 tokens
Total output is ~10,000 tokens per document
Costs include both input + output tokens

| Metric                                  | Estimate                                   |
| --------------------------------------- | ------------------------------------------ |
| 1 word ≈ 1.3–1.5 tokens                 | ⚠️ Approximate                             |
| 20 million words ≈ 28–30 million tokens | Let's use **30M tokens** as input estimate |


Ingestion steps (01–05) dont depend on each other can be run as parallel in Databricks Jobs.

               ┌────────────────────┐
               │01_ingest_word_docs │
               └────────────────────┘
               ┌────────────────────┐
               │02_ingest_xlsx_docs │
               └────────────────────┘
               ┌────────────────────┐
               │03_ingest_pptx_docs │
               └────────────────────┘
               ┌────────────────────┐
               │04_ingest_pdf_docs  │
               └────────────────────┘
               ┌────────────────────┐
               │05_ingest_msg_csv   │
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │06_redact_pii       │
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │07_local_E5_model   |
               │    classificaiton  |
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │08_embeddings       │
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │09_push_to_ai_search│
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │10_agent_query      │
               └────────────────────┘

Through the Databricks UI, you can define:

01–05 = parallel tasks

06–10 = sequential tasks

That way heavy ingestion steps scale out concurrently across clusters and the later processing stays ordered.

**Spark Driver Node vs Worker Nodes During Ingestion**

Spark Driver Node — The Brain

The driver node is responsible for:

Running your notebook/job code

Creating the SparkSession

Breaking code into logical stages and tasks

Sending tasks to the workers (executors)

Tracking progress and collecting results

The driver as the “orchestrator”.


**In short The driver:**

Parses the command

Determines the input file locations

Splits the files into chunks (partitions)

Plans a DAG (Directed Acyclic Graph) of tasks

Sends those tasks to the workers


**Spark Worker Nodes (Executors) — The Muscle**

The worker nodes (also called executors) are responsible for:

Reading the actual document data from storage

Executing transformations and computations on each data partition

Caching or persisting data in memory/disk if needed

Writing results back to storage (e.g., Delta Lake or Parquet)

Think of workers as “distributed data processors”.

**Direct Acyclic Graph of Tasks**


| Stage     | What It Does                                                               | Runs As      | Type of Spark Job                    |
| --------- | -------------------------------------------------------------------------- | ------------ | ------------------------------------ |
| **01–05** | Parallel document ingestion by file type (Word, Excel, PDF, PPTX, MSG/CSV) | Python Tasks | Heavy I/O Spark read/write jobs      |
|           |                                                                            |              | Distributed Spark job with API calls |             
| **06**    | PII redaction using Presidio detects and redacts PII Spark Job             | Python Task  | Transformation Spark job             |
| **07**    | Local E5 Model Classification                                              | Python Task  | Distributed semantic classification  |
|           |                                                                            |              | of large volumes of multilingual     |
|           |                                                                            |              | documents using E5 model in          |
|           |                                                                            |              | Databricks on Azure.                 |
| **08**    | Convert Classified content to embeddings                                   | Python Task  | Spark job text-embedding-3-large     |
| **09**    | Pull - Text to Embeddings Azure AI Search vector DB                        | AI Search    | Automatic data pull scheduled 20 mins|
| **10**    | **Optional** Agent code to query Azure AI Search (retrieval +  enrichment) | Python Task  | Light Spark/Driver job               |


**1. Databricks Job (Orchestrator / Launcher)**
A Databricks Job is the top-level execution unit — it can run:
A notebook
A Python script
A JAR or wheel
Define the tasks, dependencies, clusters, and parameters here.
Each task in the job triggers a Spark job if Spark is used inside it.
Think of it as:
"Run this workflow using Spark or Python"

**2. Spark Driver (Job Coordinator)**
When your Databricks Job runs Spark code (like reading a DataFrame), the Spark engine spins up a driver node.
The driver is responsible for:
Parsing the code
Building the logical plan (DAG)
Scheduling tasks
Tracking task progress and collecting results
It runs inside the cluster on a designated node.
Think of it as:
"Control tower of the Spark job"

**3. Spark Workers (Executors) (Task Runners)**
The worker nodes (executors) are where the actual data processing happens:
Reading and writing data
Executing .map(), .filter(), .groupBy() etc.
Writing Parquet files
Multiple executors can run in parallel on different nodes in your Databricks cluster.
Think of them as:
"Data crunchers doing the real work in parallel"

[ Databricks Job ]
       |
       ▼
[ Spark Driver ]
       |
       ▼
[ Spark Executors (Workers) ]


How They Interact (Example)

Imagine to run a Databricks Job that ingests PDFs:

df = spark.read.text("/mnt/data/*.pdf")

df = df.repartition(10)

df.write.parquet("/mnt/clean/pdf/")


Here is what happens:

| Layer               | What It Does                                                   |
| ------------------- | -------------------------------------------------------------- |
| **Databricks Job**  | Launches the script or notebook on a cluster                  |
| **Spark Driver Node**    | Parses code, builds DAG, creates 10 read/write tasks           |
| **Spark Worker Nodes (Executors)** | 10 executors read files, process them, and write Parquet files |

----------------------------------------------------------------------------------------------------------------------------------------------------------

**HashingTF explained** : 

Explain the HashingTF step with an example to make it clearer.

Recall the line: **hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)**

The purpose of HashingTF is to take a list of words (like the output from the Tokenizer) and convert it into a fixed-size numerical vector. It does this using a clever technique called the "hashing trick" to avoid having to build a huge dictionary of all unique words.

Here's how it works with an example:

Imagine you have a very small vocabulary and numFeatures is set to a small number, say 5, instead of 1000 for simplicity. This means our output vector will have 5 "bins" or dimensions.

Let's say you have a document with the following words after tokenization: ["the", "cat", "sat", "on", "the", "mat"].

The HashingTF transformer will:

**Apply a hash function to each word:** A hash function takes a piece of data (in this case, a word) and converts it into a numerical value (an integer). The key property of a good hash function is that the same input always produces the same output, and different inputs are likely to produce different outputs (though collisions can happen).

**Map the hash value to an index:** The hash value for each word is then mapped to an index within the range of 0 to numFeatures - 1 (which is 0 to 4 in our example). This is typically done using the modulo operator (%). For example, index = hash_value % numFeatures.

**Increment the count at that index:** For each word in the document, the count in the corresponding index of the output vector is incremented.
Let's illustrate with our example words and hypothetical hash values and indices (remembering numFeatures=5):

"the": hash -> 12, index -> 12 % 5 = 2. Vector: [0, 0, 1, 0, 0]

"cat": hash -> 7, index -> 7 % 5 = 2. Vector: [0, 0, 2, 0, 0] (Collision! "the" and "cat" hashed to the same index)

"sat": hash -> 9, index -> 9 % 5 = 4. Vector: [0, 0, 2, 0, 1]

"on": hash -> 3, index -> 3 % 5 = 3. Vector: [0, 0, 2, 1, 1]

"the": hash -> 12, index -> 12 % 5 = 2. Vector: [0, 0, 3, 1, 1]

"mat": hash -> 11, index -> 11 % 5 = 1. Vector: [0, 1, 3, 1, 1]

So, the output "rawFeatures" vector for this document would be [0, 1, 3, 1, 1]. This vector represents the frequency of words in the document grouped into 5 bins based on their hash values.

**Why use the Hashing Trick?**

**Scalability:** It avoids building a large vocabulary dictionary, which can be very memory-intensive when dealing with millions of documents and a vast number of unique words.

**Speed:** Hashing is generally a very fast operation.

**Fixed Size:** The output vector size is fixed (numFeatures), regardless of the number of unique words in the entire dataset.

The main drawback is the possibility of hash collisions, where different words map to the same index. This can slightly reduce the model's ability to distinguish between words, but with a sufficiently large numFeatures (like the 1000 you used), the impact is usually minimal for many tasks.

After HashingTF, the rawFeatures vector goes to the IDF step, which will re-weight these frequencies based on how common words are across all documents.

------------------------------------------------------------------------------------------------------------------------------------------------------------

**Coefficient**:
Logistic Regression model a coefficient is a numerical value that represents the weight or importance of a particular feature (input variable) in predicting the outcome.

Here's a simple way to think about it:

Imagine you are trying to predict if someone will like a certain fruit based on two features: its sweetness and its color. A simple model might look something like this:

Likelihood of liking the fruit = (Coefficient for Sweetness * Sweetness Score) + (Coefficient for Color * Color Score) + (Intercept)

Coefficients: The numbers associated with Sweetness and Color are the coefficients.
A large positive coefficient for Sweetness would mean that the sweeter the fruit, the more likely someone is to like it.
A large negative coefficient for Color might mean that a certain color makes people less likely to like the fruit.
A coefficient close to zero would mean that the feature (Sweetness or Color) has little impact on the outcome.

In this Logistic Regression model, after training, the model will have coefficients associated with each of the features generated by the TF-IDF process. These coefficients indicate how much each word or term contributes to the model's prediction of the document's class.

The model learns these coefficients during the training process by analyzing the relationship between the features (the TF-IDF vectors representing the text) and the known labels of the documents. The goal is to find the set of coefficients that best allows the model to predict the correct label for each document.






quantization methods:

Dynamic quantization – Easy, fast, good for NLP (e.g., BERT)

Static quantization – More accurate, requires calibration data

QAT (Quantization-Aware Training) – Most accurate, needs training


 №######Yet to be proved№#######
 
 step-by-step guide to deploy a quantized ONNX Transformer model (like BERT) to an Azure Machine Learning (Azure ML) real-time inference endpoint.


A quantized ONNX model (e.g., bert-base-uncased)

An inference endpoint on Azure ML (CPU-based, cost-efficient)

A working API you can call

Step 1: Setup Environment
🔧 Prerequisites
Azure subscription

Python environment (conda or venv)

Azure ML SDK

 Install Dependencies
pip install azure-ai-ml onnxruntime optimum[onnxruntime] transformers
 Login to Azure
az login
az account set --subscription "YOUR_SUBSCRIPTION_ID"
Step 2: Prepare and Quantize Your Transformer Model
Let’s use bert-base-uncased as an example.

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType

import os

# Step 1: Export to ONNX
model_id = "bert-base-uncased"
output_dir = "./onnx_model"

main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="sequence-classification"
)

# Step 2: Quantize
model_fp32_path = os.path.join(output_dir, "model.onnx")
model_int8_path = os.path.join(output_dir, "model_quant.onnx")

quantize_dynamic(
    model_fp32_path,
    model_int8_path,
    weight_type=QuantType.QInt8
)
Step 3: Set Up Azure ML Workspace
Create a config file: config.json

{
  "subscription_id": "YOUR_SUBSCRIPTION_ID",
  "resource_group": "YOUR_RESOURCE_GROUP",
  "workspace_name": "YOUR_WORKSPACE_NAME"
}
Then connect:

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(DefaultAzureCredential(), path="./config.json")
 Step 4: Create Environment for ONNX Runtime
from azure.ai.ml.entities import Environment

onnx_env = Environment(
    name="onnx-runtime-env",
    image="mcr.microsoft.com/azureml/onnxruntime:latest",
    conda_file=None,
    description="ONNX Runtime Inference"
)

ml_client.environments.create_or_update(onnx_env)
Step 5: Write the Inference Script
Save as score.py:

import json
import numpy as np
import onnxruntime
from transformers import AutoTokenizer

model_path = "model/model_quant.onnx"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
session = onnxruntime.InferenceSession(model_path)

def init():
    global session
    pass  # Model is already loaded

def run(raw_data):
    try:
        inputs = json.loads(raw_data)
        text = inputs["text"]
        tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        ort_inputs = {k: v for k, v in tokens.items()}
        outputs = session.run(None, ort_inputs)
        return {"logits": outputs[0].tolist()}
    except Exception as e:
        return {"error": str(e)}
Step 6: Register Model + Deploy
Register the quantized model
from azure.ai.ml.entities import Model

model = Model(
    path="onnx_model/model_quant.onnx",
    name="bert-quant-onnx",
    type="custom_model",
)

ml_client.models.create_or_update(model)
Create deployment config
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

endpoint = ManagedOnlineEndpoint(
    name="bert-onnx-endpoint",
    auth_mode="key"
)

deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=endpoint.name,
    model=model,
    environment=onnx_env,
    code_path="./",  # contains score.py
    scoring_script="score.py",
    instance_type="Standard_DS2_v2",
    instance_count=1
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()
ml_client.online_deployments.begin_create_or_update(deployment).result()
ml_client.online_endpoints.begin_assign_deployment(endpoint.name, "default").result()
🔍 Step 7: Test the Endpoint
endpoint = ml_client.online_endpoints.get("bert-onnx-endpoint")
print("Endpoint URL:", endpoint.scoring_uri)

# Get auth key
key = ml_client.online_endpoints.get_keys("bert-onnx-endpoint").primary_key

import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}"
}

data = {"text": "Azure ML is great for deploying ONNX models!"}
response = requests.post(endpoint.scoring_uri, headers=headers, json=data)
print(response.json())
Done! You now have:
A quantized BERT ONNX model

Running on Azure ML as a scalable REST API

On CPU (cost-efficient) or GPU (if needed)

