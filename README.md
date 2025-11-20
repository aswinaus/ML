# Machine Learning
Machine Learning experiment tracking, model checkpointing

**Training and Supervised FineTuning for a Classification Problem - Calculating the f1_macro score**

In a Supervised FineTunning model specifically in a Classification problem the F1-macro is an evaluation metric it is often monitored during supervised fine-tuning (SFT) to measure how well the encoder model is learning to classify. The F1 score is the harmonic mean of precision and recall for a class. When fine tuning a model the training objective is cross-entropy loss specifically in this case where we have multiple independent labels like problem, solution, tax type, tax topic and tax year the correct one is Binary Cross-Entropy(BCE) also can be called as Sigmoid + BCE loss which is the standard for multi-lable classificaiton and this is from where the gradient is computed and F1_macro metric is computed after each epoch (or batch) as a validation metric not as a loss like in RL where a reward signal directly drives optimization (e.g. in RLHF or GRPO), F1-macro is only used for monitoring and model selection - it does not produce gradients. It tells if the model is improving across all classes fairly.

| Stage                       | Metric used                           |
| --------------------------- | ------------------------------------- |
| **Training / Back-propagation**     | Cross-Entropy (for classification)    |
| **Validation / Evaluation** | F1-macro, Accuracy, Precision and Recall |

In short

| Type                            | Example                                                                                                       | Loss Function                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Single-label classification** | Each document has *only one* label (e.g., “Tax Type = Corporate”)                                             | **Softmax + Cross-Entropy**        |
| **Multi-label classification**  | Each document can have *multiple* labels (e.g., “has_problem=1, has_solution=1, tax_topic=‘TransferPricing’”) | **Sigmoid + Binary Cross-Entropy** |

How BCE works for tax classifier

For each label (output neuron), we compute:

<img width="983" height="226" alt="image" src="https://github.com/user-attachments/assets/468f6aca-957b-4794-8a20-cb909d27183e" />

Each label has its own independent sigmoid, so the model can output:

| Label                       | Meaning         | Example Output |
| --------------------------- | --------------- | -------------- |
| `has_problem`               | 0/1             | 0.89           |
| `has_solution`              | 0/1             | 0.65           |
| `tax_type_Corporate`        | one-hot (multi) | 0.80           |
| `tax_topic_TransferPricing` | 0/1             | 0.74           |
| `tax_year_2023`             | 0/1             | 0.55           |

As when compare softmax with CrossEntropy for  a single-label scenario

| Component          | Description                   | Softmax                     | Sigmoid                           |
| ------------------ | ----------------------------- | --------------------------- | ----------------------------------- |
| **Loss**           | Training objective            | CrossEntropy (single-label) | **BCEWithLogitsLoss (multi-label)** |
| **Activation**     | Output layer                  | Softmax                     | **Sigmoid (per label)**             |
| **Metric**         | Eval metric                   | argmax → F1                 | **sigmoid + threshold → F1_macro**  |
| **Regularization** | Weight decay + early stopping | OK                        | OK                                |


Example : Suppose a document both describes a tax problem and provides a solution. Softmax forces probabilities to sum = 1.0 -> the model must choose only one label (here “tax_problem”) even though “tax_solution” is also correct as shown in the table below where Softmax forces the probabilities to sum 1.0 meaning the model must choose only one label bere which is tax_problem even though tax_solution is also correct.

| Label        | True | Model (softmax probs) |
| ------------ | ---- | --------------------- |
| tax_problem  |  1  | 0.55                  |
| tax_solution | 1  | 0.40                  |
| tax_type     | 0    | 0.03                  |
| tax_topic    | 0    | 0.01                  |
| year         | 0    | 0.01                  |

In the case of Multi-Label where a document can contain a tax problem or solution or other class we will need Sigmoid with BinaryCrossEntropy to allow multiple labels to be "on" simultaneously so that Sigmoid gives independent probabilities in this case multiple outputs can be 1 simultaneously.

| Label        | True | Model (sigmoid probs) | Pred (> 0.5) |
| ------------ | ---- | --------------------- | ------------ |
| tax_problem  | 1  | 0.85                  | 1          |
| tax_solution | 1  | 0.74                  | 1          |
| tax_type     | 0    | 0.10                  | 0            |
| tax_topic    | 0    | 0.05                  | 0            |
| year         | 0    | 0.08                  | 0            |

Why Sigmoid with BinaryCrossEntropy classification is robust

| Property           | Softmax + Cross-Entropy  | Sigmoid + BCE                        |
| ------------------ | ------------------------ | ------------------------------------ |
| Mutual exclusivity | Forces exactly one class | Allows any combination               |
| Probabilities      | Sum = 1                  | Each label independent (0–1)         |
| Loss               | CrossEntropyLoss         | BCEWithLogitsLoss (includes sigmoid) |
| Appropriate for    | One label per sample     | Multiple possible labels per sample  |

With BCE each of five output neurons behaves like a binary detector for its label.

Sigmoid + BCE setup is ideal with each output neuron makes an independent yes/no decision allowing multiple labels to be active for the same document.

Visuallization of how multi-label document classifier works internally

<img width="596" height="553" alt="image" src="https://github.com/user-attachments/assets/bd5edd07-8fe5-4cae-ae75-fe2f298fd549" />

A realistic neural network diagram with output neuron applies Sigmoid activation producing independent probabilities (e.g., 0.87, 0.90, 0.73, 0.12, 0.58) for each of the labels and these probabilities are compared against ground truth labels using Binary Cross-Entropy (BCE) summed over all outputs.

<img width="710" height="700" alt="image" src="https://github.com/user-attachments/assets/55980f89-2a1b-44ab-ae55-2a9a73293e3e" />

With a more detailed explanation

<img width="1255" height="806" alt="image" src="https://github.com/user-attachments/assets/ab32b2c7-a6ee-440a-a276-99a226da56db" />

**F1-macro evaluation metric code**

<img width="596" height="263" alt="image" src="https://github.com/user-attachments/assets/a3cc9ed1-ad94-4116-b816-daee9a426291" />

**Training with F1 macro for evaluation + utilize the early stopping as regularization technique**

<img width="990" height="853" alt="image" src="https://github.com/user-attachments/assets/60a33364-5319-4a7d-a544-7b263627f04f" />

<img width="1022" height="151" alt="image" src="https://github.com/user-attachments/assets/7185fec8-e121-4327-a46d-09405add3e6e" />

**Make a small validation set (e.g., 10–20%):**

dataset = Dataset.from_list(train_data).train_test_split(test_size=0.15, seed=42)

tokenized_train = dataset["train"].map(preprocess_function, batched=True)

tokenized_val = dataset["test"].map(preprocess_function, batched=True)

**Targets to look for**

If val loss keeps dropping and F1_macro climbs past ~0.70+, keep training.
If val loss stops improving for ~3 evals, stop (early stopping will do it).
If training loss ↓ but val loss ↑, you’re overfitting → reduce epochs or lower LR (e.g., 1e-5) and add weight_decay.
If progress stalls Try more data per class (class balance matters).
Slightly lower LR (1e-5) or increase warmup_ratio.
Increase batch size if GPU permits (stabilizes training).
Check text length—keep max_length=256 unless your chunks are longer.

**Note: F1_macro is the macro-averaged F1 score. It is the unweighted mean of the F1 scores computed independently for each class(Problem, Solution, Topic, Tax Year) in a multi-class classification problem.**

**Why is it important?** It treats all classes equally, regardless of their frequency in the dataset.

It is especially useful when you have class imbalance, as it does not let dominant classes overshadow minority classes.

It provides a single metric that reflects the model’s ability to correctly classify all classes, not just the most common ones.

In our code: We are using f1_macro as the metric for early stopping and model selection, ensuring your model performs well across all tax-related categories, not just the majority class.

Current loss (1.11) shows learning, but it’s not “done.”

Add validation + metrics, keep training until val loss/metrics converge.

Expect clear gains with another 1–3 epochs and proper early stopping.



**After further training**

<img width="478" height="162" alt="image" src="https://github.com/user-attachments/assets/a67c8b66-2639-4ee4-92a5-4189860c7c2e" />

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
| Concept          | Embedding Similarity                 | Fine-Tuned Classifier         |
| ---------------- | ------------------------------------ | ----------------------------- |
| Output           | Vector (e.g., 768-D)                 | Logits (4-D for 4 classes)    |
| Metric           | Cosine similarity                    | Softmax + argmax              |
| Training         | Unsupervised                         | Supervised (fine-tuned)       |
| Interpretability | General similarity                   | Categorical probability       |
| Thresholding     | Manual (e.g., 0.8)                   | Confidence-based (prob > 0.5) |
| Use cases        | Semantic search, clustering          | Explicit classification       |
| Speed            | Needs multiple reference comparisons | One forward pass              |


**Intuitive Analogy**
| Analogy        | Embedding Model                                        | Classifier Model                                    |
| -------------- | ------------------------------------------------------ | --------------------------------------------------- |
| How it behaves | Measures “how close” two meanings are in general space | Decides “which bucket this text belongs to”         |
| Example        | “Are these two texts semantically alike?”              | “Is this text a problem, solution, topic, or year?” |
| Mental model   | Semantic **map** of the world                          | Decision **boundary** separating categories         |


**Practical Impact in our Case**

| Aspect       | Old (Cosine)                   | New (Classifier)            |
| ------------ | ------------------------------ | --------------------------- |
| Endpoint     | `multilingual_e5_base_service` | `finetuned_model_inference` |
| Output shape | 768-dim embeddings             | 4-class logits              |
| Evaluation   | Similarity threshold           | Softmax probability         |
| Code         | `cosine_similarity()`          | `softmax → argmax()`        |
| Use case     | Clustering, retrieval          | Direct labeling in pipeline |


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


| Stage     | What It Does                                                               | Runs As      | Type of Spark Job                     |
| --------- | -------------------------------------------------------------------------- | ------------ | ------------------------------------  |
| **01–05** | Parallel document ingestion by file type ( PDF, MSG)                       | Python Tasks | Heavy I/O Spark read/write jobs       |
|           |                                                                            |              | Distributed Spark job with API calls  |             
| **06**    | PII redaction using Presidio detects and redacts PII Spark Job             | Python Task  | Transformation Spark job              |
| **07**    | Local E5 Model Classification                                              | Python Task  | Distributed semantic classification   |
|           |                                                                            |              | of large volumes of multilingual      |
|           |                                                                            |              | documents through local language model|
|           |                                                                            |              | Databricks(Training/Serving/Inference)|
| **08**    | Convert Classified content to embeddings                                   | Python Task  | Spark job text-embedding-3-large      |
| **09**    | Pull - Text to Embeddings Azure AI Search vector DB                        | AI Search    | Automatic data pull scheduled 20 mins |
| **10**    | Agent code to query Azure AI Search (retrieval +  enrichment)              | Python Task  | Light Spark/Driver job                |
| **11**    | Expose the Azure AI KH Search Index through API                            | Azure Python | AI Search + GPT4.0 finetuned          | 
|           |                                                                            | Function     |                                       |


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

-------------------------------------------------------------------------------------------
**Hyperparameter Tuning:**
The supervised finetunning specific in the classificaiton problem which we had before computes the F1-macro and picks the model checkpoint when the F1-macro score is at the highest and it is restored at the end of training with one fixed hperparameter setup as in the code below. All these parameters are manually provided hyperparameters and none of these are tuned.

<img width="1172" height="291" alt="image" src="https://github.com/user-attachments/assets/b9c003f1-bcac-468d-ab10-61b54a60182f" />

**Ray Tune–powered hyperparameter tuning:**
This is one of the common type of hyperparameter tuning. Here we will retain SemanticTrainer, custom dataset and multi-label classification setup as we had before and run multiple training trials each with different hyperparameters with the F1-march score to tune through which we pick the best hyperparameter combination and then save the model for serving.

**In nutshell this is what Ray Tune will perform:**
Run multiple training trials each with different hyperparameters such as Learning rate, Batch size and epochs.
Report validation metrics (f1_macro) to Tune.
Pick the best hyperparameter combination.
Optionally save the best model.

**Advanced Tuning**
**Population Based Training (PBT)**

For the tax-document multi-label classification problem the learning rate, batch size and regularization strength are highly sensitive and interact in non-obvious ways so we would target those hyperparameters first.

PBT is a hyperparameter optimization + model training method where:

**1) Many model “workers” (populations) train in parallel**
Each worker starts with different hyperparameters (Learning Rate LR, batch size, etc).

**Every few iterations and the population is evaluated**
Workers doing poorly:
1) STOP
2) COPY weights from the best-performing workers
3) MUTATE their hyperparameters (slightly change LR, batch size, dropout, etc.)
   
**Bad configurations evolve into good ones**
This mimics biological evolution:
1) good solutions reproduce
2) poor solutions die
3) random mutations allow discovery of better hyperparameters over time

It is efficient than grid/random search because bad configs don’t waste compute for long.

**Population Based Training will help tax-document classifier in the following ways**


**The labels are multi-label and imbalanced**
PBT adapts LR and batch size to reduce overfitting.

**We will use a custom semantic regularization loss**
This increases sensitivity to hyperparameters for which PBT will help to stabilize it.

**For the relatively small dataset (100 samples)**
PBT helps avoid overfitting and finds gentler LRs → boosts F1.

**Early stopping alone is not enough**
Early stopping stops bad models but it does not improve hyperparameters.
