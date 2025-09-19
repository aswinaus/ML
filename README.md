# Maching Learning
Machine Learning experiment tracking, model checkpointing

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
