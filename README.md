# Maching Learning
Machine Learning experiment tracking, model checkpointing

Notebook : https://github.com/aswinaus/ML/blob/main/ADLS_Databricks_ApacheSpark.ipynb

<img width="831" height="417" alt="image" src="https://github.com/user-attachments/assets/f3fa2972-b16e-45f7-990a-0b858a9bbda7" />

The Classification Model Training explicitly uses distributed XGBoost within Databricks, leveraging multiple nodes in the cluster for scalable training.
Distributed XGBoost training in Databricks is performed using PySpark with parameters like num_workers to specify parallelism.
This enables efficient handling of large sharepoint data and faster model training times.

Notebook : ADLS_AzureSynapse_ApacheSpark.ipynb

<img width="929" height="704" alt="image" src="https://github.com/user-attachments/assets/b357d7e6-25df-45bd-a438-621f1be6ccf2" />



- Azure Blob Storage is the underlying object storage service.
- ADLS Gen2 extends Blob Storage with hierarchical namespace, fine-grained security, optimizing big data analytics.
- Azure Synapse Analytics provides a unified analytics platform combining big data (Spark Pools) and data warehousing (SQL Pools).
- Apache Spark running inside Synapse or Databricks uses Hadoop Azure filesystem connectors to read and write data from ADLS/Blob storage.
- Hadoop components (like YARN as resource manager in HDInsight) enable cluster resource management for Spark jobs.

HashingTF explained : 
Explain the HashingTF step with an example to make it clearer.

Recall the line: hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)

The purpose of HashingTF is to take a list of words (like the output from the Tokenizer) and convert it into a fixed-size numerical vector. It does this using a clever technique called the "hashing trick" to avoid having to build a huge dictionary of all unique words.

Here's how it works with an example:

Imagine you have a very small vocabulary and numFeatures is set to a small number, say 5, instead of 1000 for simplicity. This means our output vector will have 5 "bins" or dimensions.

Let's say you have a document with the following words after tokenization: ["the", "cat", "sat", "on", "the", "mat"].

The HashingTF transformer will:

Apply a hash function to each word: A hash function takes a piece of data (in this case, a word) and converts it into a numerical value (an integer). The key property of a good hash function is that the same input always produces the same output, and different inputs are likely to produce different outputs (though collisions can happen).
Map the hash value to an index: The hash value for each word is then mapped to an index within the range of 0 to numFeatures - 1 (which is 0 to 4 in our example). This is typically done using the modulo operator (%). For example, index = hash_value % numFeatures.
Increment the count at that index: For each word in the document, the count in the corresponding index of the output vector is incremented.
Let's illustrate with our example words and hypothetical hash values and indices (remembering numFeatures=5):

"the": hash -> 12, index -> 12 % 5 = 2. Vector: [0, 0, 1, 0, 0]
"cat": hash -> 7, index -> 7 % 5 = 2. Vector: [0, 0, 2, 0, 0] (Collision! "the" and "cat" hashed to the same index)
"sat": hash -> 9, index -> 9 % 5 = 4. Vector: [0, 0, 2, 0, 1]
"on": hash -> 3, index -> 3 % 5 = 3. Vector: [0, 0, 2, 1, 1]
"the": hash -> 12, index -> 12 % 5 = 2. Vector: [0, 0, 3, 1, 1]
"mat": hash -> 11, index -> 11 % 5 = 1. Vector: [0, 1, 3, 1, 1]
So, the output "rawFeatures" vector for this document would be [0, 1, 3, 1, 1]. This vector represents the frequency of words in the document, grouped into 5 bins based on their hash values.

Why use the Hashing Trick?

Scalability: It avoids building a large vocabulary dictionary, which can be very memory-intensive when dealing with millions of documents and a vast number of unique words.
Speed: Hashing is generally a very fast operation.
Fixed Size: The output vector size is fixed (numFeatures), regardless of the number of unique words in the entire dataset.
The main drawback is the possibility of hash collisions, where different words map to the same index. This can slightly reduce the model's ability to distinguish between words, but with a sufficiently large numFeatures (like the 1000 you used), the impact is usually minimal for many tasks.

After HashingTF, the rawFeatures vector goes to the IDF step, which will re-weight these frequencies based on how common words are across all documents.
