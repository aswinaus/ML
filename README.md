# Maching Learning
Machine Learning experiment tracking, model checkpointing

+--------------------+                +---------------------+
|   Azure Blob       |                | Azure Data Lake     |
|   Storage          |                | Storage Gen2 (ADLS)  |
|  (Object Storage)  |<-------------->| Built on top of Blob |
+--------------------+     Stores     | Storage with        |
                                +----| Hierarchical Namespace|
                                |    +---------------------+
                                |
                +---------------+-------------------+
                |                                   |
                |       Azure Synapse Analytics    |
                |   (Unified Analytics Workspace)  |
                |                                   |
                | +---------+      +--------------+ |
                | | Spark   |      | SQL Pools    | |  Executes Analytics Queries
                | | Pools   |      | (Dedicated & | |  and Data Processing
                | +---------+      |  Serverless) | |
                |      |           +--------------+ |
                |      | Uses Apache Spark Engine   |
                +------+----------------------------+
                       |
           +---------------------------+
           |   Apache Spark (on HDInsight, Databricks, or Synapse)      |
           |   +---------------------+                                 |
           |   | Hadoop Ecosystem    |                                 |
           |   | - HDFS             |                                 |
           |   | - YARN (Resource   |                                 |
           |   |   Manager)         |                                 |
           |   | - Hadoop Azure FS  | <-- Connects to ADLS using Hadoop |
           |   +---------------------+                                |
           +---------------------------+

- Azure Blob Storage is the underlying object storage service.
- ADLS Gen2 extends Blob Storage with hierarchical namespace, fine-grained security, optimizing big data analytics.
- Azure Synapse Analytics provides a unified analytics platform combining big data (Spark Pools) and data warehousing (SQL Pools).
- Apache Spark running inside Synapse or Databricks uses Hadoop Azure filesystem connectors to read and write data from ADLS/Blob storage.
- Hadoop components (like YARN as resource manager in HDInsight) enable cluster resource management for Spark jobs.
