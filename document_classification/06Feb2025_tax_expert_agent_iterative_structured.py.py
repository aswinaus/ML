# Databricks notebook source
# Install the required library
%pip install azure-search-documents
dbutils.library.restartPython()
# Rest of your code...

# COMMAND ----------

# MAGIC %pip install -U openai azure-ai-inference
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# =====================================================
# Agent Configuration
# =====================================================
AZURE_SEARCH_ENDPOINT = "https://my-search.search.windows.net"
AZURE_SEARCH_KEY      = ""
AZURE_SEARCH_INDEX    = "indexname"

AZURE_OPENAI_ENDPOINT = "https://my-openai.openai.azure.com/"
AZURE_OPENAI_KEY      = ""
AZURE_OPENAI_API_VER  = "2025-01-01-preview"
AZURE_OPENAI_MODEL    = "gpt-4o-mini"
AZURE_EMBED_MODEL     = "text-embedding-3-large"


# COMMAND ----------

# =====================================================
# IMPORTS
# =====================================================
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import textwrap, json, time, csv, os
from statistics import mean
from datetime import datetime

# =====================================================
# CLIENTS
# =====================================================
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

aoai = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VER
)

# =====================================================
# METRICS-ENABLED ITERATIVE TAX-EXPERT AGENT (STRUCTURED OUTPUT)
# =====================================================
def tax_expert_agent_iterative_structured(
    query: str,
    top_k: int = 5,
    max_rounds: int = 2,
    metrics_csv_path: str = "tax_agent_metrics.csv"
) -> dict:
    """
    Iterative EY Tax Expert Agent with detailed latency & token metrics.
    Returns a dict with the final answer and a metrics dictionary.
    """

    # -------------------- Metrics container --------------------
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "embedding_latency": None,
        "retrieval_latency": None,
        "time_to_first_token": None,
        "avg_inter_token_latency": None,
        "end_to_end_latency": None,
        "throughput_tokens_per_sec": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "status": "success"
    }

    # -------------------- Core retrieval & LLM call --------------------
    def retrieve_and_answer(q: str) -> str:
        try:
            # 1️⃣ Embedding latency
            t0 = time.perf_counter()
            emb_resp = aoai.embeddings.create(model=AZURE_EMBED_MODEL, input=q)
            metrics["embedding_latency"] = round(time.perf_counter() - t0, 3)
            embedding = emb_resp.data[0].embedding

            # 2️⃣ Retrieval latency
            t1 = time.perf_counter()
            results = search_client.search(
                search_text=q,
                vector_queries=[{
                    "kind": "vector",
                    "vector": embedding,
                    "fields": "embedding",
                    "k": top_k
                }],
                top=top_k,
                query_type="simple"
            )
            metrics["retrieval_latency"] = round(time.perf_counter() - t1, 3)
            docs = list(results)
            if not docs:
                metrics["status"] = "no_docs"
                return "Knowledge not found in repository. No web search attempted."

            # 3️⃣ Build context
            passages = []
            for i, d in enumerate(docs, start=1):
                raw = d.get("redacted_text") or d.get("content") or ""
                src_id = d.get("id", f"Doc-{i}")
                passages.append(f"Document {i} (Source_ID: {src_id}):\n{raw[:4000]}")
            joined_context = "\n\n".join(passages)

            # 4️⃣ Prompt (unchanged schema)
            prompt = f"""
You are an EY Tax Expert Assistant trained on internal memoranda, case analyses, and guidance documents.
You must reason from the provided excerpts to produce a structured, **elaborate EY-style output**.

User Query:
{q}

Knowledge Base Excerpts (retrieved documents):
{joined_context}

Follow these rules carefully:
1. Think like an EY tax professional writing an internal memorandum.
2. Provide detailed, narrative answers for the fields "Client_issue", "EY_IP_generated", and "References".
3. Use neutral, factual tone — **no speculation**, **no PII**, **no client identifiers**.
4. Preserve technical precision and tax terminology (e.g., motive test, asset test, subject-to-tax test, participation exemption).
5. Use multi-paragraph bullet-point structure where necessary.
6. When applicable, relate facts to specific articles, case law, or parliamentary history.
7. Do not invent facts not grounded in the documents.

Output a single JSON object in this exact schema:

{{
  "Asset": "Description of key asset(s) or participation(s)",
  "Tax_type": "Tax domain (e.g., Corporate income tax, Withholding tax, VAT)",
  "Tax_topic": "Precise topic (e.g., Article 13 CITA Participation Exemption)",
  "Tax_years": "Comma-separated list of years (e.g., 2023, 2024) or 'Unknown'",
  "Countries": "Comma-separated list of relevant jurisdictions (e.g., Netherlands, UK) only if available in the document if no country available only then  'Unknown'",
  "Date": "Date of document only if available else 'Unknown'",
  "Authors": "Comma-separated list of authors or entities if available in the document else unnamed (e.g., John Doe, EY Netherlands Tax Team)",
  "Source_ID": "EYIP-<Country>-<Year>-<Sequence> if available else 'Unknown'",
  "Client_issue": "Multi-paragraph description of the underlying factual scenario, including group structure, activities, and context — written anonymously and in EY memo style.",
  "EY_IP_generated": "Detailed explanation of the EY intellectual position, analytical reasoning, and methodology, written as an internal EY deliverable. Should include bullet points of insights or recommendations.",
  "References": "Detailed multi-line list of relevant legal articles, rulings, commentaries, and other technical sources.",
  "Confidence Score": "High | Medium | Low"
}}

Rules:
- Be factual and concise.
- Do not invent values; use "Unknown" if absent.
- Include author names or contributor names if explicitly mentioned in the document metadata (these are not considered PII for this purpose).
- Do not include or infer any client-identifying details, personal emails, phone numbers, or confidential entity identifiers.
"""

            # 5️⃣ Streaming completion (modern SDK — no .close or closing)
            start_time = time.perf_counter()
            first_token_time = None
            token_timestamps = []
            output_chunks = []

            with aoai.chat.completions.stream(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise EY Tax Expert Assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            ) as stream:
                for ev in stream:
                    if ev.type == "message.delta" and ev.delta.get("content"):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_timestamps.append(time.perf_counter())
                        output_chunks.append(ev.delta["content"])
                    elif ev.type == "error":
                        raise RuntimeError(ev.error)

            end_time = time.perf_counter()

            # 6️⃣ Compute latency metrics
            if first_token_time:
                metrics["time_to_first_token"] = round(first_token_time - start_time, 3)
            metrics["end_to_end_latency"] = round(end_time - start_time, 3)

            if len(token_timestamps) > 1:
                inter = [
                    token_timestamps[i] - token_timestamps[i - 1]
                    for i in range(1, len(token_timestamps))
                ]
                metrics["avg_inter_token_latency"] = round(mean(inter), 4)
                metrics["throughput_tokens_per_sec"] = round(
                    len(token_timestamps) / metrics["end_to_end_latency"], 2
                )

            # 7️⃣ Token usage (prompt + completion)
            usage_resp = aoai.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "token-usage-probe"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )
            usage = usage_resp.usage
            metrics["input_tokens"] = usage.prompt_tokens
            metrics["output_tokens"] = usage.completion_tokens
            metrics["total_tokens"] = usage.total_tokens

            return "".join(output_chunks)

        except Exception as e:
            metrics["status"] = f"error: {e}"
            return f"❌ Error during processing: {e}"

    # -------------------- Iterative reasoning loop --------------------
    current_query = query
    final_answer = None
    for round_idx in range(max_rounds):
        answer = retrieve_and_answer(current_query)
        if "Knowledge not found" in answer:
            final_answer = answer
            break

        # Judge completeness
        judge_prompt = f"""\
Evaluate the JSON below for factual completeness.
If it contains coherent values for Tax_type, Countries, and EY_IP_generated, reply "ACCURATE".
Else reply "INACCURATE".
Answer:
{answer}
"""
        verdict = aoai.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=[{"role": "user", "content": textwrap.dedent(judge_prompt)}],
            temperature=0
        ).choices[0].message.content.strip().upper()

        if "ACCURATE" in verdict or round_idx == max_rounds - 1:
            final_answer = answer
            break

        # Refine query for next round
        refine_prompt = f"""\
Rephrase the following question to improve retrieval precision in the tax domain.
Original query: {current_query}
Return only the improved query text.
"""
        current_query = aoai.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=[{"role": "user", "content": textwrap.dedent(refine_prompt)}],
            temperature=0.3
        ).choices[0].message.content.strip()

    # -------------------- Log metrics to CSV --------------------
    fieldnames = list(metrics.keys())
    file_exists = os.path.isfile(metrics_csv_path)
    with open(metrics_csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    return {"answer": final_answer, "metrics": metrics}


# COMMAND ----------

# =====================================================
# TEST RUN
# =====================================================
response = tax_expert_agent_iterative_structured("CryptoCom tax memorandum extraction")
print(response)