# Databricks notebook source
# MAGIC %pip install azure-search-documents
# MAGIC %pip install openai
# MAGIC %pip install --upgrade azure-search-documents
# MAGIC %pip install --pre -U "azure-search-documents>=11.6.0b1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ============================================================
# Azure AI Search + Embedding Validation Script
# ============================================================
# Purpose: Validate that vector search returns semantically
# relevant results for inference-based questions derived from
# your indexed Netherlands vs Tobacco PDF.
# ============================================================

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os, json, textwrap, requests

# -----------------------------
# 🔐 CONFIGURATION
# -----------------------------
AZURE_SEARCH_ENDPOINT = "https://docsclassifieraisearch.search.windows.net"
AZURE_SEARCH_KEY      = ""
AZURE_SEARCH_INDEX    = "docs-index"

AZURE_OPENAI_ENDPOINT = "https://tpapp.openai.azure.com/"
AZURE_OPENAI_KEY      = ""
AZURE_OPENAI_API_VER  = "2024-08-01-preview"
AZURE_EMBED_MODEL     = "text-embedding-3-large"

# -----------------------------
# 🧠 INIT CLIENTS
# -----------------------------
aoai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VER,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ============================================================
#  Inference-based validation questions + expected answers
# ============================================================
qa_pairs = [
    ("When the decision mentions belanghebbende, which entities does that term represent, and how are X1 B.V. and X2 B.V. related?",
     "Belanghebbende refers to the Dutch subsidiaries [X1] B.V. and [X2] B.V.; [X2] became head of the fiscal unity from 2013."),
    ("Who is referred to as de inspecteur throughout the judgment, and what institutional role does this party play?",
     "De inspecteur is the Dutch Tax Authority (Belastingdienst), the opposing party."),
    ("Across the 2008–2016 period, how did the corporate relationship between the Dutch subsidiaries and the UK parent influence the court’s reasoning?",
     "The entities belonged to a multinational tobacco group headed by [B] p.l.c. in the UK; intra-group financing and royalties were judged by that parent-subsidiary link."),
    ("Why did the court consider the factoring fees largely onzakelijk (non-arm’s-length)?",
     "Factoring fees were excessive versus market rates; independent firms would not pay them, proving group influence."),
    ("What rationale did the Hof give for concluding that guarantee fees were not a separate intra-group service?",
     "Because the subsidiary already benefited from the group’s implicit credit support, no extra guarantee service existed."),
    ("How did the principle of implicit support affect the assessment of the guarantee fees?",
     "Implicit support meant the company already had the group rating; additional fees lacked an arm’s-length basis."),
    ("What evidence convinced the court that the termination of licentierechten amounted to an onzakelijke onttrekking of €1.3 billion?",
     "The Dutch entity ended license rights without compensation while exploitation continued in the UK — an un-business-like withdrawal."),
    ("In which way did the cost-plus and profit-split arrangements contribute differently to the final corrections?",
     "Profit-split and cost-plus were partly upheld; cost-plus % reduced but both remained partly justified."),
    ("Under what conditions did the Hof decide that omkering en verzwaring van de bewijslast applied?",
     "Because required filings were incorrect or incomplete, reversing and intensifying the burden of proof for 2011–2016."),
    ("How did the taxpayer attempt to rebut the bewijsvermoeden under Article 8b Wet VPB, and why did the court find it insufficient?",
     "They argued ignorance of non-arm’s-length pricing; the Hof held that doubt about intent isn’t enough."),
    ("What reasoning did the Hof provide to clarify that objectivering van bewustheid means intent is no longer decisive once un-arm’s-length pricing is proven?",
     "Article 8b makes awareness objective — a deviation alone triggers the presumption of group benefit."),
    ("Why did the court find the taxpayer’s reliance on older uncertainty about implicit support only partly credible, and from which date did that defense expire?",
     "Uncertainty defense accepted only until 26 Nov 2013 (Transfer-Pricing Decree 2013); after that, rule was clear."),
    ("Which specific correction had the largest financial effect in absolute terms, and in which year did it arise?",
     "The 2016 licence-rights (‘F-exit’) correction ≈ €2.75 billion — largest adjustment."),
    ("How did the treatment of boeten evolve from 2010 to 2016, and which mitigating factor led to their reduction?",
     "Penalties reduced because of procedural-delay (overschrijding redelijke termijn)."),
    ("Why did the Hof invalidate the enormous 2016 penalty of €125 million even though it upheld other penalties?",
     "Because the taxpayer informed the inspector in advance of its position — no deliberate misconduct."),
    ("What timeline of mutual-agreement-procedure correspondence did the Hof reference, and how did that relate to the final outcome?",
     "MAP correspondence closed April 2025; it did not affect the decision."),
    ("How did the Hof use the Hornbach-arrest to rule out conflict with EU law?",
     "It held that Article 8b VPB is compatible with EU freedom of establishment."),
    ("What distinction did the court draw between onzakelijke-lening jurisprudence and transfer-pricing under Article 8b Wet VPB?",
     "Both use the arm’s-length test, but Article 8b codifies it for related-party pricing."),
    ("How did the court interpret the OECD Transfer Pricing Guidelines when defining the bandbreedte of acceptable prices?",
     "The Hof used OECD guidelines to define a range; prices outside it require correction."),
    ("In what way did the Hof emphasize that proof of a zakelijke reden (business motive) could overturn a presumption of non-arm’s-length behavior?",
     "A legitimate business motive can rebut the presumption and cancel the correction.")
]

# ============================================================
# 🔎 Semantic retrieval for each Q–A
# ============================================================
print("\n🔹 Running semantic validation across all inference questions...\n")

for i, (question, expected) in enumerate(qa_pairs, start=1):
    print(f"❓ Q{i}. {question}")
    try:
        # Create embedding for the question
        emb = aoai_client.embeddings.create(
            input=question, model=AZURE_EMBED_MODEL
        ).data[0].embedding

        # Use REST API for vector search with corrected API version
        url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2024-08-01-preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_SEARCH_KEY
        }
        payload = {
            "vector": {
                "value": emb,
                "fields": "contentVector",
                "k": 3
            },
            "select": "sourcePath,content"
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        results = response.json().get("value", [])

        # Display retrieved snippets
        for r in results:
            snippet = textwrap.shorten(r.get("content", ""), width=300, placeholder=" ...")
            print(f"   → Score: {r.get('@search.score', 0):.4f} | {r.get('sourcePath','(no path)')}")
            print(f"     Retrieved: {snippet}")

        # Display the expected answer
        print(f"\n   ✅ Expected Answer: {expected}\n{'-'*100}\n")

    except Exception as e:
        print(f"   ⚠️ Error while querying: {e}\n{'-'*100}\n")

print("\n✅ Validation run complete.")
print("Compare each retrieved snippet with the expected answer above to confirm semantic recall accuracy.")

# COMMAND ----------

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

client = SearchClient(
    endpoint="https://docsclassifieraisearch.search.windows.net",
    index_name="mydocs-knowledgeharvester-index",
    credential=AzureKeyCredential("")
)

# Generate embedding
aoai = AzureOpenAI(
    api_key="",
    azure_endpoint="https://tpapp.openai.azure.com/",
    api_version="2024-07-01-preview"
)
emb = aoai.embeddings.create(
    input="Test query about guarantee fees",
    model="text-embedding-3-large"
).data[0].embedding

# Vector search test
results = client.search(
    search_text=None,
    vector={"value": emb, "fields": "contentVector", "k": 3},
    select=["id", "sourcePath"]
)
for doc in results:
    print(doc)


# COMMAND ----------

import requests, json

AZURE_SEARCH_ENDPOINT = "https://docsclassifieraisearch.search.windows.net"
AZURE_SEARCH_KEY      = ""
AZURE_SEARCH_INDEX    = "mydocs-knowledgeharvester-index"

AZURE_OPENAI_ENDPOINT = "https://tpapp.openai.azure.com/"
AZURE_OPENAI_KEY      = ""
AZURE_OPENAI_API_VER  = "2024-08-01-preview"
AZURE_EMBED_MODEL     = "text-embedding-3-large"

search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2024-07-01"
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_KEY
}
query_text = "How did the court treat the guarantee fees in the Netherlands vs Tobacco 2025 case?"
embedding = aoai_client.embeddings.create(
    input=query_text,
    model=AZURE_EMBED_MODEL
).data[0].embedding
payload = {
    "vectorQueries": [{
        "kind": "vector",
        "vector": embedding,
        "fields": "embedding",
        "k": 3
    }],
    "select": ["id", "sourcePath", "content"]
}

resp = requests.post(search_url, headers=headers, data=json.dumps(payload))
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2)[:2000])


# COMMAND ----------

print(type(emb))
print(len(emb))
print(type(emb[0]))
