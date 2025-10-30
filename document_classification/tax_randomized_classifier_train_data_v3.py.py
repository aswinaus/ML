# Databricks notebook source
# MAGIC %pip install faker

# COMMAND ----------

import json, random, uuid, re
from faker import Faker

# ============================================================
# CONFIG
# ============================================================
fake = Faker()
OUTPUT_PATH = "synthetic_tax_memos_2000.jsonl"
NUM_MEMOS = 2000

label2id = {
    "tax_problem": 0,
    "tax_solution": 1,
    "tax_type": 2,
    "tax_topic": 3,
    "year": 4
}

TAX_TYPES = [
    "Corporate Income Tax",
    "Transfer Pricing",
    "Withholding Tax",
    "Value Added Tax",
]

TAX_TOPICS = [
    "Participation exemption",
    "Motive test",
    "Subject-to-tax test",
    "Asset test",
    "CFC rules",
]

YEARS = ["2020/21", "2021/22", "2022/23", "2023/24"]

# ============================================================
# TEMPLATE COMPONENTS
# ============================================================

def make_tax_problem():
    company = fake.company()
    jurisdiction = random.choice(["Netherlands", "Gibraltar", "Luxembourg", "Ireland"])
    issue = random.choice([
        "the applicability of the Dutch participation exemption",
        "whether intercompany financing creates a deemed portfolio investment",
        "classification of crypto mining income under Dutch tax rules",
        "eligibility of offshore income for exemption treatment",
    ])
    return (
        f"{company} faces uncertainty regarding {issue}. "
        f"The entity operates between {jurisdiction} and the Netherlands, and seeks clarity on its tax position."
    )

def make_tax_solution():
    resolution = random.choice([
        "The exemption could still apply if either the asset test or the subject-to-tax test is met.",
        "Applying the motive test indicates that the participation is not held as a portfolio investment.",
        "If the effective rate exceeds 10% under Dutch standards, the subject-to-tax test should be met.",
        "A strict cash policy and elimination of intercompany receivables support asset test compliance."
    ])
    prefix = random.choice(["EY analysis concludes: ", "Our assessment indicates: ", "It was determined that ", "Conclusion: "])
    return prefix + resolution

def make_tax_type():
    return random.choice(TAX_TYPES)

def make_tax_topic():
    topics = random.sample(TAX_TOPICS, k=random.randint(2, 4))
    return ", ".join(topics)

def make_year():
    return random.choice(YEARS)

# ============================================================
# SYNTHETIC MEMO GENERATOR
# ============================================================
def make_memo():
    memo = {
        "id": str(uuid.uuid4()),
        "tax_problem": make_tax_problem(),
        "tax_solution": make_tax_solution(),
        "tax_type": make_tax_type(),
        "tax_topic": make_tax_topic(),
        "year": make_year()
    }
    return memo

# ============================================================
# BUILD DATASET
# ============================================================
dataset = []
for _ in range(NUM_MEMOS):
    memo = make_memo()
    for key in label2id.keys():
        dataset.append({
            "id": str(uuid.uuid4()),
            "text": memo[key],
            "label": label2id[key],
            "label_name": key
        })

random.shuffle(dataset)

# ============================================================
# WRITE TO JSONL
# ============================================================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in dataset:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Generated {len(dataset)} labeled samples → {OUTPUT_PATH}")
