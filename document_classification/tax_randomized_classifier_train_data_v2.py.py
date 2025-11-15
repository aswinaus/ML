# Databricks notebook source
import random, json

label2id = {
    "problem": 0,
    "solution": 1,
    "topic": 2,
    "year": 3
}

def make_examples(base_texts, label, n):
    """Create n synthetic variations per class"""
    templates = [
        "This document {desc}.",
        "The memorandum {desc}.",
        "Our analysis {desc}.",
        "This advisory note {desc}.",
        "The report {desc}.",
        "This case study {desc}.",
        "The client memo {desc}.",
        "In this scenario, {desc}.",
        "Our opinion letter {desc}.",
        "The ruling summary {desc}."
    ]
    return [{"text": random.choice(templates).format(desc=t), "label": label2id[label]} 
            for t in random.sample(base_texts, min(len(base_texts), n))]

# --------------- Base pools for augmentation ---------------

problem_pool = [
    "describes a tax problem involving deferred tax recognition under IFRS.",
    "identifies a corporate tax problem in the application of the participation exemption.",
    "highlights a VAT problem in reverse charge mechanisms for cross-border services.",
    "addresses a tax problem with interest deductibility under thin capitalization rules.",
    "notes a compliance problem with late submission of tax returns.",
    "details a withholding tax problem on royalty payments to low-tax jurisdictions.",
    "mentions a problem in calculating the effective tax rate for crypto operations.",
    "points to a payroll tax problem with employee share options.",
    "explores a problem in applying loss carryforward limitations.",
    "outlines a tax problem with permanent establishment determination in cross-border operations."
] * 10  # replicate to enrich diversity

solution_pool = [
    "proposes a tax solution involving reclassification of intercompany financing.",
    "recommends implementing a transfer pricing adjustment to meet arm’s-length principle.",
    "details a solution to optimize VAT refund procedures across EU entities.",
    "outlines a solution applying the subject-to-tax test with new computation basis.",
    "suggests a governance solution for quarterly ETR reconciliation and monitoring.",
    "presents a tax solution to ring-fence low-taxed foreign income streams.",
    "defines a procedural solution to automate substance documentation checks.",
    "includes a remedy for payroll tax exposure on expat allowances.",
    "introduces a solution using treaty-based exemption claims for interest income.",
    "details a remediation solution to correct incorrect crypto asset classification."
] * 10

topic_pool = [
    "discusses the topic of Article 13 CITA and the motive test.",
    "reviews the topic of asset test and free portfolio investment definition.",
    "covers the topic of subject-to-tax test and base erosion principles.",
    "explains the topic of participation exemption under Dutch corporate tax law.",
    "analyzes the topic of thin capitalization and interest barrier rules.",
    "describes the topic of transfer pricing documentation under OECD guidelines.",
    "summarizes the topic of withholding tax reclaims and beneficial ownership.",
    "examines the topic of hybrid mismatch arrangements.",
    "outlines the topic of CFC legislation and exemption thresholds.",
    "focuses on the topic of fiscal unity and intra-group loss transfer."
] * 10

year_pool = [
    "refers to the tax year 2021 and related corporate filings.",
    "concerns the fiscal year 2022 and applicable reporting thresholds.",
    "summarizes the tax obligations for financial year ending 2023.",
    "applies to the period 2020–2021 under Dutch CIT law.",
    "addresses the compliance cycle for tax year 2024.",
    "covers the reporting updates for fiscal year 2025.",
    "details carryforward loss utilization through tax year 2022.",
    "discusses the adjustments made during tax year 2023.",
    "refers to the annual assessment for fiscal year 2021.",
    "mentions amendments applicable to year 2024 filings."
] * 10

# --------------- Generate 300 samples ---------------
train_data = []
train_data.extend(make_examples(problem_pool, "problem", 150))
train_data.extend(make_examples(solution_pool, "solution", 150))
train_data.extend(make_examples(topic_pool, "topic", 150))
train_data.extend(make_examples(year_pool, "year", 150))

random.shuffle(train_data)

# --------------- Save to JSONL file ---------------
output_path = "tax_classifier_train_data_v2.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for row in train_data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Generated {len(train_data)} examples → {output_path}")
