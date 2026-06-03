# Databricks notebook source
import requests
import numpy as np
import torch.nn.functional as F

SERVING_ENDPOINT = "https://adb-3249086852123311.11.azuredatabricks.net/serving-endpoints/finetuned_model_inference/invocations"
DATABRICKS_TOKEN = ""

LABELS = ["problem", "solution", "topic", "year"]

def classify_text(text, threshold=0.5):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"inputs": [text]}
    response = requests.post(SERVING_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    # ---- Normalize possible response shapes ----
    if isinstance(result, dict):
        preds = result.get("predictions") or result.get("data") or result
    else:
        preds = result

    # Case 1: list of dicts with logits
    if isinstance(preds[0], dict) and "logits" in preds[0]:
        logits = np.array(preds[0]["logits"])
    # Case 2: list of lists of floats
    elif isinstance(preds[0], list) and all(isinstance(x, (float, int)) for x in preds[0]):
        logits = np.array(preds[0])
    # Case 3: raw flat float list
    elif all(isinstance(x, (float, int)) for x in preds):
        logits = np.array(preds)
    else:
        raise ValueError(f"Unrecognized prediction format: {preds}")

    # ---- Compute softmax probabilities ----
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    best_idx = int(np.argmax(probs))
    best_label = LABELS[best_idx]
    best_prob = float(probs[best_idx])

    print(f"🧾 Input: {text[:100]}...")
    print(f"🔹 Probabilities: {dict(zip(LABELS, map(float, probs)))}")
    print(f"✅ Predicted label: {best_label} ({best_prob:.2f})")

    if best_prob < threshold:
        print("⚠️ Confidence below threshold — classify as 'Other'")
        return "Other"
    return best_label


# Example test
test_text = """Article 10a of the Dutch CITA discusses intra-group loans and abuse prevention clauses for tax avoidance."""
classify_text(test_text)
