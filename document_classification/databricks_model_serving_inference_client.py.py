# Databricks notebook source
import requests
import json

token=""
url = ""
headers = {"Authorization": f"Bearer {token}"}

payload = {
    "dataframe_records": [
        {
            "doc_text": "Here is the document text...",
            "label_text": "Discussion of a tax issue or tax solution."
        }
    ]
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())


# COMMAND ----------

# MAGIC %pip install torch
# MAGIC %pip install transformers

# COMMAND ----------

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import mlflow.pyfunc
import pandas as pd

# ---------- Your Dual Encoder Wrapper ----------
class SharedDualEncoderInference(torch.nn.Module):
    def __init__(self, base_model, num_pii_types=11):
        super().__init__()
        self.encoder = base_model
        hidden_size = base_model.config.hidden_size

        self.pii_flag_head = torch.nn.Linear(hidden_size, 1)
        self.pii_score_head = torch.nn.Linear(hidden_size, 1)
        self.pii_types_head = torch.nn.Linear(hidden_size, num_pii_types)

    def forward(self, doc_input_ids, doc_attention_mask,
                label_input_ids, label_attention_mask):

        # Encode document
        doc_emb = self.encoder(doc_input_ids, attention_mask=doc_attention_mask).pooler_output
        doc_emb = F.normalize(doc_emb, dim=-1)

        # Encode label
        label_emb = self.encoder(label_input_ids, attention_mask=label_attention_mask).pooler_output
        label_emb = F.normalize(label_emb, dim=-1)

        return {
            "doc_emb": doc_emb,
            "label_emb": label_emb,
            "pii_flag": torch.sigmoid(self.pii_flag_head(doc_emb)),
            "pii_score": torch.sigmoid(self.pii_score_head(doc_emb)),
            "pii_types": torch.sigmoid(self.pii_types_head(doc_emb)),
        }


# ---------- MLflow Model Serving Wrapper ----------
class DualEncoderServingModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        MODEL_PATH = context.artifacts["model_dir"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load base encoder
        base_model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        # Build inference model
        self.model = SharedDualEncoderInference(base_model)
        self.model.eval()

    def predict(self, context, model_input):

        # Expect DataFrame with columns: doc_text, label_text
        docs = model_input["doc_text"].tolist()
        labels = model_input["label_text"].tolist()

        # Tokenize
        doc_enc = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt", max_length=256)
        label_enc = self.tokenizer(labels, padding=True, truncation=True, return_tensors="pt", max_length=256)

        # Inference
        with torch.no_grad():
            outputs = self.model(
                doc_input_ids=doc_enc["input_ids"],
                doc_attention_mask=doc_enc["attention_mask"],
                label_input_ids=label_enc["input_ids"],
                label_attention_mask=label_enc["attention_mask"]
            )

        # Compute cosine similarity
        cos_sim = torch.sum(outputs["doc_emb"] * outputs["label_emb"], dim=1).cpu().numpy()

        return pd.DataFrame({
            "cosine_similarity": cos_sim,
            "pii_flag": outputs["pii_flag"].cpu().numpy().flatten(),
            "pii_score": outputs["pii_score"].cpu().numpy().flatten(),
            "pii_types": outputs["pii_types"].cpu().numpy().tolist()
        })


# COMMAND ----------


import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import mlflow.pyfunc
import pandas as pd
import requests

# Load model and tokenizer
MODEL_PATH = "/dbfs/tmp/pbt_pii_semantic_dual_encoder_merged"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Instantiate the shared dual‑encoder inference model
dual_encoder = SharedDualEncoderInference(base_model)
dual_encoder.eval()



# COMMAND ----------

token = ""
url = ""
headers = {"Authorization": f"Bearer {token}"}

# COMMAND ----------


payload = {
    "dataframe_records": [
        {
            "doc_text": "amount does not exceed NTD600,000, the FECOs do not need to register for the VAT purpose in Taiwan and ﬁ le/pay the VAT. Exclusions FECOs shall register for the VAT purpose in Taiwan if their annual B2C sales amount exceeds NTD480,000 (NTD600,000 effective from 7 April 2025). Thus, if the annual B2C sales amount does not exceed NTD600,000, the FECOs do not need to register for the VAT purpose in Taiwan and ﬁ le/pay the VAT. Effective date   Initially 1 May 2017; updated on 7 April 2025            EY Global Tax Alerts  Taiwan’s uniform invoice regulations require action by foreign e-commerce operators (29 Jan 2019)  Taiwan issues ruling on new tax guidelines on cross-border e-commerce transaction (4 May 2017)  Taiwan issues new tax guidelines on cross-border e-commerce transactions to be effective from 1 May 2017 (22 March 2017) ",
            "label_text": "Discussion of a tax issue or tax solution."
        }
    ]
}

response = requests.post(url, json=payload, headers=headers)
response_json = response.json()  # API response containing raw predictions

# Prepare inputs (same as those sent to the API)
docs = [record["doc_text"] for record in payload["dataframe_records"]]
labels = [record["label_text"] for record in payload["dataframe_records"]]

doc_enc = tokenizer(docs, padding=True, truncation=True, return_tensors="pt", max_length=256)
label_enc = tokenizer(labels, padding=True, truncation=True, return_tensors="pt", max_length=256)

# Run forward pass using the SharedDualEncoderInference class
with torch.no_grad():
    outputs = dual_encoder(
        doc_input_ids=doc_enc["input_ids"],
        doc_attention_mask=doc_enc["attention_mask"],
        label_input_ids=label_enc["input_ids"],
        label_attention_mask=label_enc["attention_mask"]
    )

# Compute cosine similarity from the embeddings produced by the model
cos_sim = torch.sum(outputs["doc_emb"] * outputs["label_emb"], dim=1).cpu().numpy()

# Assemble results into a DataFrame
api_result_df = pd.DataFrame({
    "cosine_similarity": [float(x) for x in cos_sim],  # Convert to float
    "pii_flag": [float(x) for x in outputs["pii_flag"].cpu().numpy().flatten()],  # Convert to float
    "pii_score": [float(x) for x in outputs["pii_score"].cpu().numpy().flatten()],  # Convert to float
    "pii_types": [str(x) for x in outputs["pii_types"].cpu().numpy().tolist()]  # Convert to string
})

display(api_result_df)

# ------------------------------------------------------------------
# (Optional) Decode the raw API response for comparison
# ------------------------------------------------------------------
# decoded_response = {
#     "cosine_similarity": [float(torch.sum(torch.tensor(response_json['predictions'][0]) *
#                                     torch.tensor(response_json['predictions'][0]), dim=0).cpu().numpy())],
#     "pii_flag": [float(torch.sigmoid(torch.tensor(response_json['predictions'][0][0])))],
#     "pii_score": [float(torch.sigmoid(torch.tensor(response_json['predictions'][0][1])))],
#     "pii_types": [str(torch.sigmoid(torch.tensor(response_json['predictions'][0][2:])))]
# }

#display(pd.DataFrame(decoded_response))