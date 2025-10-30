# Databricks notebook source
# Install required libraries for the BAAI/bge-m3 model
%pip install transformers accelerate bitsandbytes mlflow torch torchvision

# Define the model identifier to be used throughout the notebook
model_name = "BAAI/bge-m3"

# Restart the Python runtime to ensure the new packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

import os
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import mlflow
import mlflow.transformers

# COMMAND ----------

import os
import json
import torch
import mlflow
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import Any, Dict

# ============================================================
# Dual-Mode MLflow Model — Embedding + Classification
# ============================================================
class DualModeTaxModel(mlflow.pyfunc.PythonModel):
    """
    This custom MLflow model can serve in two modes:
      • mode="embedding" → returns vector embeddings
      • mode="classification" → returns logits for tax labels
    """

    def load_context(self, context):
        """
        Load model artifacts from the MLflow context.
        Disables safetensors to avoid 'header too large' issues.
        """
        # 🧩 Ensure safe deserialization
        os.environ["SAFETENSORS_FAST_GPU"] = "0"
        os.environ["SAFETENSORS_DISABLE"] = "1"

        model_path = context.artifacts["model_dir"]

        # Load tokenizer and both model heads (safetensors disabled)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            safe_serialization=False  # forces PyTorch .bin loading
        )
        self.embedder = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            safe_serialization=False  # disables safetensors
        )

        self.model.eval()
        self.embedder.eval()

    def predict(self, context, model_input):
    #def predict(self, context: mlflow.pyfunc.model.PythonModelContext, model_input: Any) -> Dict[str, Any]:

        """
        Handles both:
          - {"inputs": [...], "mode": "embedding" | "classification"}
          - [{"inputs": [...], "mode": "..."}] (Databricks Serving wrapper)
        """
        import json
        import torch

        # Handle Databricks Serving input structure
        if isinstance(model_input, list):
            model_input = model_input[0]
        elif isinstance(model_input, str):
            model_input = json.loads(model_input)

        inputs = model_input.get("inputs", [])
        mode = model_input.get("mode", "classification")

        # EMBEDDING MODE
        if mode == "embedding":
            with torch.no_grad():
                tokens = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                emb = self.embedder(**tokens).last_hidden_state[:, 0, :].numpy()
            return {"embeddings": emb.tolist()}

        # CLASSIFICATION MODE
        elif mode == "classification":
            with torch.no_grad():
                tokens = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                logits = self.model(**tokens).logits.numpy()
            return {"predictions": [{"logits": log.tolist()} for log in logits]}

        else:
            raise ValueError("mode must be either 'embedding' or 'classification'")


# ============================================================
# Local test entry point (optional)
# ============================================================
if __name__ == "__main__":
    # path to your locally fine-tuned model
    local_path = "/dbfs/tmp/BAAI/bge-m3-v2"

    # Disable safetensors globally before loading
    os.environ["SAFETENSORS_DISABLE"] = "1"

    model = DualModeTaxModel()
    try:
        model.tokenizer = AutoTokenizer.from_pretrained(local_path)
        model.model = AutoModelForSequenceClassification.from_pretrained(local_path)
        model.embedder = AutoModel.from_pretrained(local_path)
    except Exception as e:
        print(f"Warning: model partially loaded — {e}")   


# COMMAND ----------

# simple test
sample = {
    "inputs": [
        "Η κύρια πηγή εισοδήματος της Αυτοκρατορίας των Μουγάλ ήταν τα έσοδα από τη γη (mal). Το μεγαλύτερο μέρος του πληθυσμού ήταν αγρότες, επομένως η γεωργία φορολογούνταν βαριά, αλλά με έναν οργανωμένο τρόπο. Η γη εξεταζόταν και ταξινομούνταν με βάση τη γονιμότητα, τον τύπο της καλλιέργειας και την παραγωγικότητα. Οι φόροι υπολογίζονταν ανάλογα — συνήθως το ένα τρίτο της μέσης αξίας της παραγωγής. Η πληρωμή μπορούσε να γίνει σε μετρητά ή σε είδος (σιτηρά ή καλλιέργειες), αν και κατά την εποχή του Σαχ Τζαχάν, οι πληρωμές σε μετρητά ήταν πιο συνηθισμένες λόγω της νομισματοποίησης της οικονομίας."
    ], 
    "mode": "classification"
}
print(model.predict(None, sample))

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks")   # Legacy registry


# COMMAND ----------

mlflow.pyfunc.log_model(
    artifact_path="dual_mode_tax_model",
    python_model=DualModeTaxModel(),
    artifacts={"model_dir": local_path},
    registered_model_name="bge-m3_base_dualmode",
    pip_requirements=[
        # Core libraries pinned for cross-compatibility
        "transformers==4.42.0",        # supports bge-m3 + XLMRoberta, no init conflict
        "torch>=2.1.0,<2.4.0",         # ensure CUDA kernels are stable
        "mlflow>=3.5.1",
        "pandas>=2.2.0,<3.0.0",
        "numpy>=1.26.0,<2.0.0",
        "accelerate>=0.28.0",
        "sentencepiece>=0.1.99",
        "pyopenssl>=24.1.0",
        "cloudpickle==3.0.0",
        "cryptography>=43,<47"
    ]
)
