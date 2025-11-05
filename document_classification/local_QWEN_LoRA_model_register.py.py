# Databricks notebook source
# MAGIC %pip install transformers
# MAGIC # Install PyTorch and Transformers
# MAGIC %pip install torch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install 'accelerate>=0.26.0'
# MAGIC %pip install -U bitsandbytes
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoTokenizer, AutoModel

# Set the local DBFS path
local_dbfs_model_path = "/dbfs/tmp/hf_models/Qwen2.5_base"

# Download and save the tokenizer
#tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
#tokenizer.save_pretrained(local_dbfs_model_path)

# Download the model and save
#model = AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
#model.save_pretrained(local_dbfs_model_path)

# COMMAND ----------

# List files in the local DBFS path and display them
local_dbfs_model_path = "/dbfs/tmp/hf_models/Qwen2.5_base"

# Using dbutils to get file info
files = dbutils.fs.ls(f"dbfs:{local_dbfs_model_path.replace('/dbfs', '')}")

# Show the result in a rich table
display(files)

# COMMAND ----------

import mlflow
# Simple register without python model or loader module
mlflow.set_registry_uri("databricks")  # legacy workspace registry

mlflow.transformers.log_model(
    transformers_model=local_dbfs_model_path,        # path to local Qwen2.5_base model directory
    artifact_path="Qwen2_5_base", # name for the logged artifact
    registered_model_name="Qwen2_5_base", # registry name
    task="feature-extraction"             # REQUIRED when using a local checkpoint
)

# COMMAND ----------

# MAGIC %pip install peft
# MAGIC %pip install bitsandbytes
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %pip install -U bitsandbytes==0.43.1 accelerate==0.30.1 transformers==4.43.3
# MAGIC

# COMMAND ----------

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
local_dbfs_model_path = "/dbfs/tmp/hf_models/Qwen2.5_base"
dataset_path = "tax_instruct_qwen.jsonl"
output_dir = "/dbfs/tmp/qwen2p5_tax_classifier"
model_id = local_dbfs_model_path

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
ds = load_dataset("json", data_files=dataset_path)
tok = AutoTokenizer.from_pretrained(local_dbfs_model_path)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
# Use GPU automatically if available
device_map = "auto" if torch.cuda.is_available() else "cpu"
load_in_4bit = torch.cuda.is_available()  # enable 4‑bit quantization when on GPU

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    #load_in_4bit=load_in_4bit,
    torch_dtype=torch.float32,
)

# Apply LoRA
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)

# Set requires_grad to True for model parameters
for param in model.parameters():
    param.requires_grad = True

# -------------------------------------------------------
# TOKENIZE
# -------------------------------------------------------
def tok_fn(ex):
    prompt = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput:"
    output = ex["output"]
    # Combine both prompt + output
    text = f"{prompt} {output}{tok.eos_token}"
    tokenized = tok(text, truncation=True, max_length=4096)

    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    # Compute the point where output starts
    prompt_len = len(tok(prompt).input_ids)

    # Mask out the prompt part (loss not computed there)
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return tokenized

# Freeze non-LoRA weights (avoids reentrant backward)
for n, p in model.named_parameters():
    if "lora" not in n:
        p.requires_grad = False

tok_ds = ds.map(tok_fn, remove_columns=ds["train"].column_names)

# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=50,
    gradient_checkpointing=False,
    report_to="none",  # prevent wandb or mlflow auto init
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
)

trainer.train()

# COMMAND ----------

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

model_id = local_dbfs_model_path
#  loads a dataset from a JSON file named "tax_instruct_qwen.jsonl" using the load_dataset function from the datasets library.
ds = load_dataset("json", data_files="tax_instruct_qwen.jsonl")

# loads a pre-trained Qwen2.5 model and its corresponding tokenizer using the AutoModelForCausalLM and AutoTokenizer classes from the transformers library.
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", load_in_4bit=True, torch_dtype=torch.float16
)

# Apply LoRA to the pre-trained model using the get_peft_model function from the peft library. Based on my understanding.
# LoRA r=16: This is the rank of the low-rank matrices used in LoRA. A lower rank means fewer parameters to update, which can lead to faster training and inference.
# lora_alpha=32: This is the alpha value used in the LoRA algorithm, which controls the strength of the low-rank adaptation.
# lora_dropout=0.05: This is the dropout rate applied to the low-rank matrices, which helps prevent overfitting.
# target_modules=["q_proj","v_proj"]: These are the specific modules in the model that # LoRA will be applied to. In this case, it's the query projection (q_proj) and value # projection (v_proj) modules.
# task_type="CAUSAL_LM": This specifies the task type for LoRA, which in this case is # causal language modeling.

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")

# get_peft_model function is then used to apply the LoRA configuration to the model, which returns a new model with the LoRA adaptation applied.                  
model = get_peft_model(model, lora)

# We define a function tok_fn that takes an example from the dataset and tokenizes it using the AutoTokenizer. The function also creates a prompt by concatenating the instruction, input and output and then tokenizes the prompt.
def tok_fn(ex):
    prompt = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput:"
    x = tok(prompt, truncation=True, max_length=4096)
    y = tok(ex["output"], add_special_tokens=False).input_ids + [tok.eos_token_id]
    x["labels"] = x["input_ids"] + y
    return x

# Apply the tok_fn function to the entire dataset using the map method, which creates a new dataset with the tokenized examples.
tok_ds = ds.map(tok_fn, remove_columns=ds["train"].column_names)

# Set of training arguments using the TrainingArguments class from the transformers library. These arguments include the output directory, batch size, number of epochs, learning rate and other hyperparameters.
args = TrainingArguments(
    output_dir="/dbfs/tmp/qwen2p5_tax_classifier",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4, # Trainer class will use the Adam optimizer with the specified learning rate of 0.001. Optionally we can also use the SGD optimizer instead of Adam. And we can apply other optimizers such as AdamW, RMSprop, etc.
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    fp16=True,
    save_strategy="epoch",
    logging_steps=50
)
# Fine-tune the LoRA-applied model on the tokenized dataset using the Trainer class from the transformers library. The Trainer class takes care of the training loop, including batching, gradient accumulation, and saving checkpoints.
Trainer(model=model, args=args, train_dataset=tok_ds["train"]).train()

# Addtional Notes:
#Stochastic Gradient Descent (SGD) addresses this issue by computing the gradient of #the loss function using only a single example from the dataset, rather than the #entire dataset. This is done by randomly selecting a single example from the dataset, #computing the gradient of the loss function for that example, and updating the model #parameters based on this gradient.
# Momentum SGD: This adds a momentum term to the update rule, which helps to escape local minima and converge to the optimal solution.
#The key difference between standard Gradient Descent and Stochastic Gradient Descent #is that SGD uses a single example to compute the gradient, whereas standard Gradient #Descent uses the entire dataset. This makes SGD much faster and more efficient, #especially for large datasets.
