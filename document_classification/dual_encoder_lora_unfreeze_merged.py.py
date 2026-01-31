# Databricks notebook source
# MAGIC %pip install ray
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import ray
ray.shutdown()

# Get the number of GPUs available on the cluster
num_gpus = 1  # Replace with the actual number of GPUs available on the cluster

# Initialize Ray with the required configuration
ray.init(
    address=None,
    num_gpus=num_gpus,
    dashboard_host="0.0.0.0",
    dashboard_port=8265,
    _system_config={"worker_register_timeout_seconds": 120}
)

# Print the cluster resources
print(ray.cluster_resources())

# To verify parallel execution
ray.available_resources()

# COMMAND ----------

# ================================================================
# 0. Install Dependencies
# ================================================================
%pip install transformers accelerate peft datasets ray[tune] > /dev/null

# ================================================================
# 1. Imports
# ================================================================
import os, json, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
from dataclasses import dataclass

# ================================================================
# 2. Config
# ================================================================
DATA_DIR = "/Workspace/Users/aswin@eyaswin.onmicrosoft.com/synthetic_tax_dual_encoder_v3_10K/pairs"
#BASE_MODEL = "/dbfs/tmp/hf_models/bge_m3"
BASE_MODEL = "/dbfs/tmp/pbt_pii_semantic_dual_encoder_merged"
MODEL_OUT  = "/dbfs/tmp/pbt_pii_semantic_dual_encoder"
FULL_MODEL_OUT = "/dbfs/tmp/pbt_pii_semantic_dual_encoder_merged"

PII_TYPES = ["passport", "iban", "ssn","email", "credit_card","phone_number", "dob", 
             "tax_id", "driver_license", "bank_account","address"]
PII_TYPE2IDX = {k: i for i, k in enumerate(PII_TYPES)}
num_pii_types = len(PII_TYPES)

# ================================================================
# 3. Dataset & Collator
# ================================================================
def encode_pii_types(pii_list):
    vec = [0] * len(PII_TYPES)
    for t in pii_list:
        if t in PII_TYPE2IDX:
            vec[PII_TYPE2IDX[t]] = 1
    return vec

class DualEncoderDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.records = [json.loads(x) for x in open(jsonl_path)]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        label_text = " ; ".join(r.get("label_text", [])) if isinstance(r.get("label_text", []), list) else str(r.get("label_text", ""))
        doc = r["doc_text"]
        target = r.get("target", 0)
        soft = r.get("soft_score", float(target))
        pii_flag = r.get("pii_flag", 0)
        pii_score = r.get("pii_score", 0)
        pii_types = encode_pii_types(r.get("pii_types", []))

        t_doc = self.tok(doc, truncation=True, padding="max_length",
                         max_length=self.max_len, return_tensors="pt")
        t_label = self.tok(label_text, truncation=True, padding="max_length",
                           max_length=self.max_len, return_tensors="pt")

        return {
            "doc_input_ids": t_doc["input_ids"].squeeze(0),
            "doc_attention_mask": t_doc["attention_mask"].squeeze(0),
            "label_input_ids": t_label["input_ids"].squeeze(0),
            "label_attention_mask": t_label["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=torch.float),
            "soft": torch.tensor(soft, dtype=torch.float),
            "pii_flag": torch.tensor(pii_flag, dtype=torch.float),
            "pii_score": torch.tensor(pii_score, dtype=torch.float),
            "pii_types": torch.tensor(pii_types, dtype=torch.float32)
        }

@dataclass
class DualEncoderCollator:
    tok: AutoTokenizer
    def __call__(self, features):
        doc_ids = torch.stack([f["doc_input_ids"] for f in features])
        doc_mask = torch.stack([f["doc_attention_mask"] for f in features])
        label_ids = torch.stack([f["label_input_ids"] for f in features])
        label_mask = torch.stack([f["label_attention_mask"] for f in features])
        targets = torch.stack([f["target"] for f in features])
        soft = torch.stack([f["soft"] for f in features])
        return {
            "doc_input_ids": doc_ids,
            "doc_attention_mask": doc_mask,
            "label_input_ids": label_ids,
            "label_attention_mask": label_mask,
            "target": targets,
            "soft": soft
        }

def collate_fn(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out

# ================================================================
# 4. Model Definition
# ================================================================
class SharedDualEncoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.pii_flag_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.pii_score_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.pii_types_head = nn.Linear(self.encoder.config.hidden_size, num_pii_types)
        #self.doc_emb = torch.nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        #self.label_emb = torch.nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        self.config = self.encoder.config

    def forward(self, doc_input_ids, doc_attention_mask,
                label_input_ids, label_attention_mask,
                target=None, soft=None,
                pii_flag=None, pii_score=None, pii_types=None,**unused):
        doc_emb = self.encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask).pooler_output
        label_emb = self.encoder(input_ids=label_input_ids, attention_mask=label_attention_mask).pooler_output
        return {
            "pii_flag": torch.sigmoid(self.pii_flag_head(doc_emb)),
            "pii_score": torch.sigmoid(self.pii_score_head(doc_emb)),
            "pii_types": torch.sigmoid(self.pii_types_head(doc_emb)),
            "doc_emb": F.normalize(doc_emb, dim=-1),
            "label_emb": F.normalize(label_emb, dim=-1)
        }

# ================================================================
# 5. Trainer with Original Compute Loss
# ================================================================
class SemanticDualEncoderTrainer(Trainer):
    def __init__(self, *args, contrastive_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight

    def _prepare_inputs(self, inputs): return inputs
    def _move_model_to_device(self, model, device): model.to(device); return model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            doc_input_ids=inputs["doc_input_ids"],
            doc_attention_mask=inputs["doc_attention_mask"],
            label_input_ids=inputs["label_input_ids"],
            label_attention_mask=inputs["label_attention_mask"]
        )
        doc_emb, lab_emb = outputs["doc_emb"], outputs["label_emb"]

        # 1) Normalize embeddings for true cosine similarity
        doc_emb = F.normalize(doc_emb, p=2, dim=-1)
        lab_emb = F.normalize(lab_emb, p=2, dim=-1)

        #targets = inputs.get("soft", inputs.get("target"))
        targets = inputs.get("soft") if "soft" in inputs else inputs["target"]

        # 2) Cosine similarity in [-1, 1] 
        cos = (doc_emb * lab_emb).sum(-1).clamp(-1, 1)
        # If cos values are small (close to 0), then sigmoid(0) ≈ 0.5.
        # If the model starts with random weights and never moves far from zero, the gradient signal is weak, and the model doesn’t learn meaningful similarity.
        print("cos range:", cos.min().item(), cos.max().item())
        
        # Diagnostics BEFORE loss
        pm = (targets == 1)
        nm = (targets == 0)
        mean_pos_cos = cos[pm].mean().item() if pm.any() else 0.0
        mean_neg_cos = cos[nm].mean().item() if nm.any() else 0.0
        

        
        # If the cosine similarity doesn’t shift significantly during training, gradients remain tiny, and the optimizer can’t push the model toward better representations.
        
        # cos_prob = (cos + 1) / 2.0
        cos_prob = torch.sigmoid(cos)  # for BCE
        # If cos_prob is always near 0.5, that’s a red not good because it means the model’s output is not differentiating between positive and negative pairs.
        print("cos_prob range:", cos_prob.min().item(), cos_prob.max().item())

        # BCEWithLogitsLoss expects logits not values in [-1,1]
        #bce = nn.BCEWithLogitsLoss()(cos, targets)
        
        # For Cosine similarity between doc and label use BCELoss, remember if cos is then BCE loss will hover around -log(0.5) ≈ 0.693 for both classes, giving almost no incentive to separate positives from negatives.
        #bce = nn.BCELoss()(cos_prob, targets)
        
        # 3) BCE with scaled logits: widen the range to get stronger gradients 
        # alpha=5.0   
        pos_weight=None
        # contrast_weight=0.05
        alpha = getattr(self, "alpha", 5.0)
        # If negatives still hover near 0.3–0.4 late in training, consider raising margin to 0.35–0.4.
        margin = getattr(self, "margin", 0.35)
        
        contrast_weight = getattr(self, "contrast_weight", 0.05)

        scaled_cos = cos * alpha  # alpha ∈ [3, 8] — tune as needed

        if pos_weight is not None:  # e.g., to handle class imbalance
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(scaled_cos, targets.float())
        else:
            bce = nn.BCEWithLogitsLoss()(scaled_cos, targets.float())
        # Applying sigmoid to cos_prob gives us the same loss as BCEWithLogitsLoss
        #bce = nn.BCELoss()(torch.sigmoid(cos), targets)
        print("BCE Loss:", bce.item())
        
        # If cos is close to 0 then Contrastive loss will also be small because (cos - margin) or (1 - cos) barely changes.
        # 4) Contrastive loss on raw cosine (no sigmoid)
        #    Positives: penalize if similarity < 1
        #    Negatives: penalize if similarity > margin
        contrast = (targets * (1 - cos).clamp(min=0) + (1 - targets) * (cos - margin).clamp(min=0)).mean()
        # contrast = (
        #     targets * (1 - cos_prob).clamp(min=0) +
        #     (1 - targets) * (cos_prob - margin).clamp(min=0)
        # ).mean()
        print("Contrastive Loss:", contrast.item())
        
        loss = bce + contrast_weight * contrast
        print("Loss:", loss.item())
        print(f"[diag] mean_pos_cos={mean_pos_cos:.3f}, mean_neg_cos={mean_neg_cos:.3f}, "
             f"BCE={bce.item():.4f}, Contrast={contrast.item():.4f}, Total={loss.item():.4f}")
        
        # PII losses
        pii_losses = {}
        for k, criterion in [("pii_flag", nn.BCELoss()), ("pii_score", nn.MSELoss()), ("pii_types", nn.BCELoss())]:
            val = inputs.pop(k, None)
            if val is not None:
                mask = val > 0 if k != "pii_types" else val.sum(dim=1) > 0
                # loss += criterion(outputs[k].squeeze()[mask], val.float()[mask]) if mask.any() else torch.tensor(0.0, device=loss.device)
                # Do NOT add pii_losses to 'loss'
                if mask.any():
                    pii_losses[k] = criterion(outputs[k].squeeze()[mask], val.float()[mask]).item()
                else:
                    pii_losses[k] = 0.0

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

# ================================================================
# 6. PBT Training Function
# ================================================================
def train_with_pbt(config):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = SharedDualEncoder(BASE_MODEL)
    #state = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)
    #model.load_state_dict(state, strict=True)

    # Freeze all except top 4 layers
    for param in model.encoder.parameters(): param.requires_grad = False
    for layer in model.encoder.encoder.layer[-4:]:
        for param in layer.parameters(): param.requires_grad = True

    # LoRA config
    lora_cfg = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=32,
        target_modules=["query","key","value","dense"],
        layers_to_transform=list(range(20,24)),
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["pii_flag_head","pii_score_head","pii_types_head","doc_emb", "label_emb"]
    )
    #model = get_peft_model(model, lora_cfg)
    # First iteration → initializes LoRA adapters.
    # Subsequent iterations → resumes from previous trial state.
    checkpoint = session.get_checkpoint()
    if checkpoint:
        print(f"[INFO] Resuming from checkpoint: {checkpoint.path}")
        #model = SharedDualEncoder(BASE_MODEL)
        model = PeftModel.from_pretrained(model, checkpoint.path)
    else:
        print("[INFO] Starting fresh trial")
        #model = SharedDualEncoder(BASE_MODEL)
        model = get_peft_model(model, lora_cfg)  # only if no checkpoint



    train_ds = DualEncoderDataset(os.path.join(DATA_DIR, "train.jsonl"), tokenizer)
    val_ds = DualEncoderDataset(os.path.join(DATA_DIR, "val.jsonl"), tokenizer)
    
    # Datasets
    # train_ds = DualEncoderDataset("/Workspace/Users/a1172300-msp01@ey.net/trainpii.jsonl", tokenizer)
    # val_ds   = DualEncoderDataset("/Workspace/Users/a1172300-msp01@ey.net/valpii.jsonl", tokenizer)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    args = TrainingArguments(
        output_dir=session.get_trial_dir(),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        weight_decay=config["wd"],
        warmup_ratio=config["warmup"],
        num_train_epochs=2,
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=20,
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        prediction_loss_only=True,
        fp16=fp16,
        bf16=bf16
    )

    trainer = SemanticDualEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn
    )
    trainer.alpha = config["alpha"]
    trainer.margin = config["margin"]
    trainer.contrast_weight = config["contrast_weight"]

    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint.path)
    else:
        trainer.train()

    for it in range(config["max_pbt_iters"]):
        trainer.train()
        metrics = trainer.evaluate()
        score = metrics.get("eval_loss", 999)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.model.save_pretrained(tmp)
            trainer.save_model(tmp)
            trainer.save_state()  # includes optimizer & scheduler
            tokenizer.save_pretrained(tmp)
            session.report({"loss": score}, checkpoint=Checkpoint.from_directory(tmp))

# ================================================================
# 7. Dataset Preset & PBT Scheduler
# ================================================================
MEDIUM_CONFIG = dict(batch_size=16, grad_accum=1, num_workers=2,
                     eval_steps=200, save_steps=1, max_pbt_iters=2)

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="loss",
    mode="min",
    perturbation_interval=6,
    hyperparam_mutations={
        "lr": tune.loguniform(1e-6, 5e-5),
        "wd": tune.uniform(0, 0.1),
        "warmup": tune.uniform(0, 0.2),
        "lora_rank": [4,8,16,32],
        "alpha": tune.uniform(3.0, 8.0),          # logit scaling
        "margin": tune.uniform(0.25, 0.45),       # contrastive margin
        "contrast_weight": tune.uniform(0.02, 0.15) # loss balance
    }
)

# ================================================================
# 8. Start PBT
# ================================================================
analysis = tune.run(
    train_with_pbt,
    name="dual_encoder_pbt",
    scheduler=pbt,
    num_samples=4,
    max_concurrent_trials=2, 
    reuse_actors=True,
    resources_per_trial={"cpu":2,"gpu":1},
    config={"lr":2e-5, "wd":0.01, "warmup":0.06, "lora_rank":8,
            "alpha": 5.0, "margin": 0.35, "contrast_weight": 0.05, **MEDIUM_CONFIG}
)

# ================================================================
# 9. Merge Best LoRA
# ================================================================
best_trial = analysis.get_best_trial(metric="loss", mode="min")
best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="loss", mode="min")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModel.from_pretrained(BASE_MODEL, device_map="auto" if device=="cuda" else None)
peft_model = PeftModel.from_pretrained(base_model, best_checkpoint.path)
merged_model = peft_model.merge_and_unload()
torch.save(merged_model.state_dict(), os.path.join(FULL_MODEL_OUT, "pytorch_model.bin"))
#merged_model.save_pretrained(FULL_MODEL_OUT)
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(FULL_MODEL_OUT)
#tokenizer.save_pretrained(FULL_MODEL_OUT)
print("[INFO] Fully merged model saved to:", FULL_MODEL_OUT)


# COMMAND ----------

from datetime import datetime
import pytz

files = dbutils.fs.ls("dbfs:/tmp/pbt_pii_semantic_dual_encoder_merged")

rows = []
for f in files:
    est = pytz.timezone('US/Eastern')
    utc = pytz.timezone('UTC')
    utc_dt = datetime.fromtimestamp(f.modificationTime / 1000, utc)
    est_dt = utc_dt.astimezone(est)
    rows.append({
        "name": f.name,
        "path": f.path,
        "size": f.size,
        "modified_est_time": est_dt.strftime("%Y-%m-%d %H:%M:%S")
    })

df = spark.createDataFrame(rows)
display(df)

# COMMAND ----------

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# 1. Load model and tokenizer
# ------------------------------
MODEL_PATH = "/dbfs/tmp/pbt_pii_semantic_dual_encoder_merged"
# MODEL_PATH=FULL_MODEL_OUT

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load base model architecture
base_model = AutoModel.from_pretrained(
    MODEL_PATH,
    device_map="auto",  # automatically put on GPU if available
    trust_remote_code=True
)

# If you want to keep the SharedDualEncoder wrapper for PII heads:
class SharedDualEncoderInference(torch.nn.Module):
    def __init__(self, base_model, num_pii_types=11):
        super().__init__()
        self.encoder = base_model
        self.pii_flag_head = torch.nn.Linear(base_model.config.hidden_size, 1)
        self.pii_score_head = torch.nn.Linear(base_model.config.hidden_size, 1)
        self.pii_types_head = torch.nn.Linear(base_model.config.hidden_size, num_pii_types)

    def forward(self, doc_input_ids, doc_attention_mask,
                label_input_ids=None, label_attention_mask=None):
        doc_emb = self.encoder(doc_input_ids, attention_mask=doc_attention_mask).pooler_output
        doc_emb = F.normalize(doc_emb, dim=-1)

        outputs = {
            "doc_emb": doc_emb,
            "pii_flag": torch.sigmoid(self.pii_flag_head(doc_emb)),
            "pii_score": torch.sigmoid(self.pii_score_head(doc_emb)),
            "pii_types": torch.sigmoid(self.pii_types_head(doc_emb))
        }

        if label_input_ids is not None and label_attention_mask is not None:
            label_emb = self.encoder(label_input_ids, attention_mask=label_attention_mask).pooler_output
            label_emb = F.normalize(label_emb, dim=-1)
            outputs["label_emb"] = label_emb

        return outputs

# Wrap the model
model = SharedDualEncoderInference(base_model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


# doc_text = "Rate 2% of gross revenues from UK in-scope activities above threshold; however, taxpayers may apply an alternative calculation method calculated based on operating margin in respect of in-scope activities where they are loss- making or have a very low proﬁ t margin Thresholds £500m revenues from in-scope activities provided globally and £25m of revenue from in-scope activities provided to UK users per 12-month accounting period."
doc_text="The Court formulated the following principle of law for the lower court to follow: in transfer pricing cases involving intra-group transfers of goods between two low-risk companies, where risk is reduced due to the existence of a single production centre essentially producing against confirmed orders, TNMM is more appropriate than CUP because profit margin is a more indicative criterion than nominal price. In such a model, nominal price is not the result of a free market.According to the Court, TNMM with an ROS indicator is fully acceptable and may be preferred to CUP in centrally controlled, low-risk distribution models where prices are not determined by free market forces."
# doc_text="Belanghebbende is een in Nederland gevestigde vennootschap die behoort tot een internationaal opererend concern. Aan belanghebbende zijn voor de jaren 2008 tot en met 2016(navorderings)aanslagen vennootschapsbelasting (vpb) opgelegd met aanzienlijke correcties op de door belanghebbende aangegeven belastbare bedragen. Bovendien heeft de inspecteur voor de jaren 2010 en 2012 tot en met 2016 vergrijpboeten opgelegd. Voor alle jaren is in geschil of de vergoedingen (verrekenprijzen) die diverse in het buitenland gevestigde concernvennootschappen voor leveringen en diensten aan belanghebbende in rekening hebben gebracht als zakelijk kunnen worden aangemerkt. Daarnaast is voor het jaar 2016 in geschil of de beëindiging van door een gevoegde dochtermaatschappij van belanghebbende geëxploiteerde licentierechten dient te worden aangemerkt als een onzakelijke onttrekking aan het vermogen van belanghebbende."
# doc_text="Aveva, quindi, acclarato che la contribuente aveva acquistato i beni dalla capogruppo ad un prezzo lontano dal ‘valore normale’ di cui al citato art. 110, comma 7.La società impugnava, con tre distinti ricorsi, gli atti impositiviinnanzi alla Commissione tributaria provinciale di Firenze, che, previa riunione, li accoglieva: annullava il rilievo relativo al evidenziando che già era stata adottata, per altra annualità, una pronuncia favorevole alla contribuente, sulla base della non applicabilità della metodologia del TNMM e che l’Ufficio aveva errato nella scelta dei ‘comparabili’; annullava il rilievo relativo all’IVA in"
# doc_text="One of the most urgent international climate issues today is the accelerating rise in global temperatures, driven primarily by human activity. This warming trend—commonly referred to as global warming—is the result of increased concentrations of greenhouse gases such as carbon dioxide, methane, and nitrous oxide in the atmosphere. These gases trap heat, causing the Earth’s surface and oceans to warm at unprecedented rates."
# doc_text="In an increasingly interconnected world, health threats can cross borders within hours, challenging national systems and placing significant pressure on global health institutions. Whether emerging from infectious diseases, environmental hazards, geopolitical disruptions, or chronic conditions accelerated by modern lifestyles, these risks remind us that public health is a collective responsibility. This advisory outlines essential principles, recommended strategies, and coordinated actions for individuals, governments, and international organizations to safeguard public health and promote resilient health systems worldwide.1. Overview of the Global Health Landscape In the past decades, the world has seen repeated reminders of how vulnerable health systems can be. Outbreaks of infectious diseases, such as influenza variants, coronaviruses, viral hemorrhagic fevers, and antimicrobial-resistant pathogens, have emphasized the need for robust surveillance systems. Climate-driven events—including heat waves, floods, and vector expansion—have intensified the burden of diseases such as malaria, dengue, and cholera. Simultaneously, chronic illnesses such as diabetes, cardiovascular diseases, and mental health conditions continue to rise in prevalence, placing additional strain on already stressed health infrastructures."
label_text = "Discussion of a tax issue or tax solution."

encoded_doc = tokenizer(doc_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)
encoded_label = tokenizer(label_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)

# ------------------------------
# 3. Forward pass
# ------------------------------
with torch.no_grad():
    outputs = model(
        doc_input_ids=encoded_doc["input_ids"],
        doc_attention_mask=encoded_doc["attention_mask"],
        label_input_ids=encoded_label["input_ids"],
        label_attention_mask=encoded_label["attention_mask"]
    )

# ------------------------------
# 4. Extract embeddings & PII predictions
# ------------------------------
doc_emb = outputs["doc_emb"]          # shape: [1, 1024]
label_emb = outputs["label_emb"]      # shape: [1, 1024]
pii_flag = outputs["pii_flag"]        # [0-1 probability]
pii_score = outputs["pii_score"]      # [0-1 probability]
pii_types = outputs["pii_types"]      # multi-label probabilities

# ------------------------------
# 5. Compute similarity
# ------------------------------
cos_sim = torch.matmul(doc_emb, label_emb.T)  # since embeddings are normalized

print(f"Classification: \"Tax Problem\" & \"Tax Solution probability score \", {cos_sim.item()}")
#print("PII flag probability:", pii_flag.item())
#print("PII score probability:", pii_score.item())
#print("PII type probabilities:", pii_types)
