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
import os, shutil, tempfile
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
DATA_DIR = "/Workspace/Users/a1172300-msp01@ey.net/synthetic_tax_dual_encoder_v10_100/pairs"
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
        # self.pii_flag_head = nn.Linear(self.encoder.config.hidden_size, 1)
        # self.pii_score_head = nn.Linear(self.encoder.config.hidden_size, 1)
        # self.pii_types_head = nn.Linear(self.encoder.config.hidden_size, num_pii_types)
        #self.doc_emb = torch.nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        #self.label_emb = torch.nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        self.config = self.encoder.config

    def forward(self, doc_input_ids, doc_attention_mask,
                label_input_ids, label_attention_mask,
                target=None, soft=None,
                **unused):
        
        out_doc = self.encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            return_dict=True,
        )
        out_label = self.encoder(
            input_ids=label_input_ids,
            attention_mask=label_attention_mask,
            return_dict=True,
        )
        doc_emb = F.normalize(out_doc.last_hidden_state[:, 0], dim=-1)
        label_emb = F.normalize(out_label.last_hidden_state[:, 0], dim=-1)
        #doc_emb = self.doc_emb(out_doc.last_hidden_state[:, 0]) doc_out.last_hidden_state[:, 0]  # or mean pool
        #label_emb = out_label.last_hidden_state[:, 0]
        #doc_emb = self.doc_emb(doc_out.last_hidden_state[:, 0])
        #label_emb = self.label_emb(out_label.last_hidden_state[:, 0])
        # doc_emb = self.encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask).pooler_output
        # label_emb = self.encoder(input_ids=label_input_ids, attention_mask=label_attention_mask).pooler_output
        
        return {
            # "pii_flag": torch.sigmoid(self.pii_flag_head(doc_emb)),
            # "pii_score": torch.sigmoid(self.pii_score_head(doc_emb)),
            # "pii_types": torch.sigmoid(self.pii_types_head(doc_emb)),
            "doc_emb": doc_emb,
            "label_emb": label_emb,
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

        # 1) Normalize embeddings for true cosine similarity. Embeddings are typically L2-normalized. After normalization, cosine similarity is equivalent to dot product and vectors lie on the unit hypersphere(no 2 dimension or 3 dimension) meaning it is vectors are on a sphere more than 3 dimension embeddings can be 384-D, 768-D or 1024-D so in short the vectors are on a sphere means put similar things in the same direction. With normalization stable gradient flow is greatly improved.
        doc_emb = F.normalize(doc_emb, p=2, dim=-1)
        lab_emb = F.normalize(lab_emb, p=2, dim=-1)

        #targets = inputs.get("soft", inputs.get("target"))
        targets = inputs.get("soft") if "soft" in inputs else inputs["target"]
        targets = targets.float()

        # 2) Cosine similarity in [-1, 1] 
        # --- Gradient guards ---
        assert doc_emb.requires_grad, "doc_emb detached / no_grad"
        #cos = (doc_emb * lab_emb).sum(-1).clamp(-1, 1)
        cos = F.cosine_similarity(doc_emb, lab_emb)               # [B]
        assert cos.requires_grad, "cos lost grad"

        # If cos values are small (close to 0), then sigmoid(0) ≈ 0.5.
        # If the model starts with random weights and never moves far from zero, the gradient signal is weak, and the model doesn’t learn meaningful similarity.
        print("cos range:", cos.min().item(), cos.max().item())
        
        # Logging (detached) — keep training path clean
        with torch.no_grad():
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

        # The reason for using BCEWithLogitsLoss in this case is to take advantage of its numerical stability and efficiency. BCEWithLogitsLoss is designed to work with logits, which are the raw, unnormalized scores output by a model. However, it can also be used with other types of inputs, such as the scaled cosine similarity used in this code.

        # The key insight here is that the BCEWithLogitsLoss function is equivalent to the BCELoss function when the input is passed through a sigmoid function. In other words, BCEWithLogitsLoss(x) == BCELoss(torch.sigmoid(x)).

        # By using BCEWithLogitsLoss with the scaled cosine similarity, the code is effectively applying a sigmoid function to the input, which maps the cosine similarity values to a probability range between 0 and 1. This is useful because the cosine similarity values are not necessarily probabilities, but the sigmoid function helps to interpret them as such.
        
        if pos_weight is not None:  # e.g., to handle class imbalance
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(scaled_cos, targets.float())
        else:
            bce = nn.BCEWithLogitsLoss()(scaled_cos, targets.float())
        # Applying sigmoid to cos_prob gives us the same loss as BCEWithLogitsLoss
        #bce = nn.BCELoss()(torch.sigmoid(cos), targets)
        
        
        # If cos is close to 0 then Contrastive loss will also be small because (cos - margin) or (1 - cos) barely changes.
        # 4) Contrastive loss on raw cosine (no sigmoid)
        #    Positives: penalize if similarity < 1
        #    Negatives: penalize if similarity > margin
        contrast = (targets * (1 - cos).clamp(min=0) + (1 - targets) * (cos - margin).clamp(min=0)).mean()
        # contrast = (
        #     targets * (1 - cos_prob).clamp(min=0) +
        #     (1 - targets) * (cos_prob - margin).clamp(min=0)
        # ).mean()
        
        
        loss = bce + contrast_weight * contrast
        assert loss.requires_grad, "final loss does not require grad"
        print("BCE Loss:", bce.item())
        print("Contrastive Loss:", contrast.item())
        print("Loss:", loss.item())
        print(f"[diag] mean_pos_cos={mean_pos_cos:.3f}, mean_neg_cos={mean_neg_cos:.3f}, "
             f"BCE={bce.item():.4f}, Contrast={contrast.item():.4f}, Total={loss.item():.4f}")
        
        # PII losses
        # pii_losses = {}
        # for k, criterion in [("pii_flag", nn.BCELoss()), ("pii_score", nn.MSELoss()), ("pii_types", nn.BCELoss())]:
        #     #val = inputs.pop(k, None)
        #     val = inputs.get(k, None)
        #     if val is not None:
        #         mask = val > 0 if k != "pii_types" else val.sum(dim=1) > 0
        #         # loss += criterion(outputs[k].squeeze()[mask], val.float()[mask]) if mask.any() else torch.tensor(0.0, device=loss.device)
        #         # Do NOT add pii_losses to 'loss'
        #         if mask.any():
        #             pii_losses[k] = criterion(outputs[k].squeeze()[mask], val.float()[mask]).item()
        #         else:
        #             pii_losses[k] = 0.0

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
        r=32,#config["lora_rank"],
        lora_alpha=32,
        target_modules=["query","key","value","dense"],
        layers_to_transform=list(range(20,24)),
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION" # The model is used to produce embeddings the output we are expecting is representaiton not a label. We are calculating the cosine similarity between the two embeddings lable and doc text embedding and then calculating the loss using the compute loss function.
        # task_type="SEQ_CLS", We are not predicting any labels, so we don't need to specify a task type as Sequence Classification
        # modules_to_save=["pii_flag_head","pii_score_head","pii_types_head"] # removed "doc_emb", "label_emb" as they are tensors and not modules
    )
    # model = get_peft_model(model, lora_cfg)
    # First iteration → initializes LoRA adapters.
    # Subsequent iterations → resumes from previous trial state.
    # Resume from Tune checkpoint if present; else initialize LoRA
    # checkpoint = session.get_checkpoint()
    # if checkpoint:
    #     print(f"[INFO] Resuming from checkpoint: {checkpoint.path}")
    #     #model = SharedDualEncoder(BASE_MODEL)
    #     model = PeftModel.from_pretrained(model, checkpoint.path)
    # else:
   
    #model = SharedDualEncoder(BASE_MODEL)
    model = get_peft_model(model, lora_cfg)  # only if no checkpoint
    # Datasets
    train_ds = DualEncoderDataset(os.path.join(DATA_DIR, "train.jsonl"), tokenizer)
    val_ds = DualEncoderDataset(os.path.join(DATA_DIR, "val.jsonl"), tokenizer)
    # train_ds = DualEncoderDataset("/Workspace/Users/a1172300-msp01@ey.net/trainpii.jsonl", tokenizer)
    # val_ds   = DualEncoderDataset("/Workspace/Users/a1172300-msp01@ey.net/valpii.jsonl", tokenizer)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    plateau_flag = False  # Initialize before PBT loop
    recent_losses=[]
    best_loss=float("inf")
    early_stop_patience=3
    early_stop_min_delta=0.001
    # How much learning happens before PBT considers this configuration ready for exploitation?
    # This is the number of steps per PBT iteration.
    # steps_per_epoch=ceil(num_train_sample/config["batch_size"]x grad_accum)--> 200/16x1=18.75--> 19 steps
    # For Full epoc(19)--> PBT reacts too slowly so let PBT see 25-50% of the epoch
    # steps_per_pbt_iter=steps_per_epoch x fraction_of_epoch        
    steps_per_pbt_iter=200
    args = TrainingArguments(
        output_dir=session.get_trial_dir(),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        weight_decay=config["wd"],
        warmup_ratio=config["warmup"],
        # give PBT enough cycles to exploit/mutate:
        # If max steps > 0 then num_train_epochs is ignored. Training stops at max steps. HF behavior. None for now as max steps > 0
        num_train_epochs=2,  
        #evaluation_strategy="steps",
        evaluation_strategy="epoch",
        #eval_steps=config["eval_steps"],        
        #logging_strategy="steps",
        logging_strategy="epoch",
        logging_steps=10,
        #save_strategy="steps",
        save_strategy="epoch",
        #save_steps=config["save_steps"],
        #save_steps="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        prediction_loss_only=True,
        max_steps=200,        
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
    trainer.save_model(session.get_trial_dir())  # Save a checkpoint
    
    # If resuming, let HF restore optimizer/scheduler from the checkpoint folder
    #if checkpoint:
    #     print(f"[INFO] Resuming from checkpoint and let HF restore optimizer/scheduler from the checkpoint folder: {checkpoint.path}")
    #     trainer.train(resume_from_checkpoint=checkpoint.path)
    # else:
    # trainer.train()

    # --- PBT iteration loop ---
    # For each iteration: train bit more, evaluate, report a metric + checkpoint
    # NOTE: "training_iteration" increases once per tune.report() call.
    for it in range(config["max_pbt_iters"]):
        print(f"[PBT] Iteration {it+1}")
        
        # Resume from Ray Tune checkpoint if available, otherwise start fresh
        ckpt = session.get_checkpoint()
        if ckpt:
            print (f"[INFO] Resuming from checkpoint: {ckpt.path}")
            trainer.train(resume_from_checkpoint=ckpt.path)
        else:
            print("[INFO] Starting fresh trial")
            trainer.train()
        print(f"[INFO] After training: global_step = {trainer.state.global_step}")
        # Current HF step after training
        current_step = trainer.state.global_step
        steps_completed = current_step
        # steps_remaining = steps_per_trial - steps_completed

        
        print(f"[INFO] Steps completed so far: {steps_completed}")
        
        metrics = trainer.evaluate() # <-- evaluate first
        print(f"[INFO] Evaluation complete at HF step {trainer.state.global_step}, PBT iteration {it+1}")
        print(f"[INFO] Eval loss: {metrics.get('eval_loss', 'N/A')}")
        score = metrics.get("eval_loss", 999) # PBT uses 'loss' as metric in scheduler
        loss = metrics.get("eval_loss", None)

        if loss is None:
            print("[WARN] eval_loss missing, skipping early stop check")
            continue
    
        # Track losses
        recent_losses.append(loss)
        print(f"[DEBUG] recent_losses={recent_losses}, patience={early_stop_patience}, delta={early_stop_min_delta}")
        print(f"[DEBUG] Differences={[recent_losses[i]-recent_losses[i-1] for i in range(1,len(recent_losses))]}")

        if len(recent_losses) > early_stop_patience:
            recent_losses.pop(0)

        # Check improvement
        if loss < best_loss - early_stop_min_delta:
            best_loss = loss
        else:
            print(
                f"[EarlyStop] No improvement. "
                f"Best={best_loss:.4f}, Current={loss:.4f}"
            )
        
        # ---- EARLY STOP CONDITION ----
        if (
            len(recent_losses) == early_stop_patience
            and all(
                recent_losses[i] >= recent_losses[i - 1] - early_stop_min_delta
                for i in range(1, len(recent_losses))
            )
        ):
            plateau_flag = True
            print(f"[EarlyStop] Plateau detected for trial after {early_stop_patience} evals.")
            print(
                f"[EarlyStop] Triggered after {early_stop_patience} "
                f"non-improving evals: {recent_losses}"
            )
            break

        # Create a fresh checkpoint for Ray to exploit
        with tempfile.TemporaryDirectory() as tmp:
            # save model + tokenizer + HF state (so exploit works)
            trainer.model.save_pretrained(tmp)
            trainer.save_model(tmp)
            
            tokenizer.save_pretrained(tmp)
            # Copy HF trainer_state.json into tmp (save_state writes it to args.output_dir)
            trainer.save_state()               # writes to trainer.args.output_dir
            src_state = os.path.join(trainer.args.output_dir, "trainer_state.json")
            dst_state = os.path.join(tmp, "trainer_state.json")
            shutil.copy2(src_state, dst_state)

            # Optionally copy training_args.bin for completeness
            src_args = os.path.join(trainer.args.output_dir, "training_args.bin")
            if os.path.exists(src_args):
                shutil.copy2(src_args, os.path.join(tmp, "training_args.bin"))
            #session.report({"loss": score}, checkpoint=Checkpoint.from_directory(tmp))
            print(f"[INFO] Checkpoint created at HF step {trainer.state.global_step}, PBT iteration {it+1}")
            session.report(
                {"loss": score},
                #eval_f1_macro=metrics.get("eval_f1_macro", None),
                #iter=it + 1,
                checkpoint=Checkpoint.from_directory(tmp))
    
    # Save the final model state
    trainer.save_model(session.get_trial_dir())
    session.report(
        {"loss": score},
        # eval_f1_macro=metrics.get("eval_f1_macro", None),
        # iter=it + 1,
        checkpoint=Checkpoint.from_directory(session.get_trial_dir())
    )

# ================================================================
# 7. Dataset Preset & PBT Scheduler
# ================================================================
MEDIUM_CONFIG = dict(batch_size=23, grad_accum=1, num_workers=2,
                     eval_steps=10, save_steps=50, max_pbt_iters=4)
                    #  early_stop_patience=3, early_stop_min_delta=1e-4) # last N eval losses for early stopping # pbt loop runs (max_pbt_iters) times, iterations this does not affect the number of epochs or trials.

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="loss",
    mode="min",
    perturbation_interval=1, # That means PBT will consider exploitation after 3 consecutive reports. If loop runs 4(max_pbt_iters) times gives enough cycles for PBT to exploit and mutate multiple times.
    hyperparam_mutations={
        "lr": tune.loguniform(1e-6, 5e-5),
        "wd": tune.uniform(0, 0.2),
        "warmup": tune.uniform(0,0.3),        
        #"lora_rank": [4,8,16,32],
        "alpha": tune.uniform(3.0,8.0),          # logit scaling
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
    num_samples=6, # training will run for 4 trials. Each trial will train the model for 2 epochs with batch_size of 16 as mentioned in the MEDIUM_CONFIG as batch_size
    max_concurrent_trials=2, 
    reuse_actors=True,
    resources_per_trial={"cpu":2,"gpu":1},
    config=MEDIUM_CONFIG,
    stop={"training_iteration": 100},    
    # resume_from_checkpoint=checkpoint_path # This is a Ray Air Train setup
    resume="AUTO" # For Ray Tune
    # remove the initial configuration below to check if hyperparameters are perturbed properly
    # config={"lr":2e-5, "wd":0.01, "warmup":0.06, "lora_rank":8,
    #         "alpha": 5.0, "margin": 0.35, "contrast_weight": 0.05, **MEDIUM_CONFIG}
)
    
# We have num_samples=8 trials which will train the model for 2 epochs with a batch size of 30. So 8 trials X 2 epochs=16(forward and backward pass).
# Suppose we have 300 training samples in the dataset then for each epoch the model will see all the 300 training samples but in batches of 30 samples each.
# The PBT scheduler will perturb the hyperparameters every perturbation_interval=5 training iterations (i.e., every 5 batches).
# The PBT loop will run for max_pbt_iters=12 iterations, which means that the scheduler will perturb the hyperparameters up to 12 times during each trial.
# Since each trial trains the model for 2 epochs, the PBT loop will have enough time to perturb the hyperparameters multiple times during each trial.
# So, in summary:

# 16 trials will be run, each training the model for 2 epochs with a batch size of 16.
# The PBT scheduler will perturb the hyperparameters every 5 training iterations, up to 12 times during each trial.
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
# ensure that the same tokenizer is used for inference
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(FULL_MODEL_OUT)
#tokenizer.save_pretrained(FULL_MODEL_OUT)
print("[INFO] Best checkpoint saved to:", best_checkpoint.path)
print("[INFO] Fully merged model saved to:", FULL_MODEL_OUT)


# COMMAND ----------

from ray.tune import ExperimentAnalysis

analysis = ExperimentAnalysis("/root/ray_results/dual_encoder_pbt")
best_checkpoint = analysis.get_best_checkpoint(
    trial=analysis.get_best_trial(metric="loss", mode="min"),
    metric="loss",
    mode="min"
)

print(f"[INFO] Best checkpoint: {best_checkpoint}")

if os.path.exists(best_checkpoint.to_directory()):
    print(f"Resuming from checkpoint {best_checkpoint}")


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
        # self.pii_flag_head = torch.nn.Linear(base_model.config.hidden_size, 1)
        # self.pii_score_head = torch.nn.Linear(base_model.config.hidden_size, 1)
        # self.pii_types_head = torch.nn.Linear(base_model.config.hidden_size, num_pii_types)

    def forward(self, doc_input_ids, doc_attention_mask,
                label_input_ids=None, label_attention_mask=None):
        doc_emb = self.encoder(doc_input_ids, attention_mask=doc_attention_mask).pooler_output
        doc_emb = F.normalize(doc_emb, dim=-1)

        outputs = {
            "doc_emb": doc_emb
            # "pii_flag": torch.sigmoid(self.pii_flag_head(doc_emb)),
            # "pii_score": torch.sigmoid(self.pii_score_head(doc_emb)),
            # "pii_types": torch.sigmoid(self.pii_types_head(doc_emb))
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
# doc_text="The Court formulated the following principle of law for the lower court to follow: in transfer pricing cases involving intra-group transfers of goods between two low-risk companies, where risk is reduced due to the existence of a single production centre essentially producing against confirmed orders, TNMM is more appropriate than CUP because profit margin is a more indicative criterion than nominal price. In such a model, nominal price is not the result of a free market.According to the Court, TNMM with an ROS indicator is fully acceptable and may be preferred to CUP in centrally controlled, low-risk distribution models where prices are not determined by free market forces."
# doc_text="Belanghebbende is een in Nederland gevestigde vennootschap die behoort tot een internationaal opererend concern. Aan belanghebbende zijn voor de jaren 2008 tot en met 2016(navorderings)aanslagen vennootschapsbelasting (vpb) opgelegd met aanzienlijke correcties op de door belanghebbende aangegeven belastbare bedragen. Bovendien heeft de inspecteur voor de jaren 2010 en 2012 tot en met 2016 vergrijpboeten opgelegd. Voor alle jaren is in geschil of de vergoedingen (verrekenprijzen) die diverse in het buitenland gevestigde concernvennootschappen voor leveringen en diensten aan belanghebbende in rekening hebben gebracht als zakelijk kunnen worden aangemerkt. Daarnaast is voor het jaar 2016 in geschil of de beëindiging van door een gevoegde dochtermaatschappij van belanghebbende geëxploiteerde licentierechten dient te worden aangemerkt als een onzakelijke onttrekking aan het vermogen van belanghebbende."
doc_text="Aveva, quindi, acclarato che la contribuente aveva acquistato i beni dalla capogruppo ad un prezzo lontano dal ‘valore normale’ di cui al citato art. 110, comma 7.La società impugnava, con tre distinti ricorsi, gli atti impositiviinnanzi alla Commissione tributaria provinciale di Firenze, che, previa riunione, li accoglieva: annullava il rilievo relativo al evidenziando che già era stata adottata, per altra annualità, una pronuncia favorevole alla contribuente, sulla base della non applicabilità della metodologia del TNMM e che l’Ufficio aveva errato nella scelta dei ‘comparabili’; annullava il rilievo relativo all’IVA in"
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


# COMMAND ----------

