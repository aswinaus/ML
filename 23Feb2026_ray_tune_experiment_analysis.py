# Databricks notebook source

import pandas as pd
from ray.tune import ExperimentAnalysis

# Path to your Ray Tune results directory
results_path = "/root/ray_results/dual_encoder_pbt"

# Load experiment analysis
analysis = ExperimentAnalysis(results_path)

# Combine all trial dataframes into one
df_list = []
for trial_id, trial_df in analysis.trial_dataframes.items():
    trial_df["trial_id"] = trial_id
    df_list.append(trial_df)

df = pd.concat(df_list, ignore_index=True)

# Compute ETA
target_iters = 4  # From your config: max_pbt_iters
latest = df.groupby("trial_id").last().reset_index()
avg_time_per_iter = latest["time_this_iter_s"].mean()
remaining_iters = target_iters - latest["training_iteration"].mean()
eta_seconds = remaining_iters * avg_time_per_iter
eta_minutes = eta_seconds / 60

print("Current progress per trial:")
print(latest[["trial_id","training_iteration","time_total_s","loss"]])
print(f"\nEstimated remaining time: ~{eta_minutes:.2f} minutes for all trials.")


# COMMAND ----------


from ray.tune import ExperimentAnalysis
analysis = ExperimentAnalysis("/root/ray_results/dual_encoder_pbt")

# Show last config for each trial
for trial in analysis.trials:
    print(f"Trial {trial.trial_id} current config:", trial.config)


# COMMAND ----------


# Databricks-friendly plotting
%matplotlib inline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis

RESULTS_PATH = "/root/ray_results/dual_encoder_pbt"
HP_KEYS = ["lr", "wd", "warmup", "lora_rank", "alpha", "margin", "contrast_weight"]

# Load Ray Tune analysis
analysis = ExperimentAnalysis(RESULTS_PATH)

# --- Helper: ensure each trial DF has an iteration column, and locate per-iteration HP columns if logged ---
def ensure_iteration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "training_iteration" not in df.columns:
        df["training_iteration"] = np.arange(1, len(df) + 1)
    return df

def find_hp_columns(df: pd.DataFrame) -> dict:
    hp_cols = {}
    for k in HP_KEYS:
        for candidate in [f"config/{k}", f"params.{k}", k]:
            if candidate in df.columns:
                hp_cols[k] = candidate
                break
    return hp_cols

# --- Build “long” dataframe with (trial_id, iteration, hyperparam, value) ---
records = []
loss_rows = []

for trial_id, trial_df in analysis.trial_dataframes.items():
    df = ensure_iteration(trial_df)

    # Capture loss per iteration (if present)
    if "loss" in df.columns:
        loss_vals = pd.to_numeric(df["loss"], errors="coerce")
        for it, v in zip(df["training_iteration"].values, loss_vals.values):
            loss_rows.append({"trial_id": trial_id, "iteration": int(it), "loss": float(v) if pd.notna(v) else np.nan})

    # Try to read hyperparameters logged per-iteration
    hp_map = find_hp_columns(df)

    if not hp_map:
        # Fall back: broadcast the *current* config to all rows (gives latest values; not historic)
        trial_obj = next(t for t in analysis.trials if t.trial_id == trial_id)
        cfg = trial_obj.config
        for k in HP_KEYS:
            df[k] = cfg.get(k, np.nan)
        hp_map = {k: k for k in HP_KEYS}

    for k, col in hp_map.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        for it, v in zip(df["training_iteration"].values, vals.values):
            records.append({
                "trial_id": trial_id,
                "iteration": int(it),
                "hyperparam": k,
                "value": float(v) if pd.notna(v) else np.nan
            })

hp_long = pd.DataFrame(records)
loss_df = pd.DataFrame(loss_rows)

# If nothing was collected (edge case), stop early with a friendly message
if hp_long.empty:
    raise RuntimeError("No hyperparameter values found. Check that your Tune logs include config/params columns or provide current configs.")

# --- Graph 1: Combined overlay (normalized) ---
def zscore(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    mu = g["value"].mean()
    sd = g["value"].std()
    if not sd or np.isnan(sd) or sd == 0:
        sd = 1.0
    g["z"] = (g["value"] - mu) / sd
    return g

hp_norm = hp_long.groupby(["trial_id", "hyperparam"]).apply(zscore).reset_index(drop=True)

plt.figure(figsize=(13, 6))
for (trial_id, hp), g in hp_norm.groupby(["trial_id", "hyperparam"]):
    plt.plot(g["iteration"], g["z"], label=f"Iteration {g['iteration'].max()} - {trial_id}:{hp}", alpha=0.85)
plt.title("PBT Hyperparameters (normalized) over iterations")
plt.xlabel("Iteration")
plt.ylabel("Normalized value (z-score)")
plt.grid(True, alpha=0.3)
plt.legend(ncol=3, fontsize=8, frameon=False)
plt.show()

# --- Graph 2: Multi-panel dashboard with actual values ---
hp_order = HP_KEYS
n = len(hp_order)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows), sharex=True)
axes = axes.flatten()

for i, hp in enumerate(hp_order):
    ax = axes[i]
    for trial_id, g in hp_long[hp_long["hyperparam"] == hp].groupby("trial_id"):
        ax.plot(g["iteration"], g["value"], marker="o", ms=3, label=trial_id, alpha=0.9)
    ax.set_title(hp)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(hp)
    ax.grid(True, alpha=0.3)

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
fig.suptitle("PBT Hyperparameter Evolution (actual values) per trial", y=1.02)
plt.tight_layout()
plt.show()

# --- (Optional) Graph 3: add loss overlay on each panel via a twin axis ---
if not loss_df.empty:
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows), sharex=True)
    axes = axes.flatten()

    for i, hp in enumerate(hp_order):
        ax = axes[i]
        for trial_id, g in hp_long[hp_long["hyperparam"] == hp].groupby("trial_id"):
            ax.plot(g["iteration"], g["value"], marker="o", ms=3, label=f"{trial_id}", alpha=0.9)
        ax.set_title(f"{hp} (loss overlay)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(hp)
        ax.grid(True, alpha=0.3)

        # Twin axis for loss
        ax2 = ax.twinx()
        for trial_id, lg in loss_df.groupby("trial_id"):
            ax2.plot(lg["iteration"], lg["loss"], color="red", linestyle="--", alpha=0.5, label=f"{trial_id} loss")
        ax2.set_ylabel("loss", color="red")
        ax2.tick_params(axis='y', labelcolor="red")

    # Shared legend (combine handles)
    fig.suptitle("Hyperparameters vs Loss per trial", y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("Loss not found in trial logs; skipping loss overlay chart.")


# COMMAND ----------

import json

def parse_result_json(file_path, default_max_iters=5):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None, None

    if not lines:
        return None, None, None

    entry = lines[-1]  # last report
    train_iter = entry.get('training_iteration')
    elapsed = entry.get('time_total_s')
    max_iters = entry.get('config', {}).get('max_pbt_iters', default_max_iters)

    return train_iter, elapsed, max_iters

def compute_remaining(train_iter, elapsed, max_iters):
    if train_iter is None or elapsed is None or max_iters is None:
        return None, None
    remaining = max(max_iters - train_iter, 0)
    if train_iter == 0:
        return remaining, None
    avg_time = elapsed / train_iter
    return remaining, remaining * avg_time

# Usage
file_path = '/root/ray_results/dual_encoder_pbt/train_with_pbt_609b7_00000_0_2025-12-21_19-00-55/result.json'
train_iter, elapsed, max_iters = parse_result_json(file_path)

if train_iter and elapsed and max_iters:
    remaining, remaining_time = compute_remaining(train_iter, elapsed, max_iters)
    print(f"Completed loops: {train_iter}/{max_iters}")
    print(f"Elapsed time: {elapsed:.0f}s ({elapsed/60:.1f} min, {elapsed/3600:.2f} hr)")
    print(f"Remaining loops: {remaining}")
    print(f"Estimated time left: {remaining_time:.0f}s ({remaining_time/60:.1f} min, {remaining_time/3600:.2f} hr)")
else:
    print("Could not extract values. Check file path or JSON format.")


# COMMAND ----------

# Databricks-friendly plotting
%matplotlib inline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis

RESULTS_PATH = "/root/ray_results/dual_encoder_pbt"
HP_KEYS = ["lr", "wd", "warmup", "lora_rank", "alpha", "margin", "contrast_weight"]

# Load Ray Tune analysis
analysis = ExperimentAnalysis(RESULTS_PATH)

# --- Helper: ensure iteration column ---
def ensure_iteration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "training_iteration" not in df.columns:
        df["training_iteration"] = np.arange(1, len(df) + 1)
    return df

# --- Detect hyperparameter columns ---
def find_hp_columns(df: pd.DataFrame) -> dict:
    hp_cols = {}
    for k in HP_KEYS:
        for candidate in [f"config/{k}", f"params.{k}", k]:
            if candidate in df.columns:
                hp_cols[k] = candidate
                break
    return hp_cols

# --- Detect loss column dynamically ---
def find_loss_column(df: pd.DataFrame) -> str:
    for candidate in ["loss", "eval_loss", "train_loss", "objective"]:
        if candidate in df.columns:
            return candidate
    return None

# --- Build long dataframe ---
records = []
loss_rows = []

for trial_id, trial_df in analysis.trial_dataframes.items():
    df = ensure_iteration(trial_df)

    # Capture loss per iteration
    loss_col = find_loss_column(df)
    if loss_col:
        loss_vals = pd.to_numeric(df[loss_col], errors="coerce")
        for it, v in zip(df["training_iteration"].values, loss_vals.values):
            loss_rows.append({"trial_id": trial_id, "iteration": int(it), "loss": float(v) if pd.notna(v) else np.nan})

    # Capture hyperparameters
    hp_map = find_hp_columns(df)
    if not hp_map:
        trial_obj = next(t for t in analysis.trials if t.trial_id == trial_id)
        cfg = trial_obj.config
        for k in HP_KEYS:
            df[k] = cfg.get(k, np.nan)
        hp_map = {k: k for k in HP_KEYS}

    for k, col in hp_map.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        for it, v in zip(df["training_iteration"].values, vals.values):
            records.append({
                "trial_id": trial_id,
                "iteration": int(it),
                "hyperparam": k,
                "value": float(v) if pd.notna(v) else np.nan
            })

hp_long = pd.DataFrame(records)
loss_df = pd.DataFrame(loss_rows)

if hp_long.empty:
    raise RuntimeError("No hyperparameter values found. Check Tune logs or configs.")

# --- Summary Table with Start/End Time ---
summary = []
for tid, group in loss_df.groupby('trial_id'):
    sorted_grp = group.sort_values('iteration')
    start_loss = sorted_grp['loss'].iloc[0]
    end_loss = sorted_grp['loss'].iloc[-1]
    improvement = (start_loss - end_loss) / start_loss * 100

    # Extract timing info from trial dataframe
    trial_df = analysis.trial_dataframes[tid]
    start_time = trial_df['time_total_s'].iloc[0] if 'time_total_s' in trial_df.columns else None
    end_time = trial_df['time_total_s'].iloc[-1] if 'time_total_s' in trial_df.columns else None
    duration_sec = end_time - start_time if start_time and end_time else None

    # Hyperparameters
    hp_vals = hp_long[hp_long['trial_id'] == tid].groupby('hyperparam')['value'].last().to_dict()

    record = {
        'trial_id': tid,
        'start_loss': start_loss,
        'end_loss': end_loss,
        'pct_improvement': improvement,
        # 'start_time_s': start_time,
        # 'end_time_s': end_time,
        'duration_sec': duration_sec
    }
    record.update(hp_vals)
    summary.append(record)

summary_df = pd.DataFrame(summary)

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(summary_df)

# Optional: Export to CSV
summary_df.to_csv("/dbfs/FileStore/trial_summary.csv", index=False)

# --- Plot improvement with annotations ---
fig, ax = plt.subplots(figsize=(10, 6))
df_plot = summary_df.dropna(subset=['pct_improvement'])
bars = ax.bar(df_plot['trial_id'], df_plot['pct_improvement'], color='skyblue')

# Add annotations above bars
for bar, pct in zip(bars, df_plot['pct_improvement']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{pct:.1f}%", ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Trial ID')
ax.set_ylabel('Percentage Improvement of Loss')
ax.set_title('Loss Improvement per Trial')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# COMMAND ----------

# Databricks-friendly plotting
%matplotlib inline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis

RESULTS_PATH = "/root/ray_results/dual_encoder_pbt"
HP_KEYS = ["lr", "wd", "warmup", "lora_rank", "alpha", "margin", "contrast_weight"]

# Load Ray Tune analysis
analysis = ExperimentAnalysis(RESULTS_PATH)

# --- Helper: ensure iteration column ---
def ensure_iteration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "training_iteration" not in df.columns:
        df["training_iteration"] = np.arange(1, len(df) + 1)
    return df

# --- Detect hyperparameter columns ---
def find_hp_columns(df: pd.DataFrame) -> dict:
    hp_cols = {}
    for k in HP_KEYS:
        for candidate in [f"config/{k}", f"params.{k}", k]:
            if candidate in df.columns:
                hp_cols[k] = candidate
                break
    return hp_cols

# --- Detect loss column dynamically ---
def find_loss_column(df: pd.DataFrame) -> str:
    for candidate in ["loss", "eval_loss", "train_loss", "objective"]:
        if candidate in df.columns:
            return candidate
    return None

# --- Build long dataframe ---
records = []
loss_rows = []

for trial_id, trial_df in analysis.trial_dataframes.items():
    df = ensure_iteration(trial_df)

    # Capture loss per iteration
    loss_col = find_loss_column(df)
    if loss_col:
        loss_vals = pd.to_numeric(df[loss_col], errors="coerce")
        for it, v in zip(df["training_iteration"].values, loss_vals.values):
            loss_rows.append({"trial_id": trial_id, "iteration": int(it), "loss": float(v) if pd.notna(v) else np.nan})

    # Capture hyperparameters
    hp_map = find_hp_columns(df)
    if not hp_map:
        trial_obj = next(t for t in analysis.trials if t.trial_id == trial_id)
        cfg = trial_obj.config
        for k in HP_KEYS:
            df[k] = cfg.get(k, np.nan)
        hp_map = {k: k for k in HP_KEYS}

    for k, col in hp_map.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        for it, v in zip(df["training_iteration"].values, vals.values):
            records.append({
                "trial_id": trial_id,
                "iteration": int(it),
                "hyperparam": k,
                "value": float(v) if pd.notna(v) else np.nan
            })

hp_long = pd.DataFrame(records)
loss_df = pd.DataFrame(loss_rows)

if hp_long.empty:
    raise RuntimeError("No hyperparameter values found. Check Tune logs or configs.")

# --- Summary Table with Start/End Time ---
summary = []
for tid, group in loss_df.groupby('trial_id'):
    sorted_grp = group.sort_values('iteration')
    start_loss = sorted_grp['loss'].iloc[0]
    end_loss = sorted_grp['loss'].iloc[-1]
    improvement = (start_loss - end_loss) / start_loss * 100

    # Extract timing info from trial dataframe
    trial_df = analysis.trial_dataframes[tid]
    start_time = trial_df['time_total_s'].iloc[0] if 'time_total_s' in trial_df.columns else None
    end_time = trial_df['time_total_s'].iloc[-1] if 'time_total_s' in trial_df.columns else None
    duration_sec = end_time - start_time if start_time and end_time else None

    # Hyperparameters
    hp_vals = hp_long[hp_long['trial_id'] == tid].groupby('hyperparam')['value'].last().to_dict()

    record = {
        'trial_id': tid,
        'start_loss': start_loss,
        'end_loss': end_loss,
        'pct_improvement': improvement,
        'duration_sec': duration_sec
    }
    record.update(hp_vals)
    summary.append(record)

summary_df = pd.DataFrame(summary)

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(summary_df)

# Optional: Export to CSV
summary_df.to_csv("/dbfs/FileStore/trial_summary.csv", index=False)

# --- Plot improvement with annotations ---
fig, ax = plt.subplots(figsize=(10, 6))
df_plot = summary_df.dropna(subset=['pct_improvement'])
bars = ax.bar(df_plot['trial_id'], df_plot['pct_improvement'], color='skyblue')

# Add annotations above bars
for bar, pct in zip(bars, df_plot['pct_improvement']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{pct:.1f}%", ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Trial ID')
ax.set_ylabel('Percentage Improvement of Loss')
ax.set_title('Loss Improvement per Trial')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# COMMAND ----------

%pip install gputil

# COMMAND ----------

import torch
import GPUtil

# Get the GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the GPU memory usage
gpu_mem_usage = GPUtil.showUtilization()

print(gpu_mem_usage)

# Get the GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the GPU specs
gpu_specs = GPUtil.getGPUs()

for gpu in gpu_specs:
    print(f"GPU Name: {gpu.name}")
    print(f"GPU Memory: {gpu.memoryTotal} MB")
    print(f"GPU Memory Usage: {gpu.memoryUsed} MB")
    print(f"GPU Utilization: {gpu.load * 100}%")

# COMMAND ----------

# To Check remining iterations in training

from ray.tune import ExperimentAnalysis

analysis = ExperimentAnalysis("/root/ray_results/dual_encoder_pbt")

for trial in analysis.trials:
    current_iter = trial.last_result.get("training_iteration", 0)
    max_iters = trial.config.get("max_pbt_iters", 0)
    remaining = max_iters - current_iter
    print(f"Trial {trial.trial_id}: {remaining} iterations left (current: {current_iter}/{max_iters})")


# COMMAND ----------


for trial in analysis.trials:
    max_pbt_iters = trial.config.get("max_pbt_iters", 0)
    exploit_done = trial.last_result.get("pbt_exploit_count", 0)  # may be missing
    remaining = max_pbt_iters - exploit_done
    print(f"Trial {trial.trial_id}: {remaining} perturbations left (done: {exploit_done}/{max_pbt_iters})")


# COMMAND ----------

# Databricks-friendly plotting
%matplotlib inline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
import smtplib
from email.mime.text import MIMEText

RESULTS_PATH = "/root/ray_results/dual_encoder_pbt"
HP_KEYS = ["lr", "wd", "warmup", "lora_rank", "alpha", "margin", "contrast_weight"]

# Email configuration
SMTP_SERVER = "smtp.xyz.com"
SMTP_PORT = 465
SENDER_EMAIL = ""
EMAIL_PASSWORD = ""  # Use environment variable for security
RECIPIENT_EMAIL = ""

def send_email(subject, body, to_email):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())



# Load Ray Tune analysis
analysis = ExperimentAnalysis(RESULTS_PATH)

# --- Helper: ensure iteration column ---
def ensure_iteration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "training_iteration" not in df.columns:
        df["training_iteration"] = np.arange(1, len(df) + 1)
    return df

# --- Detect hyperparameter columns ---
def find_hp_columns(df: pd.DataFrame) -> dict:
    hp_cols = {}
    for k in HP_KEYS:
        for candidate in [f"config/{k}", f"params.{k}", k]:
            if candidate in df.columns:
                hp_cols[k] = candidate
                break
    return hp_cols

# --- Detect loss column dynamically ---
def find_loss_column(df: pd.DataFrame) -> str:
    for candidate in ["loss", "eval_loss", "train_loss", "objective"]:
        if candidate in df.columns:
            return candidate
    return None

# --- Build long dataframe ---
records = []
loss_rows = []

for trial_id, trial_df in analysis.trial_dataframes.items():
    df = ensure_iteration(trial_df)

    # Capture loss per iteration
    loss_col = find_loss_column(df)
    if loss_col:
        loss_vals = pd.to_numeric(df[loss_col], errors="coerce")
        for it, v in zip(df["training_iteration"].values, loss_vals.values):
            loss_rows.append({"trial_id": trial_id, "iteration": int(it), "loss": float(v) if pd.notna(v) else np.nan})

    # Capture hyperparameters
    hp_map = find_hp_columns(df)
    if not hp_map:
        trial_obj = next(t for t in analysis.trials if t.trial_id == trial_id)
        cfg = trial_obj.config
        for k in HP_KEYS:
            df[k] = cfg.get(k, np.nan)
        hp_map = {k: k for k in HP_KEYS}

    for k, col in hp_map.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        for it, v in zip(df["training_iteration"].values, vals.values):
            records.append({
                "trial_id": trial_id,
                "iteration": int(it),
                "hyperparam": k,
                "value": float(v) if pd.notna(v) else np.nan
            })

hp_long = pd.DataFrame(records)
loss_df = pd.DataFrame(loss_rows)

if hp_long.empty:
    raise RuntimeError("No hyperparameter values found. Check Tune logs or configs.")

# --- Summary Table with Start/End Time ---
summary = []
for tid, group in loss_df.groupby('trial_id'):
    sorted_grp = group.sort_values('iteration')
    start_loss = sorted_grp['loss'].iloc[0]
    end_loss = sorted_grp['loss'].iloc[-1]
    improvement = (start_loss - end_loss) / start_loss * 100

    # Extract timing info from trial dataframe
    trial_df = analysis.trial_dataframes[tid]
    start_time = trial_df['time_total_s'].iloc[0] if 'time_total_s' in trial_df.columns else None
    end_time = trial_df['time_total_s'].iloc[-1] if 'time_total_s' in trial_df.columns else None
    duration_sec = end_time - start_time if start_time and end_time else None

    # Hyperparameters
    hp_vals = hp_long[hp_long['trial_id'] == tid].groupby('hyperparam')['value'].last().to_dict()

    record = {
        'trial_id': tid,
        'start_loss': start_loss,
        'end_loss': end_loss,
        'pct_improvement': improvement,        
        'duration_sec': duration_sec
    }
    record.update(hp_vals)
    summary.append(record)

    # --- Send Email Notification ---
    email_body = f"""
Trial Completed: {tid}
Start Loss: {start_loss}
End Loss: {end_loss}
Improvement: {improvement:.2f}%
Start Time (s): {start_time}
End Time (s): {end_time}
Duration (sec): {duration_sec}
Hyperparameters: {hp_vals}
"""
    send_email(subject=f"Trial {tid} Completed", body=email_body, to_email=RECIPIENT_EMAIL)
    # dbutils.notebook.send_mail(
    #     to=["aswin.bhaskaran@ey.com"],
    #     subject=f"Trial {tid} Completed",
    #     body=email_body
    #     )
summary_df = pd.DataFrame(summary)

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(summary_df)

# Optional: Export to CSV
summary_df.to_csv("/dbfs/FileStore/trial_summary.csv", index=False)

# --- Plot improvement with annotations ---
fig, ax = plt.subplots(figsize=(10, 6))
df_plot = summary_df.dropna(subset=['pct_improvement'])
bars = ax.bar(df_plot['trial_id'], df_plot['pct_improvement'], color='skyblue')

# Add annotations above bars
for bar, pct in zip(bars, df_plot['pct_improvement']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{pct:.1f}%", ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Trial ID')
ax.set_ylabel('Percentage Improvement of Loss')
ax.set_title('Loss Improvement per Trial')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# COMMAND ----------


# Databricks-friendly plotting
%matplotlib inline

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis

RESULTS_PATH = "/root/ray_results/dual_encoder_pbt"

# Load Ray Tune analysis
analysis = ExperimentAnalysis(RESULTS_PATH)

# Build a tidy dataframe of the LAST row per trial (for quick sanity display)
latest_rows = []
all_rows = []

for trial_id, trial_df in analysis.trial_dataframes.items():
    # Ensure iteration column exists; Ray often logs 'training_iteration'
    if "training_iteration" not in trial_df.columns:
        # Fall back to row index as iteration (best-effort)
        trial_df = trial_df.copy()
        trial_df["training_iteration"] = np.arange(1, len(trial_df) + 1)

    # Attach the static config values to each row (so we can plot configs over time)
    cfg = analysis.get_best_config(metric="loss", mode="min")  # just a placeholder accessor
    # Better: read the trial's own config (available on analysis.trials)
    trial_obj = next(t for t in analysis.trials if t.trial_id == trial_id)
    cfg = trial_obj.config

    # Broadcast config fields to every row so we can plot evolution per iteration
    for k, v in cfg.items():
        # Only include numeric hyperparameters we care about
        if k in ["lr", "wd", "warmup", "lora_rank", "alpha", "margin", "contrast_weight"]:
            trial_df[k] = v

    trial_df["trial_id"] = trial_id
    all_rows.append(trial_df)

    # Keep a compact “latest” view for console display
    latest = trial_df.iloc[-1]
    if latest is not None: latest_rows.append({
        "trial_id": trial_id,
        "training_iteration": int(latest["training_iteration"]),
        "loss": float(latest.get("loss", np.nan)),
        "time_total_s": float(latest.get("time_total_s", np.nan)),
        "lr": cfg.get("lr"),
        "wd": cfg.get("wd"),
        "warmup": cfg.get("warmup"),
        "lora_rank": cfg.get("lora_rank"),
        "alpha": cfg.get("alpha"),
        "margin": cfg.get("margin"),
        "contrast_weight": cfg.get("contrast_weight"),
    })

latest_df = pd.DataFrame(latest_rows).sort_values("trial_id")
full_df = pd.concat(all_rows, ignore_index=True)

print("Latest per-trial snapshot:")
display(latest_df)
