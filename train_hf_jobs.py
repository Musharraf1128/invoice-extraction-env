"""
ESCTR Training Script — HuggingFace Jobs (T4-medium)
Model: Qwen3-1.7B + QLoRA (4-bit)
Platform: HF Jobs T4-medium (~$0.60/hr, ~$3-5 total)
Episodes: 500
Run with:
    hf jobs run python:3.11 --flavor t4-medium \
        --env HF_TOKEN=$HF_TOKEN \
        --env HF_REPO_ID=musharraf7/esctr-grpo-1.7b-lora \
        -- python train_hf_jobs.py
"""

import os
import subprocess
import sys

# ── 0. Install deps ──────────────────────────────────────────────────────────
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("torch==2.4.0")
install("transformers>=4.51.0")
install("trl>=0.17.0")
install("peft>=0.14.0")
install("bitsandbytes>=0.43.0")
install("datasets>=3.0.0")
install("accelerate>=1.0.0")
install("openenv>=0.4.0")

# Install our environment from HF Space
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://huggingface.co/spaces/musharraf7/esctr-environment"
])

# ── 1. Imports ───────────────────────────────────────────────────────────────
import json
import math
import time
from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import HfApi

# ── 2. Config ────────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen3-1.7B"
HF_REPO_ID   = os.getenv("HF_REPO_ID", "musharraf7/esctr-grpo-1.7b-lora")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
EPISODES     = 500
TASK         = "procurement_reconciliation"
OUTPUT_DIR   = "/tmp/esctr-1.7b-lora"
MAX_STEPS    = 500

print(f"[CONFIG] Model: {MODEL_ID}")
print(f"[CONFIG] Episodes: {EPISODES}, Task: {TASK}")
print(f"[CONFIG] Output: {OUTPUT_DIR}")
print(f"[CONFIG] Push to: {HF_REPO_ID}")
print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── 3. Environment factory ───────────────────────────────────────────────────
from server.environment import ESCTREnvironment
from server.models import Action

SYSTEM_PROMPT = """You are an autonomous financial auditor operating inside an Enterprise Resource Planning (ERP) system.

Your job is to investigate financial discrepancies using the tools available to you, then submit a precise financial decision.

TOOLS:
- query_database: Search corporate databases (purchase_orders, invoices, shipping_logs, sla_contracts, vendor_records)
- read_document: Retrieve full document text by document_id
- communicate_vendor: Send a message to the vendor
- submit_financial_decision: Submit your final adjustment amount (this ends the episode)

RULES:
1. Always investigate before submitting. Query at least 2 databases.
2. Your adjustment amount must be mathematically precise.
3. Respond ONLY with a valid JSON tool call. No prose, no <think> tags.

FORMAT:
{"action_type": "query_database", "query_parameters": {"table": "purchase_orders"}}
{"action_type": "submit_financial_decision", "adjustment_amount": -450.00, "adjustment_reason": "Overcharge on 50 units at $5/unit difference"}
"""

def environment_factory():
    """Create a fresh ESCTR environment for each episode."""
    env = ESCTREnvironment()
    return env

def make_dataset(n=EPISODES):
    """Generate n prompts for procurement_reconciliation task."""
    import random
    data = []
    for i in range(n):
        seed = random.randint(1, 999999)
        data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"[NEW EPISODE]\ntask: {TASK}\nseed: {seed}\n\nBegin your investigation."}
            ],
            "seed": seed,
        })
    return Dataset.from_list(data)

# ── 4. Shaped reward function ────────────────────────────────────────────────
def shaped_reward_fn(completions, env_outputs, **kwargs):
    """
    Process reward shaping:
    - +0.05 per unique valid investigation tool call
    - Full env reward on top
    Ensures GRPO always has gradient signal.
    """
    rewards = []
    for completion, env_out in zip(completions, env_outputs):
        base_reward = env_out.get("reward", 0.0) if env_out else 0.0

        # Count investigation steps from completion
        investigation_bonus = 0.0
        seen_tools = set()
        text = completion if isinstance(completion, str) else str(completion)
        for tool in ["query_database", "read_document", "communicate_vendor"]:
            if tool in text and tool not in seen_tools:
                investigation_bonus += 0.05
                seen_tools.add(tool)

        total = base_reward + investigation_bonus
        rewards.append(min(total, 1.0))  # cap at 1.0
    return rewards

# ── 5. QLoRA config ──────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── 6. GRPO config ───────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,           # K rollouts per prompt
    temperature=1.4,             # exploration
    max_completion_length=768,
    max_prompt_length=512,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=5,
    save_steps=100,
    report_to="none",            # no external trackers needed
    remove_unused_columns=False,
    max_tool_calling_iterations=8,
)

# ── 7. Train ─────────────────────────────────────────────────────────────────
print("\n[TRAIN] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("[TRAIN] Building dataset...")
dataset = make_dataset(EPISODES)

print("[TRAIN] Starting GRPOTrainer...")
start_time = time.time()

trainer = GRPOTrainer(
    model=MODEL_ID,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=shaped_reward_fn,
    environment_factory=environment_factory,
    peft_config=lora_config,
    model_init_kwargs={
        "quantization_config": bnb_config,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    },
)

trainer.train()

elapsed = time.time() - start_time
print(f"\n[DONE] Training complete in {elapsed/60:.1f} minutes")

# ── 8. Save and push ─────────────────────────────────────────────────────────
print(f"[SAVE] Saving LoRA adapters to {OUTPUT_DIR}/final...")
trainer.save_model(f"{OUTPUT_DIR}/final")

if HF_TOKEN and HF_REPO_ID:
    print(f"[PUSH] Pushing to HF Hub: {HF_REPO_ID}")
    trainer.model.push_to_hub(
        HF_REPO_ID,
        token=HF_TOKEN,
        commit_message=f"ESCTR 1.7B QLoRA — {EPISODES} episodes, TRL GRPO"
    )
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    print(f"[PUSH] Done! Model at: https://huggingface.co/{HF_REPO_ID}")

# ── 9. Save training log ─────────────────────────────────────────────────────
log = {
    "model": MODEL_ID,
    "episodes": EPISODES,
    "elapsed_minutes": round(elapsed / 60, 1),
    "timestamp": datetime.utcnow().isoformat(),
    "repo": HF_REPO_ID,
}
with open(f"{OUTPUT_DIR}/run_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(json.dumps(log, indent=2))
