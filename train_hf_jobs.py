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

# Pin torch 2.6 first (TRL 0.17+ needs FSDPModule from torch >= 2.6)
install("torch>=2.6.0")
install("transformers>=4.51.0")
install("trl>=0.17.0")
install("peft>=0.14.0")
install("bitsandbytes>=0.43.0")
install("datasets>=3.0.0")
install("accelerate>=1.0.0")
install("openenv")
install("jmespath")

# Install our environment package from the cloned repo
script_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", script_dir])
sys.path.insert(0, script_dir)

# ── 1. Imports ───────────────────────────────────────────────────────────────
import json
import time
from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

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

# ── 3. Environment wrapper (TRL-compatible) ──────────────────────────────────
from server.environment import ESCTREnvironment
from server.models import ESCTRAction

class ESCTRToolEnv:
    """TRL-compatible wrapper. Public methods with docstrings become tools."""

    def __init__(self):
        self.env = ESCTREnvironment()
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        """Reset the environment and return the initial briefing."""
        import random
        seed = random.randint(0, 100_000)
        obs = self.env.reset(task_name="procurement_reconciliation", seed=seed)
        self.reward = 0.0
        self.done = False
        return obs.system_response

    def query_database(self, table: str) -> str:
        """Query a corporate database table to discover available records.

        Args:
            table: The database table to query. One of: 'purchase_orders', 'invoices', 'shipping_logs', 'sla_contracts', 'warehouse_logs'

        Returns:
            A summary of records found in the specified table.
        """
        if self.done:
            return "Episode is over."
        action = ESCTRAction(action_type="query_database", query_parameters={"table": table})
        obs = self.env.step(action)
        self.reward = obs.reward
        self.done = obs.done
        return obs.system_response

    def read_document(self, document_id: str) -> str:
        """Read a specific document by its unique identifier to see full details.

        Args:
            document_id: The document ID to read, e.g. 'PO-2024-0055' or 'INV-2024-0055'

        Returns:
            The full contents of the requested document.
        """
        if self.done:
            return "Episode is over."
        action = ESCTRAction(action_type="read_document", document_id=document_id)
        obs = self.env.step(action)
        self.reward = obs.reward
        self.done = obs.done
        return obs.system_response

    def communicate_vendor(self, message_content: str) -> str:
        """Send a message to the vendor during a dispute negotiation.

        Args:
            message_content: The message to send to the vendor.

        Returns:
            The vendor's response to your message.
        """
        if self.done:
            return "Episode is over."
        action = ESCTRAction(action_type="communicate_vendor", message_content=message_content)
        obs = self.env.step(action)
        self.reward = obs.reward
        self.done = obs.done
        return obs.system_response

    def submit_financial_decision(self, adjustment_amount: float, adjustment_reason: str) -> str:
        """Submit the final financial adjustment. This ends the episode.

        Args:
            adjustment_amount: The exact monetary adjustment amount as a float.
            adjustment_reason: A brief explanation of why this adjustment is correct.

        Returns:
            The grading result with your score and feedback.
        """
        if self.done:
            return "Episode is over."
        action = ESCTRAction(
            action_type="submit_financial_decision",
            adjustment_amount=adjustment_amount,
            adjustment_reason=adjustment_reason,
        )
        obs = self.env.step(action)
        self.reward = obs.reward
        self.done = obs.done
        return obs.system_response

    def _get_reward(self) -> float:
        return self.reward

    def _is_done(self) -> bool:
        return self.done

def make_dataset(n=EPISODES):
    """Generate n prompts for procurement_reconciliation task."""
    import random
    data = []
    for i in range(n):
        seed = random.randint(1, 999999)
        data.append({
            "prompt": [
                {"role": "system", "content": "You are an autonomous financial auditor. Use the provided tools to investigate discrepancies and submit a precise financial decision."},
                {"role": "user",   "content": f"Begin your investigation. Find any overcharges or discrepancies."}
            ],
        })
    return Dataset.from_list(data)

# ── 4. Shaped reward function ────────────────────────────────────────────────
def shaped_reward_fn(environments, **kwargs) -> list[float]:
    """Shaped reward for GRPO — gives partial credit for investigation progress.

    Without shaping, the model must call submit_financial_decision to get ANY
    reward. This creates variance between rollouts even without submission.
    """
    rewards = []
    for env in environments:
        # Base: the environment's graded reward (non-zero only if submitted)
        r = env.reward

        # Shaping: credit for investigation effort
        step_count = env.env._state.step_count if hasattr(env.env, '_state') else 0
        submitted = env.env._state.outcome_submitted if hasattr(env.env, '_state') else False

        # Small per-step bonus for using tools (caps at 0.20)
        investigation_bonus = min(step_count * 0.05, 0.20)

        # Bonus for actually submitting (even with wrong amount)
        submit_bonus = 0.15 if submitted else 0.0

        rewards.append(r + investigation_bonus + submit_bonus)
    return rewards

# ── 5. LoRA config ───────────────────────────────────────────────────────────
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
    max_tool_calling_iterations=8,
    learning_rate=5e-5,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    log_completions=True,
    num_completions_to_print=1,
    chat_template_kwargs={"enable_thinking": False},
    report_to="none",
    push_to_hub=False,
    remove_unused_columns=False,
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
    reward_funcs=shaped_reward_fn,
    train_dataset=dataset,
    args=grpo_config,
    environment_factory=ESCTRToolEnv,
    peft_config=lora_config,
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
