#!/usr/bin/env python3
"""
ESCTR 4B Training — GRPO + LoRA on RTX 4090 (24 GB)
====================================================

Self-contained training script. No imports from other train scripts.

Memory budget (RTX 4090, 24 GB):
    Qwen3-4B in bf16       ≈  8 GB
    LoRA adapters           ≈  0.05 GB
    KV cache (K=2, 512 tok) ≈  3 GB
    Grad checkpointing      ≈  3 GB
    Optimizer (LoRA only)   ≈  0.2 GB
    ──────────────────────────────────
    Total                   ≈ 14 GB   (plenty of headroom)

Usage on RunPod:
    chmod +x setup_runpod.sh && ./setup_runpod.sh
    python train_4b.py
"""

import os
import sys
import random
import time

# ── Memory + cache config (must be set BEFORE any torch import) ───────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Redirect HF + torch caches to /workspace to avoid filling the container disk
if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/workspace/torch_cache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# ── Import ESCTR environment (in-process, no server) ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.environment import ESCTREnvironment
from server.models import ESCTRAction


# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous Financial Controller AI operating within an enterprise ERP system.

Your job is to investigate financial discrepancies in procurement records by using the available tools, then submit a precise monetary adjustment.

INVESTIGATION WORKFLOW:
1. Query databases to discover what records exist (purchase_orders, invoices, shipping_logs, sla_contracts, warehouse_logs)
2. Read specific documents to get full details
3. Compare line items, delivery dates, and contract terms
4. Calculate the exact adjustment amount
5. Submit your financial decision with the calculated amount and reasoning

CRITICAL RULES:
- Always query AND read documents before submitting. Never guess.
- Your adjustment_amount must be the EXACT monetary difference you calculated.
- Show your arithmetic in the adjustment_reason.
- If a vendor offers a settlement, verify their claims against internal records before accepting.

You have access to the following tools. Call them to interact with the ERP system."""


# ── Task config ───────────────────────────────────────────────────────────
TRAIN_TASKS = [
    t.strip()
    for t in os.environ.get(
        "ESCTR_TASKS",
        os.environ.get("ESCTR_TASK", "procurement_reconciliation")
    ).split(",")
    if t.strip()
]


# ── TRL environment wrapper ──────────────────────────────────────────────
class ESCTRToolEnv:
    """TRL-compatible wrapper.

    Public methods with docstrings become tools. TRL handles the multi-turn
    loop automatically via environment_factory.
    """

    def __init__(self):
        self.env = ESCTREnvironment()
        self.reward = 0.0
        self.done = False
        self._tasks = TRAIN_TASKS or ["procurement_reconciliation"]

    def reset(self, **kwargs) -> str | None:
        """Reset the environment and return the initial briefing."""
        seed = random.randint(0, 100_000)
        task = random.choice(self._tasks)
        obs = self.env.reset(task_name=task, seed=seed)
        self.reward = 0.0
        self.done = False
        return obs.system_response

    def query_database(self, table: str) -> str:
        """
        Query a corporate database table to discover available records.

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
        """
        Read a specific document by its unique identifier to see full details.

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
        """
        Send a message to the vendor during a dispute negotiation.

        Args:
            message_content: The message to send to the vendor, such as requesting clarification or rejecting a settlement offer.

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
        """
        Submit the final financial adjustment. This is the terminal action that ends the episode.

        Args:
            adjustment_amount: The exact monetary adjustment amount as a float (e.g. 450.00). Must be calculated from the documents.
            adjustment_reason: A brief explanation of why this adjustment is correct, including your arithmetic.

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


# ── Reward function ───────────────────────────────────────────────────────
def reward_func(environments, **kwargs) -> list[float]:
    """Extract reward from each environment instance after episode completion."""
    return [env.reward for env in environments]


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ── User-configurable via env vars ────────────────────────────────────
    model_name = os.environ.get("ESCTR_MODEL", "Qwen/Qwen3-4B")
    num_episodes = int(os.environ.get("ESCTR_EPISODES", "300"))
    max_len = int(os.environ.get("ESCTR_MAX_COMPLETION_LENGTH", "512"))
    lora_r = int(os.environ.get("ESCTR_LORA_R", "16"))
    grad_accum = int(os.environ.get("ESCTR_GRAD_ACCUM", "4"))

    # Output goes to /workspace on RunPod so it persists across restarts
    default_out = "/workspace/esctr-4b-lora" if os.path.isdir("/workspace") else "./esctr-4b-lora"
    output_dir = os.environ.get("ESCTR_OUTPUT", default_out)
    os.makedirs(output_dir, exist_ok=True)

    # ── Preflight checks ─────────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA not available — this script requires a GPU."
    gpu = torch.cuda.get_device_properties(0)
    vram_gb = round(gpu.total_memory / 1024**3, 1)
    print(f"\n{'='*60}")
    print(f"  GPU: {gpu.name}  |  VRAM: {vram_gb} GB")
    print(f"  Model: {model_name}")
    print(f"  Tasks: {', '.join(TRAIN_TASKS)}")
    print(f"  Episodes: {num_episodes}")
    print(f"  LoRA rank: {lora_r}  |  Max completion: {max_len} tokens")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": SYSTEM_PROMPT}]] * num_episodes
    })

    # ── LoRA config ───────────────────────────────────────────────────────
    # Target all linear layers for maximum expressiveness within budget.
    # TRL automatically uses base model as reference when peft_config is set,
    # so we do NOT need a separate ref model copy in VRAM.
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_r * 2,       # standard 2x multiplier
        target_modules="all-linear",  # TRL recommended
        lora_dropout=0.05,
        bias="none",
    )

    # ── Detect precision ──────────────────────────────────────────────────
    # RTX 4090 (Ada Lovelace) has native bf16 support.
    # Fall back to fp16 if something is wrong.
    use_bf16 = os.environ.get("ESCTR_BF16", "1") == "1"
    try:
        if use_bf16:
            torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
    except Exception:
        print("⚠️  bf16 not supported on this GPU, falling back to fp16")
        use_bf16 = False

    # ── GRPO config ───────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        # Schedule
        num_train_epochs=1,
        learning_rate=2e-5,          # higher LR for LoRA (only adapter weights)
        warmup_steps=10,
        max_grad_norm=1.0,
        optim="adamw_torch",

        # Batching — keep batch=1, accumulate for effective batch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,

        # GRPO
        num_generations=2,           # K=2 rollouts per prompt (minimum for GRPO)
        max_completion_length=max_len,
        log_completions=True,
        num_completions_to_print=1,

        # Memory
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=not use_bf16,

        # Logging — "none" avoids Trackio integration crashes (known issue)
        output_dir=output_dir,
        report_to="none",
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,

        # Do NOT push to hub during training (avoids quota/auth crashes)
        push_to_hub=False,
    )

    # ── Create trainer ────────────────────────────────────────────────────
    print("Loading model + LoRA adapters...")
    t0 = time.time()

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=ESCTRToolEnv,
        peft_config=peft_config,
    )

    load_time = time.time() - t0
    used = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"✅ Model loaded in {load_time:.0f}s  |  VRAM used: {used} GB / {vram_gb} GB\n")

    # ── Train! ────────────────────────────────────────────────────────────
    print("🚀 Starting GRPO training...\n")
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0

    peak = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  Wall time: {elapsed/60:.1f} minutes")
    print(f"  Peak VRAM: {peak} GB / {vram_gb} GB")
    if hasattr(stats, 'metrics'):
        rt = stats.metrics.get('train_runtime', elapsed)
        print(f"  Train runtime: {rt:.0f}s")
    print(f"{'='*60}\n")

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, "final")
    trainer.save_model(save_path)
    print(f"💾 LoRA adapters saved to: {save_path}")
    print(f"   (To push to Hub later: trainer.push_to_hub())")


if __name__ == "__main__":
    main()
