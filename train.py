#!/usr/bin/env python3
"""
ESCTR Training Script — GRPO with TRL + vLLM
=============================================

Train an LLM to be an autonomous financial controller using
Group Relative Policy Optimization (GRPO) against the ESCTR environment.

Usage (Colab / HF Jobs):
    pip install -Uq "trl[vllm]" trackio datasets
    pip install -e .        # install esctr-environment package
    python train.py

The environment runs in-process (no HTTP server needed during training).
The HF Space deployment is only for judges to test the environment interactively.
"""

import random
import sys
import os

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Import ESCTR environment (runs in-process, no server needed)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.environment import ESCTREnvironment
from server.models import ESCTRAction


# ---------------------------------------------------------------------------
# System prompt — tells the model what it is and what tools are available
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ESCTR Environment wrapper for TRL environment_factory
# ---------------------------------------------------------------------------
# TRL discovers public methods (with docstrings) as callable tools.
# The model generates tool calls; TRL executes them and feeds results back.
# ---------------------------------------------------------------------------

# Task to train on — start with the easiest task for stable training
TRAIN_TASK = os.environ.get("ESCTR_TASK", "procurement_reconciliation")


class ESCTRToolEnv:
    """TRL-compatible wrapper around the ESCTR environment.

    Public methods with docstrings are auto-discovered as tools by TRL's
    environment_factory. The trainer handles the multi-turn loop automatically.
    """

    def __init__(self):
        self.env = ESCTREnvironment()
        self.reward = 0.0
        self.done = False
        self._task = TRAIN_TASK

    def reset(self, **kwargs) -> str | None:
        """Reset the environment and return the initial briefing."""
        seed = random.randint(0, 100_000)
        obs = self.env.reset(
            task_name=self._task,
            seed=seed,
        )
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
            raise ValueError("Episode is over. No more actions allowed.")

        action = ESCTRAction(
            action_type="query_database",
            query_parameters={"table": table},
        )
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
            raise ValueError("Episode is over. No more actions allowed.")

        action = ESCTRAction(
            action_type="read_document",
            document_id=document_id,
        )
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
            raise ValueError("Episode is over. No more actions allowed.")

        action = ESCTRAction(
            action_type="communicate_vendor",
            message_content=message_content,
        )
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
            raise ValueError("Episode is over. No more actions allowed.")

        action = ESCTRAction(
            action_type="submit_financial_decision",
            adjustment_amount=adjustment_amount,
            adjustment_reason=adjustment_reason,
        )
        obs = self.env.step(action)
        self.reward = obs.reward
        self.done = obs.done
        return obs.system_response


# ---------------------------------------------------------------------------
# Reward function — reads from env instances after each episode
# ---------------------------------------------------------------------------

def reward_func(environments, **kwargs) -> list[float]:
    """Extract reward from each environment instance after episode completion."""
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def main():
    # Model selection — Qwen3-1.7B is efficient on T4 GPU
    model_name = os.environ.get("ESCTR_MODEL", "Qwen/Qwen3-1.7B")
    output_dir = os.environ.get("ESCTR_OUTPUT", "esctr-grpo-trained")
    num_episodes = int(os.environ.get("ESCTR_EPISODES", "1000"))

    # Create dataset — each entry triggers one rollout episode
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": SYSTEM_PROMPT}]] * num_episodes
    })

    # GRPO configuration
    grpo_config = GRPOConfig(
        # Training schedule
        num_train_epochs=1,
        learning_rate=1e-6,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        warmup_steps=10,
        optim="adamw_torch",
        max_grad_norm=1.0,

        # GRPO settings
        num_generations=2,
        max_completion_length=2048,
        log_completions=True,
        num_completions_to_print=2,
        chat_template_kwargs={"enable_thinking": False},

        # Logging
        output_dir=output_dir,
        report_to="trackio",
        trackio_space_id=output_dir,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,

        # Memory optimization
        gradient_checkpointing=True,
        bf16=False,
        fp16=True,

        # Hub integration
        push_to_hub=True,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=ESCTRToolEnv,
    )

    # Show GPU stats before training
    import torch
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    print(f"\n{'='*60}")
    print(f"ESCTR Training — {model_name}")
    print(f"Task: {TRAIN_TASK}")
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Train!
    trainer_stats = trainer.train()

    # Show training stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"\nTraining completed in {trainer_stats.metrics['train_runtime']:.0f} seconds")
        print(f"Peak GPU memory: {used_memory} GB / {max_memory} GB")

    # Save and push
    trainer.save_model(output_dir)
    trainer.push_to_hub()
    print(f"\nModel saved to {output_dir} and pushed to Hub!")


if __name__ == "__main__":
    main()
