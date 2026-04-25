import os
import random
import sys
import torch

from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# Add standard local path to use ESCTR env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.environment import ESCTREnvironment
from server.models import ESCTRAction

# Import the environment and tools from our original train.py
from train import SYSTEM_PROMPT, ESCTRToolEnv, reward_func, TRAIN_TASKS

def main():
    # User requested Gemma 7B/9B. Google's Gemma-2-9B-It is SOTA for this class.
    model_name = os.environ.get("ESCTR_MODEL", "google/gemma-2-9b-it")
    output_dir = os.environ.get("ESCTR_OUTPUT", "esctr-gemma-grpo")
    num_episodes = int(os.environ.get("ESCTR_EPISODES", "250"))
    
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": SYSTEM_PROMPT}]] * num_episodes
    })

    # =========================================================================
    # RTX 4090 MEMORY OPTIMIZATIONS (24GB VRAM)
    # 1. 4-Bit Quantization via BitsAndBytes (shrinks 9B model from ~18GB -> ~5GB)
    # =========================================================================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # =========================================================================
    # 2. LoRA (Low-Rank Adaptation)
    # TRL GRPO natively supports PEFT. Instead of training 9 billion parameters,
    # we only train ~15 million adapter parameters. 
    # =========================================================================
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    grpo_config = GRPOConfig(
        num_train_epochs=1,
        learning_rate=2e-5, # LoRA LR typically higher than full finetuning
        gradient_accumulation_steps=8, 
        per_device_train_batch_size=1,
        
        # Generation config (keep length low to avoid KV cache explosion)
        max_completion_length=1024,
        num_generations=4, # Crucial: enough to do group relative optimization
        
        # Logging & saving
        output_dir=output_dir,
        report_to="trackio",
        trackio_space_id=output_dir,
        logging_steps=1,
        
        # Additional Memory Savings
        bf16=True, # Gemma 2 performs exceptionally well with bfloat16
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=ESCTRToolEnv,
        peft_config=peft_config, # Pass LoRA config
    )

    print(f"\n{'='*60}")
    print(f"🚀 RTX 4090 GRPO Training — {model_name}")
    print(f"Modes: 4-Bit LoRA Quantization")
    print(f"Tasks: {', '.join(TRAIN_TASKS)}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}\n")

    trainer.train()
    
    print("\n✅ Training finished! Saving LoRA adapters...")
    trainer.save_model(f"{output_dir}-lora")

if __name__ == "__main__":
    main()
