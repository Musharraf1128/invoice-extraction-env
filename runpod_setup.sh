#!/bin/bash

echo "🚀 Setting up ESCTR Training Environment for RunPod..."

# Install PyTorch requirements for 4-bit LoRA (bitsandbytes, peft)
pip install -U "trl[vllm]" peft accelerate bitsandbytes trackio datasets

# Install our custom environment requirements
pip install -r requirements.txt
pip install -e .

echo ""
echo "✅ Setup Complete!"
echo "⚠️ IMPORTANT: We are using Gemma-2-9B-It, which is a gated model."
echo "Please run this command to log into Hugging Face before starting training:"
echo "    huggingface-cli login"
echo ""
echo "Once logged in, start training with:"
echo "    python train_runpod.py"
