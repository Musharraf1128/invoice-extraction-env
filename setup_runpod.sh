#!/bin/bash
set -e

echo ""
echo "============================================================"
echo "  ESCTR — RunPod Setup (RTX 4090 / 24 GB)"
echo "============================================================"
echo ""

# ── Redirect caches to /workspace (persists across pod restarts) ──────────
export HF_HOME=/workspace/hf_cache
export TORCH_HOME=/workspace/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ── Install training dependencies ─────────────────────────────────────────
# Pin versions known to work together (avoids Trackio/transformers mismatches)
pip install -q --upgrade pip

echo "📦 Installing TRL + PEFT + dependencies..."
pip install -q \
    "trl>=0.18" \
    "transformers>=4.51" \
    "peft>=0.15" \
    "accelerate>=1.5" \
    "datasets>=3.0" \
    "jmespath" \
    "sentencepiece"

# ── Install ESCTR environment package ─────────────────────────────────────
echo "📦 Installing ESCTR environment..."
pip install -q -e .

# ── Verify everything loads ───────────────────────────────────────────────
echo ""
echo "🔍 Verifying installation..."
python -c "
import torch, trl, peft, transformers, datasets
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  trl:          {trl.__version__}')
print(f'  peft:         {peft.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

# Verify ESCTR environment loads
from server.environment import ESCTREnvironment
env = ESCTREnvironment()
obs = env.reset(task_name='procurement_reconciliation', seed=42)
print(f'  ESCTR env:    ✅ (briefing length: {len(obs.system_response)} chars)')
"

echo ""
echo "============================================================"
echo "  ✅ Setup complete! Now run:"
echo ""
echo "     python train_4b.py"
echo ""
echo "  Optional env vars:"
echo "     ESCTR_EPISODES=300     (default: 300)"
echo "     ESCTR_MODEL=Qwen/Qwen3-4B  (default)"
echo "     ESCTR_LORA_R=16       (default: 16)"
echo "     ESCTR_TASKS=procurement_reconciliation,sla_enforcement"
echo "============================================================"
echo ""
