---
title: "Training Autonomous Financial Auditors with RLVR"
thumbnail: https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/reward_curve_4b.png
authors:
  - user: musharraf7
date: 2026-04-26
tags:
  - reinforcement-learning
  - openenv
  - grpo
  - tool-use
  - finance
---

# Training Autonomous Financial Auditors with RLVR

> What if we could train an LLM to investigate procurement fraud, enforce SLA penalties, and reject bad vendor settlements — autonomously?

That's what we built for the [OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv). **ESCTR** (Enterprise Supply Chain & Tax Reconciliation) is a stateful environment where an LLM agent operates as a **financial controller**, navigating a multi-step audit pipeline with 4 ERP tools, adversarial vendors, and mathematically precise reward verification.

🏢 **Environment**: [musharraf7/esctr-environment](https://huggingface.co/spaces/musharraf7/esctr-environment)
🧠 **Trained Model**: [musharraf7/esctr-grpo-4b-lora](https://huggingface.co/musharraf7/esctr-grpo-4b-lora)
📊 **Training Dashboard**: [Trackio](https://huggingface.co/spaces/musharraf7/esctr-grpo-trained)

---

## The Problem: Why Financial Auditing Needs RL

Every day, enterprises process millions of procurement transactions. Between Purchase Orders, shipping manifests, SLA contracts, and vendor invoices, discrepancies inevitably arise:

- A vendor bills $45/unit instead of the contracted $40
- A shipment arrives 5 days late, triggering penalty clauses
- The vendor disputes the penalty, claiming your warehouse rejected delivery

Resolving these disputes requires humans to **manually cross-reference siloed databases**, interpret contract clauses, and perform precise arithmetic. It's slow, expensive, and error-prone.

Current LLMs can't solve this reliably because it requires:
1. **Multi-step tool use** (querying databases, reading documents, communicating with vendors)
2. **Precise arithmetic** under contract constraints
3. **Adversarial reasoning** (rejecting bad settlement offers)
4. **State tracking** across 10-20 interaction steps

This is exactly the kind of capability that **Reinforcement Learning with Verifiable Rewards (RLVR)** was designed to teach.

---

## The Environment: Three Tasks, Escalating Difficulty

ESCTR provides 3 tasks with escalating complexity:

| Task | Difficulty | What the Agent Must Do |
|------|-----------|----------------------|
| **Procurement Reconciliation** | Easy | Find overcharged line items, calculate exact overcharge |
| **SLA Enforcement** | Medium | Discover late shipments, retrieve SLA contract, compute penalty |
| **Adversarial Auditing** | Hard | All of the above + disprove vendor claims using warehouse logs |

The agent interacts through **4 ERP tools**:
- `query_database` — search shipping logs, purchase orders, invoices
- `read_document` — retrieve full document text
- `communicate_vendor` — negotiate with an adversarial vendor
- `submit_financial_decision` — submit the final adjustment (terminal action)

Every scenario is **procedurally generated from a seed**, enabling infinite training configurations with deterministic, reproducible grading.

---

## Reward Design: Dense, Verifiable, Hard to Game

Following the RLVR paradigm (Wen et al., ICLR 2026), our reward is:

```
R_total = α·R_outcome + β·R_trajectory − penalties
```

- **R_outcome** (60-70%): Binary — did the agent submit the exact correct adjustment amount?
- **R_trajectory** (30-40%): Did the agent follow proper investigative procedure?
- **Penalties**: Step costs (-0.005/step), hallucination (-0.02), gullibility (-0.20 for accepting bad settlements)

The correct answer is always a **precise floating-point number** derived from contract terms. No LLM-as-judge, no fuzzy evaluation — pure programmatic verification.

---

## Training: From 0.6B to 4B — The Hard Way

### Phase 1: Proof of Concept (Qwen3-0.6B)

We first validated the training loop with a 0.6B model on a T4 GPU using TRL's `GRPOTrainer` with `environment_factory`.

**Result:** The model went from 0.09 → 0.30 reward (+222%) in 500 episodes. It perfectly learned the investigation procedure (query PO → query Invoice → read documents → submit) with zero tool failures.

![0.6B Reward Curve](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/reward_curve.png)

![Training Dashboard](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/training_dashboard.png)

### Phase 2: Scaling to 4B — and Hitting a Wall

We then tried to scale to **Qwen3-4B** on an RTX 4090 (24GB VRAM) with LoRA adapters. The first three attempts **completely failed** — loss flat at 0.0, zero learning.

**What went wrong:**

1. **Token Budget Exhaustion**: Qwen3-4B produces massive `<think>` reasoning blocks by default. It would exhaust the entire 512-token generation budget on internal monologue before making a single tool call.

2. **Deterministic Starvation**: Even after fixing the thinking issue, at `temperature=1.0` all K=4 rollouts were identical. The model deterministically made exactly 3 investigation calls and stopped, never calling `submit_financial_decision`. With zero reward variance, GRPO had **zero gradient signal**.

This was the core engineering challenge. We spent ~4 hours debugging completion traces before discovering the root cause.

### Phase 2.5: The Fix — Shaped Rewards + Forced Exploration

We implemented two key changes:

1. **Process Reward Shaping**: Instead of only rewarding the final submission, we injected `+0.05` partial credit for each valid investigation step. This gave GRPO the gradient signal it needed.

2. **High-Temperature Exploration**: Raised `temperature=1.5` and kept `K=4` rollouts to force diversity in the group sampling.

### Phase 3: Success — 4B Training in 71 Minutes

With shaped rewards and forced exploration, the 4B model finally learned:

![4B Reward Curve](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/reward_curve_4b.png)

![4B Tool Discipline](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/tool_calls_4b.png)

**Key Results:**
- Peak reward: **0.27** (vs 0.09 baseline)
- Tool calls converged to exactly **4.0 per episode** (the expected investigation + submit sequence)
- **Zero tool failures** across 300 episodes
- Peak VRAM: only **19.74 GB** on a 24GB GPU
- Total training time: **71.3 minutes**

The tool execution graph tells the clearest story: early on, the model varies wildly between 2-4.25 tool calls. By the end, it rigidly locks onto exactly 4.0 — having learned the optimal investigation → submission pipeline.

---

## What the Agent Actually Learned

| Metric | Baseline (untrained) | Trained (4B, 300 ep) |
|--------|---------------------|---------------------|
| Mean Reward | 0.09 | 0.20 (peak 0.27) |
| Tool Success Rate | 60% | 100% |
| Investigation Completeness | 40% | 100% |
| Tool Calls/Episode | erratic (1-4) | stable 4.0 |
| Tool Failures | frequent | 0 |

The baseline model jumps to a decision with no investigation. The trained agent follows a principled audit path: query the PO, query the invoice, read the relevant documents, then submit with evidence.

---

## Technical Details

| Parameter | 0.6B Run | 4B Run |
|-----------|----------|--------|
| Model | Qwen/Qwen3-0.6B | Qwen/Qwen3-4B |
| GPU | T4 (Colab) | RTX 4090 (RunPod) |
| Quantization | None | 4-bit (BitsAndBytes) |
| Adapter | Full model | LoRA (r=16, all-linear) |
| Episodes | 500 | 300 |
| Training Time | ~2 hours | ~71 minutes |
| Peak VRAM | ~14 GB | 19.74 GB |
| Framework | TRL GRPOTrainer | TRL GRPOTrainer |

---

## Why This Matters

ESCTR demonstrates that **RLVR can teach LLMs enterprise-grade financial reasoning** — a domain nearly absent from existing RL/LLM training benchmarks. Unlike game environments (chess, snake, tic-tac-toe), our environment:

- Tests **real-world professional skills** (procurement auditing, SLA enforcement)
- Requires **adversarial reasoning** (vendor negotiation with settlement traps)
- Has **verifiable, precise rewards** (exact floating-point amounts from contract math)
- Could **plug into production systems** (SAP/Oracle) as a pre-audit layer

We believe this is the kind of environment that pushes the frontier of what we can train LLMs to do — not just playing games, but performing the complex, multi-step reasoning that enterprises actually need.

---

## Links

- 🏢 **Environment Space**: [musharraf7/esctr-environment](https://huggingface.co/spaces/musharraf7/esctr-environment)
- 🧠 **Trained LoRA Weights**: [musharraf7/esctr-grpo-4b-lora](https://huggingface.co/musharraf7/esctr-grpo-4b-lora)
- 📊 **Training Dashboard**: [Trackio Space](https://huggingface.co/spaces/musharraf7/esctr-grpo-trained)
- 💻 **Source Code**: [GitHub](https://github.com/Musharraf1128/esctr-environment)
- 🏋️ **Training Scripts**: [`train.py`](https://github.com/Musharraf1128/esctr-environment/blob/main/train.py) (0.6B) · [`train_4b.py`](https://github.com/Musharraf1128/esctr-environment/blob/main/train_4b.py) (4B)

*Built for the [OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv) by Musharraf Shah.*
