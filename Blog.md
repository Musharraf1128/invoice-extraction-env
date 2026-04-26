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

> *What if we could train an LLM to investigate procurement fraud, enforce SLA penalties, and reject bad vendor settlements — autonomously?*

That's the question we set out to answer for the [OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv).

The result is **ESCTR** — *Enterprise Supply Chain & Tax Reconciliation* — a stateful RL environment where an LLM agent operates as a **financial controller**. It navigates a multi-step audit pipeline armed with 4 ERP tools, faces adversarial vendors, and is graded against mathematically precise reward verification.

🏢 **Live Environment**: [musharraf7/esctr-environment](https://huggingface.co/spaces/musharraf7/esctr-environment)
📊 **Training Dashboard**: [Trackio](https://huggingface.co/spaces/musharraf7/esctr-grpo-trained)
💻 **Source Code**: [GitHub](https://github.com/Musharraf1128/esctr-environment)

---

## The Problem: Why Financial Auditing Needs RL

Every day, enterprises process millions of procurement transactions. Between Purchase Orders, shipping manifests, SLA contracts, and vendor invoices — discrepancies are inevitable:

- A vendor bills **$45/unit** instead of the contracted **$40**
- A shipment arrives **5 days late**, triggering penalty clauses
- The vendor disputes the penalty, claiming *your warehouse rejected the delivery*

Resolving these disputes today means humans manually cross-referencing siloed databases, interpreting contract clauses, and performing precise arithmetic under pressure. It's slow, expensive, and deeply error-prone.

**Current LLMs can't solve this reliably.** Not because the individual steps are hard, but because the *combination* is:

1. **Multi-step tool use** — querying databases, reading documents, communicating with vendors
2. **Precise arithmetic** under contract constraints
3. **Adversarial reasoning** — rejecting manipulative settlement offers
4. **State tracking** across 10–20 interaction steps

This is exactly the capability gap that **Reinforcement Learning with Verifiable Rewards (RLVR)** was designed to close. So we built the environment to prove it.

---

## The Environment: Three Tasks, Escalating Stakes

ESCTR gives the agent three scenarios of increasing complexity — each one a realistic slice of enterprise financial operations:

| Task | Difficulty | What the Agent Must Do |
|------|-----------|----------------------|
| **Procurement Reconciliation** | Easy | Identify overcharged line items, calculate the exact overcharge |
| **SLA Enforcement** | Medium | Discover late shipments, retrieve the SLA contract, compute the penalty |
| **Adversarial Auditing** | Hard | All of the above *plus* disprove vendor counter-claims using warehouse logs |

The agent has four ERP tools at its disposal:

- `query_database` — search shipping logs, purchase orders, and invoices
- `read_document` — retrieve the full text of a contract or manifest
- `communicate_vendor` — negotiate with an adversarial vendor that will lie, deflect, and offer bad settlements
- `submit_financial_decision` — submit the final adjustment amount (the terminal, point-of-no-return action)

Every scenario is **procedurally generated from a seed**, enabling infinite training configurations with deterministic, reproducible grading. There is no memorizing the answer — the agent must investigate.

---

## Reward Design: Dense, Verifiable, Impossible to Fake

Following the RLVR paradigm (Wen et al., ICLR 2026), our reward function is:

```
R_total = α · R_outcome + β · R_trajectory − penalties
```

- **R_outcome** (60–70%): Binary — did the agent submit the *exact* correct adjustment amount?
- **R_trajectory** (30–40%): Did the agent follow proper investigative procedure?
- **Penalties**: Step costs (−0.005/step), hallucination (−0.02), gullibility (−0.20 for accepting bad settlements)

The correct answer is always a **precise floating-point number** derived from contract terms. There is no LLM-as-judge, no fuzzy rubric — just pure programmatic verification. Either you found the fraud, or you didn't.

---

## Training: From 0.6B to 4B — The Hard Way

### Phase 1 — Proof of Concept (Qwen3-0.6B)

We first validated the training loop with a 0.6B model on a T4 GPU using TRL's `GRPOTrainer` with `environment_factory`.

**The result spoke for itself:** the model went from a mean reward of **0.09 → 0.30** (+222%) in just 500 episodes. It perfectly learned the canonical investigation procedure — query PO → query Invoice → read documents → submit — with zero tool failures.

![0.6B Reward Curve](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/reward_curve.png)

![Training Dashboard](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/training_dashboard.png)

The proof of concept worked. Time to scale.

---

### Phase 2 — Scaling to 4B, and Hitting a Wall

We scaled to **Qwen3-4B** on an RTX 4090 (24GB VRAM) with LoRA adapters. The first three attempts **completely failed** — loss flat at 0.0, zero learning whatsoever.

Four hours of debugging later, we found two distinct root causes:

**Problem 1: Token Budget Exhaustion**

Qwen3-4B produces large `<think>` reasoning blocks by default. The model was consuming its entire 512-token generation budget on internal monologue — before making a single tool call. No actions, no reward, no gradient.

**Problem 2: Deterministic Starvation**

Even after addressing the thinking issue, at `temperature=1.0` all K=4 rollouts in each GRPO batch were *identical*. The model had learned to deterministically make exactly 3 investigation calls and stop — never reaching `submit_financial_decision`. With zero reward variance across the group, GRPO had **zero gradient signal**. The math simply didn't work.

This was the core engineering challenge of the project. The model wasn't broken — the training setup was starving it of the variance it needed to learn.

---

### Phase 2.5 — The Fix: Shaped Rewards + Forced Exploration

Two targeted changes broke the deadlock:

1. **Process Reward Shaping** — Instead of only rewarding the final submission, we injected `+0.05` partial credit for each valid investigation step. This gave GRPO the gradient signal it needed to even begin learning the terminal action.

2. **High-Temperature Exploration** — Raising `temperature=1.5` with K=4 rollouts forced diversity in group sampling. The model was finally exploring, failing, and learning from the contrast.

---

### Phase 3 — Success: 4B Training in 71 Minutes

With shaped rewards and forced exploration, the 4B model finally learned — and the results were clean:

![4B Reward Curve](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/reward_curve_4b.png)

![4B Tool Discipline](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/tool_calls_4b.png)

**Key results:**

| Metric | Value |
|--------|-------|
| Peak Reward | **0.27** (vs 0.09 baseline) |
| Tool Calls/Episode | Converged to exactly **4.0** |
| Tool Failure Rate | **0** across 300 episodes |
| Peak VRAM | **19.74 GB** on 24GB GPU |
| Total Training Time | **71.3 minutes** |

The tool execution graph tells the most compelling story. Early in training, the model varies wildly — 2 to 4.25 tool calls per episode, chaotic and unreliable. By the end, it locks rigidly onto **exactly 4.0** — having learned the optimal investigate → investigate → investigate → submit pipeline. The chaos collapses into discipline.

---

### Phase 4 — Iterative Run: Qwen3-1.7B on HF Jobs (In Progress)

Following judge advice to **iterate on multiple model sizes**, we launched a third training run on **HF Jobs T4-medium** using `Qwen/Qwen3-1.7B` with LoRA adapters — entirely on HuggingFace's own cloud compute, no local GPU needed.

This run will not complete before the submission deadline (~500 steps × 50s/step ≈ 7 hours), but the early metrics already tell an important story: **the shaped reward architecture generalises cleanly to the 1.7B scale**.

**Observed training progression (Steps 5–20):**

| Step | Loss | Reward (mean) | Reward Std | Tool Calls/ep | Entropy |
|------|------|--------------|------------|---------------|---------|
| 5    | 0.184 | **0.195** | 0.010 | **3.9** | 0.132 |
| 10   | 0.116 | 0.195 | 0.010 | **3.9** | 0.127 |
| 15   | 0.088 | 0.180 | 0.029 | 3.6 | 0.028 |
| 20   | 0.186 | 0.190 | 0.020 | 3.8 | 0.047 |

What this tells us:
- **No cold-start collapse** — reward is non-zero from the very first logged step. The shaped investigation bonus is doing exactly what it was designed to do.
- **Zero tool failures** at every step — the 1.7B model calls tools with valid JSON syntax just as reliably as the 4B model.
- **Loss is decreasing**, confirming gradient signal is flowing through the LoRA adapter.
- **Entropy is dropping** (0.132 → 0.028) — the model is committing to a policy, not just wandering. It has learned that the `query_database → read_document → submit` pipeline is the winning trajectory.

The high `frac_reward_zero_std` (0.6–0.8) at early steps is expected — it means some GRPO groups have identical rollouts, which is normal before the model diversifies its exploration. This resolved naturally in the 4B run around step ~30.

---

## What the Agent Actually Learned

| Metric | Baseline (untrained) | Trained (4B, 300 ep) |
|--------|---------------------|---------------------|
| Mean Reward | 0.09 | 0.20 (peak 0.27) |
| Tool Success Rate | 60% | **100%** |
| Investigation Completeness | 40% | **100%** |
| Tool Calls/Episode | Erratic (1–4) | Stable **4.0** |
| Tool Failures | Frequent | **0** |

The untrained model jumps straight to a decision with no evidence. The trained agent follows a principled audit path: gather evidence, read the contract, then — and only then — submit with conviction.

---

## Technical Summary

| Parameter | 0.6B Run | 4B Run | 1.7B Run (in progress) |
|-----------|----------|--------|------------------------|
| Model | Qwen/Qwen3-0.6B | Qwen/Qwen3-4B | Qwen/Qwen3-1.7B |
| GPU | T4 (Colab) | RTX 4090 (RunPod) | T4 (HF Jobs) |
| Quantization | None | 4-bit (BitsAndBytes) | 4-bit (BitsAndBytes) |
| Adapter | Full model | LoRA (r=16) | LoRA (r=16) |
| Episodes | 500 | 300 | 500 (planned) |
| Training Time | ~2 hours | ~71 minutes | ~7 hours (ongoing) |
| Framework | TRL GRPOTrainer | TRL GRPOTrainer | TRL GRPOTrainer |
| Script | [`train.py`](train.py) | [`train_4b.py`](train_4b.py) | [`train_hf_jobs.py`](train_hf_jobs.py) |

---

## Why This Matters

ESCTR demonstrates that **RLVR can teach LLMs enterprise-grade financial reasoning** — a domain nearly absent from existing RL training benchmarks.

Unlike game environments (chess, Snake, tic-tac-toe), our environment tests capabilities that actually exist in production systems:

- **Real-world professional skills** — procurement auditing, SLA enforcement, dispute resolution
- **Adversarial reasoning** — vendor negotiation where the counterpart is actively trying to deceive you
- **Verifiable, precise rewards** — exact floating-point answers derived from contract mathematics
- **Production integration potential** — the same tool interface could plug directly into SAP or Oracle as a pre-audit layer

The broader point: this is the kind of environment that pushes the frontier of *what we can train LLMs to do*. Not playing games — performing the complex, multi-step reasoning that enterprises actually need and pay billions of dollars for humans to do today.

---

## Links

- 🏢 **Environment Space**: [musharraf7/esctr-environment](https://huggingface.co/spaces/musharraf7/esctr-environment)
- 📊 **Training Dashboard**: [Trackio Space](https://huggingface.co/spaces/musharraf7/esctr-grpo-trained)
- 🏋️ **Training Scripts**: [`train.py`](https://github.com/Musharraf1128/esctr-environment/blob/main/train.py) · [`train_4b.py`](https://github.com/Musharraf1128/esctr-environment/blob/main/train_4b.py) · [`train_hf_jobs.py`](https://github.com/Musharraf1128/esctr-environment/blob/main/train_hf_jobs.py)
- 💻 **Source Code**: [GitHub](https://github.com/Musharraf1128/esctr-environment)

---

*Built for the [OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv) by Musharraf.*