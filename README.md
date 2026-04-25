---
title: ESCTR Environment
emoji: 🏢
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# 🏢 ESCTR: Enterprise Supply Chain & Tax Reconciliation

> **Training LLMs to be autonomous financial auditors** — an OpenEnv environment for teaching AI agents to investigate procurement discrepancies, enforce SLA penalties, and navigate adversarial vendor disputes using **Reinforcement Learning with Verifiable Rewards (RLVR)**.

**Space URL:** [`musharraf7/esctr-environment`](https://huggingface.co/spaces/musharraf7/esctr-environment) · **Training Dashboard:** [Trackio](https://huggingface.co/spaces/musharraf7/esctr-grpo-trained) · **Training Script:** [`train.py`](train.py)

---

## The Problem

Every day, global enterprises process millions of procurement transactions. Between the Purchase Order, the shipping manifest, the SLA contract, and the final vendor invoice, discrepancies **inevitably** arise:

- A vendor bills $45/unit instead of the contracted $40
- A shipment arrives 5 days late, triggering SLA penalty clauses
- A vendor disputes the penalty, claiming *your* warehouse rejected the delivery

Resolving these disputes currently requires human financial controllers to **manually cross-reference multiple siloed databases**, interpret complex contract clauses, perform precise arithmetic, and negotiate with adversarial counterparties. It's slow, expensive, and error-prone.

**What if we could train LLMs to do this autonomously?**

## The Environment

ESCTR provides a stateful sandbox where an LLM agent operates as an **autonomous financial controller**. Rather than just extracting data from a document, the agent must:

1. **Investigate** — query procurement databases, shipping logs, SLA contracts
2. **Reason** — cross-reference documents, calculate penalties, verify claims
3. **Negotiate** — handle adversarial vendor communications
4. **Decide** — submit a mathematically precise financial adjustment

### Three Tasks, Escalating Complexity

| Task | Difficulty | Max Steps | What the Agent Must Do |
|------|-----------|-----------|----------------------|
| **Procurement Reconciliation** | Easy | 10 | Find an overcharged line item between PO and Invoice, calculate the exact overcharge |
| **SLA Enforcement** | Medium | 15 | Discover a late shipment, retrieve the SLA contract, calculate the penalty from contract terms |
| **Adversarial Auditing** | Hard | 20 | All of the above + verify warehouse logs to disprove vendor's claim + reject a settlement offer |

### The Tool Suite

The agent interacts through **4 ERP tools**, each requiring precise parameters:

| Tool | Purpose | Parameters |
|------|---------|------------|
| `query_database` | Search corporate databases | `{"table": "shipping_logs"}` |
| `read_document` | Retrieve full document text | `document_id: "PO-2024-1234"` |
| `communicate_vendor` | Negotiate with adversarial vendor | `message_content: "We reject..."` |
| `submit_financial_decision` | Submit final adjustment (terminal) | `adjustment_amount: -450.00` |

### Procedural Generation

Every scenario is generated from a seed — **same seed = same scenario = deterministic grading**. This enables:
- Infinite training configurations (no memorization)
- Reproducible evaluation
- Fair comparison between models

## Design Rationale

ESCTR is built on three foundational principles from recent RL and agent research:

1. **RLVR Paradigm**: Following Wen et al. (ICLR 2026), our environment uses rule-based, externally verifiable reward functions that incentivize multi-step reasoning — no LLM-as-judge, no fuzzy evaluation. The correct adjustment amount is always a precise floating-point number derived deterministically from contract terms.

2. **Dense Process Rewards**: Inspired by Agent-RLVR (2025) and RLVRR's reward chain decomposition, we augment sparse verifiable rewards (correct penalty ✓/✗) with process-level environment rewards (investigation milestones, tool-use discipline) to make RL effective in long-horizon financial auditing tasks.

3. **GRPO Training**: We adopt Group Relative Policy Optimization via TRL's `GRPOTrainer` with `environment_factory`, leveraging its theoretical success amplification properties under verifiable rewards as analyzed by Mroueh (2025). Our group sampling (K rollouts per prompt, deterministic pass/fail reward) follows the DeepSeek-R1 paradigm.

> *"ESCTR is to procurement and tax reconciliation what FinToolBench is to market-driven finance: a runnable environment with auditable tool traces and domain-specific compliance constraints."*

## Reward Architecture (RLVR-Inspired)

Following the RLVR paradigm and RLVRR's reward chain concept, we decompose rewards into **outcome verification** (content-like) and **investigation quality** (process-like) components:

```
R_total = α·R_outcome + β·R_trajectory − penalties
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **R_outcome** | 60-70% | Did the agent submit the correct adjustment amount? (Binary verifier) |
| **R_trajectory** | 30-40% | Did the agent follow proper investigative procedure? (Checklist-style subgoals) |
| **Efficiency penalty** | -0.005/step | Encourages shortest path to resolution |
| **Hallucination penalty** | -0.02 | Invalid queries, nonexistent documents |
| **Gullibility penalty** | -0.20 | Accepting adversarial settlement offers (Task 3) |
| **Evidence bonus** | +0.05 | Citing warehouse logs as evidence (Task 3) |

### Why This Reward Design Matters

- **Dense, not sparse**: Trajectory milestones reward correct investigative behavior (querying the right databases, reading the right documents) even if the final answer is wrong — following Agent-RLVR's guidance signal approach
- **Hard to game**: An agent that spams queries gets penalized by step costs; an agent that submits without investigating gets 0 trajectory reward
- **Verifiable**: The correct answer is always a precise floating-point number derived from contract terms — no subjective evaluation, aligned with RLVR's programmatic verification requirement
- **Risk-aware**: Following Chen et al. (2025), we evaluate not only correctness but also risk measures such as over-penalization, under-penalization, and reliance on unverified vendor claims

## Training Results

We trained **Qwen3-0.6B** on the Procurement Reconciliation task using **TRL's GRPOTrainer** with `environment_factory`, running 500 episodes on a T4 GPU (~2 hours).

### Reward Curve

The model improved from near-zero reward to a stable 0.30 within the first 100 training steps, representing a **222% improvement** in mean reward:

![Reward curve over 500 training steps](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/reward_curve.png)

### Training Dashboard

Four-panel view showing reward, policy entropy, tool usage convergence, and completion length:

![ESCTR GRPO Training Dashboard](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/training_dashboard.png)

### Baseline vs Trained Comparison

| Metric | Baseline (untrained) | Trained (500 episodes) | Δ |
|--------|---------------------|----------------------|---|
| Mean Reward | 0.09 | 0.30 | **+222%** |
| Tool Success Rate | 60% | 100% | **+67%** |
| Investigation Completeness | 40% | 100% | **+150%** |
| Tool Calls/Episode | erratic (1-4) | stable 3.0 | converged |
| Tool Failures | frequent | 0 | eliminated |

![Baseline vs Trained comparison](https://raw.githubusercontent.com/Musharraf1128/esctr-environment/main/plots/comparison_chart.png)

### Key Findings

1. **Tool mastery learned**: The model converged to exactly 3 tool calls per episode with zero failures — it learned the correct investigation pattern (query PO → query Invoice → read documents → submit)
2. **Trajectory reward captured**: The 0.30 plateau corresponds to perfect trajectory score (all investigation milestones hit) but without solving the final arithmetic — showing the reward decomposition works as designed
3. **Policy entropy stable**: Entropy did not collapse to zero, indicating the model maintains exploration capacity for future training with larger models
4. **Scaling hypothesis**: The 0.6B model learned *investigation procedure* but not *arithmetic reasoning* — we predict larger models (3B+) will break through the 0.30 plateau to achieve outcome rewards

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-0.6B` |
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Framework | TRL `GRPOTrainer` + `environment_factory` |
| Episodes | 500 |
| GPU | NVIDIA T4 (Colab) |
| Training Time | ~2 hours |
| Max Completion Length | 768 tokens |

📊 **Live training dashboard**: [Trackio Space](https://huggingface.co/spaces/musharraf7/esctr-grpo-trained)

## Quick Start

### Run the environment
```bash
# Docker
docker build -t esctr-env .
docker run -p 7860:7860 esctr-env

# Or locally
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Connect an agent
```python
import requests

url = "http://localhost:7860"

# Reset with a task
r = requests.post(f"{url}/reset", json={"task_name": "sla_enforcement", "seed": 42})
briefing = r.json()["observation"]["system_response"]

# Query a database
r = requests.post(f"{url}/step", json={
    "action": {
        "action_type": "query_database",
        "query_parameters": {"table": "shipping_logs"}
    }
})
result = r.json()["observation"]["system_response"]

# Submit financial decision
r = requests.post(f"{url}/step", json={
    "action": {
        "action_type": "submit_financial_decision",
        "adjustment_amount": -450.00,
        "adjustment_reason": "Late delivery penalty per SLA clause"
    }
})
score = r.json()["reward"]
```

### Run baseline inference
```bash
export ENV_URL="http://localhost:7860"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="your_token"
python inference.py
```

## Why This Matters

| Question | Answer |
|----------|--------|
| *Does this teach an LLM something it can't do well?* | Yes — multi-step financial reasoning with tool use is a known weakness of current LLMs |
| *Is the domain underexplored?* | Yes — supply chain auditing + adversarial negotiation is nearly absent from RL/LLM training benchmarks. Like EconAgentBench (ICLR 2026), we instantiate economic decision processes under partial information |
| *Could a researcher write a paper about this?* | Yes — training autonomous financial auditors has direct commercial and academic value, bridging FinToolBench-style tool evaluation with RLVR-driven policy optimization |
| *Is the reward hard to game?* | Yes — the correct answer is always a precise number from contract math; trajectory rewards require specific database queries |
| *Path to production?* | ESCTR could plug into real procurement systems (SAP/Oracle) as a pre-audit layer, flagging discrepancies before human review |

## Case Study: Trained Agent vs Baseline

A single episode on seed `42` (Procurement Reconciliation):

| Step | Baseline (untrained) | Trained (GRPO, 500 ep) |
|------|---------------------|------------------------|
| 1 | Submits random amount immediately | `query_database(table="purchase_orders")` |
| 2 | — | `query_database(table="invoices")` |
| 3 | — | `read_document(document_id="PO-2025-XXXX")` |
| 4 | — | `submit_financial_decision(amount=..., reason="...")` |
| **Reward** | **0.00** | **0.30** |
| **Investigation** | Skipped | Query PO ✓, Query Invoice ✓, Read docs ✓ |
| **Risk** | High (no evidence gathered) | Low (full audit trail) |

The baseline model jumps to a decision with no investigation, while the trained agent follows a principled audit path — exactly the behavioral shift RLVR incentivizes.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset with task + seed |
| `/step` | POST | Execute an action |
| `/state` | GET | Current state |
| `/schema` | GET | Action/Observation/State schemas |
| `/metadata` | GET | Environment metadata |
| `/ws` | WebSocket | Persistent session |

## Project Structure

```
├── server/
│   ├── __init__.py
│   ├── app.py             # FastAPI application
│   ├── environment.py     # Core stateful environment + tool handlers
│   ├── procedural.py      # Deterministic scenario generation engine
│   ├── graders.py         # Multi-axis deterministic graders (3 tasks)
│   └── models.py          # Pydantic Action/Observation/State schemas
├── plots/
│   ├── reward_curve.png   # Training reward over steps
│   ├── training_dashboard.png  # Multi-panel training metrics
│   └── comparison_chart.png    # Baseline vs Trained comparison
├── train.py               # TRL GRPO training script (environment_factory)
├── inference.py           # Baseline inference script
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Package config
├── requirements.txt       # Dependencies
├── Dockerfile             # Container definition
└── README.md              # This file
```

## Themes Alignment

- **🌐 World Modeling (Professional Tasks)** — Real interaction with tools and dynamic databases
- **📋 Long-Horizon Planning** — Multi-step investigation requiring state tracking across 10-20 steps
- **🤝 Multi-Agent Interactions** — Adversarial vendor negotiation with settlement dynamics
- **📈 Self-Improvement** — Escalating difficulty curriculum (Easy → Medium → Hard)

## Limitations & Future Work

- **Model scale**: Training on 0.6B showed tool mastery but not arithmetic reasoning; we predict 3B+ models will break through the 0.30 reward plateau to capture outcome rewards
- **Single-task**: Current training focuses on Task 1 (Procurement Reconciliation); extending to SLA Enforcement and Adversarial Auditing requires curriculum-based training
- **Vendor agent**: The adversarial vendor follows rule-based policies; replacing with a second LLM (à la MultiAgentBench/TAMAS) would create a truly competitive multi-agent dynamic
- **Risk metrics**: Following Chen et al. (2025), adding explicit over/under-penalization tracking would strengthen compliance evaluation

## References

| Paper | Relevance to ESCTR |
|-------|--------------------|
| Wen et al., "RLVR Implicitly Incentivizes Correct Reasoning" (ICLR 2026) | Foundational paradigm — binary verifiable rewards |
| Mroueh, "GRPO's Effective Loss and Success Amplification" (2025) | Theoretical justification for GRPO under verifiable rewards |
| Agent-RLVR (2025) | Dense guidance signals for sparse multi-step environments |
| RLVRR — "From Verifiable Dot to Reward Chain" (2025) | Reward decomposition into content + style components |
| FinToolBench (2026) | Financial tool-use benchmark with auditable traces |
| Chen et al., "Auditing LLM Agents in Finance Must Prioritize Risk" (2025) | Risk-first evaluation framework |
| EconAgentBench (ICLR 2026) | Economic decision processes under partial information |
| TL-GRPO — "Turn-Level RL for Iterative Optimization" (2026) | Turn-level RL variant for persistent-state environments |
| MultiAgentBench / TAMAS (2025-2026) | Competitive multi-agent evaluation frameworks |
