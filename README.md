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

> **Training LLMs to be autonomous financial auditors** — an OpenEnv environment for teaching AI agents to investigate procurement discrepancies, enforce SLA penalties, and navigate adversarial vendor disputes.

**Space URL:** `https://huggingface.co/spaces/musharraf7/invoice-extraction-env`

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

Each scenario generates: company profiles, product catalogs with contracted pricing, purchase orders, vendor invoices (with seeded discrepancies), SLA contracts (linear/tiered penalty structures), shipping telemetry, and warehouse access logs.

## Reward Architecture (RLVR-Inspired)

```
R_total = α·R_outcome + β·R_trajectory − penalties
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **R_outcome** | 60-70% | Did the agent submit the correct adjustment amount? |
| **R_trajectory** | 30-40% | Did the agent follow proper investigative procedure? |
| **Efficiency penalty** | -0.005/step | Encourages shortest path to resolution |
| **Hallucination penalty** | -0.02 | Invalid queries, nonexistent documents |
| **Gullibility penalty** | -0.20 | Accepting adversarial settlement offers (Task 3) |
| **Evidence bonus** | +0.05 | Citing warehouse logs as evidence (Task 3) |

### Why This Reward Design Matters

- **Dense, not sparse**: Trajectory milestones reward correct investigative behavior (querying the right databases, reading the right documents) even if the final answer is wrong
- **Hard to game**: An agent that spams queries gets penalized by step costs; an agent that submits without investigating gets 0 trajectory reward
- **Verifiable**: The correct answer is always a precise floating-point number derived from contract terms — no subjective evaluation

## Results

*Training evidence and reward plots will be added during the onsite hackathon (April 25-26) when compute credits are provided.*

<!-- Placeholder for training results -->
<!-- ![Reward curves](plots/reward_curves.png) -->

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
| *Is the domain underexplored?* | Yes — supply chain auditing + adversarial negotiation is nearly absent from RL/LLM training benchmarks |
| *Could a researcher write a paper about this?* | Yes — training autonomous financial auditors has direct commercial and academic value |
| *Is the reward hard to game?* | Yes — the correct answer is always a precise number from contract math; trajectory rewards require specific database queries |

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
