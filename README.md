---
title: Invoice Extraction Environment
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Invoice Extraction Environment

An OpenEnv-compliant environment where AI agents extract structured data from unstructured invoice and receipt documents. Features **5 difficulty tiers** — from clean invoices to adversarial documents with decoy fields, OCR corruption, and hidden calculations — with **procedural document generation** for virtually infinite training configurations and an **RLVR-inspired composite reward architecture**.

**Space URL:** `https://huggingface.co/spaces/musharraf7/invoice-extraction-env`

```python
import requests

# Connect to the environment
url = "https://musharraf7-invoice-extraction-env.hf.space"
r = requests.post(f"{url}/reset", json={"task_name": "simple_invoice"})
print(r.json())
```

## Why This Environment?

Invoice data extraction is a **$5B+ industry** problem faced daily by every business. This environment provides:

- **Real RL training signal**: Per-field partial-credit scoring gives dense reward gradients via RLVR-inspired composite rewards
- **Infinite training data**: Procedural document generation creates unique invoices from any seed — eliminating overfitting to a static corpus
- **Genuine difficulty progression**: From clean invoices to adversarial traps that challenge frontier models
- **Multi-tool agentic workflow**: Hard tasks feature database queries, calculation verification, and discrepancy detection tools — training agents for multi-step reasoning
- **Reward shaping**: Trajectory milestones, consistency bonuses, efficiency signals, and improvement tracking provide rich learning signals beyond simple field matching
- **Production relevance**: The task directly models what commercial document processing systems must solve

## Reward Architecture (RLVR-Inspired)

The environment uses a composite reward function inspired by Reinforcement Learning with Verifiable Rewards:

```
R_total = α·R_outcome + β·R_trajectory + bonuses
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **R_outcome** | α = 0.70 | Weighted extraction accuracy (financial fields 1.5×, line items 2.0×) |
| **R_trajectory** | β = 0.30 | Micro-rewards for information gathering milestones |
| **Consistency bonus** | +0.03 | Agent's subtotal + tax = total |
| **Efficiency bonus** | +0.01–0.02 | Solution found in ≤5 steps |
| **Improvement bonus** | up to +0.02 | Score improves on retry |
| **Step cost** | -0.005/step | Encourages efficient exploration |
| **Hallucination penalty** | -0.02 | Invalid JSON or unknown commands |

### Trajectory Milestones

| Action | Micro-reward | Purpose |
|--------|-------------|---------|
| `view_document` | +0.01 | Evidence gathering |
| `view_fields` | +0.01 | Understanding requirements |
| `get_feedback` | +0.005 | Learning from errors |
| `query_related_documents` | +0.015 | Cross-referencing (hard tasks) |
| `verify_calculations` | +0.01 | Mathematical verification |
| `check_discrepancies` | +0.015 | Anomaly detection |

## Action Space

The agent sends an `InvoiceAction` with a `command` and optional `payload`:

| Command | Description | Payload | Available Tasks |
|---------|-------------|---------|-----------------|
| `view_document` | View the raw document text | — | All |
| `view_fields` | See required fields with descriptions | — | All |
| `extract` | Submit extracted fields | JSON string | All |
| `get_feedback` | Get detailed per-field feedback | — | All |
| `query_related_documents` | Retrieve PO, credit memos, etc. | — | multi_document, adversarial |
| `verify_calculations` | Submit arithmetic for verification | JSON string | multi_document, adversarial |
| `check_discrepancies` | Flag inconsistencies in documents | — | multi_document, adversarial |

### Action Schema
```json
{
  "command": "extract",
  "payload": "{\"invoice_number\": \"INV-2024-001\", \"date\": \"2024-01-15\", ...}"
}
```

## Observation Space

Each step returns an `InvoiceObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Response text from the environment |
| `task_name` | string | Current task name |
| `current_score` | float | Best score achieved so far |
| `attempts_remaining` | int | Remaining extraction attempts |
| `required_fields` | list | Fields to extract |
| `done` | bool | Whether the episode has ended |
| `reward` | float | Reward signal (0.01–0.99) |
| `last_action_status` | string | "success" or "error" |
| `error_message` | string | Diagnostic error message (if error) |
| `current_step` | int | Step number within episode |
| `accumulated_reward` | float | Total reward accumulated so far |

## Tasks (5 Difficulty Tiers)

### 1. `simple_invoice` (Easy) — 3 attempts
Clean, well-formatted invoices with clear field labels.

**Required fields:** `invoice_number`, `date`, `vendor_name`, `customer_name`, `subtotal`, `tax`, `total`, `line_items`

### 2. `messy_invoice` (Medium) — 3 attempts
Same fields but from messy, inconsistently formatted documents with abbreviations, typos, and non-standard layouts.

**Required fields:** Same as simple_invoice

### 3. `multi_document` (Hard) — 5 attempts
Complex multi-section documents containing a purchase order, invoice, and credit memo/payment receipt. The agent must cross-reference sections. **Advanced tools available** (`query_related_documents`, `verify_calculations`, `check_discrepancies`).

**Required fields:** All basic fields + `po_number`, `adjustment_reason`, `adjusted_total`

### 4. `corrupted_scan` (Very Hard) — 4 attempts
Simulates OCR-scanned/faxed invoices with systematic character errors:
- Character substitutions: `0`↔`O`, `1`↔`l`↔`I`, `5`↔`S`, `8`↔`B`
- Garbled sections and scan artifacts
- The agent must **reason through noise** to recover the true values

**Required fields:** Same as simple_invoice

### 5. `adversarial_invoice` (Expert) — 6 attempts
Adversarial documents designed to trap and challenge frontier models:
- **Decoy fields**: Multiple invoice numbers — only one is current
- **Hidden calculations**: Discounts the agent must compute
- **Contradictory sections**: PO vs invoice disagreements
- **Budget variance alerts**: Agent must identify and explain discrepancies

**Advanced tools available** for investigation.

**Required fields:** All basic fields + `po_number`, `discount_amount`, `original_total`, `discrepancy_notes`

## Procedural Document Generation

The environment features a **procedural generation engine** that creates unique invoice documents from any seed value:

- **15 vendor profiles** with addresses across the US
- **15 customer profiles** with realistic business names
- **25+ product catalog items** spanning hardware, software, and services
- **10 tax rate configurations** (5%–10%)
- **Deterministic**: Same seed always produces the same document
- **Infinite variety**: Seeds 0–2 use static test fixtures; seeds ≥ 3 generate novel documents

```python
# Use seed to get different documents
r = requests.post(f"{url}/reset", json={"task_name": "simple_invoice", "seed": 42})
r = requests.post(f"{url}/reset", json={"task_name": "simple_invoice", "seed": 100})
```

## Per-Field Scoring

- **Text fields**: Fuzzy matching with SequenceMatcher (0.0–1.0)
- **Numeric fields**: Exact match (1.0), within 1% (0.9), within 5% (0.5), within 10% (0.2)
- **Date fields**: Normalized comparison (YYYY-MM-DD) with format tolerance
- **Line items**: Best-fit matching of description, qty, price, amount (weighted 2.0×)
- **Reasoning fields** (discrepancy_notes): Fuzzy matching with lower threshold
- **Financial fields** (subtotal, tax, total): Weighted 1.5× for importance

## Setup Instructions

### Run with Docker
```bash
docker build -t invoice-extraction-env .
docker run -p 7860:7860 invoice-extraction-env
```

### Run locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with uv
```bash
uv run server
```

### Run inference
```bash
export ENV_URL="http://localhost:7860"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset with task selection |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current state |
| `/schema` | GET | Get action/observation schemas |
| `/metadata` | GET | Get environment metadata |
| `/ws` | WebSocket | Persistent session |

## Project Structure
```
├── server/
│   ├── __init__.py
│   ├── app.py             # FastAPI application
│   ├── environment.py     # Core environment logic + RLVR reward architecture
│   ├── documents.py       # 15-document corpus across 5 difficulty tiers
│   ├── procedural.py      # Procedural document generation engine
│   ├── graders.py         # Field-level scoring with weighted fuzzy matching
│   └── models.py          # Pydantic Action/Observation/State types
├── __init__.py            # Package declaration
├── inference.py           # Baseline inference script (all 5 tasks)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Package configuration
├── requirements.txt       # Dependencies
├── Dockerfile             # Container definition
└── README.md              # This file
```
