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

An OpenEnv-compliant environment where AI agents extract structured data from unstructured invoice and receipt documents. Features **5 difficulty tiers** — from clean invoices to adversarial documents with decoy fields, OCR corruption, and hidden calculations.

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

- **Real RL training signal**: Per-field partial-credit scoring gives dense reward gradients
- **Genuine difficulty progression**: From clean invoices to adversarial traps that challenge frontier models
- **Reward shaping**: Consistency bonuses, efficiency signals, and improvement tracking provide rich learning signals beyond simple field matching
- **Production relevance**: The task directly models what commercial document processing systems must solve

## Action Space

The agent sends an `InvoiceAction` with a `command` and optional `payload`:

| Command | Description | Payload |
|---------|-------------|---------|
| `view_document` | View the raw document text | — |
| `view_fields` | See required fields with descriptions | — |
| `extract` | Submit extracted fields | JSON string |
| `get_feedback` | Get detailed per-field feedback | — |

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

## Tasks (5 Difficulty Tiers)

### 1. `simple_invoice` (Easy) — 3 attempts
Clean, well-formatted invoices with clear field labels.

**Required fields:** `invoice_number`, `date`, `vendor_name`, `customer_name`, `subtotal`, `tax`, `total`, `line_items`

### 2. `messy_invoice` (Medium) — 3 attempts
Same fields but from messy, inconsistently formatted documents with abbreviations, typos, and non-standard layouts.

**Required fields:** Same as simple_invoice

### 3. `multi_document` (Hard) — 5 attempts
Complex multi-section documents containing a purchase order, invoice, and credit memo/payment receipt. The agent must cross-reference sections.

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

**Required fields:** All basic fields + `po_number`, `discount_amount`, `original_total`, `discrepancy_notes`

## Reward Design

### Per-Field Scoring (Base Score)
- **Text fields**: Fuzzy matching with SequenceMatcher (0.0–1.0)
- **Numeric fields**: Exact match (1.0), within 1% (0.9), within 5% (0.5)
- **Date fields**: Normalized comparison (YYYY-MM-DD)
- **Line items**: Best-fit matching of description, qty, price, amount
- **Reasoning fields** (discrepancy_notes): Fuzzy matching with lower threshold

### Reward Shaping Bonuses
| Bonus | Value | Trigger |
|-------|-------|---------|
| **Consistency** | +0.03 | Agent's subtotal + tax = total |
| **Efficiency** | +0.01–0.02 | Solution found in ≤5 steps |
| **Improvement** | up to +0.02 | Score improves on retry |

### Episode Mechanics
- **Best score tracked** across all extraction attempts
- **Partial progress** feedback identifies weak fields for refinement
- **Early termination** at score ≥ 0.95
- **All scores** clamped to strict (0.01, 0.99) range

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
│   ├── environment.py     # Core environment logic + reward shaping
│   ├── documents.py       # 15-document corpus across 5 difficulty tiers
│   ├── graders.py         # Field-level scoring with fuzzy matching
│   └── models.py          # Pydantic Action/Observation/State types
├── __init__.py            # Package declaration
├── inference.py           # Baseline inference script (all 5 tasks)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Package configuration
├── requirements.txt       # Dependencies
├── uv.lock                # Dependency lock file
├── Dockerfile             # Container definition
└── README.md              # This file
```
