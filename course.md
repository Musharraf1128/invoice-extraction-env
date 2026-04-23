# ESCTR: The Full Story — From Invoice Extraction to Enterprise Supply Chain Auditing

> This document captures the entire journey: the problem we set out to solve, the research we did, the approaches we tried, and how we arrived at the final ESCTR environment.

---

## Table of Contents

1. [The Starting Point — OpenEnv Hackathon](#1-the-starting-point)
2. [Round 1 — Invoice Extraction Environment](#2-round-1--invoice-extraction-environment)
3. [Research Phase — What Would Win Round 2?](#3-research-phase)
4. [The Pivot Decision — Why ESCTR](#4-the-pivot-decision)
5. [Architecture Deep Dive — How ESCTR Works](#5-architecture-deep-dive)
6. [Reward Design — RLVR Principles](#6-reward-design)
7. [What We Learned](#7-what-we-learned)

---

## 1. The Starting Point

### What is the OpenEnv Hackathon?

The **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology** is a hackathon focused on building **RL training environments for LLMs**. The core idea: instead of training LLMs on static datasets, we build interactive environments where agents learn through Reinforcement Learning with Verifiable Rewards (RLVR).

**OpenEnv** is a framework by Meta PyTorch and HuggingFace that treats RL environments as isolated microservices — the training loop (client) is completely decoupled from the environment simulation (server). The environment exposes standard HTTP endpoints (`/reset`, `/step`, `/state`) and the agent interacts through typed Actions and Observations.

### The Challenge

Build an OpenEnv-compliant environment that:
- Simulates a task humans actually perform
- Has programmatic, deterministic grading (no LLM-as-judge)
- Provides dense reward signals (not just 0/1 at the end)
- Supports multiple difficulty tiers
- Runs within 2 vCPU / 8GB RAM constraints
- Is deployable as a Docker container on HuggingFace Spaces

---

## 2. Round 1 — Invoice Extraction Environment

### The Original Idea

Our Round 1 submission was an **Invoice Extraction Environment** — an environment where an AI agent extracts structured data (vendor name, invoice number, line items, totals, etc.) from unstructured invoice documents.

### What We Built

- **5 difficulty tiers**: simple_invoice → messy_invoice → multi_document → corrupted_scan → adversarial_invoice
- **15 static documents** across the 5 tiers
- **Fuzzy string matching** for text fields, numeric tolerance for amounts
- **Multi-step interaction**: view_document → view_fields → extract → get_feedback → refine
- **OpenEnv compliance**: FastAPI server, typed Pydantic models, Docker deployment

### Round 1 Enhancements (Pre-Pivot)

Before Round 2 guidelines dropped, we upgraded the Round 1 environment with:

1. **Procedural Document Generation** (`procedural.py`): A seed-based engine generating infinite invoice variations — 15 vendor profiles, 15 customers, 25 products, OCR corruption simulation. This eliminated the overfitting risk of a 15-document static corpus.

2. **RLVR Composite Rewards**: Instead of a simple extraction score, we implemented:
   ```
   R_total = 0.70 × R_outcome + 0.30 × R_trajectory + bonuses
   ```
   With trajectory milestones (micro-rewards for viewing documents, getting feedback), efficiency bonuses, consistency bonuses (subtotal + tax = total), and penalties.

3. **Weighted Grading**: Financial fields scored 1.5×, line items 2.0×, with built-in cross-field arithmetic verification.

4. **Multi-Tool Workflow**: For hard tasks (multi_document, adversarial_invoice), we added `query_related_documents`, `verify_calculations`, and `check_discrepancies` tools.

### Why Round 1 Wasn't Enough

The enhanced invoice extraction was technically solid — all tests passed, good reward design, infinite procedural data. **But it wasn't going to win Round 2.**

---

## 3. Research Phase

### RESEARCH_1: The ESCTR Blueprint

We conducted deep research into what would maximize hackathon scoring. The key findings:

**The Core Problem with Invoice Extraction:**

| Vulnerability | Why It Hurts |
|--------------|-------------|
| **Saturated domain** | Document extraction is a well-trodden path. Judges have seen it before. |
| **Shallow interaction** | View document → extract → done. No real multi-step reasoning. |
| **Text-centric abstraction** | Pre-parsed text removes any visual/spatial reasoning challenge. |
| **Low novelty ceiling** | Even with procedural generation, the core task is "fill in the JSON fields." |

**What Frontier AI Research Demands:**

Drawing from the **OLMo 3 technical report** and RLVR research, we identified that winning environments need:
- **Long-horizon planning**: Agents that plan across 10-20 steps, not 3-5
- **Tool orchestration**: Multiple heterogeneous tools, not just "view" and "extract"
- **Partial observability**: Information spread across multiple databases, not one document
- **Adversarial dynamics**: Active counterparties that resist the agent's goal
- **Deterministic verification**: Correct answers that are mathematically provable, not fuzzy-matched

**The Proposed Solution: Enterprise Supply Chain & Tax Reconciliation (ESCTR)**

The research proposed pivoting from "extract data from an invoice" to "act as an autonomous financial controller investigating procurement discrepancies." This transforms a simple NLP extraction task into a genuine **agentic workflow** that maps to real enterprise operations worth trillions of dollars annually.

### RESEARCH_2: Supporting Analysis

The supplementary research validated the ESCTR concept against:
- Amazon's agentic AI evaluation practices
- Multi-agent negotiation frameworks
- The credit assignment problem in long-horizon RL
- Rubric-based reward systems for domains beyond simple verification

### Key Insight from Research

> "An environment that challenges frontier 72B models at 40% success rate on its hardest task provides more training headroom than one where 8B models already score 80%."

This directly informed our task difficulty design — Task 3 (Adversarial Auditing) is deliberately hard enough that a model must:
1. Query 5 different databases
2. Cross-reference shipping dates against SLA penalty clauses
3. Verify warehouse logs to disprove a vendor's false claim
4. Navigate a multi-turn negotiation
5. Reject a settlement offer
6. Calculate the exact penalty amount to 2 decimal places

---

## 4. The Pivot Decision

### Round 2 Guidelines Changed Everything

When the Round 2 guidelines arrived, the scoring criteria shifted dramatically:

| Criterion | Round 1 Weight | Round 2 Weight |
|-----------|---------------|---------------|
| Environment Innovation | ~30% | **40%** |
| Storytelling & Presentation | 0% | **30%** |
| Training Evidence (reward curves) | 0% | **20%** |
| Reward & Training Pipeline | ~25% | **10%** |

**70% of the score** now depends on innovation + storytelling. The guidelines explicitly warned:

> *"A messy but ambitious environment with real training evidence beats a polished but boring one."*
> *"Judges have seen a lot of chess, snake, tic-tac-toe, and grid-world clones."*

### The Decision Matrix

| Factor | Invoice Extraction | ESCTR |
|--------|-------------------|-------|
| Innovation (40%) | ⚠️ Known domain, seen before | ✅ Novel — supply chain auditing is unexplored in RL |
| Storytelling (30%) | ⚠️ Hard to make exciting | ✅ Strong narrative — "training autonomous financial controllers" |
| Training Evidence (20%) | Equal | Equal |
| Theme Alignment | Weak — barely touches themes | ✅ Hits Theme #3.1 (World Modeling), #2 (Long-Horizon), #1 (Multi-Agent) |
| Technical Depth | Good but shallow | ✅ 4 tools, 5 databases, adversarial negotiation |

### Decision: Full ESCTR Pivot

We chose **Option A: Full ESCTR Pivot** because:
1. The innovation ceiling is dramatically higher
2. The storytelling angle is compelling and unique
3. Our existing RLVR reward architecture transfers directly
4. The procedural generation concept transfers directly
5. We had 2 days pre-onsite + 2 days onsite to build it

The risk was real — a complete rewrite — but a "polished but boring" environment was guaranteed to lose.

---

## 5. Architecture Deep Dive

### How ESCTR Works

The agent is presented with a **discrepancy alert** and must use 4 ERP tools to investigate:

```
┌─────────────────────────────────────────┐
│           ESCTR Environment             │
│                                         │
│  ┌─────────┐  ┌──────────┐  ┌────────┐│
│  │ Purchase │  │ Shipping │  │  SLA   ││
│  │  Orders  │  │   Logs   │  │Contract││
│  └────┬─────┘  └────┬─────┘  └───┬────┘│
│       │              │            │      │
│  ┌────┴──────────────┴────────────┴────┐│
│  │         Tool Dispatcher              ││
│  │  query_database | read_document      ││
│  │  communicate_vendor                  ││
│  │  submit_financial_decision           ││
│  └────────────────┬─────────────────────┘│
│                   │                      │
│  ┌────────────────┴─────────────────────┐│
│  │         Grader Engine                ││
│  │  R = α·outcome + β·trajectory − pen  ││
│  └──────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### The Three Tasks

**Task 1 — Procurement Reconciliation (Easy)**
- A vendor invoices at higher prices than contracted
- Agent must: Query PO → Query Invoice → Compare line items → Find overcharge → Submit correction
- Grading: Correct line item ID + exact adjustment amount = 1.0

**Task 2 — SLA Enforcement (Medium)**
- A shipment arrived late, vendor demands full payment
- Agent must: Query shipping logs → Discover delay → Query SLA contract → Calculate penalty per terms → Submit deduction
- Grading: Exact penalty calculation = 1.0, within 5% = 0.7, within 10% = 0.4

**Task 3 — Adversarial Auditing (Hard)**
- Vendor disputes the late delivery, claims warehouse rejected shipment
- Agent must: Verify shipping delay → Get SLA terms → Query warehouse logs (prove dock was open) → Engage vendor → Reject settlement offer → Enforce full penalty
- Grading: Multi-axis — outcome (60%) + trajectory (40%) − gullibility penalty + evidence bonus

### Procedural Generation

Every scenario is generated from a seed using deterministic randomization:
- **15 vendor profiles** with US addresses
- **15 buyer profiles** with realistic business names
- **20 products** across hardware, electrical, IT, machinery categories
- **5 SLA penalty structures** (linear and tiered)
- Same seed → identical scenario → reproducible evaluation

### The Vendor Negotiation System

Task 3 features a **3-phase adversarial vendor**:

1. **Phase 1 — The Excuse**: Vendor claims your warehouse rejected delivery
2. **Phase 2 — The Settlement Offer**: Vendor offers 40-55% of the penalty as a "goodwill credit"
3. **Phase 3 — Concession or Persistence**: If agent rejects firmly + cites evidence, vendor concedes

The agent is penalized −0.20 for **gullibility** (accepting the settlement) and rewarded +0.05 for **evidence citation** (mentioning warehouse logs in the adjustment reason).

---

## 6. Reward Design

### RLVR Principles Applied

Our reward design follows principles from the OLMo 3 technical report:

```
R_total = α · R_outcome + β · R_trajectory − penalties
```

**Why not just binary rewards?**
- Sparse rewards (0 or 1 at the end) make credit assignment intractable in 15-20 step episodes
- The agent can't tell which of its 15 actions contributed to success or failure
- Dense trajectory rewards act as "algorithmic breadcrumbs" guiding policy gradients

**Trajectory Milestones:**

| Milestone | Meaning |
|-----------|---------|
| `retrieved_po` | Agent queried the purchase order database |
| `retrieved_invoice` | Agent queried the invoice database |
| `retrieved_shipping` | Agent discovered the shipping delay |
| `retrieved_sla` | Agent found the penalty terms |
| `checked_warehouse` | Agent verified internal records |
| `vendor_negotiation` | Agent engaged with the adversarial vendor |
| `calculated_penalty` | Agent performed penalty arithmetic |

**Penalties:**
- Step cost: −0.005 per action (encourages efficiency)
- Hallucination: −0.02 for invalid queries or nonexistent documents
- Gullibility: −0.20 for accepting adversarial settlements (Task 3)

**Why These Specific Values?**
- Step cost is small enough that investigation is still rewarded
- Hallucination penalty is 4× the step cost — bad actions are much worse than slow actions
- Gullibility penalty is massive (−0.20) because accepting a fraudulent claim is the worst possible failure mode in financial auditing

---

## 7. What We Learned

### Technical Lessons

1. **Procedural generation is non-negotiable** for RL environments. Static corpora get memorized instantly. Our engine generates unique scenarios from any seed.

2. **Tool restriction per task** is important. Easy tasks shouldn't have tools the agent can't meaningfully use — it creates noise in the reward signal.

3. **Adversarial dynamics create genuine difficulty.** A vendor that lies and offers settlements tests the agent's reasoning in ways static documents never can.

4. **Composite rewards require careful balancing.** If trajectory reward is too high, agents learn to query everything without ever submitting. If too low, they learn to guess without investigating.

### Strategic Lessons

1. **Read the scoring rubric backwards.** Don't start with what you want to build — start with what gets scored highest and work backwards.

2. **Innovation (40%) + Storytelling (30%) = 70%.** A technically perfect but boring environment loses to a messy but ambitious one with a great narrative.

3. **The pivot was worth the risk.** Rewriting 1000+ lines of code in 2 days was aggressive, but staying with invoice extraction would have capped us at "top 10, not first."

4. **Domain choice matters enormously.** Supply chain auditing is a multi-trillion dollar problem that's underexplored in AI training — this gives us both novelty and real-world utility.

---

## Appendix: File History

| Phase | Files Created/Modified | Purpose |
|-------|----------------------|---------|
| Round 1 | `server/documents.py` (15 static docs) | Original invoice corpus |
| Round 1 | `server/graders.py` (fuzzy matching) | Text extraction grading |
| Enhancement | `server/procedural.py` v1 (invoice generator) | Infinite invoice variations |
| Enhancement | `server/environment.py` v1 (6 tools) | Multi-tool invoice extraction |
| **ESCTR Pivot** | `server/models.py` (ESCTRAction/Obs/State) | ERP tool schemas |
| **ESCTR Pivot** | `server/procedural.py` v2 (corporate graphs) | Supply chain scenario generation |
| **ESCTR Pivot** | `server/graders.py` v2 (3 task graders) | Deterministic multi-axis scoring |
| **ESCTR Pivot** | `server/environment.py` v2 (4 tools + vendor AI) | Full ESCTR environment |
| **ESCTR Pivot** | `inference.py` v2 (financial controller) | Baseline agent script |
| **ESCTR Pivot** | Removed `server/documents.py` | No longer needed |
