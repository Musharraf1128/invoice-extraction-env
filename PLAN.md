# 🎯 ESCTR 30-Hour Battle Plan

**Start:** April 25, 2:30 PM IST (after lunch + ceremony)
**Deadline:** April 26, 3:00 PM IST (submission deadline)
**Available:** ~24.5 hours of real work time

---

## Current Status Audit

| Component | Status | Notes |
|-----------|--------|-------|
| Environment (server/) | ✅ DONE | 3 tasks, 4 tools, adversarial vendor, procedural gen |
| OpenEnv compliance | ✅ DONE | reset/step/state, typed schemas, openenv.yaml |
| HF Space deployed | ✅ DONE | `musharraf7/esctr-environment` |
| Inference script | ✅ DONE | Multi-turn, task-specific prompts, [START/STEP/END] |
| Training script | ✅ DONE | `train.py` — TRL GRPO with environment_factory |
| Training evidence (plots) | ❌ MISSING | **Non-negotiable requirement** |
| Baseline vs Trained comparison | ❌ MISSING | **Non-negotiable requirement** |
| Blog / Video / Slides | ❌ MISSING | **Non-negotiable requirement** |
| README (storytelling) | ⚠️ NEEDS UPDATE | Has structure but needs plots, links, market data |

---

## Scoring Breakdown & Strategy

| Criterion | Weight | Our Current Score | Target | How |
|-----------|--------|------------------|--------|-----|
| Environment Innovation | 40% | 35/40 | 38/40 | Already strong; polish README framing |
| Storytelling & Presentation | 30% | 5/30 | 25/30 | README rewrite + video/slides + pitch |
| Showing Training Improvement | 20% | 0/20 | 16/20 | Training + plots + before/after table |
| Reward & Training Pipeline | 10% | 2/10 | 8/10 | Working TRL/Unsloth script + Colab |

**Current estimated: ~42/100 → Target: ~87/100**

---

## The Plan

### BLOCK 1: Hours 0-3 (2:30 PM - 5:30 PM, Apr 25)
**Goal: Get training loop working**

- [x] Claim HF compute credits ($30): https://huggingface.co/coupons/claim/hf-openenv-community
- [x] Claim Cursor credits: https://tinyurl.com/sclr-openenv-dashboard
- [x] Study the reference training scripts (TRL OpenEnv docs, Wordle GRPO, environment_factory pattern)
- [x] Build `train.py`:
  - TRL GRPOTrainer with `environment_factory=ESCTRToolEnv`
  - Environment runs **in-process** (no HTTP needed)
  - Model: Qwen/Qwen3-1.7B (efficient on T4 with vLLM colocate)
  - 4 tool methods: query_database, read_document, communicate_vendor, submit_financial_decision
  - Start with Task 1 ONLY (procurement_reconciliation)
- [ ] Run smoke test: verify rewards flow on 5-10 episodes

### BLOCK 2: Hours 3-6 (5:30 PM - 8:30 PM, Apr 25)
**Goal: Training is running and producing data**

- [ ] Fix any bugs from smoke test
- [ ] Start real training on Task 1 (procurement_reconciliation)
  - Log: per-episode reward, episode length, task name, seed, milestones
  - Use HF Jobs with T4 GPU (small/medium) for training
- [ ] Run baseline evaluation (save these numbers!):
  - Run untrained model on 10-20 fixed seeds for each task
  - Record: mean reward, exact-correct rate per task
  - Save as `baseline_results.json`
- [ ] Let training run overnight → **CHECK BEFORE DINNER** that it's not crashed

### BLOCK 3: Hours 6-8 (8:30 PM - 10:30 PM, Apr 25)
**Goal: README v2 + verify training is alive**

- [ ] Check training progress — is reward curve moving up?
- [ ] Start README rewrite (storytelling format from strategy doc):
  1. One-paragraph problem hook with market data ($11.7B AI audit, $96.9B AI accounting)
  2. Two-paragraph environment summary
  3. Reward architecture (why it's hard to game)
  4. [PLACEHOLDER for training plots]
  5. [PLACEHOLDER for before/after table]
  6. Links to Space, training notebook, video
- [ ] Use the positioning lines from `esctr_hackathon_strategy.md`

### BLOCK 4: Hours 8-14 (10:30 PM - 4:30 AM, Apr 26)
**Goal: Extended training + sleep in shifts**

- [ ] If training stable: extend to Tasks 1+2 (add sla_enforcement)
- [ ] Only add Task 3 (adversarial_auditing) if pipeline is stable
- [ ] Check training every 2 hours for crashes
- [ ] Get some sleep! Set alarm for 6 AM

### BLOCK 5: Hours 14-18 (6:00 AM - 10:00 AM, Apr 26)  
**Goal: Harvest training results**

- [ ] Stop training, save final checkpoint
- [ ] Run trained model evaluation on same fixed seeds as baseline:
  - Mean reward by task
  - Exact-correct rate by task
  - Settlement gullibility rate (Task 3)
- [ ] Generate plots:
  - Reward curve over training steps (PNG, labeled axes)
  - Loss curve over training steps
  - Before/after comparison bar chart
- [ ] Create comparison table:
  ```
  | Metric | Baseline | Trained | Δ |
  |--------|----------|---------|---|
  | Task 1 avg reward | 0.xx | 0.xx | +xx% |
  | Task 2 avg reward | 0.xx | 0.xx | +xx% |
  | Task 3 avg reward | 0.xx | 0.xx | +xx% |
  ```
- [ ] Commit plots as PNG in `plots/` directory

### BLOCK 6: Hours 18-22 (10:00 AM - 2:00 PM, Apr 26)
**Goal: Storytelling artifacts + final README**

- [ ] Final README with embedded plots and comparison table
- [ ] Produce ONE of:
  - **Option A (fastest):** 3-5 slide deck (Google Slides) — Problem → Environment → Training → Results → Impact
  - **Option B:** <2 min screen recording showing environment + training curves
  - **Option C:** Mini HF blog post
- [ ] Link everything from README:
  - HF Space URL
  - Training notebook / script
  - Video / slides / blog
  - Plots
- [ ] Prepare 90-second verbal pitch (even if not presented live):
  - "ESCTR trains LLMs to be autonomous financial controllers..."
  - "We applied RLVR to enterprise supply chain auditing..."
  - "The trained model improved X% on reward and stopped accepting bad vendor settlements..."

### BLOCK 7: Hours 22-24 (2:00 PM - 3:00 PM, Apr 26)
**Goal: Final polish + submission**

- [ ] Final git push to GitHub
- [ ] Final push to HuggingFace Space
- [ ] Verify HF Space is building and healthy
- [ ] Open README in fresh browser — can a judge understand everything in 3 minutes?
- [ ] Verify ALL links work (Space, notebook, video/slides)
- [ ] **SUBMIT before 3:00 PM**

---

## Non-Negotiables (if time gets tight, these CANNOT be dropped)

1. ✅ Working training script connected to environment
2. ✅ At least ONE readable reward plot from a real run
3. ✅ Baseline vs trained comparison (table or chart)
4. ✅ README links to ALL assets (Space, notebook, video/slides)
5. ✅ Short memorable narrative about supply chain auditing

## Things to DROP if behind schedule

- Multi-task training (just do Task 1 if needed)
- Fancy video (use slides instead — 20 min to make, link from README)
- Perfect plots (ugly but real beats beautiful but fake)
- Environment polish (don't touch server/ code — it's done)

---

## Key Resources

| Resource | URL |
|----------|-----|
| HF Credits | https://huggingface.co/coupons/claim/hf-openenv-community |
| Cursor Credits | https://tinyurl.com/sclr-openenv-dashboard |
| TRL Wordle GRPO | https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb |
| TRL Sudoku GRPO | https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_sudoku_grpo.ipynb |
| Unsloth 2048 | https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/examples/unsloth_2048.ipynb |
| HF Jobs Docs | https://huggingface.co/docs/hub/jobs |
| Our HF Space | https://huggingface.co/spaces/musharraf7/esctr-environment |
| TRL OpenEnv Docs | https://huggingface.co/docs/trl/en/openenv |

---

## Quick Decision Rules

- **"Should I add a feature to the environment?"** → NO. Environment is frozen.
- **"Training is crashing, what do I prioritize?"** → Fix training. It's 30% of score (20% evidence + 10% pipeline).
- **"I have 2 hours left, what do I do?"** → Commit plots + update README + push. Everything must be visible in the repo.
- **"Plots are ugly"** → Ship them. Ugly real plots > no plots.
- **"Should I train on all 3 tasks?"** → Only if Task 1 is stable. Task 1 alone is enough.
