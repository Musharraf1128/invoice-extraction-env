"""
Deterministic Graders for the ESCTR Environment.

Each task has a specific grader that scores the agent's performance
using verifiable, programmatic criteria — no subjective evaluation.

Scoring is always in the strict range (0.01, 0.99) to satisfy OpenEnv validators.
"""

from typing import Any, Dict, List, Tuple

from .procedural import Scenario


def clamp_score(score: float) -> float:
    """Clamp score to strict (0.01, 0.99) range."""
    return round(max(0.01, min(0.99, score)), 4)


# ---------------------------------------------------------------------------
# Task 1: Procurement Reconciliation
# ---------------------------------------------------------------------------

def grade_task1(
    scenario: Scenario,
    submitted_amount: float,
    submitted_line_item: str = None,
    milestones: List[str] = None,
    steps_taken: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """Grade the procurement reconciliation task.

    Perfect score requires:
    - Correct discrepant line item identified
    - Exact adjustment amount (overcharge value, negative)

    Partial credit:
    - Correct line item but wrong amount → 0.5
    - Wrong line item → 0.0 outcome
    """
    milestones = milestones or []
    feedback = {"task": "procurement_reconciliation"}

    # Outcome scoring (weight: 0.70)
    correct_amount = scenario.correct_adjustment
    correct_item = scenario.discrepant_line_item_id

    outcome_score = 0.0
    item_correct = (submitted_line_item == correct_item) if submitted_line_item and correct_item else False
    amount_correct = abs(submitted_amount - correct_amount) < 0.02 if submitted_amount is not None else False

    if item_correct and amount_correct:
        outcome_score = 1.0
        feedback["outcome"] = "PERFECT — correct line item and exact adjustment amount"
    elif item_correct and not amount_correct:
        outcome_score = 0.5
        feedback["outcome"] = f"PARTIAL — correct line item but wrong amount (expected {correct_amount:.2f}, got {submitted_amount:.2f})"
    elif not item_correct and amount_correct:
        outcome_score = 0.4
        feedback["outcome"] = f"PARTIAL — correct amount but wrong line item (expected {correct_item})"
    else:
        outcome_score = 0.0
        feedback["outcome"] = "FAIL — wrong line item and wrong amount"

    # Trajectory scoring (weight: 0.30)
    trajectory_score = 0.0
    trajectory_details = []
    if "retrieved_po" in milestones:
        trajectory_score += 0.4
        trajectory_details.append("Retrieved PO ✓")
    if "retrieved_invoice" in milestones:
        trajectory_score += 0.4
        trajectory_details.append("Retrieved Invoice ✓")
    if "compared_documents" in milestones:
        trajectory_score += 0.2
        trajectory_details.append("Compared documents ✓")

    trajectory_score = min(1.0, trajectory_score)
    feedback["trajectory"] = trajectory_details

    # Efficiency penalty
    max_steps = 10
    efficiency_penalty = max(0, (steps_taken - max_steps) * 0.02)

    # Composite
    alpha, beta = 0.70, 0.30
    raw_score = alpha * outcome_score + beta * trajectory_score - efficiency_penalty
    final_score = clamp_score(raw_score)

    feedback["outcome_score"] = outcome_score
    feedback["trajectory_score"] = trajectory_score
    feedback["efficiency_penalty"] = efficiency_penalty
    feedback["final_score"] = final_score
    feedback["correct_adjustment"] = correct_amount
    feedback["correct_line_item"] = correct_item

    return final_score, feedback


# ---------------------------------------------------------------------------
# Task 2: SLA Enforcement
# ---------------------------------------------------------------------------

def grade_task2(
    scenario: Scenario,
    submitted_amount: float,
    milestones: List[str] = None,
    steps_taken: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """Grade the SLA enforcement task.

    Perfect score requires:
    - Exact penalty amount calculated from shipping delay + SLA terms

    Partial credit:
    - Within 5% of correct penalty → 0.7
    - Within 10% → 0.4
    - Approved invoice without penalty → 0.0
    """
    milestones = milestones or []
    feedback = {"task": "sla_enforcement"}

    correct_penalty = scenario.penalty_amount or 0.0
    correct_adjustment = scenario.correct_adjustment  # negative

    # Outcome scoring (weight: 0.60)
    outcome_score = 0.0
    if submitted_amount is not None and correct_adjustment != 0:
        error = abs(submitted_amount - correct_adjustment)
        error_pct = error / abs(correct_adjustment) if correct_adjustment != 0 else float('inf')

        if error < 0.02:
            outcome_score = 1.0
            feedback["outcome"] = "PERFECT — exact penalty amount"
        elif error_pct <= 0.05:
            outcome_score = 0.7
            feedback["outcome"] = f"CLOSE — within 5% (expected {correct_adjustment:.2f}, got {submitted_amount:.2f})"
        elif error_pct <= 0.10:
            outcome_score = 0.4
            feedback["outcome"] = f"PARTIAL — within 10% (expected {correct_adjustment:.2f}, got {submitted_amount:.2f})"
        else:
            outcome_score = 0.1
            feedback["outcome"] = f"INCORRECT — expected {correct_adjustment:.2f}, got {submitted_amount:.2f}"
    elif submitted_amount == 0 or submitted_amount is None:
        outcome_score = 0.0
        feedback["outcome"] = "FAIL — approved invoice without applying penalty"

    # Trajectory scoring (weight: 0.40)
    trajectory_score = 0.0
    trajectory_details = []
    if "retrieved_shipping" in milestones:
        trajectory_score += 0.30
        trajectory_details.append("Retrieved shipping log ✓")
    if "retrieved_sla" in milestones:
        trajectory_score += 0.30
        trajectory_details.append("Retrieved SLA contract ✓")
    if "retrieved_po" in milestones:
        trajectory_score += 0.15
        trajectory_details.append("Retrieved PO ✓")
    if "retrieved_invoice" in milestones:
        trajectory_score += 0.15
        trajectory_details.append("Retrieved Invoice ✓")
    if "calculated_penalty" in milestones:
        trajectory_score += 0.10
        trajectory_details.append("Performed penalty calculation ✓")

    trajectory_score = min(1.0, trajectory_score)
    feedback["trajectory"] = trajectory_details

    # Efficiency
    max_steps = 15
    efficiency_penalty = max(0, (steps_taken - max_steps) * 0.02)

    alpha, beta = 0.60, 0.40
    raw_score = alpha * outcome_score + beta * trajectory_score - efficiency_penalty
    final_score = clamp_score(raw_score)

    feedback["outcome_score"] = outcome_score
    feedback["trajectory_score"] = trajectory_score
    feedback["efficiency_penalty"] = efficiency_penalty
    feedback["final_score"] = final_score
    feedback["correct_adjustment"] = correct_adjustment
    feedback["penalty_amount"] = correct_penalty

    return final_score, feedback


# ---------------------------------------------------------------------------
# Task 3: Adversarial Auditing
# ---------------------------------------------------------------------------

def grade_task3(
    scenario: Scenario,
    submitted_amount: float,
    rejected_settlement: bool = False,
    cited_evidence: bool = False,
    milestones: List[str] = None,
    steps_taken: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """Grade the adversarial auditing task.

    Perfect score requires:
    - Full penalty enforced (rejected vendor's bogus claim)
    - Settlement offer rejected
    - Warehouse logs cited as evidence

    Severe deductions for:
    - Accepting settlement (gullibility)
    - Not checking warehouse logs before rejecting
    """
    milestones = milestones or []
    feedback = {"task": "adversarial_auditing"}

    correct_adjustment = scenario.correct_adjustment

    # Outcome scoring (weight: 0.60)
    outcome_score = 0.0
    if submitted_amount is not None and correct_adjustment != 0:
        error = abs(submitted_amount - correct_adjustment)
        error_pct = error / abs(correct_adjustment) if correct_adjustment != 0 else float('inf')

        if error < 0.02:
            outcome_score = 1.0
            feedback["outcome"] = "PERFECT — full contractual penalty enforced"
        elif error_pct <= 0.05:
            outcome_score = 0.6
            feedback["outcome"] = f"CLOSE — nearly correct (expected {correct_adjustment:.2f}, got {submitted_amount:.2f})"
        else:
            outcome_score = 0.2
            feedback["outcome"] = f"INCORRECT — expected {correct_adjustment:.2f}, got {submitted_amount:.2f}"
    else:
        outcome_score = 0.0
        feedback["outcome"] = "FAIL — no financial decision submitted"

    # Gullibility penalty: accepted adversarial settlement
    gullibility_penalty = 0.0
    if not rejected_settlement and "vendor_negotiation" in milestones:
        gullibility_penalty = 0.20
        feedback["gullibility"] = "PENALIZED — accepted vendor's settlement offer"
    elif rejected_settlement:
        feedback["gullibility"] = "GOOD — rejected vendor's settlement offer"

    # Evidence citation bonus
    evidence_bonus = 0.0
    if cited_evidence and "checked_warehouse" in milestones:
        evidence_bonus = 0.05
        feedback["evidence"] = "GOOD — cited warehouse logs as evidence"

    # Trajectory scoring (weight: 0.40)
    trajectory_score = 0.0
    trajectory_details = []
    if "retrieved_shipping" in milestones:
        trajectory_score += 0.20
        trajectory_details.append("Retrieved shipping log ✓")
    if "retrieved_sla" in milestones:
        trajectory_score += 0.20
        trajectory_details.append("Retrieved SLA contract ✓")
    if "checked_warehouse" in milestones:
        trajectory_score += 0.25
        trajectory_details.append("Checked warehouse access logs ✓")
    if "vendor_negotiation" in milestones:
        trajectory_score += 0.15
        trajectory_details.append("Engaged in vendor negotiation ✓")
    if "retrieved_po" in milestones:
        trajectory_score += 0.10
        trajectory_details.append("Retrieved PO ✓")
    if "retrieved_invoice" in milestones:
        trajectory_score += 0.10
        trajectory_details.append("Retrieved Invoice ✓")

    trajectory_score = min(1.0, trajectory_score)
    feedback["trajectory"] = trajectory_details

    # Efficiency
    max_steps = 20
    efficiency_penalty = max(0, (steps_taken - max_steps) * 0.015)

    alpha, beta = 0.60, 0.40
    raw_score = (alpha * outcome_score + beta * trajectory_score
                 + evidence_bonus - gullibility_penalty - efficiency_penalty)
    final_score = clamp_score(raw_score)

    feedback["outcome_score"] = outcome_score
    feedback["trajectory_score"] = trajectory_score
    feedback["gullibility_penalty"] = gullibility_penalty
    feedback["evidence_bonus"] = evidence_bonus
    feedback["efficiency_penalty"] = efficiency_penalty
    feedback["final_score"] = final_score
    feedback["correct_adjustment"] = correct_adjustment

    return final_score, feedback
