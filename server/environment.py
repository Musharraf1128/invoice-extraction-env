"""
Invoice Extraction Environment — Core Implementation.

A stateful environment where an AI agent extracts structured data
from unstructured invoice/receipt documents through a multi-step
interaction loop with RLVR-inspired dense reward signals.

Reward Architecture:
    R_total = α·R_outcome + β·R_trajectory + R_penalties
    α = 0.70 (outcome dominates)
    β = 0.30 (trajectory contributes)
    Penalties: step cost, hallucination penalties
"""

import json
from typing import Any, Optional
from uuid import uuid4

from .models import InvoiceAction, InvoiceObservation, InvoiceState
from .documents import get_document, TASK_REQUIRED_FIELDS
from .graders import grade_extraction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = {
    "simple_invoice": 3,
    "messy_invoice": 3,
    "multi_document": 5,
    "corrupted_scan": 4,
    "adversarial_invoice": 6,
}

# Reward architecture coefficients
ALPHA = 0.70   # outcome weight
BETA = 0.30    # trajectory weight

# Trajectory micro-rewards
REWARD_VIEW_DOC = 0.01
REWARD_VIEW_FIELDS = 0.01
REWARD_GET_FEEDBACK = 0.005
REWARD_QUERY_RELATED = 0.015
REWARD_VERIFY_CALC = 0.01
REWARD_CHECK_DISCREP = 0.015

# Penalties
PENALTY_PER_STEP = -0.005
PENALTY_INVALID_JSON = -0.02
PENALTY_UNKNOWN_CMD = -0.02
PENALTY_INVALID_CALC = -0.01

# Tasks that support advanced tool commands
TOOL_ENABLED_TASKS = {"multi_document", "adversarial_invoice"}

VALID_TASKS = list(TASK_REQUIRED_FIELDS.keys())


class InvoiceExtractionEnvironment:
    """Environment for extracting structured data from invoice documents.

    The agent interacts through these commands:
      - view_document: See the raw document text
      - view_fields: See the list of required fields
      - extract: Submit extracted fields as JSON
      - get_feedback: Get detailed feedback on last extraction
      - query_related_documents: Retrieve cross-reference documents
      - verify_calculations: Submit arithmetic for verification
      - check_discrepancies: Request environment to flag inconsistencies

    Reward design follows RLVR principles:
      R_total = α·R_outcome + β·R_trajectory + R_penalties
    """

    def __init__(self):
        self._state = InvoiceState(episode_id=str(uuid4()))
        self._document_text = ""
        self._ground_truth = {}
        self._required_fields = []
        self._last_feedback = {}
        self._last_extracted = {}
        self._initialized = False
        self._trajectory_reward = 0.0
        self._milestones = set()  # tracks which trajectory milestones agent has hit
        self._related_docs_text = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "simple_invoice",
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Reset the environment with a new task and document."""
        if task_name not in VALID_TASKS:
            task_name = "simple_invoice"

        doc_index = seed if seed is not None else 0
        doc_data = get_document(task_name, doc_index)
        max_attempts = MAX_ATTEMPTS.get(task_name, 3)

        self._state = InvoiceState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            document_id=doc_data["id"],
            best_score=0.0,
            attempts_used=0,
            max_attempts=max_attempts,
            accumulated_reward=0.0,
        )

        self._document_text = doc_data["text"]
        self._ground_truth = doc_data["ground_truth"]
        self._required_fields = doc_data["required_fields"]
        self._last_feedback = {}
        self._last_extracted = {}
        self._initialized = True
        self._trajectory_reward = 0.0
        self._milestones = set()
        self._related_docs_text = self._build_related_docs(task_name, doc_data)

        tool_hint = ""
        if task_name in TOOL_ENABLED_TASKS:
            tool_hint = (
                "\nAdvanced tools available for this task:\n"
                "  - 'query_related_documents': Retrieve PO, credit memos, etc.\n"
                "  - 'verify_calculations': Submit arithmetic for verification\n"
                "  - 'check_discrepancies': Flag inconsistencies in the document\n"
            )

        return InvoiceObservation(
            done=False,
            reward=0.0,
            text=(
                f"Invoice Extraction Environment ready.\n"
                f"Task: {task_name}\n"
                f"Document ID: {doc_data['id']}\n"
                f"Fields to extract: {len(self._required_fields)}\n"
                f"Max attempts: {max_attempts}\n\n"
                f"Use 'view_document' to see the document text.\n"
                f"Use 'view_fields' to see the required fields.\n"
                f"Use 'extract' with a JSON payload to submit your extraction.\n"
                f"Use 'get_feedback' to see feedback on your last attempt."
                f"{tool_hint}"
            ),
            task_name=task_name,
            current_score=0.0,
            attempts_remaining=max_attempts,
            required_fields=self._required_fields,
            current_step=0,
            accumulated_reward=0.0,
            last_action_status="success",
        )

    def _build_related_docs(self, task_name: str, doc_data: dict) -> str:
        """Build related documents text for cross-referencing tasks."""
        gt = doc_data["ground_truth"]
        if task_name not in TOOL_ENABLED_TASKS:
            return ""

        parts = []
        if "po_number" in gt:
            parts.append(
                f"=== PURCHASE ORDER ===\n"
                f"PO Number: {gt.get('po_number', 'N/A')}\n"
                f"Vendor: {gt.get('vendor_name', 'N/A')}\n"
                f"Buyer: {gt.get('customer_name', 'N/A')}\n"
            )
            if "line_items" in gt:
                for item in gt["line_items"]:
                    parts.append(
                        f"  - {item['quantity']}x {item['description']} "
                        f"@ ${item['unit_price']:.2f} = ${item['amount']:.2f}"
                    )
            parts.append("")

        if gt.get("adjustment_reason"):
            parts.append(
                f"=== ADJUSTMENT MEMO ===\n"
                f"Reason: {gt['adjustment_reason']}\n"
            )
            if gt.get("adjusted_total"):
                parts.append(f"Adjusted Total: ${gt['adjusted_total']:,.2f}")
            parts.append("")

        if gt.get("discount_amount") and gt["discount_amount"] > 0:
            parts.append(
                f"=== DISCOUNT APPLIED ===\n"
                f"Discount: ${gt['discount_amount']:,.2f}\n"
                f"Original Total: ${gt.get('original_total', 0):,.2f}\n"
            )

        return "\n".join(parts) if parts else "No related documents found for this invoice."

    def step(
        self,
        action: InvoiceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Execute a step in the environment."""
        if not self._initialized:
            return InvoiceObservation(
                done=True,
                reward=0.0,
                text="Error: Environment not initialized. Call reset() first.",
                metadata={"error": "not_initialized"},
                last_action_status="error",
                error_message="Environment not initialized. Call reset() first.",
            )

        self._state.step_count += 1
        command = action.command.lower().strip()

        # Apply per-step cost (encourages efficiency)
        self._trajectory_reward += PENALTY_PER_STEP

        handlers = {
            "view_document": self._handle_view_document,
            "view_fields": self._handle_view_fields,
            "extract": lambda: self._handle_extract(action.payload),
            "get_feedback": self._handle_get_feedback,
            "query_related_documents": self._handle_query_related,
            "verify_calculations": lambda: self._handle_verify_calculations(action.payload),
            "check_discrepancies": self._handle_check_discrepancies,
        }

        handler = handlers.get(command)
        if handler:
            return handler()
        else:
            # Unknown command penalty
            self._trajectory_reward += PENALTY_UNKNOWN_CMD
            self._state.accumulated_reward += PENALTY_UNKNOWN_CMD
            return self._make_obs(
                done=False,
                reward=0.0,
                text=(
                    f"Unknown command: '{command}'. "
                    f"Valid commands: {', '.join(handlers.keys())}"
                ),
                status="error",
                error_msg=f"Unknown command: '{command}'",
            )

    def _make_obs(
        self,
        done: bool,
        reward: float,
        text: str,
        status: str = "success",
        error_msg: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> InvoiceObservation:
        """Build a standardized observation."""
        return InvoiceObservation(
            done=done,
            reward=round(max(0.0, min(1.0, reward)), 4) if reward >= 0 else round(max(0.0, reward), 4),
            text=text,
            task_name=self._state.task_name,
            current_score=self._state.best_score,
            attempts_remaining=self._state.max_attempts - self._state.attempts_used,
            required_fields=self._required_fields,
            metadata=metadata or {},
            last_action_status=status,
            error_message=error_msg,
            current_step=self._state.step_count,
            accumulated_reward=round(self._state.accumulated_reward, 4),
        )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_view_document(self) -> InvoiceObservation:
        """Return the current document text (trajectory milestone)."""
        if "view_document" not in self._milestones:
            self._milestones.add("view_document")
            self._trajectory_reward += REWARD_VIEW_DOC
            self._state.accumulated_reward += REWARD_VIEW_DOC
        return self._make_obs(done=False, reward=0.0, text=self._document_text)

    def _handle_view_fields(self) -> InvoiceObservation:
        """Return the list of required fields with descriptions."""
        if "view_fields" not in self._milestones:
            self._milestones.add("view_fields")
            self._trajectory_reward += REWARD_VIEW_FIELDS
            self._state.accumulated_reward += REWARD_VIEW_FIELDS

        field_descriptions = {
            "invoice_number": "The invoice/document number (string)",
            "date": "Invoice date in YYYY-MM-DD format (use reissue date if applicable)",
            "vendor_name": "Name of the vendor/seller/supplier",
            "customer_name": "Name of the customer/buyer/bill-to party",
            "subtotal": "Subtotal before tax, after discounts (number)",
            "tax": "Tax amount (number)",
            "total": "Total amount due (number)",
            "line_items": "Array of items: [{description, quantity, unit_price, amount}]",
            "po_number": "Purchase order reference number (string)",
            "adjustment_reason": "Reason for any adjustments/credits (string)",
            "adjusted_total": "Final adjusted total after credits/payments (number)",
            "discount_amount": "Monetary discount value applied (number, 0 if none)",
            "original_total": "What the total would have been without adjustments (number)",
            "discrepancy_notes": "Free-text description of all discrepancies, adjustments, and anomalies found",
        }

        lines = ["Required fields to extract:\n"]
        for field in self._required_fields:
            desc = field_descriptions.get(field, "No description available")
            lines.append(f"  - {field}: {desc}")

        lines.append(f"\nSubmit your extraction using the 'extract' command.")
        lines.append(f"Payload must be a valid JSON string with these field names.")

        return self._make_obs(done=False, reward=0.0, text="\n".join(lines))

    def _handle_query_related(self) -> InvoiceObservation:
        """Return cross-reference documents (PO, credit memos, etc.)."""
        if self._state.task_name not in TOOL_ENABLED_TASKS:
            return self._make_obs(
                done=False, reward=0.0,
                text="This command is not available for the current task.",
                status="error",
                error_msg="query_related_documents only available for multi_document and adversarial_invoice tasks",
            )

        if "query_related" not in self._milestones:
            self._milestones.add("query_related")
            self._trajectory_reward += REWARD_QUERY_RELATED
            self._state.accumulated_reward += REWARD_QUERY_RELATED

        return self._make_obs(
            done=False, reward=0.0,
            text=self._related_docs_text or "No related documents found.",
        )

    def _handle_verify_calculations(self, payload: str) -> InvoiceObservation:
        """Verify arithmetic submitted by the agent."""
        if self._state.task_name not in TOOL_ENABLED_TASKS:
            return self._make_obs(
                done=False, reward=0.0,
                text="This command is not available for the current task.",
                status="error",
                error_msg="verify_calculations only available for multi_document and adversarial_invoice tasks",
            )

        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            self._trajectory_reward += PENALTY_INVALID_CALC
            self._state.accumulated_reward += PENALTY_INVALID_CALC
            return self._make_obs(
                done=False, reward=0.0,
                text="Invalid JSON payload for verify_calculations.",
                status="error",
                error_msg="Payload must be valid JSON with numeric fields to verify",
            )

        if "verify_calc" not in self._milestones:
            self._milestones.add("verify_calc")
            self._trajectory_reward += REWARD_VERIFY_CALC
            self._state.accumulated_reward += REWARD_VERIFY_CALC

        results = []
        gt = self._ground_truth
        checks = {
            "subtotal_plus_tax": (
                lambda: round(gt.get("subtotal", 0) + gt.get("tax", 0), 2),
                gt.get("total"),
            ),
        }

        sub = data.get("subtotal")
        tax = data.get("tax")
        total = data.get("total")

        if sub is not None and tax is not None:
            computed = round(float(sub) + float(tax), 2)
            if total is not None:
                match = abs(computed - float(total)) < 0.02
                results.append(
                    f"subtotal ({sub}) + tax ({tax}) = {computed} | "
                    f"your total ({total}) — {'MATCH ✓' if match else 'MISMATCH ✗'}"
                )
            else:
                results.append(f"subtotal ({sub}) + tax ({tax}) = {computed}")

        if not results:
            results.append("No recognizable calculations found. Submit fields like: subtotal, tax, total")

        return self._make_obs(
            done=False, reward=0.0,
            text="Calculation verification:\n" + "\n".join(results),
        )

    def _handle_check_discrepancies(self) -> InvoiceObservation:
        """Flag inconsistencies in the document."""
        if self._state.task_name not in TOOL_ENABLED_TASKS:
            return self._make_obs(
                done=False, reward=0.0,
                text="This command is not available for the current task.",
                status="error",
                error_msg="check_discrepancies only available for multi_document and adversarial_invoice tasks",
            )

        if "check_discrep" not in self._milestones:
            self._milestones.add("check_discrep")
            self._trajectory_reward += REWARD_CHECK_DISCREP
            self._state.accumulated_reward += REWARD_CHECK_DISCREP

        gt = self._ground_truth
        hints = []

        if gt.get("discount_amount") and gt["discount_amount"] > 0:
            hints.append("⚠ A discount has been applied to this invoice.")
        if gt.get("adjustment_reason"):
            hints.append("⚠ There is an adjustment/credit memo affecting the final amount.")
        if gt.get("po_number"):
            hints.append("⚠ This invoice references a purchase order — cross-check quantities and amounts.")
        if gt.get("original_total") and gt.get("total"):
            if abs(gt["original_total"] - gt["total"]) > 0.01:
                hints.append("⚠ The final total differs from the original total — investigate adjustments.")

        if not hints:
            hints.append("No obvious discrepancies detected.")

        return self._make_obs(
            done=False, reward=0.0,
            text="Discrepancy analysis:\n" + "\n".join(hints),
        )

    def _handle_extract(self, payload: str) -> InvoiceObservation:
        """Process an extraction attempt with RLVR-style composite reward."""
        attempts_remaining = self._state.max_attempts - self._state.attempts_used

        if attempts_remaining <= 0:
            return self._make_obs(
                done=True,
                reward=self._state.best_score,
                text="No attempts remaining. Episode is complete.",
                metadata={"final_score": self._state.best_score},
            )

        # Parse the JSON payload
        try:
            extracted = json.loads(payload)
            if not isinstance(extracted, dict):
                raise ValueError("Payload must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            self._state.attempts_used += 1
            self._trajectory_reward += PENALTY_INVALID_JSON
            self._state.accumulated_reward += PENALTY_INVALID_JSON
            attempts_remaining = self._state.max_attempts - self._state.attempts_used
            done = attempts_remaining <= 0

            return self._make_obs(
                done=done,
                reward=0.0,
                text=f"Invalid JSON payload: {str(e)}\nPlease submit a valid JSON object.",
                status="error",
                error_msg=f"Invalid JSON: {str(e)}",
                metadata={"error": "invalid_json"},
            )

        # Grade the extraction
        self._state.attempts_used += 1
        base_score, feedback = grade_extraction(
            extracted, self._ground_truth, self._required_fields
        )

        # === COMPOSITE REWARD (RLVR-inspired) ===

        # R_outcome: base extraction score
        r_outcome = base_score

        # R_trajectory: accumulated from milestones
        r_trajectory = max(0.0, self._trajectory_reward)

        # Improvement bonus
        improvement_bonus = 0.0
        if self._state.attempts_used > 1 and base_score > self._state.best_score:
            improvement_bonus = min(base_score - self._state.best_score, 0.02)

        # Step efficiency bonus
        efficiency_bonus = 0.0
        if self._state.step_count <= 3:
            efficiency_bonus = 0.02
        elif self._state.step_count <= 5:
            efficiency_bonus = 0.01

        # Consistency bonus (subtotal + tax ≈ total)
        consistency_bonus = 0.0
        ext_sub = _safe_float(extracted.get("subtotal"))
        ext_tax = _safe_float(extracted.get("tax"))
        ext_total = _safe_float(extracted.get("total"))
        if ext_sub is not None and ext_tax is not None and ext_total is not None:
            computed = round(ext_sub + ext_tax, 2)
            if abs(computed - ext_total) < 0.02:
                consistency_bonus = 0.03

        # Composite reward
        bonus = improvement_bonus + efficiency_bonus + consistency_bonus
        score = round(max(0.01, min(0.99, ALPHA * r_outcome + BETA * r_trajectory + bonus)), 4)

        # Track
        self._state.best_score = max(self._state.best_score, score)
        self._state.accumulated_reward += score
        self._last_feedback = feedback
        self._last_extracted = extracted

        attempts_remaining = self._state.max_attempts - self._state.attempts_used
        done = attempts_remaining <= 0 or score >= 0.95

        # Build feedback text
        matched = sum(1 for f in feedback.values() if f.get("matched", False))
        total_fields = len(feedback)
        bonus_details = []
        if consistency_bonus > 0:
            bonus_details.append(f"consistency: +{consistency_bonus:.3f}")
        if improvement_bonus > 0:
            bonus_details.append(f"improvement: +{improvement_bonus:.3f}")
        if efficiency_bonus > 0:
            bonus_details.append(f"efficiency: +{efficiency_bonus:.3f}")
        if r_trajectory > 0:
            bonus_details.append(f"trajectory: {r_trajectory:.3f}")

        feedback_text = (
            f"Extraction scored: {score:.4f} "
            f"(outcome: {r_outcome:.4f} × {ALPHA}, trajectory: {r_trajectory:.3f} × {BETA})\n"
            f"Fields matched: {matched}/{total_fields}\n"
            f"Best score so far: {self._state.best_score:.4f}\n"
            f"Attempts remaining: {attempts_remaining}\n"
        )

        if bonus_details:
            feedback_text += f"Reward bonuses: {', '.join(bonus_details)}\n"

        if not done and score < 0.95:
            weak_fields = [
                name for name, data in feedback.items()
                if not data.get("matched", False)
            ]
            if weak_fields:
                feedback_text += f"\nFields needing improvement: {', '.join(weak_fields)}"
                feedback_text += "\nUse 'get_feedback' for detailed per-field scores."

        if done:
            feedback_text += f"\n\nEpisode complete. Final score: {self._state.best_score:.4f}"

        return self._make_obs(
            done=done,
            reward=score,
            text=feedback_text,
            metadata={
                "score": score,
                "base_score": base_score,
                "r_outcome": r_outcome,
                "r_trajectory": r_trajectory,
                "bonus": bonus,
                "bonus_details": bonus_details,
                "best_score": self._state.best_score,
                "field_scores": {k: v["score"] for k, v in feedback.items()},
            },
        )

    def _handle_get_feedback(self) -> InvoiceObservation:
        """Return detailed feedback on the last extraction attempt."""
        if not self._last_feedback:
            return self._make_obs(
                done=False,
                reward=0.0,
                text="No extraction attempt yet. Use 'extract' to submit your extraction first.",
            )

        if "get_feedback" not in self._milestones:
            self._milestones.add("get_feedback")
            self._trajectory_reward += REWARD_GET_FEEDBACK
            self._state.accumulated_reward += REWARD_GET_FEEDBACK

        lines = ["Detailed feedback on last extraction:\n"]
        for field, data in self._last_feedback.items():
            score = data.get("score", 0.0)
            matched = "✓" if data.get("matched", False) else "✗"
            field_type = data.get("expected_type", "unknown")
            lines.append(f"  [{matched}] {field} ({field_type}): {score:.2f}")

        lines.append(f"\nOverall best score: {self._state.best_score:.2f}")
        lines.append(f"Attempts remaining: {self._state.max_attempts - self._state.attempts_used}")

        return self._make_obs(
            done=False,
            reward=0.0,
            text="\n".join(lines),
            metadata={"field_feedback": self._last_feedback},
        )

    @property
    def state(self) -> InvoiceState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up resources."""
        self._initialized = False


def _safe_float(value) -> float:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re
        cleaned = re.sub(r"[$ ,]", "", value.strip())
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None
