"""
Invoice Extraction Environment — Core Implementation.

A stateful environment where an AI agent extracts structured data
from unstructured invoice/receipt documents through a multi-step
interaction loop with partial-credit reward signals.
"""

import json
from typing import Any, Optional
from uuid import uuid4

from .models import InvoiceAction, InvoiceObservation, InvoiceState
from .documents import get_document, TASK_REQUIRED_FIELDS
from .graders import grade_extraction


# Max extraction attempts per difficulty
MAX_ATTEMPTS = {
    "simple_invoice": 3,
    "messy_invoice": 3,
    "multi_document": 5,
    "corrupted_scan": 4,
    "adversarial_invoice": 6,
}

VALID_TASKS = list(TASK_REQUIRED_FIELDS.keys())


class InvoiceExtractionEnvironment:
    """Environment for extracting structured data from invoice documents.

    The agent interacts through these commands:
      - view_document: See the raw document text
      - view_fields: See the list of required fields
      - extract: Submit extracted fields as JSON
      - get_feedback: Get detailed feedback on last extraction

    Rewards are in [0.0, 1.0] and accumulate based on best score.
    """

    def __init__(self):
        self._state = InvoiceState(episode_id=str(uuid4()))
        self._document_text = ""
        self._ground_truth = {}
        self._required_fields = []
        self._last_feedback = {}
        self._last_extracted = {}
        self._initialized = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "simple_invoice",
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Reset the environment with a new task and document.

        Args:
            seed: Random seed for document selection
            episode_id: Optional custom episode ID
            task_name: One of 'simple_invoice', 'messy_invoice', 'multi_document'

        Returns:
            Initial observation with task information
        """
        if task_name not in VALID_TASKS:
            task_name = "simple_invoice"

        # Select document (deterministic based on seed)
        doc_index = (seed or 0) % 3
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
        )

        self._document_text = doc_data["text"]
        self._ground_truth = doc_data["ground_truth"]
        self._required_fields = doc_data["required_fields"]
        self._last_feedback = {}
        self._last_extracted = {}
        self._initialized = True

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
            ),
            task_name=task_name,
            current_score=0.0,
            attempts_remaining=max_attempts,
            required_fields=self._required_fields,
        )

    def step(
        self,
        action: InvoiceAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Execute a step in the environment.

        Args:
            action: InvoiceAction with command and optional payload
            timeout_s: Unused, kept for interface compatibility

        Returns:
            InvoiceObservation with response and reward
        """
        if not self._initialized:
            return InvoiceObservation(
                done=True,
                reward=0.0,
                text="Error: Environment not initialized. Call reset() first.",
                metadata={"error": "not_initialized"},
            )

        self._state.step_count += 1
        command = action.command.lower().strip()

        if command == "view_document":
            return self._handle_view_document()
        elif command == "view_fields":
            return self._handle_view_fields()
        elif command == "extract":
            return self._handle_extract(action.payload)
        elif command == "get_feedback":
            return self._handle_get_feedback()
        else:
            return InvoiceObservation(
                done=False,
                reward=0.0,
                text=(
                    f"Unknown command: '{command}'. "
                    f"Valid commands: view_document, view_fields, extract, get_feedback"
                ),
                task_name=self._state.task_name,
                current_score=self._state.best_score,
                attempts_remaining=self._state.max_attempts - self._state.attempts_used,
                required_fields=self._required_fields,
                metadata={"error": "unknown_command"},
            )

    def _handle_view_document(self) -> InvoiceObservation:
        """Return the current document text."""
        return InvoiceObservation(
            done=False,
            reward=0.0,
            text=self._document_text,
            task_name=self._state.task_name,
            current_score=self._state.best_score,
            attempts_remaining=self._state.max_attempts - self._state.attempts_used,
            required_fields=self._required_fields,
        )

    def _handle_view_fields(self) -> InvoiceObservation:
        """Return the list of required fields with descriptions."""
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

        return InvoiceObservation(
            done=False,
            reward=0.0,
            text="\n".join(lines),
            task_name=self._state.task_name,
            current_score=self._state.best_score,
            attempts_remaining=self._state.max_attempts - self._state.attempts_used,
            required_fields=self._required_fields,
        )

    def _handle_extract(self, payload: str) -> InvoiceObservation:
        """Process an extraction attempt."""
        attempts_remaining = self._state.max_attempts - self._state.attempts_used

        if attempts_remaining <= 0:
            return InvoiceObservation(
                done=True,
                reward=self._state.best_score,
                text="No attempts remaining. Episode is complete.",
                task_name=self._state.task_name,
                current_score=self._state.best_score,
                attempts_remaining=0,
                required_fields=self._required_fields,
                metadata={"final_score": self._state.best_score},
            )

        # Parse the JSON payload
        try:
            extracted = json.loads(payload)
            if not isinstance(extracted, dict):
                raise ValueError("Payload must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            self._state.attempts_used += 1
            attempts_remaining = self._state.max_attempts - self._state.attempts_used
            done = attempts_remaining <= 0

            return InvoiceObservation(
                done=done,
                reward=0.0,
                text=f"Invalid JSON payload: {str(e)}\nPlease submit a valid JSON object.",
                task_name=self._state.task_name,
                current_score=self._state.best_score,
                attempts_remaining=attempts_remaining,
                required_fields=self._required_fields,
                metadata={"error": "invalid_json", "details": str(e)},
            )

        # Grade the extraction
        self._state.attempts_used += 1
        base_score, feedback = grade_extraction(
            extracted, self._ground_truth, self._required_fields
        )

        # === REWARD SHAPING BONUSES ===
        bonus = 0.0
        bonus_details = []

        # 1. Mathematical consistency bonus: subtotal + tax ≈ total
        ext_sub = _safe_float(extracted.get("subtotal"))
        ext_tax = _safe_float(extracted.get("tax"))
        ext_total = _safe_float(extracted.get("total"))
        if ext_sub is not None and ext_tax is not None and ext_total is not None:
            computed = round(ext_sub + ext_tax, 2)
            if abs(computed - ext_total) < 0.02:
                bonus += 0.03
                bonus_details.append("consistency_check: +0.03")

        # 2. Improvement tracking: rewarding learning from feedback
        prev_score = self._state.best_score
        if self._state.attempts_used > 1 and base_score > prev_score:
            improvement = min(base_score - prev_score, 0.02)
            bonus += improvement
            bonus_details.append(f"improvement: +{improvement:.3f}")

        # 3. Step efficiency signal: fewer steps = small bonus
        steps_used = self._state.step_count
        if steps_used <= 3:
            bonus += 0.02  # Very efficient
            bonus_details.append("efficiency: +0.02")
        elif steps_used <= 5:
            bonus += 0.01  # Moderately efficient
            bonus_details.append("efficiency: +0.01")

        # Apply bonus (clamped to strict (0, 1))
        score = round(max(0.01, min(0.99, base_score + bonus)), 4)

        # Track best score
        self._state.best_score = max(self._state.best_score, score)
        self._last_feedback = feedback
        self._last_extracted = extracted

        attempts_remaining = self._state.max_attempts - self._state.attempts_used
        done = attempts_remaining <= 0 or score >= 0.95

        # Build feedback text
        matched = sum(1 for f in feedback.values() if f.get("matched", False))
        total = len(feedback)
        feedback_text = (
            f"Extraction scored: {score:.4f} (base: {base_score:.4f}, bonus: {bonus:.3f})\n"
            f"Fields matched: {matched}/{total}\n"
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

        return InvoiceObservation(
            done=done,
            reward=score,
            text=feedback_text,
            task_name=self._state.task_name,
            current_score=self._state.best_score,
            attempts_remaining=attempts_remaining,
            required_fields=self._required_fields,
            metadata={
                "score": score,
                "base_score": base_score,
                "bonus": bonus,
                "bonus_details": bonus_details,
                "best_score": self._state.best_score,
                "field_scores": {k: v["score"] for k, v in feedback.items()},
            },
        )

    def _handle_get_feedback(self) -> InvoiceObservation:
        """Return detailed feedback on the last extraction attempt."""
        if not self._last_feedback:
            return InvoiceObservation(
                done=False,
                reward=0.0,
                text="No extraction attempt yet. Use 'extract' to submit your extraction first.",
                task_name=self._state.task_name,
                current_score=self._state.best_score,
                attempts_remaining=self._state.max_attempts - self._state.attempts_used,
                required_fields=self._required_fields,
            )

        lines = ["Detailed feedback on last extraction:\n"]
        for field, data in self._last_feedback.items():
            score = data.get("score", 0.0)
            matched = "Y" if data.get("matched", False) else "N"
            field_type = data.get("expected_type", "unknown")
            lines.append(f"  [{matched}] {field} ({field_type}): {score:.2f}")

        lines.append(f"\nOverall best score: {self._state.best_score:.2f}")
        lines.append(f"Attempts remaining: {self._state.max_attempts - self._state.attempts_used}")

        return InvoiceObservation(
            done=False,
            reward=0.0,
            text="\n".join(lines),
            task_name=self._state.task_name,
            current_score=self._state.best_score,
            attempts_remaining=self._state.max_attempts - self._state.attempts_used,
            required_fields=self._required_fields,
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
