"""
Grading logic for the Invoice Extraction Environment.

Provides field-level scoring with fuzzy matching for text fields
and exact matching for numeric/date fields. All scores are in [0.0, 1.0].
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove common punctuation variations
    text = text.replace(".", "").replace(",", "").replace("'", "").replace('"', "")
    return text


def normalize_number(value: Any) -> Optional[float]:
    """Normalize a numeric value: strip currency symbols, parse to float."""
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        # Remove currency symbols, commas, whitespace
        cleaned = re.sub(r"[$ ,]", "", value.strip())
        try:
            return round(float(cleaned), 2)
        except (ValueError, TypeError):
            return None
    return None


def normalize_date(date_str: str) -> Optional[str]:
    """Normalize date to YYYY-MM-DD format."""
    if not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    # Already in YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # MM/DD/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_str)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    # DD-Mon-YYYY or Mon DD, YYYY etc - try common patterns
    month_map = {
        "jan": "01", "january": "01", "feb": "02", "february": "02",
        "mar": "03", "march": "03", "apr": "04", "april": "04",
        "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
        "aug": "08", "august": "08", "sep": "09", "september": "09",
        "oct": "10", "october": "10", "nov": "11", "november": "11",
        "dec": "12", "december": "12",
    }

    # "January 15, 2024" or "Jan 15 2024"
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s*'?(\d{2,4})$", date_str, re.IGNORECASE)
    if m:
        month = month_map.get(m.group(1).lower())
        if month:
            year = m.group(3)
            if len(year) == 2:
                year = "20" + year
            return f"{year}-{month}-{int(m.group(2)):02d}"

    # "15-Feb-2024" or "20-Feb-2024"
    m = re.match(r"(\d{1,2})-(\w+)-(\d{4})$", date_str, re.IGNORECASE)
    if m:
        month = month_map.get(m.group(2).lower())
        if month:
            return f"{m.group(3)}-{month}-{int(m.group(1)):02d}"

    return date_str  # Return as-is if no pattern matches


def grade_text(actual: Any, expected: Any) -> float:
    """Grade a text field using fuzzy matching. Returns 0.0-1.0."""
    if actual is None or expected is None:
        return 0.0 if actual != expected else 1.0

    norm_actual = normalize_text(str(actual))
    norm_expected = normalize_text(str(expected))

    if norm_actual == norm_expected:
        return 1.0

    # Use SequenceMatcher for fuzzy comparison
    ratio = SequenceMatcher(None, norm_actual, norm_expected).ratio()

    # Apply a threshold: below 0.4 similarity = 0 score
    if ratio < 0.4:
        return 0.0

    return round(ratio, 4)


def grade_numeric(actual: Any, expected: Any) -> float:
    """Grade a numeric field. Returns 1.0 for exact match, partial for close."""
    norm_actual = normalize_number(actual)
    norm_expected = normalize_number(expected)

    if norm_actual is None or norm_expected is None:
        return 0.0

    if norm_actual == norm_expected:
        return 1.0

    # Partial credit for being close (within 5%)
    if norm_expected != 0:
        error_pct = abs(norm_actual - norm_expected) / abs(norm_expected)
        if error_pct <= 0.01:
            return 0.9  # Very close
        elif error_pct <= 0.05:
            return 0.5  # Somewhat close
        elif error_pct <= 0.10:
            return 0.2  # In the ballpark

    return 0.0


def grade_date(actual: Any, expected: Any) -> float:
    """Grade a date field after normalization. Returns 0.0 or 1.0."""
    if actual is None:
        return 0.0

    norm_actual = normalize_date(str(actual))
    norm_expected = normalize_date(str(expected))

    if norm_actual == norm_expected:
        return 1.0

    # Partial credit for getting the right date with wrong format
    if norm_actual and norm_expected:
        # Remove separators and compare
        a = re.sub(r"[^0-9]", "", norm_actual)
        e = re.sub(r"[^0-9]", "", norm_expected)
        if a == e:
            return 0.8

    return 0.0


def grade_line_items(actual: Any, expected: Any) -> float:
    """Grade line items extraction. Checks description, qty, price, amount."""
    if not isinstance(actual, list) or not isinstance(expected, list):
        return 0.0

    if len(actual) == 0:
        return 0.0

    total_score = 0.0
    matched_expected = set()

    for act_item in actual:
        if not isinstance(act_item, dict):
            continue

        best_score = 0.0
        best_idx = -1

        for idx, exp_item in enumerate(expected):
            if idx in matched_expected:
                continue
            if not isinstance(exp_item, dict):
                continue

            # Score each field of the line item
            desc_score = grade_text(
                act_item.get("description", ""),
                exp_item.get("description", ""),
            )
            qty_score = grade_numeric(
                act_item.get("quantity"),
                exp_item.get("quantity"),
            )
            price_score = grade_numeric(
                act_item.get("unit_price"),
                exp_item.get("unit_price"),
            )
            amt_score = grade_numeric(
                act_item.get("amount"),
                exp_item.get("amount"),
            )

            item_score = (desc_score * 0.3 + qty_score * 0.2 +
                          price_score * 0.2 + amt_score * 0.3)

            if item_score > best_score:
                best_score = item_score
                best_idx = idx

        if best_idx >= 0:
            matched_expected.add(best_idx)
            total_score += best_score

    # Normalize by expected count, penalize missing/extra items
    expected_count = len(expected)
    if expected_count == 0:
        return 1.0 if len(actual) == 0 else 0.0

    # Score = matched items score / expected count
    # Penalize for extra items (max penalty = 0.2)
    extra_penalty = max(0, len(actual) - expected_count) * 0.05
    extra_penalty = min(extra_penalty, 0.2)

    score = (total_score / expected_count) - extra_penalty
    return max(0.0, min(1.0, round(score, 4)))


def grade_extraction(
    extracted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    required_fields: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """Grade the full extraction against ground truth.

    Args:
        extracted: The agent's extracted fields
        ground_truth: The correct field values
        required_fields: List of field names to grade

    Returns:
        Tuple of (overall_score, field_feedback)
        overall_score is in [0.0, 1.0]
        field_feedback maps field names to {score, expected, actual}
    """
    field_scores = {}
    feedback = {}

    numeric_fields = {"total", "subtotal", "tax", "adjusted_total"}
    date_fields = {"date", "due_date"}
    list_fields = {"line_items"}

    for field in required_fields:
        expected = ground_truth.get(field)
        actual = extracted.get(field)

        if field in list_fields:
            score = grade_line_items(actual, expected)
        elif field in numeric_fields:
            score = grade_numeric(actual, expected)
        elif field in date_fields:
            score = grade_date(actual, expected)
        else:
            score = grade_text(actual, expected)

        field_scores[field] = score
        feedback[field] = {
            "score": score,
            "expected_type": "list" if field in list_fields else
                            "number" if field in numeric_fields else
                            "date" if field in date_fields else "text",
            "matched": score >= 0.8,
        }

    # Overall score = weighted average
    if not field_scores:
        return 0.0, feedback

    overall = sum(field_scores.values()) / len(field_scores)
    overall = round(max(0.0, min(1.0, overall)), 4)

    return overall, feedback
