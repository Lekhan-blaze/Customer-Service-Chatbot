"""Mismatch detection between OCR-extracted data and user-stated data."""

import re
from typing import Any, Dict, List, Optional

from support_bot.files.models import ConflictReport, OcrExtractedFields


def _normalize_amount(s: str) -> Optional[float]:
    """Parse amount string to float, ignoring currency symbols and commas."""
    if not s:
        return None
    cleaned = re.sub(r"[^\d.]", "", s)
    try:
        return float(cleaned)
    except ValueError:
        return None


def detect_conflicts(
    ocr_fields: OcrExtractedFields,
    user_claims: Dict[str, Any],
    confidence_threshold: float = 0.5,
) -> List[ConflictReport]:
    """Compare OCR-extracted fields vs user-stated claims.
    
    Returns list of ConflictReport for mismatches or low-confidence extractions.
    Comparison rules:
    - order_id: string equality (case-insensitive, ignoring dashes/spaces)
    - amount: numeric comparison; flag if >10% difference
    - low confidence < threshold: request clarification on all extracted fields
    """
    conflicts: List[ConflictReport] = []

    # Low confidence overall -> request clarification on everything extracted
    if ocr_fields.confidence < confidence_threshold:
        conflicts.append(ConflictReport(
            field="confidence",
            user_stated="",
            ocr_extracted=f"overall confidence={ocr_fields.confidence:.2f}",
            severity="medium",
            clarification_prompt=(
                "The uploaded document is difficult to read clearly. "
                "Could you confirm your order ID, amount charged, and transaction date?"
            ),
        ))
        return conflicts

    # Order ID mismatch
    if ocr_fields.order_id and "order_id" in user_claims:
        u_id = re.sub(r"[\s\-]", "", str(user_claims["order_id"])).upper()
        o_id = re.sub(r"[\s\-]", "", str(ocr_fields.order_id)).upper()
        if u_id and o_id and u_id != o_id:
            conflicts.append(ConflictReport(
                field="order_id",
                user_stated=user_claims["order_id"],
                ocr_extracted=ocr_fields.order_id,
                severity="high",
                clarification_prompt=(
                    f"The order ID in your document ({ocr_fields.order_id}) doesn't match "
                    f"the order ID you mentioned ({user_claims['order_id']}). "
                    "Could you confirm which is correct?"
                ),
            ))

    # Amount mismatch (> 10% difference)
    if ocr_fields.amount and "amount" in user_claims:
        u_amt = _normalize_amount(str(user_claims["amount"]))
        o_amt = _normalize_amount(str(ocr_fields.amount))
        if u_amt is not None and o_amt is not None and u_amt > 0:
            diff_ratio = abs(u_amt - o_amt) / u_amt
            if diff_ratio > 0.10:
                conflicts.append(ConflictReport(
                    field="amount",
                    user_stated=str(user_claims["amount"]),
                    ocr_extracted=ocr_fields.amount,
                    severity="high" if diff_ratio > 0.30 else "medium",
                    clarification_prompt=(
                        f"The amount in your document (₹{ocr_fields.amount}) differs from what you mentioned "
                        f"(₹{user_claims['amount']}). Could you confirm the charged amount?"
                    ),
                ))

    return conflicts
