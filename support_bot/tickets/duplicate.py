"""Duplicate ticket detection and related issue grouping engine."""

import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


def _text_similarity(t1: str, t2: str) -> float:
    """Compute word token Jaccard similarity between two issue descriptions."""
    _STOP_WORDS = {"the", "a", "an", "is", "for", "in", "to", "my", "i", "and", "or", "on", "it", "with", "this", "please"}
    w1 = set(re.findall(r"\b\w+\b", t1.lower())) - _STOP_WORDS
    w2 = set(re.findall(r"\b\w+\b", t2.lower())) - _STOP_WORDS
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def check_duplicate_or_related_tickets(
    conn: sqlite3.Connection,
    customer_name: str,
    order_or_enrollment_id: str,
    product_or_course: str,
    description: str,
    similarity_threshold: float = 0.5,
) -> Tuple[Optional[str], List[str]]:
    """Check for duplicate or related tickets in SQLite.
    
    Returns (duplicate_of_ticket_id, list_of_related_ticket_ids).
    - If customer, order/enrollment ID, product match AND description similarity >= threshold:
      returns (original_ticket_id, [])
    - If customer, order/enrollment ID, product match BUT description similarity < threshold:
      returns (None, [related_ticket_id, ...]) -> Group related issues
    - Otherwise: (None, []) -> Separate independent ticket
    """
    rows = conn.execute(
        "SELECT id, description, status FROM tickets "
        "WHERE order_or_enrollment_id = ? "
        "AND LOWER(product_or_course) = LOWER(?) "
        "AND duplicate_of_ticket_id IS NULL "
        "AND status NOT IN ('resolved', 'closed')",
        (order_or_enrollment_id, product_or_course),
    ).fetchall()

    if not rows:
        return None, []

    related_ids = []
    for r in rows:
        tid = r["id"]
        existing_desc = r["description"]
        sim = _text_similarity(description, existing_desc)
        if sim >= similarity_threshold:
            # Found duplicate
            return tid, []
        else:
            # Different issue for same customer & order -> related issue
            related_ids.append(tid)

    return None, related_ids
