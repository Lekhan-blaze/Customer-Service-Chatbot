"""Entity extraction, preservation, and restoration for multilingual conversations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Regex patterns for high-priority domain entities
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Explicit Order / Enrollment / Payment / Ticket IDs (must contain at least one digit)
    (
        "id",
        re.compile(
            r"\b(?:ORD|ENR|PAY|TICK|TXN|REF)[-_]?[A-Za-z0-9]*\d[A-Za-z0-9]*\b",
            re.IGNORECASE,
        ),
    ),
    # Contextual IDs (e.g. "order 4521", "order # 4512", "enrollment id 9081" - must contain digits)
    (
        "contextual_id",
        re.compile(
            r"\b(?:order|enrollment|payment|ticket|txn|acct)\s*(?:id|number|no|#|reference|ref)?\s*[:#]?\s*([A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)\b",
            re.IGNORECASE,
        ),
    ),
    # Course and product codes (e.g. PY-101, DS-202, ML-300, COURSE-42)
    (
        "product_code",
        re.compile(
            r"\b(?:PY|DS|ML|AI|CS|WEB|DEV|COURSE|PROD)[-_]\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    # Currency amounts (e.g. ₹5000, Rs. 2,500, $150, 4999 INR, 5000 rupees)
    (
        "currency",
        re.compile(
            r"(?:₹|Rs\.?|INR|\$|USD|€)\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:rupees|inr|dollars|usd|rs)\b",
            re.IGNORECASE,
        ),
    ),
    # Standard ISO and common dates (e.g. 2025-01-06, 15/08/2025, 15-08-2025, Jan 15, 2025)
    (
        "date",
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class ProtectedText:
    """Represents text with entities replaced by durable placeholders."""

    text: str
    entity_map: dict[str, str] = field(default_factory=dict)
    entities_by_type: dict[str, list[str]] = field(default_factory=dict)

    def restore(self, target_text: str) -> str:
        """Replace all placeholders in *target_text* with their original values."""
        if not target_text or not self.entity_map:
            return target_text

        result = target_text
        for placeholder, original in self.entity_map.items():
            # Match placeholder exactly or with relaxed spacing/casing
            clean_token = re.escape(placeholder)
            # Allow optional surrounding spaces inside the underscores e.g. __ ENT_ID_0 __
            relaxed = re.compile(
                clean_token.replace(r"\_", r"_?\s*"),
                re.IGNORECASE,
            )
            if placeholder in result:
                result = result.replace(placeholder, original)
            else:
                result = relaxed.sub(original, result)
        return result


def protect_entities(
    text: str,
    extra_entities: dict[str, str] | None = None,
) -> ProtectedText:
    """Extract known entity patterns and replace them with unique placeholders.

    Any entities passed in *extra_entities* (e.g. from the analyzer or session facts)
    are protected first.
    """
    if not text:
        return ProtectedText(text="", entity_map={}, entities_by_type={})

    entity_map: dict[str, str] = {}
    entities_by_type: dict[str, list[str]] = {}
    counter = 0
    protected = text

    # 1. Protect explicit extra entities (e.g. names or identified keys)
    if extra_entities:
        for ent_type, val in extra_entities.items():
            if val and val in protected:
                placeholder = f"__ENT_{ent_type.upper()}_{counter}__"
                protected = protected.replace(val, placeholder)
                entity_map[placeholder] = val
                entities_by_type.setdefault(ent_type, []).append(val)
                counter += 1

    # 2. Match pattern-based entities
    for ent_type, pattern in _ENTITY_PATTERNS:
        # Use finditer to find exact spans
        matches = list(pattern.finditer(protected))
        # Process in reverse order to preserve string indices
        for match in reversed(matches):
            raw_match = match.group(0)
            # Skip if already a placeholder
            if raw_match.startswith("__ENT_") and raw_match.endswith("__"):
                continue

            # For contextual IDs like "order 4521", we protect the specific ID value if group(1) exists
            if match.lastindex and match.lastindex >= 1:
                val = match.group(1)
                if "__ENT_" in val:
                    continue
                placeholder = f"__ENT_ID_{counter}__"
                # Replace only the ID within this contextual match
                start, end = match.span(1)
                protected = protected[:start] + placeholder + protected[end:]
                entity_map[placeholder] = val
                entities_by_type.setdefault("id", []).append(val)
                counter += 1
            else:
                val = raw_match
                if "__ENT_" in val:
                    continue
                placeholder = f"__ENT_{ent_type.upper()}_{counter}__"
                start, end = match.span(0)
                protected = protected[:start] + placeholder + protected[end:]
                entity_map[placeholder] = val
                entities_by_type.setdefault(ent_type, []).append(val)
                counter += 1

    return ProtectedText(
        text=protected,
        entity_map=entity_map,
        entities_by_type=entities_by_type,
    )


def restore_entities(text: str, entity_map: dict[str, str]) -> str:
    """Convenience helper to restore entity placeholders into a target string."""
    if not entity_map or not text:
        return text
    pt = ProtectedText(text="", entity_map=entity_map)
    return pt.restore(text)
