"""PII and payment-data masking — reusable function and logging filter."""

from __future__ import annotations

import logging
import re

# ── Patterns ────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Email
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    # Phone numbers — Indian 10-digit with optional +91 / 0 prefix
    ("phone", re.compile(r"(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}"), "[PHONE]"),
    # Credit/debit card numbers (13-19 digits, optionally grouped by - or spaces)
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    # Indian bank account (9-18 digits)
    ("bank_account", re.compile(r"\b\d{9,18}\b(?=.*IFSC|.*ifsc)?"), None),  # handled below
    # IFSC code
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[IFSC]"),
    # UPI ID
    ("upi", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z]{2,}\b"), None),  # overlap with email handled by order
    # API keys (generic long alphanum tokens)
    ("api_key", re.compile(r"\b(?:AIza|sk-|ghp_|gho_|AKIA)[A-Za-z0-9_-]{16,}\b"), "[API_KEY]"),
]

# Bank account pattern needs context — only mask if "account" appears nearby
_BANK_ACCT_CONTEXT = re.compile(
    r"(?:account|a/c|acct)[\s#.:]*(\d{9,18})", re.IGNORECASE
)

# Refined UPI pattern (user@provider, not regular email)
_UPI_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@(?:upi|paytm|oksbi|okaxis|okicici|ybl|ibl|axl|apl)\b",
    re.IGNORECASE,
)


def mask_pii(text: str) -> str:
    """Replace PII and payment data in *text* with placeholder tokens."""
    if not text:
        return text

    # Order matters:
    # 1. API keys (high entropy, prefixes like sk-, AIza)
    text = _PATTERNS[6][1].sub("[API_KEY]", text)
    # 2. IFSC codes (alphanumeric pattern like HDFC0001234)
    text = _PATTERNS[4][1].sub("[IFSC]", text)
    # 3. Bank account with context (e.g. Account: 1234567890) before generic cards
    text = _BANK_ACCT_CONTEXT.sub(lambda m: m.group(0).replace(m.group(1), "[BANK_ACCT]"), text)
    # 4. Card numbers (13-19 digits)
    text = _PATTERNS[2][1].sub("[CARD]", text)
    # 5. UPI IDs (before email, since UPI handles @upi, @okaxis etc.)
    text = _UPI_PATTERN.sub("[UPI]", text)
    # 6. Email addresses
    text = _PATTERNS[0][1].sub("[EMAIL]", text)
    # 7. Phone numbers
    text = _PATTERNS[1][1].sub("[PHONE]", text)

    return text


class MaskingFilter(logging.Filter):
    """Logging filter that masks PII in log records before they are emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_pii(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: mask_pii(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    mask_pii(str(a)) if isinstance(a, str) else a for a in record.args
                )
        return True
