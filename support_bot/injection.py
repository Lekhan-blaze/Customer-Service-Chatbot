"""Prompt-injection detector — returns a risk score and matched reasons."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Heuristic rules ─────────────────────────────────────────────────────

_RULES: list[tuple[str, re.Pattern, float]] = [
    (
        "system_prompt_override",
        re.compile(
            r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above)\s+"
            r"(?:instructions|prompts|rules|context)",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "role_hijack",
        re.compile(
            r"(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you(?:'re|\s+are)))\s+",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "delimiter_injection",
        re.compile(r"```|<\|.*?\|>|<<SYS>>|\[INST\]", re.IGNORECASE),
        0.6,
    ),
    (
        "data_exfil",
        re.compile(
            r"(?:print|reveal|show|output|display|repeat)\s+(?:(?:your|the|system|internal|all)\s+)*"
            r"(?:prompt|instructions|rules|secrets|password|api.?key)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "encoding_evasion",
        re.compile(r"(?:base64|rot13|hex|unicode)\s*(?:decode|encode)", re.IGNORECASE),
        0.5,
    ),
    (
        "jailbreak_keyword",
        re.compile(r"\b(?:DAN|DUDE|jailbreak|bypass|override)\b", re.IGNORECASE),
        0.7,
    ),
]


@dataclass
class InjectionResult:
    """Outcome of a prompt-injection scan."""

    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def is_risky(self) -> bool:
        return self.score >= self._threshold

    _threshold: float = field(default=0.6, repr=False)


def detect_injection(
    text: str,
    *,
    threshold: float = 0.6,
    extra_rules: list[tuple[str, re.Pattern, float]] | None = None,
) -> InjectionResult:
    """Scan *text* for prompt-injection patterns.

    Returns an :class:`InjectionResult` with the aggregate score (max of
    matched rule scores) and the list of matched rule names.
    """
    if not text:
        return InjectionResult(_threshold=threshold)

    rules = list(_RULES)
    if extra_rules:
        rules.extend(extra_rules)

    matched: list[str] = []
    max_score = 0.0
    for name, pattern, weight in rules:
        if pattern.search(text):
            matched.append(name)
            max_score = max(max_score, weight)

    return InjectionResult(score=max_score, reasons=matched, _threshold=threshold)


class PromptInjectionDetector:
    """Configurable OOP wrapper around detect_injection."""

    def __init__(self, config: dict | None = None, threshold: float = 0.6):
        self.config = config or {}
        self.threshold = self.config.get("injection", {}).get("threshold", threshold)

    def detect(self, text: str, source_type: str = "document"):
        r = detect_injection(text, threshold=self.threshold)
        class AdaptedResult:
            def __init__(self, result: InjectionResult, st: str):
                self.is_injection = result.is_risky
                self.risk_score = result.score
                self.matched_reasons = result.reasons
                self.source_type = st
        return AdaptedResult(r, source_type)
