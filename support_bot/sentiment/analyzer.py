"""Model-based multi-label sentiment analyzer interface and implementations."""

from __future__ import annotations

import abc
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent.parent / '.env'
    if _env.exists():
        load_dotenv(_env)
    else:
        load_dotenv()
except Exception:
    pass

from support_bot.sentiment.calibration import SentimentCalibrator

# Ordered label set used everywhere
LABELS = ["positive", "neutral", "negative", "frustrated", "urgent", "sarcastic"]

# Risk categories that can trigger immediate escalation
RISK_CATEGORIES = [
    "account_compromise",
    "duplicate_payment",
    "legal_threat",
    "data_breach",
    "general_risk",
]


@dataclass
class SentimentResult:
    """Structured sentiment analysis result for a single message."""

    # Raw scores from model (0-1 per label)
    raw_scores: dict[str, float] = field(default_factory=dict)
    # Platt-scaled calibrated scores
    calibrated_scores: dict[str, float] = field(default_factory=dict)
    # Single dominant label (highest calibrated score)
    dominant_label: str = "neutral"
    # Risk category detected (or None)
    risk_category: str | None = None
    risk_score: float = 0.0
    # Tone instruction for reply generation
    tone_instruction: str = ""
    # Was confidence too low to act on?
    low_confidence: bool = False


class SentimentAnalyzer(abc.ABC):
    """Abstract interface for sentiment analysis — inject in tests or production."""

    @abc.abstractmethod
    def analyze(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> SentimentResult:
        """Return multi-label sentiment result for a message."""

    def analyze_conversation(
        self,
        messages: list[str],
        config: dict[str, Any] | None = None,
    ) -> SentimentResult:
        """Aggregate sentiment over recent messages (last N from config)."""
        cfg = config or {}
        n = cfg.get("sentiment", {}).get("recent_messages_count", 5)
        recent = messages[-n:]
        if not recent:
            return SentimentResult()

        # Analyze each, then average calibrated scores
        results = [self.analyze(m, config=config) for m in recent]
        avg: dict[str, float] = {}
        for label in LABELS:
            avg[label] = sum(r.calibrated_scores.get(label, 0.0) for r in results) / len(results)

        dominant = max(avg, key=lambda k: avg[k])
        # Propagate highest risk from any message
        max_risk = max(results, key=lambda r: r.risk_score)
        return SentimentResult(
            raw_scores=avg,
            calibrated_scores=avg,
            dominant_label=dominant,
            risk_category=max_risk.risk_category,
            risk_score=max_risk.risk_score,
            tone_instruction=results[-1].tone_instruction,
            low_confidence=all(r.low_confidence for r in results),
        )


def _tone_from_dominant(dominant: str) -> str:
    """Map dominant sentiment label to a tone instruction for the LLM."""
    if dominant in ("frustrated", "negative"):
        return "empathetic"
    if dominant == "urgent":
        return "concise"
    return "neutral"


class GeminiSentimentAnalyzer(SentimentAnalyzer):
    """Production sentiment analyzer using Google Gemini structured JSON output."""

    def __init__(self, model_name: str = "gemini-3.8-flash", api_key: str | None = None):
        from support_bot.llm import make_gemini_llm

        self.llm = make_gemini_llm(
            temperature=0.0,
            timeout=30.0,
            api_key=api_key,
            model_name=model_name,
        )

    def analyze(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> SentimentResult:
        cfg = config or {}
        calibrator = SentimentCalibrator.from_config(cfg)
        sarcasm_threshold = cfg.get("sentiment", {}).get("sarcasm_threshold", 0.60)
        low_conf_threshold = cfg.get("sentiment", {}).get("low_confidence_threshold", 0.45)
        ctx_summary = context.get("summary", "") if context else ""

        prompt = f"""You are a multilingual customer service sentiment and risk analyzer.
Analyze this customer message in any language (English, Hindi, Tamil, Kannada, Hinglish, etc.):
Message: "{text}"
Conversation context: "{ctx_summary}"

CRITICAL INSTRUCTIONS FOR RISK EVALUATION:
The risk_category and risk_score MUST reflect the presence and severity of the risk event itself, completely INDEPENDENT OF EMOTIONAL TONE or volume. Even if the customer is calm, polite, or matter-of-fact, a genuine risk event must receive a significant risk_score:

- "duplicate_payment": Customer was debited twice, charged multiple times, paid double fee, or shows evidence of duplicate billing.
  * Calm example: "I think I've been charged twice for the enrollment. Please check." (risk_score: 0.50 to 0.85)
  * Loud example: "You stole my money two times! Refund immediately!" (risk_score: 0.90 to 0.99)

- "account_compromise": Unauthorized login, password change by unknown party, hacked account, or security lockout.
  * Calm example: "I noticed a login from an unfamiliar IP and my profile email was changed." (risk_score: 0.75 to 0.90)
  * Loud example: "Someone hacked my account right now! Help!" (risk_score: 0.95 to 1.0)

- "legal_threat": Explicit or implicit reference to legal counsel, consumer court, lawyer, litigation, police complaint, or regulatory body.
  * Calm example: "If this cannot be resolved, I will need to consult my legal advisor or consumer forum." (risk_score: 0.70 to 0.85)
  * Loud example: "I am suing you in high court tomorrow!" (risk_score: 0.95 to 1.0)

- "data_breach": Leak or unauthorized disclosure of sensitive personal, financial, or student data.
  * Calm example: "I noticed other students' personal records visible on my portal." (risk_score: 0.80 to 0.95)

- "general_risk": Other substantial business or safety hazards.

If no specific risk is mentioned, risk_category must be null and risk_score 0.0.
A calm question about general fee structure or standard refund policy without duplicate charge is NOT high risk (risk_category: null, risk_score: 0.0).

SENTIMENT SCORES:
All sentiment scores (positive, neutral, negative, frustrated, urgent, sarcastic) must be between 0.0 and 1.0.
A sarcastic message that appears positive but is actually criticizing service must have high sarcastic and negative scores.

Return ONLY a valid JSON object:
{{
  "positive": 0.0,
  "neutral": 0.0,
  "negative": 0.0,
  "frustrated": 0.0,
  "urgent": 0.0,
  "sarcastic": 0.0,
  "risk_category": null or one of ["account_compromise","duplicate_payment","legal_threat","data_breach","general_risk"],
  "risk_score": 0.0
}}
"""
        try:
            from support_bot.llm import llm_text

            raw = llm_text(self.llm.invoke(prompt))
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-zA-Z]*\n", "", clean)
                clean = re.sub(r"\n```$", "", clean)
            data = json.loads(clean.strip())

            raw_scores = {label: float(data.get(label, 0.0)) for label in LABELS}
            calibrated = calibrator.calibrate_scores(raw_scores)

            # Sarcasm guard: if sarcastic is high, suppress positive
            if calibrated.get("sarcastic", 0.0) >= sarcasm_threshold:
                calibrated["positive"] = min(calibrated.get("positive", 0.0), 0.3)

            dominant = max(calibrated, key=lambda k: calibrated[k])
            risk_cat = data.get("risk_category")
            if risk_cat not in RISK_CATEGORIES:
                risk_cat = None
            risk_score = float(data.get("risk_score", 0.0))
            low_conf = max(calibrated.values(), default=0.0) < low_conf_threshold

            return SentimentResult(
                raw_scores=raw_scores,
                calibrated_scores=calibrated,
                dominant_label=dominant,
                risk_category=risk_cat,
                risk_score=risk_score,
                tone_instruction=_tone_from_dominant(dominant),
                low_confidence=low_conf,
            )
        except Exception:
            # A temporary provider outage must not turn every message into an
            # unusable all-zero result.  Keep the support flow functional with
            # the same deterministic classifier used by offline tests; it can
            # still identify common high-risk messages such as duplicate
            # charges and account compromise.
            return FakeSentimentAnalyzer().analyze(text, context=context, config=cfg)


class FakeSentimentAnalyzer(SentimentAnalyzer):
    """Deterministic sentiment analyzer for unit tests — no network calls."""

    def __init__(
        self,
        default_result: SentimentResult | None = None,
        custom_fn: Callable[[str], SentimentResult] | None = None,
    ):
        self.default_result = default_result
        self.custom_fn = custom_fn
        self.calls: list[str] = []

    def analyze(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> SentimentResult:
        self.calls.append(text)
        if self.custom_fn:
            return self.custom_fn(text)
        if self.default_result:
            return self.default_result

        # Heuristic-based synthetic analysis
        return self._heuristic(text, config or {})

    def _heuristic(self, text: str, config: dict) -> SentimentResult:
        low_conf_threshold = config.get("sentiment", {}).get("low_confidence_threshold", 0.45)
        sarcasm_threshold = config.get("sentiment", {}).get("sarcasm_threshold", 0.60)

        text_lower = text.lower()
        raw: dict[str, float] = {l: 0.0 for l in LABELS}
        risk_cat: str | None = None
        risk_score = 0.0

        # Simple keyword-based scoring (for tests only)
        if any(w in text_lower for w in ["angry", "furious", "terrible", "worst", "disgusting"]):
            raw["frustrated"] = 0.85
            raw["negative"] = 0.80
        elif any(w in text_lower for w in ["frustrated", "annoyed", "upset", "disappointed"]):
            raw["frustrated"] = 0.75
            raw["negative"] = 0.65
        elif any(w in text_lower for w in ["urgent", "asap", "immediately", "emergency"]):
            raw["urgent"] = 0.80
            raw["negative"] = 0.50
        elif any(w in text_lower for w in ["great", "excellent", "happy", "thank"]):
            raw["positive"] = 0.85
        elif any(w in text_lower for w in ["sue", "lawyer", "legal", "court"]):
            raw["negative"] = 0.75
            risk_cat = "legal_threat"
            risk_score = 0.85
        elif any(w in text_lower for w in ["account hacked", "hacked", "stolen", "breach"]):
            raw["urgent"] = 0.80
            raw["negative"] = 0.70
            risk_cat = "account_compromise"
            risk_score = 0.90
        elif any(w in text_lower for w in ["charged twice", "duplicate", "double charge"]):
            raw["frustrated"] = 0.70
            raw["negative"] = 0.60
            risk_cat = "duplicate_payment"
            risk_score = 0.80
        elif any(w in text_lower for w in ["oh sure", "yeah right", "fantastic service"]) and any(
            w in text_lower for w in ["wait", "still", "never", "always"]
        ):
            raw["sarcastic"] = 0.80
            raw["negative"] = 0.70
            raw["frustrated"] = 0.50
        else:
            raw["neutral"] = 0.80

        # Sarcasm guard
        if raw.get("sarcastic", 0.0) >= sarcasm_threshold:
            raw["positive"] = min(raw.get("positive", 0.0), 0.3)

        # Use raw scores directly as calibrated (test doesn't need Platt)
        calibrated = {l: round(v, 4) for l, v in raw.items()}
        dominant = max(calibrated, key=lambda k: calibrated[k])
        low_conf = max(calibrated.values(), default=0.0) < low_conf_threshold

        return SentimentResult(
            raw_scores=raw,
            calibrated_scores=calibrated,
            dominant_label=dominant,
            risk_category=risk_cat,
            risk_score=risk_score,
            tone_instruction=_tone_from_dominant(dominant),
            low_confidence=low_conf,
        )
