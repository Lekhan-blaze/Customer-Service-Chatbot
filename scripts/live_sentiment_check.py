#!/usr/bin/env python3
"""Live sentiment check — runs GeminiSentimentAnalyzer on 8 sample messages.

Never prints the API key.
Usage: PYTHONPATH=. python scripts/live_sentiment_check.py
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()
except Exception:
    pass


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import sys
import json

# Ensure PYTHONPATH includes project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support_bot.sentiment.analyzer import GeminiSentimentAnalyzer

# 8 sample messages — one per required category
SAMPLES = [
    {
        "label": "sarcastic",
        "text": "Oh fantastic, my course is still locked after paying. Great service as always!",
    },
    {
        "label": "calm_high_risk (duplicate payment)",
        "text": "I think I've been charged twice for the Data Science course enrollment. Please check.",
    },
    {
        "label": "urgent",
        "text": "My account was hacked and I can't log in! Please help immediately!",
    },
    {
        "label": "Hindi",
        "text": "मुझे इस कोर्स की फीस के बारे में जानकारी चाहिए। क्या आप बता सकते हैं?",
    },
    {
        "label": "Tamil",
        "text": "இந்த கோர்ஸ் எடுக்க என்ன தகுதி வேண்டும்? கட்டணம் எவ்வளவு?",
    },
    {
        "label": "Kannada",
        "text": "ಈ ಕೋರ್ಸ್‌ಗೆ ಸೇರಲು ಯಾವ ಅರ್ಹತೆ ಬೇಕು? ದಯವಿಟ್ಟು ತಿಳಿಸಿ.",
    },
    {
        "label": "Hinglish",
        "text": "Bhai mujhe pata nahi tha ki fees itni zyada hogi. Refund milega kya?",
    },
    {
        "label": "neutral",
        "text": "What are the prerequisites for the Python programming course?",
    },
]

CONFIG = {
    "sentiment": {
        "sarcasm_threshold": 0.60,
        "low_confidence_threshold": 0.45,
        "calibration": {"enabled": True},
    }
}


def main():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    print("API key: [REDACTED]")
    print(f"Model: gemini-3.8-flash\n")
    print("=" * 70)

    analyzer = GeminiSentimentAnalyzer()
    results = []

    for sample in SAMPLES:
        label = sample["label"]
        text = sample["text"]
        print(f"\n[{label}]")
        print(f"  Input: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            result = analyzer.analyze(text, config=CONFIG)
            scores_str = ", ".join(
                f"{k}={v:.2f}" for k, v in sorted(result.calibrated_scores.items(), key=lambda x: -x[1])
            )
            print(f"  Dominant: {result.dominant_label}")
            print(f"  Scores:   {scores_str}")
            if result.risk_category:
                print(f"  Risk:     {result.risk_category} ({result.risk_score:.2f})")
            print(f"  Tone:     {result.tone_instruction}")
            print(f"  Low-conf: {result.low_confidence}")
            results.append({"label": label, "dominant": result.dominant_label, "ok": True})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"label": label, "dominant": None, "ok": False, "error": str(e)})

    print("\n" + "=" * 70)
    print("Summary:")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        dom = r.get("dominant", "N/A")
        print(f"  [{status}] {r['label']}: dominant={dom}")


if __name__ == "__main__":
    main()
