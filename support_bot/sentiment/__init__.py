"""Sentiment analysis package — model-based, multi-label, calibrated."""

from support_bot.sentiment.analyzer import (
    FakeSentimentAnalyzer,
    GeminiSentimentAnalyzer,
    SentimentAnalyzer,
    SentimentResult,
)
from support_bot.sentiment.calibration import SentimentCalibrator

__all__ = [
    "SentimentAnalyzer",
    "GeminiSentimentAnalyzer",
    "FakeSentimentAnalyzer",
    "SentimentResult",
    "SentimentCalibrator",
]
