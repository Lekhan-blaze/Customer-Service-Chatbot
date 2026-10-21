"""Confidence calibration utilities for sentiment scoring (Platt scaling and ECE)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def platt_scale(raw_score: float, a: float, b: float) -> float:
    """Apply 1D Platt scaling sigmoid: 1 / (1 + exp(-(a * score + b)))."""
    try:
        logit = -(a * raw_score + b)
        # Avoid math overflow
        if logit > 30.0:
            return 0.0
        if logit < -30.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(logit))
    except (OverflowError, ValueError):
        return max(0.0, min(1.0, raw_score))


def compute_ece(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across *n_bins* equally spaced bins."""
    if not y_true or not y_prob or len(y_true) != len(y_prob):
        return 0.0

    n = len(y_true)
    ece = 0.0
    bin_size = 1.0 / n_bins

    for i in range(n_bins):
        bin_lower = i * bin_size
        bin_upper = (i + 1) * bin_size
        # Find points in this bin
        bin_indices = [
            idx
            for idx, p in enumerate(y_prob)
            if (bin_lower <= p < bin_upper) or (i == n_bins - 1 and p == bin_upper)
        ]
        if not bin_indices:
            continue

        bin_true = [y_true[idx] for idx in bin_indices]
        bin_probs = [y_prob[idx] for idx in bin_indices]

        acc = sum(bin_true) / len(bin_true)
        conf = sum(bin_probs) / len(bin_probs)
        weight = len(bin_indices) / n

        ece += weight * abs(acc - conf)

    return ece


@dataclass
class SentimentCalibrator:
    """Applies calibration parameters to multi-label sentiment predictions."""

    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> SentimentCalibrator:
        """Initialize from runtime configuration dict."""
        cfg = config or {}
        calib_cfg = cfg.get("sentiment", {}).get("calibration", {})
        enabled = calib_cfg.get("enabled", True)
        weights = calib_cfg.get(
            "weights",
            {
                "positive": {"a": 2.1, "b": -0.8},
                "neutral": {"a": 1.8, "b": -0.4},
                "negative": {"a": 2.2, "b": -0.7},
                "frustrated": {"a": 2.3, "b": -0.9},
                "urgent": {"a": 2.4, "b": -1.0},
                "sarcastic": {"a": 2.5, "b": -1.1},
            },
        )
        return cls(weights=weights, enabled=enabled)

    def calibrate(self, label: str, raw_score: float) -> float:
        """Calibrate confidence for a single label."""
        if not self.enabled or label not in self.weights:
            return round(max(0.0, min(1.0, raw_score)), 4)
        params = self.weights[label]
        a = params.get("a", 1.0)
        b = params.get("b", 0.0)
        return round(platt_scale(raw_score, a, b), 4)

    def calibrate_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Calibrate a full dictionary of multi-label sentiment scores."""
        return {
            label: self.calibrate(label, score)
            for label, score in scores.items()
        }
