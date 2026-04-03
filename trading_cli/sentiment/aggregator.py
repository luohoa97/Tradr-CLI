"""Aggregate FinBERT per-headline results into a single symbol-level score.

Supports event-type weighting (earnings/executive/product/macro/generic)
and temporal decay (newer headlines have more impact).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from trading_cli.sentiment.news_classifier import EventType, EventClassification, DEFAULT_WEIGHTS

LABEL_DIRECTION = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def aggregate_scores(results: list[dict]) -> float:
    """
    Weighted average of label directions, weighted by confidence score.

    Returns float in [-1.0, +1.0]:
      +1.0 = all headlines strongly positive
      -1.0 = all headlines strongly negative
       0.0 = neutral or empty
    """
    if not results:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for r in results:
        label = r.get("label", "neutral")
        score = float(r.get("score", 0.5))
        direction = LABEL_DIRECTION.get(label, 0.0)
        weighted_sum += direction * score
        total_weight += score
    if total_weight == 0.0:
        return 0.0
    return max(-1.0, min(1.0, weighted_sum / total_weight))


def aggregate_scores_weighted(
    results: list[dict],
    classifications: list[EventClassification] | None = None,
    timestamps: list[float] | None = None,
    event_weights: dict[EventType, float] | None = None,
    half_life_hours: float = 24.0,
) -> float:
    """
    Weighted sentiment aggregation with event-type and temporal decay.

    Args:
        results: List of FinBERT results with "label" and "score" keys.
        classifications: Optional event classifications for each headline.
        timestamps: Optional Unix timestamps for each headline (for temporal decay).
        event_weights: Custom event type weight multipliers.
        half_life_hours: Hours for temporal half-life decay. Default 24h.

    Returns float in [-1.0, +1.0].
    """
    if not results:
        return 0.0

    now = time.time()
    total_weight = 0.0
    weighted_sum = 0.0
    weights = event_weights or DEFAULT_WEIGHTS

    for i, r in enumerate(results):
        label = r.get("label", "neutral")
        score = float(r.get("score", 0.5))
        direction = LABEL_DIRECTION.get(label, 0.0)

        # Base weight from FinBERT confidence
        w = score

        # Event type weight multiplier
        if classifications and i < len(classifications):
            ec = classifications[i]
            w *= weights.get(ec.event_type, 1.0)

        # Temporal decay: newer headlines weight more
        if timestamps and i < len(timestamps):
            ts = timestamps[i]
            age_hours = (now - ts) / 3600.0
            # Exponential decay: weight halves every half_life_hours
            decay = 0.5 ** (age_hours / half_life_hours)
            w *= decay

        weighted_sum += direction * w
        total_weight += w

    if total_weight == 0.0:
        return 0.0
    return max(-1.0, min(1.0, weighted_sum / total_weight))
 
 
def get_sentiment_summary(results: list[dict]) -> dict:
    """Return counts, dominant label, and aggregate score."""
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        label = r.get("label", "neutral")
        if label in counts:
            counts[label] += 1
    dominant = max(counts, key=lambda k: counts[k]) if results else "neutral"
    return {
        "score": aggregate_scores(results),
        "positive_count": counts["positive"],
        "negative_count": counts["negative"],
        "neutral_count": counts["neutral"],
        "total": len(results),
        "dominant": dominant,
    }
 
 
def score_to_bar(score: float, width: int = 20) -> str:
    """Render a text gauge like:  ──────●──────────  for display in terminals."""
    clamped = max(-1.0, min(1.0, score))
    mid = width // 2
    pos = int(mid + clamped * mid)
    pos = max(0, min(width - 1, pos))
    bar = list("─" * width)
    bar[mid] = "┼"
    bar[pos] = "●"
    return "".join(bar)
