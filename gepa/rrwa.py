"""
RRWA: Rank-Reciprocal Weighted Average.

Combines several repeated judge scores for the same translation into
one stable number: remove anything more than two standard deviations
from the mean, then weight the rest by rank so the most consistent
scores count for more. This is pure statistics, language agnostic by
construction, no per-language config needed here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class RRWAResult:
    final_score: float
    mean: float
    stability_index: float  # 1.0 = perfect agreement, lower = more disagreement
    raw_scores: list[float]
    kept_scores: list[float]
    discarded_scores: list[float]


def compute_rrwa(scores: list[float]) -> RRWAResult:
    if not scores:
        raise ValueError("Cannot aggregate an empty list of scores.")
    if len(scores) == 1:
        return RRWAResult(
            final_score=scores[0],
            mean=scores[0],
            stability_index=1.0,
            raw_scores=scores,
            kept_scores=scores,
            discarded_scores=[],
        )

    mean = statistics.fmean(scores)
    std_dev = statistics.pstdev(scores)

    if std_dev == 0:
        kept, discarded = scores, []
    else:
        kept = [s for s in scores if abs(s - mean) <= 2 * std_dev]
        discarded = [s for s in scores if abs(s - mean) > 2 * std_dev]
        if not kept:  # every score was somehow an outlier of itself; keep all
            kept, discarded = scores, []

    ranked = sorted(kept, reverse=True)
    weights = [1.0 / rank for rank in range(1, len(ranked) + 1)]
    weighted_sum = sum(s * w for s, w in zip(ranked, weights))
    final_score = weighted_sum / sum(weights)

    # Stability index: how tightly the kept scores agree, scaled 0-1.
    # A max plausible spread of 10 points is assumed (typical 0-10 scale);
    # adjust if a different scale is used elsewhere.
    kept_std = statistics.pstdev(kept) if len(kept) > 1 else 0.0
    stability_index = max(0.0, 1 - (kept_std / 5))

    return RRWAResult(
        final_score=final_score,
        mean=mean,
        stability_index=stability_index,
        raw_scores=scores,
        kept_scores=kept,
        discarded_scores=discarded,
    )
