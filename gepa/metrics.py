"""
Reference-based companion metrics.

Every language config already declared `companion_metric: chrf++`, but
nothing computed it — the config was promising something the pipeline
didn't deliver. This module closes that gap.

chrF++ is deliberately a *companion*, not a replacement for the judge.
It compares character and word n-grams against a reference translation,
so it's cheap, deterministic, and carries no model's opinion — but it
also can't tell a good paraphrase from a bad one, and it can reward a
translation that shares surface characters while missing the meaning.
Its value is as an independent second signal: when the judge and chrF++
disagree sharply on the same example, that example is worth a look.
"""

from __future__ import annotations

from dataclasses import dataclass

from sacrebleu.metrics import CHRF

# word_order=2 is what makes this chrF++ rather than plain chrF.
_CHRF_PLUS_PLUS = CHRF(word_order=2)


@dataclass
class CompanionScore:
    metric: str
    score: float | None
    note: str = ""


def compute_chrf_plus_plus(hypothesis: str, reference: str) -> CompanionScore:
    """
    Score one translation against its reference, 0-100 (higher better).
    Returns a None score rather than raising when there's no usable
    reference, since plenty of real examples legitimately lack one.
    """
    if not reference or not reference.strip():
        return CompanionScore("chrf++", None, "no reference available for this example")
    if not hypothesis or not hypothesis.strip():
        return CompanionScore("chrf++", 0.0, "empty hypothesis")

    result = _CHRF_PLUS_PLUS.sentence_score(hypothesis.strip(), [reference.strip()])
    return CompanionScore("chrf++", round(result.score, 4))


def compute_companion(metric_name: str | None, hypothesis: str, reference: str) -> CompanionScore:
    """Dispatch on whatever a language's config asked for."""
    if not metric_name:
        return CompanionScore("none", None, "no companion metric configured")
    normalized = metric_name.strip().lower()
    if normalized in {"chrf++", "chrf_plus_plus", "chrfpp"}:
        return compute_chrf_plus_plus(hypothesis, reference)
    return CompanionScore(normalized, None, f"unsupported companion metric '{metric_name}'")
