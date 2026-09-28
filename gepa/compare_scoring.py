"""
Score the same translations under both scoring methods and compare.

This exists so the choice between flat and severity-weighted scoring is
settled by measurement rather than by citation. Both methods judge
*identical* translations — generated once, then scored twice — so any
difference is the scoring method and nothing else.

What the comparison is actually looking for:

  * Spread. A scoring method that puts every translation in a narrow
    band cannot tell an optimizer which prompt is better, however
    reasonable each individual score looks. This pipeline's first real
    run scored 0.983 before any optimization had happened.
  * Agreement with an independent signal. chrF++ knows nothing about
    either judge. If one method tracks it more closely, that method is
    responding to something real rather than to its own habits.

Neither result is assumed. If severity scoring does not separate the
examples better, that is the finding and it should be reported as one.
"""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass

from judge import judge_translation
from metrics import compute_companion


@dataclass
class ScoredExample:
    source: str
    translation: str
    reference: str
    flat_score: float | None
    severity_score: float | None
    companion_score: float | None
    severity_errors: list[str]


@dataclass
class MethodSummary:
    method: str
    scores: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores)

    @property
    def spread(self) -> float:
        """Standard deviation: how much room the method leaves to tell examples apart."""
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def score_range(self) -> tuple[float, float]:
        return (min(self.scores), max(self.scores))

    @property
    def share_near_top(self) -> float:
        """
        Fraction scoring 9 or above. The ceiling effect made concrete:
        when most examples are already near-perfect, an optimizer that
        skips perfect scores has almost nothing left to learn from.
        """
        return sum(1 for s in self.scores if s >= 9.0) / len(self.scores)


def _with_scoring_method(config: dict, method: str) -> dict:
    clone = copy.deepcopy(config)
    clone["scoring_method"] = method
    return clone


def score_both_ways(
    examples: list[dict],
    translations: list[str],
    config: dict,
    config_dir: str,
    target_language: str = "",
) -> list[ScoredExample]:
    """Judge each already-generated translation under both methods."""
    flat_config = _with_scoring_method(config, "flat")
    severity_config = _with_scoring_method(config, "severity")
    companion_name = config.get("companion_metric")

    scored = []
    for example, translation in zip(examples, translations):
        source = example["source"]
        reference = example.get("reference", "")

        flat = judge_translation(source, translation, flat_config, config_dir, target_language)
        severity = judge_translation(
            source, translation, severity_config, config_dir, target_language
        )
        companion = compute_companion(companion_name, translation, reference)

        # The individual errors are the point of severity scoring, so
        # they get carried through rather than collapsed into a number.
        errors: list[str] = []
        for model_result in severity.per_model:
            errors.extend(model_result.invalid_reasons)

        scored.append(
            ScoredExample(
                source=source,
                translation=translation,
                reference=reference,
                flat_score=flat.score,
                severity_score=severity.score,
                companion_score=companion.score,
                severity_errors=errors,
            )
        )
    return scored


def summarize(scored: list[ScoredExample]) -> dict:
    """Both methods' distributions, plus how each tracks the companion metric."""
    from validate_judge import _spearman

    flat = [s.flat_score for s in scored if s.flat_score is not None]
    severity = [s.severity_score for s in scored if s.severity_score is not None]

    # Correlation needs both values present on the same example, so the
    # pairs are built together rather than from the two lists above.
    paired = [
        (s.companion_score, s.flat_score, s.severity_score)
        for s in scored
        if s.companion_score is not None
        and s.flat_score is not None
        and s.severity_score is not None
    ]

    summary = {
        "n": len(scored),
        "flat": None,
        "severity": None,
        "flat_vs_companion_spearman": None,
        "severity_vs_companion_spearman": None,
        "unjudged_flat": sum(1 for s in scored if s.flat_score is None),
        "unjudged_severity": sum(1 for s in scored if s.severity_score is None),
    }

    if flat:
        f = MethodSummary("flat", flat)
        summary["flat"] = {
            "mean": round(f.mean, 3),
            "stdev": round(f.spread, 3),
            "min": f.score_range[0],
            "max": f.score_range[1],
            "share_9_or_above": round(f.share_near_top, 3),
        }
    if severity:
        s = MethodSummary("severity", severity)
        summary["severity"] = {
            "mean": round(s.mean, 3),
            "stdev": round(s.spread, 3),
            "min": s.score_range[0],
            "max": s.score_range[1],
            "share_9_or_above": round(s.share_near_top, 3),
        }

    if len(paired) > 2:
        companion = [p[0] for p in paired]
        # _spearman returns None when a series has no variance at all,
        # which is itself a finding: a method that scored everything
        # identically cannot correlate with anything.
        for key, index in (("flat", 1), ("severity", 2)):
            rho = _spearman(companion, [p[index] for p in paired])
            summary[f"{key}_vs_companion_spearman"] = None if rho is None else round(rho, 3)

    return summary


def verdict(summary: dict) -> str:
    """
    State what the numbers support, including when they support nothing.
    A comparison that quietly recommends the method its author already
    preferred is not a comparison.
    """
    flat, severity = summary.get("flat"), summary.get("severity")
    if not flat or not severity:
        return "Not enough valid judgments from one of the methods to compare."

    if summary["n"] < 10:
        caveat = f"Only {summary['n']} examples; treat this as directional, not conclusive.\n"
    else:
        caveat = ""

    wider = severity["stdev"] > flat["stdev"]
    less_ceilinged = severity["share_9_or_above"] < flat["share_9_or_above"]

    f_corr = summary["flat_vs_companion_spearman"]
    s_corr = summary["severity_vs_companion_spearman"]
    tracks_better = f_corr is not None and s_corr is not None and s_corr > f_corr

    if wider and less_ceilinged and tracks_better:
        return caveat + "Severity scoring separates the examples better and tracks the independent metric more closely. Adopt it."
    if wider and less_ceilinged:
        return caveat + "Severity scoring separates the examples better, but does not track the independent metric more closely. Worth adopting for the added headroom, with the correlation rechecked on more examples."
    if not wider and not less_ceilinged:
        return caveat + "Severity scoring does not separate the examples any better than flat scoring here. The ceiling effect is not explained by the scoring method alone; look at the data and the judge model next."
    return caveat + "Mixed result: the two methods differ but neither is clearly better on these examples. Re-run on a larger sample before deciding."
