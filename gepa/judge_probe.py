"""
Measure whether the judge can detect a fault it is shown on purpose.

Why this exists. Comparing two scoring methods on real translations
told us they disagree, but not which is right, because nothing in that
comparison knew the true quality of anything. Both methods put almost
every example near the top, and there are two completely different
explanations for that:

  * the translations really are good, and a high score is correct, or
  * the judge is lenient and would score anything highly.

Those need opposite fixes, so guessing between them is expensive. This
module removes the guess by constructing the ground truth instead of
inferring it: take a translation the judge already scored highly, break
it in one specific, known way, and score it again. The drop is the
judge's sensitivity to that fault.

A judge that scores a translation with its negation flipped the same as
the original is not strict or lenient — it is not measuring meaning at
all. That is worth knowing before spending a single further run on
prompt optimization.

This is the same idea as the quality bands used when building a human
review set, applied to the judge rather than to a human reviewer. The
difference is that here we know exactly what we broke.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from judge import judge_translation

# How much a competent judge should mark each fault down, on a 0-10
# scale. These are expectations to compare against, not thresholds the
# pipeline enforces — the point is to see the shape of the response,
# not to pass or fail a specific number.
EXPECTED_MINIMUM_DROP = {
    "negation_flip": 5.0,  # the meaning is now the opposite
    "entity_swap": 4.0,  # a reader is misinformed about a fact
    "clause_omission": 3.0,  # part of the meaning is simply gone
    "untranslated": 7.0,  # not a translation at all
    "truncation": 3.0,  # the second half is missing
    "punctuation_only": 0.0,  # a control: should barely move
}


@dataclass
class Degradation:
    name: str
    description: str
    apply: Callable[[str], str | None]


@dataclass
class ProbeResult:
    source: str
    original: str
    degradation: str
    degraded: str
    original_score: float | None
    degraded_score: float | None

    @property
    def drop(self) -> float | None:
        if self.original_score is None or self.degraded_score is None:
            return None
        return self.original_score - self.degraded_score

    @property
    def expected_drop(self) -> float:
        return EXPECTED_MINIMUM_DROP.get(self.degradation, 0.0)

    @property
    def detected(self) -> bool | None:
        """Did the judge mark this fault down as much as it should have?"""
        if self.drop is None:
            return None
        if self.expected_drop == 0.0:
            # The control: "detecting" a punctuation change as a serious
            # fault would itself be a problem, so passing means staying
            # roughly where it was.
            return self.drop < 1.0
        return self.drop >= self.expected_drop


# --- The faults --------------------------------------------------------
#
# Each returns None when it cannot be applied to a given sentence, so a
# probe is skipped rather than reported as a pass it did not earn. A
# degradation that silently no-ops would make the judge look perfect.

_CJK = re.compile(r"[一-鿿㐀-䶿]")

# Latin-script negations are matched on word boundaries and
# case-insensitively, because "No tenemos" at the start of a sentence is
# the most common shape and a naive lowercase substring misses it — which
# would silently skip the single most important fault in the probe.
_NEGATIONS_LATIN = [
    (r"\bno\b\s*", ""),
    (r"\bnunca\b", "siempre"),
    (r"\bnada\b", "todo"),
    (r"\bnever\b", "always"),
    (r"\bn't\b", ""),
]

# Chinese negation particles, matched as plain substrings since there
# are no word boundaries to anchor to.
_NEGATIONS_ZH = [("没有", "有"), ("不是", "是"), ("不", ""), ("无法", "能够")]

_ENTITIES = {
    "Uganda": "Rwanda",
    "San Francisco": "Los Angeles",
    "Jordan": "Lebanon",
    "Asunción": "Montevideo",
    "Gaddafi": "Mubarak",
}


def _flip_negation(text: str) -> str | None:
    for pattern, replacement in _NEGATIONS_LATIN:
        flipped, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if count:
            # Removing a leading "No " leaves the next word lowercase.
            # Left alone that is a second, unintended fault, and the
            # probe is supposed to change exactly one thing.
            return flipped[:1].upper() + flipped[1:] if flipped else None
    for marker, replacement in _NEGATIONS_ZH:
        if marker in text:
            return text.replace(marker, replacement, 1)
    return None


def _swap_entity(text: str) -> str | None:
    for original, replacement in _ENTITIES.items():
        if original in text:
            return text.replace(original, replacement, 1)
    # Fall back to swapping any number, which is the same class of
    # fault: a reader is confidently misinformed about a fact.
    match = re.search(r"\b(\d+)\b", text)
    if match:
        wrong = str(int(match.group(1)) + 7)
        return text[: match.start()] + wrong + text[match.end() :]
    return None


def _omit_clause(text: str) -> str | None:
    # Fullwidth punctuation as well as ASCII: without it every Chinese
    # sentence skips this fault, and Mandarin is the direction that
    # matters most here.
    parts = [p for p in re.split(r"[,，、]\s*", text) if p.strip()]
    if len(parts) < 2:
        return None
    kept = "，".join(parts[:-1]) if "，" in text else ", ".join(parts[:-1])
    return kept + ("。" if _CJK.search(text) else ".")


def _truncate(text: str) -> str | None:
    # CJK packs far more meaning per character, so a fixed character
    # floor tuned for Latin script would exclude most Chinese sentences.
    minimum = 12 if _CJK.search(text) else 40
    if len(text) < minimum:
        return None
    return text[: int(len(text) * 0.55)].rstrip()


def _leave_untranslated(text: str, source: str = "") -> str | None:
    return source or None


def _punctuation_only(text: str) -> str | None:
    for mark in (",", "，", "、"):
        if mark in text:
            return text.replace(mark, "", 1)
    if text and text[-1] in ".。！!?？":
        return text[:-1]
    return None


DEGRADATIONS = [
    Degradation("negation_flip", "meaning reversed", _flip_negation),
    Degradation("entity_swap", "wrong name, place or number", _swap_entity),
    Degradation("clause_omission", "a clause removed", _omit_clause),
    Degradation("truncation", "cut off partway", _truncate),
    Degradation("punctuation_only", "control: one punctuation mark removed", _punctuation_only),
]


def probe_example(
    source: str,
    translation: str,
    config: dict,
    config_dir: str,
    target_language: str = "",
) -> list[ProbeResult]:
    """Score one translation, then score each broken version of it."""
    baseline = judge_translation(source, translation, config, config_dir, target_language)

    results = []
    for degradation in DEGRADATIONS:
        broken = degradation.apply(translation)
        if broken is None or broken.strip() == translation.strip():
            continue  # not applicable to this sentence; skip rather than fake a pass
        verdict = judge_translation(source, broken, config, config_dir, target_language)
        results.append(
            ProbeResult(
                source=source,
                original=translation,
                degradation=degradation.name,
                degraded=broken,
                original_score=baseline.score,
                degraded_score=verdict.score,
            )
        )

    # The untranslated case is built from the source rather than from
    # the translation, so it does not fit the transform signature above.
    if source.strip() and source.strip() != translation.strip():
        verdict = judge_translation(source, source, config, config_dir, target_language)
        results.append(
            ProbeResult(
                source=source,
                original=translation,
                degradation="untranslated",
                degraded=source,
                original_score=baseline.score,
                degraded_score=verdict.score,
            )
        )

    return results


def summarize(results: list[ProbeResult]) -> dict:
    """Detection rate per fault type, and the mean drop the judge applied."""
    by_fault: dict[str, list[ProbeResult]] = {}
    for result in results:
        by_fault.setdefault(result.degradation, []).append(result)

    summary = {}
    for fault, group in by_fault.items():
        drops = [r.drop for r in group if r.drop is not None]
        detected = [r.detected for r in group if r.detected is not None]
        summary[fault] = {
            "n": len(group),
            "mean_drop": round(sum(drops) / len(drops), 2) if drops else None,
            "expected_min_drop": EXPECTED_MINIMUM_DROP.get(fault, 0.0),
            "detection_rate": round(sum(detected) / len(detected), 2) if detected else None,
        }
    return summary


def verdict(summary: dict) -> str:
    """
    Say what this means for the pipeline, including the uncomfortable
    reading. A probe that can only conclude "the judge is fine" is not
    a probe.
    """
    serious = [f for f in ("negation_flip", "entity_swap", "untranslated") if f in summary]
    if not serious:
        return "No serious faults could be applied to these sentences; the probe is inconclusive."

    rates = [
        summary[f]["detection_rate"] for f in serious if summary[f]["detection_rate"] is not None
    ]
    if not rates:
        return "The judge produced no valid scores on the degraded outputs; fix that before reading anything into this."

    worst = min(rates)
    mean_rate = sum(rates) / len(rates)

    control = summary.get("punctuation_only", {}).get("detection_rate")
    control_note = ""
    if control is not None and control < 0.5:
        control_note = (
            " The control also failed: the judge moved a lot on a punctuation change alone, "
            "which means its scores are noisy rather than merely lenient."
        )

    if mean_rate >= 0.8 and worst >= 0.6:
        return (
            "The judge detects serious faults reliably. The high scores on real translations "
            "are therefore evidence that those translations are good, not that the judge is "
            "lenient — so the ceiling is a property of the data, and the fix is harder "
            "examples, not a different judge." + control_note
        )
    if mean_rate < 0.4:
        return (
            "The judge largely fails to mark down faults it was shown directly, including "
            "reversed meaning and wrong entities. Its scores on real translations cannot be "
            "trusted, and no amount of prompt optimization against it is meaningful until "
            "this is fixed." + control_note
        )
    return (
        "The judge detects some serious faults and misses others. Its scores carry partial "
        "signal. Worth identifying which fault types it is blind to and addressing those "
        "specifically before optimizing against it." + control_note
    )
