"""
GEMBA-style, reference-free LLM judge.

Two things this module deliberately does not do:

  * It has no per-language branches. Which model judges, how many runs,
    whether there's an extra rubric, and whether there's one judge or
    several all come from that language's config. Adding a language's
    quirk means adding a config file, never editing this code.
  * It never coerces a failed judgment into a score. A malformed
    response, an out-of-range number, or a zero that contradicts its
    own stated reasoning gets excluded and recorded — not silently
    averaged in as if the translation were bad. Treating a judge
    failure as a quality signal is how you end up optimizing against
    your own bugs.
"""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass, field

import json_repair

from clients import get_client
from rrwa import RRWAResult, compute_rrwa

BASE_RUBRIC = """You are an expert translation quality judge. Score the following
translation from 0 to 10 and explain your reasoning.

Evaluate:
- Semantic accuracy: does it convey the same meaning, with no omissions or hallucinations
- Fluency: does it read naturally, without awkward phrasing or unnatural word order
- Naturalness: does it sound like a native speaker, not a word-for-word translation
- Cultural appropriateness: are idioms, register, and formality level correct for context
"""

OUTPUT_FORMAT_INSTRUCTION = """
Respond with ONLY a JSON object in this exact shape, no other text before or after it:
{"score": <number 0-10>, "feedback": "<your written explanation>"}
"""


@dataclass
class JudgeCallResult:
    score: float
    feedback: str


@dataclass
class JudgeModelResult:
    """One judge model's verdict, after its repeated runs are aggregated."""

    model: str
    score: float | None
    feedback: str
    valid_runs: int
    total_runs: int
    invalid_reasons: list[str] = field(default_factory=list)
    suspicious_zero: bool = False


@dataclass
class JudgeVerdict:
    """The final answer handed to the optimizer, plus why it says that."""

    score: float | None
    feedback: str
    needs_review: bool
    per_model: list[JudgeModelResult] = field(default_factory=list)
    disagreement: float | None = None  # spread between judges, when >1 judge ran


def _load_rubric_extension(path: str | None, config_dir: str) -> str:
    if not path:
        return ""
    full_path = os.path.join(config_dir, path)
    if not os.path.exists(full_path):
        return ""  # referenced but not written yet: fail soft, not hard
    with open(full_path, encoding="utf-8") as f:
        return "\n" + f.read()


def _build_system_prompt(judge_spec: dict, config_dir: str) -> str:
    return (
        BASE_RUBRIC
        + _load_rubric_extension(judge_spec.get("rubric_extension"), config_dir)
        + OUTPUT_FORMAT_INSTRUCTION
    )


def _parse_judge_response(raw_text: str) -> JudgeCallResult:
    """
    Judges wrap JSON in prose and markdown fences despite instructions,
    and their feedback — being natural language about quoted words — is
    full of unescaped quotes and literal newlines that make it
    technically invalid JSON while remaining perfectly unambiguous to a
    reader. json_repair handles exactly that; strict json.loads does
    not, and was observed failing on real judge output.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {raw_text[:200]!r}")

    parsed = json_repair.loads(match.group(0))
    if not isinstance(parsed, dict) or "score" not in parsed or "feedback" not in parsed:
        raise ValueError(f"Judge response missing expected fields: {raw_text[:200]!r}")

    score = float(parsed["score"])
    if not (0.0 <= score <= 10.0):
        raise ValueError(f"Judge score {score} outside the valid 0-10 range")

    return JudgeCallResult(score=score, feedback=str(parsed["feedback"]))


def judge_once(source: str, translation: str, judge_spec: dict, config_dir: str) -> JudgeCallResult:
    client = get_client(judge_spec["provider"], judge_spec["model"])
    raw = client.complete(
        system=_build_system_prompt(judge_spec, config_dir),
        user=f"Source: {source}\nTranslation: {translation}",
        temperature=0.7,
    )
    return _parse_judge_response(raw)


def _is_suspicious_zero(result: JudgeCallResult) -> bool:
    """
    A zero score paired with feedback that reports no actual problem is
    an internal contradiction — the number and the model's own reasoning
    disagree. Far more often a parsing or generation artifact than a
    genuine verdict, so it gets flagged rather than trusted.
    """
    if result.score != 0.0:
        return False
    text = result.feedback.lower()
    negative_markers = ("error", "wrong", "incorrect", "missing", "poor", "bad", "fail", "omit")
    return not any(marker in text for marker in negative_markers)


def judge_with_model(
    source: str, translation: str, judge_spec: dict, config_dir: str
) -> JudgeModelResult:
    """Run one judge model N times and aggregate its runs via RRWA."""
    n_runs = judge_spec.get("runs_per_example", 3)
    scores, feedbacks, invalid = [], [], []
    suspicious = False

    for _ in range(n_runs):
        try:
            result = judge_once(source, translation, judge_spec, config_dir)
        except Exception as e:
            invalid.append(f"{type(e).__name__}: {str(e)[:120]}")
            continue

        if _is_suspicious_zero(result):
            suspicious = True
            invalid.append("suspicious_zero: score 0 with no problem described in feedback")
            continue

        scores.append(result.score)
        feedbacks.append(result.feedback)

    if not scores:
        return JudgeModelResult(
            model=judge_spec["model"],
            score=None,
            feedback="",
            valid_runs=0,
            total_runs=n_runs,
            invalid_reasons=invalid,
            suspicious_zero=suspicious,
        )

    rrwa: RRWAResult = compute_rrwa(scores)
    closest = min(range(len(scores)), key=lambda i: abs(scores[i] - rrwa.final_score))
    return JudgeModelResult(
        model=judge_spec["model"],
        score=rrwa.final_score,
        feedback=feedbacks[closest],
        valid_runs=len(scores),
        total_runs=n_runs,
        invalid_reasons=invalid,
        suspicious_zero=suspicious,
    )


def judge_translation(
    source: str, translation: str, config: dict, config_dir: str
) -> JudgeVerdict:
    """
    Score a translation using however many judges the language config
    declares. Single-judge and multi-judge configs take the same path —
    callers never branch on which shape a language uses.

    With several judges, the median is used rather than the mean: one
    judge failing badly on an example shouldn't drag the verdict down
    with it, which is exactly the failure mode multi-judge averaging
    was found to introduce in earlier research.
    """
    from reliability import iter_judges

    results = [
        judge_with_model(source, translation, spec, config_dir) for spec in iter_judges(config)
    ]
    usable = [r for r in results if r.score is not None]

    if not usable:
        return JudgeVerdict(
            score=None,
            feedback="No judge produced a valid score for this example.",
            needs_review=True,
            per_model=results,
        )

    scores = [r.score for r in usable]
    final_score = statistics.median(scores)
    disagreement = (max(scores) - min(scores)) if len(scores) > 1 else None

    # Surface the feedback from whichever judge landed closest to the
    # final verdict, so the written explanation matches the number.
    closest = min(usable, key=lambda r: abs(r.score - final_score))

    return JudgeVerdict(
        score=final_score,
        feedback=closest.feedback,
        needs_review=False,
        per_model=results,
        disagreement=disagreement,
    )
