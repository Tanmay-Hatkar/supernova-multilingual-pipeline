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

# --- Severity-weighted scoring (MQM) ------------------------------------
#
# Asking a model for one holistic number produces a score that barely
# moves: in this pipeline's own runs the flat judge put almost every
# translation between 9 and 10 while a reference metric on the same
# sentences ranged from 26 to 77. A score with no spread gives an
# optimizer nothing to optimize against.
#
# The fix is not to ask harder. It's to stop asking for a number at all:
# the judge lists the errors it actually found and labels each one's
# severity, and the score is computed from those labels. That makes the
# scale mean something specific, and it makes one serious error cost
# more than many trivial ones — which is the behavior we want an
# optimizer to chase.
#
# Weights are the standard MQM ones used by GEMBA-MQM.

SEVERITY_WEIGHTS = {"minor": 1, "major": 5, "critical": 25}

# One critical error is enough to exhaust the budget, so a translation
# whose meaning is destroyed scores zero regardless of what else is
# right about it.
MAX_PENALTY = 25

SEVERITY_RUBRIC = """You are an expert translation quality judge. Do not give an
overall score. Instead, identify every error in the translation and label how
severe each one is.

Severity levels:
- "minor": awkward, unidiomatic, or stylistically off, but the meaning survives
  intact. A reader would understand correctly.
- "major": part of the meaning is wrong, missing, or added. A reader would be
  misled about something, but not about the main point.
- "critical": the meaning is destroyed, reversed, or replaced. This includes
  wrong entities, numbers, or negation; output in the wrong language or script;
  and text left untranslated.

Judge the translation against the SOURCE. Different wording from a reference is
not an error. Proper nouns and acronyms conventionally left in Latin script are
not errors.

If the translation is correct, return an empty errors list. Do not invent errors
to appear thorough, and do not overlook real ones to appear generous.
"""

SEVERITY_OUTPUT_FORMAT = """
Respond with ONLY a JSON object in this exact shape, no other text before or after it:
{"errors": [{"severity": "minor|major|critical", "span": "<the problematic text>",
"explanation": "<what is wrong with it>"}], "feedback": "<overall assessment>"}
"""


@dataclass
class JudgeError:
    """One error the judge reported, under severity-weighted scoring."""

    severity: str
    span: str
    explanation: str

    @property
    def weight(self) -> int:
        return SEVERITY_WEIGHTS[self.severity]


@dataclass
class JudgeCallResult:
    score: float
    feedback: str
    # Empty under flat scoring, which reports no structure — only a
    # number. Populated under severity scoring, where the errors are
    # the actual judgment and the score is derived from them.
    errors: list[JudgeError] = field(default_factory=list)


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


def _build_system_prompt(judge_spec: dict, config_dir: str, scoring_method: str) -> str:
    extension = _load_rubric_extension(judge_spec.get("rubric_extension"), config_dir)
    if scoring_method == "severity":
        return SEVERITY_RUBRIC + extension + SEVERITY_OUTPUT_FORMAT
    return BASE_RUBRIC + extension + OUTPUT_FORMAT_INSTRUCTION


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


SCORING_METHODS = ("flat", "severity")


def _resolve_scoring_method(config: dict) -> str:
    """
    Read `scoring_method` from a language config, rejecting anything
    unrecognized. This key sat in every config file for weeks while
    nothing read it, so a typo here should fail loudly rather than
    silently fall back to the method it was written to replace.
    """
    method = str(config.get("scoring_method", "flat")).strip().lower()
    if method not in SCORING_METHODS:
        raise ValueError(
            f"Unknown scoring_method {method!r} in language config; "
            f"expected one of {list(SCORING_METHODS)}"
        )
    return method


def score_from_errors(errors: list[JudgeError]) -> float:
    """
    Turn a list of labelled errors into a 0-10 score.

    The penalty is the MQM weighted sum, capped so that one critical
    error exhausts the budget on its own. Capping rather than letting
    the penalty run away matters: without it, a translation with three
    critical errors and one with one critical error both score zero
    anyway, but the uncapped version makes the arithmetic look more
    precise than the judgment behind it actually is.
    """
    penalty = min(sum(error.weight for error in errors), MAX_PENALTY)
    return round(10.0 * (1.0 - penalty / MAX_PENALTY), 4)


def _parse_severity_response(raw_text: str) -> JudgeCallResult:
    """
    Parse a severity-tagged judgment and compute its score here, rather
    than trusting a number the judge supplies. The judge's job is to
    find and label errors; converting that to a score is arithmetic and
    belongs on our side, where it is consistent across models and runs.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {raw_text[:200]!r}")

    parsed = json_repair.loads(match.group(0))
    if not isinstance(parsed, dict) or "errors" not in parsed:
        raise ValueError(f"Judge response missing an 'errors' list: {raw_text[:200]!r}")

    raw_errors = parsed["errors"]
    if not isinstance(raw_errors, list):
        raise ValueError(f"Judge 'errors' was not a list: {raw_text[:200]!r}")

    errors = []
    for item in raw_errors:
        if not isinstance(item, dict):
            raise ValueError(f"Judge error entry was not an object: {item!r}")
        severity = str(item.get("severity", "")).strip().lower()
        if severity not in SEVERITY_WEIGHTS:
            # An unrecognized severity can't be weighted, and guessing
            # one would silently invent a number. Reject the whole
            # judgment so it's excluded rather than quietly distorted.
            raise ValueError(
                f"Unknown severity {severity!r}; expected one of {sorted(SEVERITY_WEIGHTS)}"
            )
        errors.append(
            JudgeError(
                severity=severity,
                span=str(item.get("span", "")),
                explanation=str(item.get("explanation", "")),
            )
        )

    feedback = str(parsed.get("feedback", "")).strip()
    if not feedback and errors:
        # The written explanation is what GEPA revises prompts from, so
        # a judgment without one is only half useful. Rebuild it from
        # the errors rather than handing the optimizer an empty string.
        feedback = "; ".join(f"[{e.severity}] {e.span}: {e.explanation}" for e in errors)

    if not feedback and not errors:
        # An empty error list with no written justification is a
        # non-answer, not a perfect translation — but under severity
        # scoring the two are arithmetically identical, both landing on
        # 10.0. A disengaged or truncated judge response would therefore
        # score full marks, which is the single most expensive way for a
        # judge to fail: it awards the top score to anything.
        #
        # Rejecting it sends the example down the judge-failure path,
        # where it is excluded and counted rather than silently trusted.
        # The flat path cannot hit this, because it requires a score
        # field to be present at all.
        raise ValueError(
            "Judge reported no errors and gave no explanation; treating as a "
            "non-answer rather than a perfect score"
        )

    return JudgeCallResult(score=score_from_errors(errors), feedback=feedback, errors=errors)


def _build_user_prompt(source: str, translation: str, target_language: str) -> str:
    """
    Name the target language explicitly.

    Without it, a judge handed the English source back as the
    "translation" sees perfect meaning preservation and says so. A
    sensitivity probe on this pipeline found exactly that: six out of
    six untranslated outputs scored ten out of ten, and the mean score
    went *up* when the Spanish was replaced by the original English.
    The judge was not being lenient — it was never told what language
    it was supposed to be looking at.
    """
    if not target_language:
        return f"Source: {source}\nTranslation: {translation}"
    return (
        f"Target language: {target_language}\n"
        f"Source (English): {source}\n"
        f"Translation (should be in {target_language}): {translation}\n\n"
        f"If the translation is not in {target_language}, that is a critical failure "
        f"regardless of how well it conveys the meaning."
    )


def judge_once(
    source: str,
    translation: str,
    judge_spec: dict,
    config_dir: str,
    scoring_method: str = "flat",
    target_language: str = "",
) -> JudgeCallResult:
    client = get_client(judge_spec["provider"], judge_spec["model"])
    raw = client.complete(
        system=_build_system_prompt(judge_spec, config_dir, scoring_method),
        user=_build_user_prompt(source, translation, target_language),
        temperature=0.7,
    )
    if scoring_method == "severity":
        return _parse_severity_response(raw)
    return _parse_judge_response(raw)


def _is_suspicious_zero(result: JudgeCallResult) -> bool:
    """
    A zero score paired with feedback that reports no actual problem is
    an internal contradiction — the number and the model's own reasoning
    disagree. Far more often a parsing or generation artifact than a
    genuine verdict, so it gets flagged rather than trusted.

    Severity scoring makes this check nearly redundant by construction,
    since a zero there is arithmetic over errors the judge listed
    explicitly. The keyword heuristic is kept only for flat scoring,
    where a bare number is all there is to sanity-check against.
    """
    if result.score != 0.0:
        return False
    if result.errors:
        return False  # severity scoring: the zero is backed by listed errors
    text = result.feedback.lower()
    negative_markers = ("error", "wrong", "incorrect", "missing", "poor", "bad", "fail", "omit")
    return not any(marker in text for marker in negative_markers)


def judge_with_model(
    source: str,
    translation: str,
    judge_spec: dict,
    config_dir: str,
    scoring_method: str = "flat",
    target_language: str = "",
) -> JudgeModelResult:
    """Run one judge model N times and aggregate its runs via RRWA."""
    n_runs = judge_spec.get("runs_per_example", 3)
    scores, feedbacks, invalid = [], [], []
    suspicious = False

    for _ in range(n_runs):
        try:
            result = judge_once(
                source, translation, judge_spec, config_dir, scoring_method, target_language
            )
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
    source: str,
    translation: str,
    config: dict,
    config_dir: str,
    target_language: str = "",
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

    scoring_method = _resolve_scoring_method(config)
    # Fall back to the language named in the config, so a caller that
    # forgets to pass it still gets a judge that knows what language it
    # is looking at. Silently omitting this is what made the judge score
    # untranslated English output ten out of ten.
    language = target_language or config.get("language_name", "")
    results = [
        judge_with_model(source, translation, spec, config_dir, scoring_method, language)
        for spec in iter_judges(config)
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
