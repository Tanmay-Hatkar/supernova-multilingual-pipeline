"""
Tests for severity-weighted (MQM) scoring.

The point of this scoring method is discrimination: a flat holistic
score put nearly every translation in this pipeline between 9 and 10
while a reference metric on the same sentences ranged from 26 to 77.
So these tests check two things — that the arithmetic matches the
published MQM weights, and that a malformed or unweighable judgment is
rejected outright rather than converted into a plausible-looking number.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge import (
    MAX_PENALTY,
    SEVERITY_WEIGHTS,
    JudgeError,
    _is_suspicious_zero,
    _parse_severity_response,
    _resolve_scoring_method,
    score_from_errors,
)


def _error(severity: str) -> JudgeError:
    return JudgeError(severity=severity, span="x", explanation="y")


def test_no_errors_scores_ten():
    assert score_from_errors([]) == 10.0


def test_weights_match_published_mqm_values():
    # minor 1, major 5, critical 25 — the GEMBA-MQM weights the pod's
    # own scoring research converged on independently.
    assert SEVERITY_WEIGHTS == {"minor": 1, "major": 5, "critical": 25}


def test_one_critical_error_exhausts_the_budget():
    assert score_from_errors([_error("critical")]) == 0.0


def test_one_major_costs_five_times_one_minor():
    minor_penalty = 10.0 - score_from_errors([_error("minor")])
    major_penalty = 10.0 - score_from_errors([_error("major")])
    assert major_penalty == pytest.approx(5 * minor_penalty)


def test_many_minor_errors_still_beat_one_critical():
    # The behavior the whole method exists for: avoiding one serious
    # error is worth more than polishing several small ones.
    ten_minors = score_from_errors([_error("minor")] * 10)
    one_critical = score_from_errors([_error("critical")])
    assert ten_minors > one_critical
    assert ten_minors == pytest.approx(6.0)


def test_penalty_is_capped_rather_than_going_negative():
    assert score_from_errors([_error("critical")] * 4) == 0.0
    assert MAX_PENALTY == 25


def test_parses_clean_severity_response():
    raw = (
        '{"errors": [{"severity": "major", "span": "Rwanda", '
        '"explanation": "source says Uganda"}], "feedback": "Wrong country."}'
    )
    result = _parse_severity_response(raw)
    assert len(result.errors) == 1
    assert result.errors[0].severity == "major"
    assert result.score == 8.0
    assert result.feedback == "Wrong country."


def test_empty_error_list_is_a_perfect_score_not_a_failure():
    result = _parse_severity_response('{"errors": [], "feedback": "Accurate and natural."}')
    assert result.score == 10.0
    assert result.errors == []


def test_parses_severity_response_wrapped_in_prose_and_fences():
    raw = (
        "Here is my assessment:\n```json\n"
        '{"errors": [{"severity": "minor", "span": "asi", "explanation": "missing accent"}],'
        ' "feedback": "Minor orthography issue."}\n```\nHope that helps.'
    )
    result = _parse_severity_response(raw)
    assert result.score == 9.6
    assert result.errors[0].severity == "minor"


def test_missing_feedback_is_rebuilt_from_the_errors():
    # GEPA revises prompts from the written feedback, so an empty
    # explanation is worth reconstructing rather than passing through.
    raw = '{"errors": [{"severity": "major", "span": "no", "explanation": "negation dropped"}]}'
    result = _parse_severity_response(raw)
    assert "negation dropped" in result.feedback


def test_unknown_severity_is_rejected_not_guessed():
    raw = '{"errors": [{"severity": "moderate", "span": "x", "explanation": "y"}], "feedback": "z"}'
    with pytest.raises(ValueError, match="Unknown severity"):
        _parse_severity_response(raw)


def test_missing_errors_key_is_rejected():
    with pytest.raises(ValueError, match="errors"):
        _parse_severity_response('{"score": 8, "feedback": "flat-format response"}')


def test_errors_must_be_a_list():
    with pytest.raises(ValueError):
        _parse_severity_response('{"errors": "none", "feedback": "x"}')


def test_zero_backed_by_listed_errors_is_not_suspicious():
    # Under flat scoring a zero with no stated problem is treated as an
    # artifact. Under severity scoring a zero is arithmetic over errors
    # the judge listed, so it must not be thrown away.
    raw = (
        '{"errors": [{"severity": "critical", "span": "all", '
        '"explanation": "output left in English"}], "feedback": "Untranslated."}'
    )
    result = _parse_severity_response(raw)
    assert result.score == 0.0
    assert not _is_suspicious_zero(result)


def test_scoring_method_must_be_recognized():
    assert _resolve_scoring_method({"scoring_method": "severity"}) == "severity"
    assert _resolve_scoring_method({}) == "flat"
    with pytest.raises(ValueError, match="Unknown scoring_method"):
        _resolve_scoring_method({"scoring_method": "sevrity"})
