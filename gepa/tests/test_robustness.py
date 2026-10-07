"""
Tests for the pieces added to make this pipeline robust rather than
merely functional: distinguishing a retryable rate limit from a
structural rejection, refusing to turn judge failures into scores, and
handling single- and multi-judge configs through one code path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients import _is_structural, _parse_retry_after_seconds
from judge import JudgeCallResult, _is_suspicious_zero, _parse_judge_response
from metrics import compute_chrf_plus_plus, compute_companion
from reliability import iter_judges

# --- retryable vs structural: the distinction that cost real debugging time ---


def test_structural_rejection_is_recognized():
    message = (
        "Request too large for model `qwen/qwen3.6-27b` on output tokens per minute "
        "(OTPM): Limit 1000, Requested 4096. The request's expected output tokens "
        "exceed the enforced limit; reduce max_tokens and try again."
    )
    assert _is_structural(message)


def test_transient_rate_limit_is_not_treated_as_structural():
    message = (
        "Rate limit reached for model `qwen/qwen3.6-27b` on tokens per minute (TPM): "
        "Limit 8000, Used 7984. Please try again in 12.4s."
    )
    assert not _is_structural(message)


def test_retry_after_parses_seconds_and_minutes():
    assert _parse_retry_after_seconds("please try again in 12.4s") == pytest.approx(12.4)
    assert _parse_retry_after_seconds("please try again in 2m3.5s") == pytest.approx(123.5)


def test_retry_after_falls_back_when_message_format_changes():
    # Provider wording drifts; a missing pattern shouldn't crash the run.
    assert _parse_retry_after_seconds("slow down") > 0


# --- judge output validation: never invent a score from a failure ---


def test_out_of_range_score_is_rejected():
    with pytest.raises(ValueError, match="outside the valid"):
        _parse_judge_response('{"score": 42, "feedback": "great"}')


def test_negative_score_is_rejected():
    with pytest.raises(ValueError, match="outside the valid"):
        _parse_judge_response('{"score": -3, "feedback": "bad"}')


def test_suspicious_zero_flagged_when_feedback_describes_no_problem():
    result = JudgeCallResult(score=0.0, feedback="The translation reads naturally.")
    assert _is_suspicious_zero(result)


def test_genuine_zero_with_stated_problem_is_not_suspicious():
    result = JudgeCallResult(score=0.0, feedback="Completely wrong meaning, major error.")
    assert not _is_suspicious_zero(result)


def test_nonzero_score_is_never_suspicious():
    assert not _is_suspicious_zero(JudgeCallResult(score=7.0, feedback="fine"))


# --- single and multi judge configs travel the same path ---


def test_single_judge_config_normalizes_to_a_list():
    config = {"judge": {"provider": "groq", "model": "a"}}
    assert iter_judges(config) == [{"provider": "groq", "model": "a"}]


def test_multi_judge_config_returns_all_judges():
    config = {"judges": [{"model": "a"}, {"model": "b"}, {"model": "c"}]}
    assert len(iter_judges(config)) == 3


def test_config_with_no_judge_raises_clearly():
    with pytest.raises(ValueError, match="neither"):
        iter_judges({"task_model": {"model": "x"}})


# --- companion metric: the config promised chrF++, now it delivers ---


def test_chrf_scores_identical_text_highly():
    result = compute_chrf_plus_plus(
        "El planeta madre se está agotando.", "El planeta madre se está agotando."
    )
    assert result.score == pytest.approx(100.0, abs=0.01)


def test_chrf_scores_unrelated_text_low():
    result = compute_chrf_plus_plus("Buenos días amigo.", "El planeta madre se está agotando.")
    assert result.score < 30


def test_chrf_handles_missing_reference_without_raising():
    result = compute_chrf_plus_plus("alguna traducción", "")
    assert result.score is None
    assert "no reference" in result.note


def test_empty_hypothesis_scores_zero_not_none():
    # An empty translation is a real failure, distinct from a missing reference.
    result = compute_chrf_plus_plus("", "El planeta madre se está agotando.")
    assert result.score == 0.0


def test_companion_dispatch_handles_unknown_metric_gracefully():
    result = compute_companion("bleu-9000", "hola", "hola")
    assert result.score is None
    assert "unsupported" in result.note


def test_companion_dispatch_accepts_chrf_aliases():
    for alias in ("chrf++", "CHRF++", "chrf_plus_plus"):
        assert compute_companion(alias, "hola mundo", "hola mundo").score == pytest.approx(
            100.0, abs=0.01
        )
