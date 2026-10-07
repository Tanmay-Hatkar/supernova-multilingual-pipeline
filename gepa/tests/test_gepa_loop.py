"""
Tests for the loop's scoring logic, with no network and no API spend.

Every rule in here was previously verified only by paying for a real
run, which meant it was verified rarely and slowly. These are the rules
that decide whether a reported number means anything:

  * a judge failure must not be scored as a bad translation
  * a failed deterministic check must be scored as a real zero
  * unjudgeable examples must be excluded from the reported mean
  * a run must abort rather than report a score built on a broken judge

All four are silent when wrong. A run with a broken judge still prints
a confident number, which is exactly why they are pinned here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gepa_loop
from gepa_loop import MAX_JUDGE_FAILURE_RATE, NEUTRAL_SCORE, PoolScore, RunState
from judge import JudgeVerdict


class FakePrediction:
    """Stands in for what dspy.Predict returns."""

    def __init__(self, translation: str):
        self.translation = translation


class FakeExample:
    def __init__(self, source: str, reference: str = "", target_language: str = "Spanish"):
        self.source = source
        self.reference = reference
        self.target_language = target_language


class FakeProgram:
    """Returns a canned translation per source, so a pool can be scored offline."""

    def __init__(self, outputs: dict[str, str]):
        self.outputs = outputs

    def __call__(self, source: str, target_language: str) -> FakePrediction:
        return FakePrediction(self.outputs[source])


def _verdict(score: float | None, needs_review: bool = False) -> JudgeVerdict:
    return JudgeVerdict(
        score=score,
        feedback="fake feedback mentioning an error" if score == 0.0 else "fake feedback",
        needs_review=needs_review,
    )


@pytest.fixture
def config():
    # No output_checks beyond the universal ones, so most tests exercise
    # the judge path rather than the check path.
    return {"language_name": "Spanish", "scoring_method": "flat", "companion_metric": None}


def _metric_with_judge(monkeypatch, config, verdicts):
    """Build the metric with judge_translation replaced by a canned sequence."""
    calls = iter(verdicts)
    monkeypatch.setattr(gepa_loop, "judge_translation", lambda *a, **k: next(calls))
    state = RunState()
    return gepa_loop._build_metric(config, config_dir=".", state=state), state


# --- A judge failure is a failure of measurement, not of translation ---


def test_judge_failure_is_not_scored_as_zero(monkeypatch, config):
    metric, state = _metric_with_judge(monkeypatch, config, [_verdict(None, needs_review=True)])
    result = metric(FakeExample("Hello."), FakePrediction("Hola."))

    assert result.score != 0.0, "a parsing failure must not look like a terrible translation"
    assert result.score == NEUTRAL_SCORE
    assert state.judge_failures == 1
    assert state.per_example[0]["outcome"] == "judge_failed"


def test_judge_failure_returns_the_running_mean_once_scores_exist(monkeypatch, config):
    metric, state = _metric_with_judge(
        monkeypatch,
        config,
        [_verdict(8.0), _verdict(6.0), _verdict(None, needs_review=True)],
    )
    metric(FakeExample("one"), FakePrediction("uno"))
    metric(FakeExample("two"), FakePrediction("dos"))
    result = metric(FakeExample("three"), FakePrediction("tres"))

    # Mean of 0.8 and 0.6: neutral with respect to the scores so far,
    # rather than dragging the candidate down.
    assert result.score == pytest.approx(0.7)


def test_an_empty_translation_is_a_genuine_zero(monkeypatch, config):
    metric, state = _metric_with_judge(monkeypatch, config, [])
    result = metric(FakeExample("Hello."), FakePrediction("   "))

    assert result.score == 0.0
    assert state.empty_translations == 1
    assert state.judge_failures == 0, "an empty output is the program's fault, not the judge's"


# --- Deterministic checks run first and are not negotiable ------------


def test_a_failed_check_scores_zero_without_consulting_the_judge(monkeypatch):
    config = {"language_name": "Mandarin", "output_checks": ["simplified_script"]}

    def explode(*a, **k):
        raise AssertionError("the judge must not be called when a check has failed")

    monkeypatch.setattr(gepa_loop, "judge_translation", explode)
    state = RunState()
    metric = gepa_loop._build_metric(config, config_dir=".", state=state)

    result = metric(FakeExample("The museum opens."), FakePrediction("博物館的新展覽"))

    assert result.score == 0.0
    assert state.check_failures == 1
    assert "simplified_script" in state.per_example[0]["checks"]["failed_checks"]


def test_source_copy_is_caught_for_any_language(monkeypatch, config):
    monkeypatch.setattr(
        gepa_loop, "judge_translation", lambda *a, **k: pytest.fail("judge should not run")
    )
    state = RunState()
    metric = gepa_loop._build_metric(config, config_dir=".", state=state)

    source = "Please water the plants."
    result = metric(FakeExample(source), FakePrediction(source))

    assert result.score == 0.0
    assert state.check_failures == 1


# --- Reported means exclude what could not be judged -------------------


def test_unjudged_examples_are_excluded_from_the_pool_mean(monkeypatch, config):
    metric, state = _metric_with_judge(
        monkeypatch,
        config,
        [_verdict(10.0), _verdict(None, needs_review=True), _verdict(8.0)],
    )
    examples = [FakeExample("a"), FakeExample("b"), FakeExample("c")]
    program = FakeProgram({"a": "uno", "b": "dos", "c": "tres"})

    score = gepa_loop._score_pool(program, examples, metric, state, "test")

    # Mean of 1.0 and 0.8 only. Averaging the failure in as a zero would
    # report 0.6 and understate the prompt by a third.
    assert score.mean == pytest.approx(0.9)
    assert score.scored == 2
    assert score.unjudged == 1
    assert score.total == 3


def test_empty_translations_are_included_in_the_pool_mean(monkeypatch, config):
    metric, state = _metric_with_judge(monkeypatch, config, [_verdict(10.0)])
    examples = [FakeExample("a"), FakeExample("b")]
    program = FakeProgram({"a": "uno", "b": ""})

    score = gepa_loop._score_pool(program, examples, metric, state, "test")

    # 1.0 and a real 0.0: the empty output is the program's failure.
    assert score.mean == pytest.approx(0.5)
    assert score.scored == 2
    assert score.unjudged == 0


def test_pool_records_are_labelled_with_their_pool(monkeypatch, config):
    metric, state = _metric_with_judge(monkeypatch, config, [_verdict(9.0)])
    gepa_loop._score_pool(
        FakeProgram({"a": "uno"}), [FakeExample("a")], metric, state, "sealed_test"
    )
    assert state.per_example[0]["pool"] == "sealed_test"


# --- A broken judge stops the run rather than producing a number ------


def test_run_aborts_above_the_judge_failure_threshold():
    state = RunState()
    state.metric_calls = 10
    state.judge_failures = 2  # 20%, above the 10% limit

    with pytest.raises(RuntimeError, match="Fix the judge"):
        gepa_loop._abort_if_judge_unreliable(state)


def test_run_continues_below_the_threshold():
    state = RunState()
    state.metric_calls = 100
    state.judge_failures = 5
    gepa_loop._abort_if_judge_unreliable(state)  # must not raise


def test_threshold_is_not_tripped_by_an_empty_run():
    gepa_loop._abort_if_judge_unreliable(RunState())


def test_failure_rate_counts_both_failure_kinds():
    state = RunState()
    state.metric_calls = 10
    state.judge_failures = 1
    state.empty_translations = 1
    assert state.failure_rate == pytest.approx(0.2)
    assert MAX_JUDGE_FAILURE_RATE == 0.10


# --- Reporting ---------------------------------------------------------


def test_pool_score_with_nothing_judged_reports_none_not_zero():
    score = PoolScore(mean=None, scored=0, unjudged=3, total=3)
    assert score.as_dict()["mean"] is None, "no data is not the same as a score of zero"


def test_config_fingerprint_is_stable_and_order_independent():
    a = gepa_loop._config_fingerprint({"judge": "x", "scoring_method": "flat"})
    b = gepa_loop._config_fingerprint({"scoring_method": "flat", "judge": "x"})
    c = gepa_loop._config_fingerprint({"judge": "y", "scoring_method": "flat"})
    assert a == b, "key order must not change the fingerprint"
    assert a != c, "a different config must produce a different fingerprint"
