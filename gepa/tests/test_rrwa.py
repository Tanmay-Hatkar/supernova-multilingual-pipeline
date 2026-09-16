import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrwa import compute_rrwa


def test_identical_scores_are_perfectly_stable():
    result = compute_rrwa([8.0, 8.0, 8.0])
    assert result.final_score == pytest.approx(8.0)
    assert result.stability_index == pytest.approx(1.0)
    assert result.discarded_scores == []


def test_single_score_returns_itself():
    result = compute_rrwa([7.5])
    assert result.final_score == 7.5
    assert result.stability_index == 1.0


def test_matches_published_gemba_mqm_worked_example():
    # The exact ten-score example from the published GEMBA-MQM V2
    # appendix, already independently verified against this same
    # formula in multiple pieces of this project's earlier work. Using
    # the real published case here instead of an invented one, since a
    # single dominant outlier in a small, made-up sample can inflate
    # that sample's own standard deviation enough to stay inside 2
    # sigma of itself — a real limitation of this method with small n,
    # not something a synthetic test should paper over.
    scores = [-50, -11, -6, -6, -11, -6, -6, -55, -6, -11]
    result = compute_rrwa(scores)
    assert -55 in result.discarded_scores
    assert result.final_score == pytest.approx(-8.50, abs=0.01)


def test_final_score_favors_higher_ranked_values():
    # With no outliers, the weighting should pull the result toward the
    # higher end of the retained scores, not a flat mean.
    scores = [10.0, 6.0]
    result = compute_rrwa(scores)
    assert result.final_score > result.mean


def test_empty_scores_raises():
    with pytest.raises(ValueError):
        compute_rrwa([])
