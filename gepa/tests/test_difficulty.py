"""
Tests for the difficulty heuristic.

These pin the *ordering* the heuristic exists to get right, not the
absolute numbers. The weights are judgement rather than fitted
parameters, so asserting exact scores would lock in arbitrary values
and break on every reasonable tuning. What must hold is that an idiom
outranks a long plain sentence, since getting that backwards is
precisely what length-based stratification did.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from difficulty import (
    DIFFICULTY_BANDS,
    band_of,
    distribution,
    score_difficulty,
    select_stratified,
)

SHORT_IDIOM = "You gotta make things better and just hope for the best."
LONG_PLAIN = (
    "The company announced that the new office building will open in the spring "
    "and will provide space for about four hundred people who currently work "
    "from several smaller locations across the city area."
)
TRIVIAL = "All ads by AutoLease"


def test_a_short_idiom_outranks_a_long_plain_sentence():
    # The ordering length-based stratification got backwards, and the
    # whole reason this module exists.
    assert score_difficulty(SHORT_IDIOM).total > score_difficulty(LONG_PLAIN).total


def test_trivial_text_scores_near_zero():
    assert score_difficulty(TRIVIAL).total < 2.0
    assert band_of(TRIVIAL) == "easy"


def test_empty_input_does_not_crash():
    assert score_difficulty("").total == 0.0
    assert score_difficulty("   ").total == 0.0


def test_negation_raises_the_score():
    plain = "I have time to finish this today."
    negated = "I do not have time to finish this today."
    assert score_difficulty(negated).total > score_difficulty(plain).total


def test_named_entities_and_numbers_raise_the_score():
    plain = "The press reported that the operation included several agents."
    loaded = "The press in Asuncion reported that the operation included 12 agents."
    assert score_difficulty(loaded).total > score_difficulty(plain).total


def test_subordinate_clauses_raise_the_score():
    plain = "The practice of farming marine organisms has raised concerns."
    tangled = (
        "The practice of farming marine organisms, although promising, "
        "has nevertheless raised concerns."
    )
    assert score_difficulty(tangled).total > score_difficulty(plain).total


def test_bands_are_ordered_consistently_with_scores():
    scores = {band: [] for band in DIFFICULTY_BANDS}
    for text in (TRIVIAL, LONG_PLAIN, SHORT_IDIOM):
        result = score_difficulty(text)
        scores[result.band].append(result.total)
    for easy in scores["easy"]:
        for hard in scores["hard"]:
            assert easy < hard


def test_features_are_reported_not_just_a_total():
    # A score you cannot decompose is a score you cannot debug.
    result = score_difficulty(SHORT_IDIOM)
    assert result.features
    assert abs(sum(result.features.values()) - result.total) < 0.01


def test_distribution_counts_every_example_exactly_once():
    sources = [TRIVIAL, LONG_PLAIN, SHORT_IDIOM, "Short one.", "Another short one."]
    counts = distribution(sources)
    assert sum(counts.values()) == len(sources)


def test_stratified_selection_takes_an_even_spread_not_the_hardest():
    # Selecting only hard examples would let a prompt win on them while
    # regressing on simple sentences, with nothing to reveal it.
    candidates = (
        [{"source": TRIVIAL} for _ in range(20)]
        + [{"source": LONG_PLAIN} for _ in range(20)]
        + [{"source": SHORT_IDIOM} for _ in range(20)]
    )
    selected = select_stratified(candidates, per_band=5, rng=random.Random(42))
    bands = distribution([c["source"] for c in selected])
    assert bands["easy"] == 5
    assert bands["hard"] == 5


def test_selection_annotates_each_example_with_its_band_and_score():
    candidates = [{"source": SHORT_IDIOM}, {"source": TRIVIAL}]
    selected = select_stratified(candidates, per_band=1, rng=random.Random(0))
    for item in selected:
        assert item["difficulty_band"] in DIFFICULTY_BANDS
        assert isinstance(item["difficulty_score"], float)
