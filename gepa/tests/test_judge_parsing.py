"""
Tests for judge response parsing. These use canned strings simulating
messy real model output, no API call required, mirroring the exact
failure modes already observed live in this project's history.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge import _parse_judge_response


def test_parses_clean_json():
    result = _parse_judge_response('{"score": 7.5, "feedback": "Solid translation."}')
    assert result.score == 7.5
    assert result.feedback == "Solid translation."


def test_parses_json_wrapped_in_markdown_fence():
    raw = '```json\n{"score": 8, "feedback": "Good, minor tone issue."}\n```'
    result = _parse_judge_response(raw)
    assert result.score == 8
    assert "tone" in result.feedback


def test_parses_json_with_prose_before_and_after():
    raw = (
        "Sure, here is my evaluation:\n"
        '{"score": 6, "feedback": "Slightly stiff register."}\n'
        "Let me know if you need anything else."
    )
    result = _parse_judge_response(raw)
    assert result.score == 6


def test_parses_feedback_with_unescaped_quotes():
    raw = '{"score": 5, "feedback": "Uses "muy" correctly but tone is off."}'
    result = _parse_judge_response(raw)
    assert result.score == 5
    assert "muy" in result.feedback


def test_parses_feedback_with_literal_newline():
    raw = '{"score": 4,\n"feedback": "First point.\nSecond point on register."}'
    result = _parse_judge_response(raw)
    assert result.score == 4
    assert "Second point" in result.feedback


def test_raises_clearly_when_no_json_present():
    with pytest.raises(ValueError):
        _parse_judge_response("I think this translation is pretty good, no JSON here.")


def test_raises_when_repaired_json_missing_expected_fields():
    with pytest.raises(ValueError):
        _parse_judge_response('{"rating": 8, "comment": "wrong field names entirely"}')


# --- The judge must be told what language it is looking at -------------
#
# A sensitivity probe found the judge scoring untranslated English
# output ten out of ten, six times out of six, with the mean score
# rising when the Spanish was replaced by the original English. The
# cause was in the prompt: it named neither the target language nor the
# fact that the output was supposed to be in it. These tests pin that
# shut, because the failure is invisible in any output the judge
# produces — it looks like a confident, well-reasoned high score.

from judge import _build_user_prompt


def test_user_prompt_names_the_target_language():
    prompt = _build_user_prompt("Hello there.", "Hola.", "Spanish")
    assert prompt.count("Spanish") >= 2


def test_user_prompt_says_wrong_language_is_a_failure():
    prompt = _build_user_prompt("Hello there.", "Hola.", "Mandarin")
    assert "critical failure" in prompt.lower()
    assert "not in Mandarin" in prompt


def test_user_prompt_still_works_without_a_language():
    # Back-compatible, but deliberately does not pretend to name one.
    prompt = _build_user_prompt("Hello there.", "Hola.", "")
    assert "Hello there." in prompt and "Hola." in prompt
    assert "critical failure" not in prompt.lower()


def test_source_and_translation_are_labelled_distinctly():
    # The judge has to be able to tell which text is which; a probe
    # case that scored 10/10 was the source pasted in as the output.
    prompt = _build_user_prompt("The cat sat.", "El gato se sentó.", "Spanish")
    assert prompt.index("The cat sat.") < prompt.index("El gato se sentó.")
