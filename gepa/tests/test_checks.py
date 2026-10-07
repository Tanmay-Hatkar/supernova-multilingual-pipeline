"""
Tests for the deterministic output checks.

The fixtures are failure patterns observed in real evaluation output,
not invented ones. Each of these passes an automated gate that counts
CJK characters, and several were scored ten out of ten by an LLM judge.
If these tests pass, those patterns can no longer slip through.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checks import (
    CHECK_REGISTRY,
    check_no_cantonese_markers,
    check_no_residual_latin,
    check_not_empty,
    check_not_source_copy,
    check_simplified_script,
    run_checks,
)


# --- Script -------------------------------------------------------------

def test_simplified_output_passes_script_check():
    assert check_simplified_script("博物馆的新展览下个月开幕。").passed


def test_fully_traditional_output_is_flagged():
    # The kind of output a semantic judge scores 10/10: fluent,
    # grammatical, and entirely the wrong script.
    result = check_simplified_script("博物館的新展覽下個月開幕。")
    assert not result.passed
    assert result.evidence


def test_single_traditional_character_is_flagged():
    # One Traditional character in otherwise Simplified output, which no
    # ratio-based heuristic can see.
    result = check_simplified_script("你拿到钱了")
    assert result.passed  # this one is already Simplified
    result = check_simplified_script("你拿到錢了")
    assert not result.passed
    assert "錢" in result.evidence


def test_latin_text_is_unaffected_by_the_script_check():
    assert check_simplified_script("Estuve con ella antes.").passed


# --- Chinese variety ----------------------------------------------------

def test_cantonese_vernacular_is_flagged():
    # Valid Chinese characters, wrong variety.
    result = check_no_cantonese_markers("我今日冇時間搞掂呢件事")
    assert not result.passed
    assert "冇" in result.evidence


def test_standard_mandarin_passes_the_variety_check():
    assert check_no_cantonese_markers("我今天没有时间处理这件事。").passed


# --- Residual Latin -----------------------------------------------------

def test_english_noun_left_inside_chinese_is_flagged():
    # A single English noun inside Chinese: a ratio-based gate sees one
    # Latin word and passes it.
    result = check_no_residual_latin(
        "你最好不要 obsession 在这件事上", "You'd better not be so fixated on this."
    )
    assert not result.passed
    assert "obsession" in result.evidence


def test_proper_noun_kept_from_the_source_is_not_leakage():
    result = check_no_residual_latin("Tim 昨天来了。", "Tim came yesterday.")
    assert result.passed


def test_acronyms_are_not_flagged():
    result = check_no_residual_latin("联合国和 NATO 发表了声明。", "A statement was issued.")
    assert result.passed


def test_clean_chinese_output_passes():
    result = check_no_residual_latin("会议将于下周举行。", "The meeting will be held next week.")
    assert result.passed


def test_non_cjk_output_skips_the_latin_check():
    # Spanish output is all Latin script; the check must not fire on it.
    result = check_no_residual_latin("Estuve con ella antes.", "I was with her earlier.")
    assert result.passed


# --- Source copy and empty ---------------------------------------------

def test_untranslated_source_copy_is_flagged():
    source = "Please remember to water the plants while I'm away."
    assert not check_not_source_copy(source, source).passed


def test_source_copy_is_caught_despite_punctuation_differences():
    assert not check_not_source_copy("7 francs", "7 francs.").passed


def test_a_real_translation_is_not_a_source_copy():
    assert check_not_source_copy("Estuve con ella antes.", "I was with her earlier.").passed


def test_empty_output_is_flagged():
    assert not check_not_empty("   ").passed
    assert check_not_empty("algo").passed


# --- Configuration ------------------------------------------------------

def test_universal_checks_run_even_when_config_asks_for_nothing():
    report = run_checks("", "some source", {})
    assert not report.passed
    assert "not_empty" in [r.name for r in report.failures]


def test_config_selects_which_checks_apply():
    mandarin = {"output_checks": ["simplified_script", "no_cantonese_markers"]}
    report = run_checks("博物館的新展覽", "The museum's new exhibit", mandarin)
    assert not report.passed
    assert "simplified_script" in report.as_dict()["failed_checks"]

    # The same output under a config that does not ask for script
    # checking passes, because which checks apply is a property of the
    # target language, not of this code.
    spanish = {"output_checks": []}
    assert run_checks("博物館的新展覽", "The museum's new exhibit", spanish).passed


def test_unknown_check_name_fails_loudly():
    with pytest.raises(ValueError, match="Unknown output_checks"):
        run_checks("texto", "text", {"output_checks": ["no_such_check"]})


def test_every_registered_check_is_callable_with_the_same_signature():
    for name, check in CHECK_REGISTRY.items():
        result = check("你好", "hello")
        assert result.name, f"{name} returned a result with no name"
