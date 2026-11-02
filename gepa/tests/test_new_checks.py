"""
Tests for the four checks added after the deterministic set proved
cheaper and more reliable than the judge at the failures it misses.

Each check has two test groups, and the second matters more. A missed
failure leaves the judge a chance to catch it. A false positive scores
a correct translation zero and hands the optimizer a written reason
that is not true, which steers prompt rewriting toward fixing a
problem that was never there.

So the "must not fire" cases here are not padding. They are the cases
that nearly made each check unusable: locale decimal marks, Chinese
spelled-out numerals, a dropped full stop, Chinese reduplication, and
a decimal point inside a number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from checks import (
    CHECK_REGISTRY,
    check_cjk_punctuation,
    check_no_degeneration,
    check_not_truncated,
    check_numbers_preserved,
    run_checks,
)

# --- numbers: must fire -----------------------------------------------


def test_a_changed_number_is_caught():
    result = check_numbers_preserved("Llegaré a las 5.", "I will arrive at 9.")
    assert not result
    assert "9" in result.evidence


def test_a_dropped_number_is_caught():
    result = check_numbers_preserved("Costó mucho dinero.", "It cost 400 dollars.")
    assert not result
    assert "400" in result.evidence


def test_a_transposed_number_is_caught():
    result = check_numbers_preserved("Hay 21 personas.", "There are 12 people.")
    assert not result


# --- numbers: must not fire -------------------------------------------


def test_matching_numerals_pass():
    assert check_numbers_preserved("Llegaré a las 9 en punto.", "I will arrive at 9 sharp.")


def test_locale_grouping_marks_are_not_a_mismatch():
    # 1,000 and 1.000 are the same quantity written two ways. Comparing
    # digits avoids having to know which language uses which.
    assert check_numbers_preserved("Costó 1.000 dólares.", "It cost 1,000 dollars.")


def test_locale_decimal_marks_are_not_a_mismatch():
    assert check_numbers_preserved("Pesa 3,5 kilos.", "It weighs 3.5 kilos.")


@pytest.mark.parametrize(
    "translation,source",
    [
        ("十二个人", "12 people"),
        ("五本书", "5 books"),
        ("二十本书", "20 books"),
        ("三十五个", "35 of them"),
    ],
)
def test_a_number_spelled_out_in_chinese_is_not_a_mismatch(translation, source):
    # The false positive that would have made this check unusable for
    # Mandarin: writing 12 as 十二 is correct, not a dropped number.
    assert check_numbers_preserved(translation, source)


def test_separate_numbers_are_not_merged_into_one():
    assert check_numbers_preserved("Elige 1, 2 o 3.", "Pick 1, 2 or 3.")


def test_text_with_no_numbers_at_all_passes():
    assert check_numbers_preserved("Hola, mundo.", "Hello, world.")


def test_a_number_only_in_the_translation_is_not_reported():
    # Deliberate, and measured. Extra numbers on real references are
    # dominated by paragraph numbering, footnote markers and date
    # reordering rather than by invented quantities, so reporting them
    # failed more correct translations than it caught wrong ones.
    assert check_numbers_preserved("37. Costó mucho dinero.", "It cost a lot of money.")


def test_a_paragraph_number_on_one_side_only_is_not_a_mismatch():
    # The largest single source of false reports before it was handled:
    # UN-style documents carry a numbering prefix on one side only.
    assert check_numbers_preserved("37. Hay 9 personas.", "There are 9 people.")
    assert check_numbers_preserved("Hay 9 personas.", "37. There are 9 people.")


def test_a_dotted_date_is_split_rather_than_concatenated():
    # 28.10.04 is three components. Joining their digits invented the
    # number 281004, which appeared in neither text, and reported it as
    # missing. The components are now compared individually.
    #
    # A two-digit year written out in full on the other side (04 against
    # 2004) is still reported. That is a known and accepted limit: it
    # occurred once in 496 reference translations, and the machinery to
    # resolve it would be guessing which century a number belongs to.
    result = check_numbers_preserved("2004年10月28日。", "Speech of 28.10.04.")
    assert "281004" not in result.evidence
    assert "28" not in result.evidence
    assert "10" not in result.evidence


@pytest.mark.parametrize(
    "translation,source",
    [
        ("2 900万起新感染", "prevent 29 million new infections"),
        ("1亿3千万年", "130 million years"),
        ("有200万难民", "2 million refugees"),
    ],
)
def test_chinese_myriad_grouping_is_not_compared(translation, source):
    # Chinese groups by ten thousand, not by thousand, so a correct
    # translation carries different digits for the same quantity. No
    # digit comparison can resolve that, so the check stands down
    # rather than reporting what it cannot prove.
    assert check_numbers_preserved(translation, source)


def test_a_dropped_number_is_still_caught_in_chinese_without_scale_characters():
    # The stand-down above must not disable the check for Mandarin
    # generally.
    assert not check_numbers_preserved(
        "增强妇女权力网络指导小组会议", "Steering Group, 31 May 2007"
    )


# --- truncation: must fire --------------------------------------------


@pytest.mark.parametrize("ending", [",", ";", "，", "、", "：", "-"])
def test_output_ending_mid_clause_is_caught(ending):
    assert not check_not_truncated(f"La reunión empieza y después{ending}")


def test_an_unclosed_bracket_is_caught():
    result = check_not_truncated("Dijo (y esto es importante")
    assert not result
    assert "()" in result.evidence


def test_an_unclosed_cjk_quote_is_caught():
    assert not check_not_truncated("他说「我们明天出发")


# --- truncation: must not fire ----------------------------------------


def test_a_complete_sentence_passes():
    assert check_not_truncated("La reunión empieza a las nueve.")


def test_a_missing_full_stop_is_not_truncation():
    # Deliberate. Models drop the final mark on complete output often
    # enough that failing it would cost more good translations than it
    # would catch truncated ones.
    assert check_not_truncated("La reunión empieza a las nueve")


def test_balanced_brackets_pass():
    assert check_not_truncated("Dijo (y esto es importante) que vendría.")


def test_a_sentence_ending_in_a_closing_quote_passes():
    assert check_not_truncated("他说「我们明天出发」。")


def test_a_source_that_itself_ends_on_a_colon_is_not_truncation():
    # A sentence introducing a list legitimately ends on a colon, and a
    # translation that does the same has preserved the source. Every
    # truncation reported on real Mandarin references was of this kind.
    assert check_not_truncated("意见中还应在适当时表明：", "The opinion shall also state whether:")


def test_a_source_list_item_ending_on_a_semicolon_is_not_truncation():
    assert check_not_truncated("- 法律和政治地位；", "- Legal and political status;")


def test_truncation_is_still_caught_when_the_source_ends_cleanly():
    assert not check_not_truncated("意见中还应在适当时表明：", "The opinion shall also state this.")


# --- degeneration: must fire ------------------------------------------


def test_a_looping_fragment_is_caught():
    result = check_no_degeneration("No puedo ir no puedo ir no puedo ir no puedo ir")
    assert not result


def test_a_looping_cjk_fragment_is_caught():
    assert not check_no_degeneration("我不知道我不知道我不知道")


def test_a_repeated_sentence_is_caught():
    result = check_no_degeneration("Vendré mañana. Hace frío hoy. Vendré mañana.")
    assert not result


# --- degeneration: must not fire --------------------------------------


def test_ordinary_text_passes():
    assert check_no_degeneration("Vendré mañana si hace buen tiempo.")


@pytest.mark.parametrize("text", ["看看这个", "我想慢慢走", "他高高兴兴地回家了", "谢谢你"])
def test_chinese_reduplication_is_not_degeneration(text):
    # Doubling is ordinary Chinese morphology. Requiring three
    # consecutive repeats rather than two is what keeps these passing.
    assert check_no_degeneration(text)


def test_a_word_repeated_twice_for_emphasis_passes():
    assert check_no_degeneration("Muy, muy bien.")


def test_short_repeated_fragments_between_sentences_pass():
    assert check_no_degeneration("Sí. No. Sí. No.")


# --- cjk punctuation: must fire ---------------------------------------


def test_a_halfwidth_comma_after_chinese_is_caught():
    result = check_cjk_punctuation("你好, 世界。")
    assert not result
    assert "," in result.evidence


def test_a_halfwidth_full_stop_after_chinese_is_caught():
    assert not check_cjk_punctuation("我明天去北京.")


def test_a_halfwidth_question_mark_after_chinese_is_caught():
    assert not check_cjk_punctuation("你确定吗?")


# --- cjk punctuation: must not fire -----------------------------------


def test_fullwidth_punctuation_passes():
    assert check_cjk_punctuation("你好，世界。你确定吗？")


def test_a_decimal_point_inside_a_number_passes():
    # The marks are between digits, not touching CJK, so they are not
    # reported.
    assert check_cjk_punctuation("它重3.5公斤，很轻。")


def test_non_cjk_output_is_not_checked():
    assert check_cjk_punctuation("Hello, world. Are you sure?")


# --- registry wiring --------------------------------------------------


@pytest.mark.parametrize(
    "name", ["numbers_preserved", "not_truncated", "no_degeneration", "cjk_punctuation"]
)
def test_each_new_check_is_requestable_by_config(name):
    report = run_checks("你好，世界。", "Hello, world.", {"output_checks": [name]})
    assert name in [r.name for r in report.results]
    assert CHECK_REGISTRY[name]("你好，世界。", "Hello, world.") is not None


def test_a_failing_new_check_fails_the_whole_report():
    report = run_checks(
        "Llegaré a las 5.", "I will arrive at 9.", {"output_checks": ["numbers_preserved"]}
    )
    assert not report.passed
    assert "numbers_preserved" in report.as_dict()["failed_checks"]
