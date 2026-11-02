"""
Deterministic output checks: the failures a judge should not be asked
to catch.

An LLM judge scored an output left entirely in English at ten out of
ten, and another written entirely in Traditional characters at ten out
of ten. Both were caught instantly by a string comparison. The lesson is
not that the judge is bad — it is that asking a probabilistic grader to
police a deterministic property is the wrong tool for the job.

Everything here is a string operation: no model, no API call, no cost,
no variance. The checks are also the *cheapest* thing in the pipeline
and catch the failure class that costs the most, so they run first and
their verdict is not negotiable by a judge's opinion.

What these checks deliberately cannot do:

  * The Cantonese marker set is a curated list of high-signal
    characters, not a linguistic model. It catches vernacular Cantonese
    written in Chinese characters; it will not catch Cantonese phrased
    entirely in words shared with Mandarin, and it cannot distinguish
    a deliberate quotation from an error.
  * The Latin-token check exempts acronyms and anything already present
    in the source, which means a proper noun the model invented in
    Latin script passes. That is the right trade: flagging every
    legitimately-kept name would make the check unusable.
  * Script detection tells you a character has a Simplified form that
    differs. A handful of characters are genuinely shared or ambiguous,
    so a single flagged character is weaker evidence than a sentence
    full of them, and the count is reported for that reason.
  * Number preservation only sees numerals. A quantity spelled out in
    words in either language is invisible to it, so "twelve" rendered
    as 12 passes unexamined and so does 1st rendered as primero.
  * Truncation only reports endings that cannot finish a sentence. An
    output missing its final full stop is not reported, because models
    drop that mark on complete sentences often enough that failing them
    would cost more good translations than it caught truncated ones.

Every check here errs toward passing. A false positive scores a correct
translation zero and feeds the optimizer a reason that is not true,
which is worse than missing a failure the judge may still catch.

These limits are written down rather than discovered later, because a
check whose blind spots are undocumented is how a gate ends up under-
reporting its own failure rate while looking like it works.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Characters that appear in vernacular written Cantonese and effectively
# never in Standard Written Chinese. Curated and deliberately short:
# every addition trades a missed case for a false positive, and a false
# positive on this check blocks a correct translation.
CANTONESE_MARKERS = frozenset("係佢嘅冇咗喺唔啲嗰乜嘢咁噉睇畀嚟攞諗郁掂嘥揾邊咩")

# Latin-script tokens that are legitimate inside Chinese output. Kept
# tiny on purpose; anything in the source text is exempted separately,
# which covers the vast majority of real cases.
_ALWAYS_ALLOWED_LATIN = frozenset({"ok", "tv", "dna", "ai", "app", "email", "wifi"})

_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_CJK = re.compile(r"[一-鿿㐀-䶿]")

# One number, including any grouping or decimal marks inside it. The
# separators are stripped afterwards rather than parsed, so no locale's
# convention has to be hardcoded.
_NUMBER_RUN = re.compile(r"\d(?:[\d.,]*\d)?")

_CHINESE_DIGITS = "零一二三四五六七八九"

# Chinese scale characters. Their presence means the translation is
# grouping by ten thousand rather than by thousand, which makes a
# digit-level comparison against a Western-grouped source meaningless.
_CJK_SCALE = re.compile(r"[万萬亿億千百]")

# Characters that cannot end a finished sentence in any language here.
# A missing full stop is deliberately absent: models drop it routinely
# on complete output, so treating that as truncation would fail good
# translations far more often than it caught cut-off ones.
_MID_CLAUSE_ENDINGS = frozenset(",;:，、；：-–—/")

# A leading list or paragraph marker: "37.", "1.2", "j.", "- ", "• ".
# Stripped before numbers are compared because a numbering prefix on
# one side only is a property of the document, not of the translation.
_ENUMERATION_PREFIX = re.compile(r"^\s*(?:[-•*]\s*)?(?:\d+(?:\.\d+)*|[a-zA-Z])[.、)]\s+")
_PAIRED = {"(": ")", "[": "]", "{": "}", "（": "）", "「": "」", "『": "』", "《": "》"}

# Three or more consecutive repeats of the same fragment. Three rather
# than two so Chinese reduplication and ordinary doubling pass.
_LOOP = re.compile(r"(\S.{1,39}?)\1{2,}")

_SENTENCE_SPLIT = re.compile(r"[.!?。！？]+")

_ASCII_PUNCT = frozenset(",.!?;:")


@dataclass
class CheckResult:
    """One check's verdict on one output."""

    name: str
    passed: bool
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class CheckReport:
    """Every check's verdict on one output, plus the overall answer."""

    results: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed_checks": [r.name for r in self.failures],
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                    "evidence": r.evidence,
                }
                for r in self.results
            ],
        }


def _normalize(text: str) -> str:
    """Strip case, punctuation and spacing, for comparisons that should ignore them."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def check_simplified_script(text: str) -> CheckResult:
    """
    Flag Traditional characters in output that should be Simplified.

    Uses OpenCC's t2s conversion: any character that changes under it
    has a distinct Simplified form and therefore was not written in
    Simplified. This is the check that catches a fluent, grammatical,
    entirely wrong-script translation — which a semantic judge reads as
    perfectly good Chinese, because it is.
    """
    try:
        from opencc import OpenCC
    except ImportError:
        return CheckResult(
            "simplified_script",
            True,
            "skipped: opencc is not installed, so script was not checked",
        )

    converted = OpenCC("t2s").convert(text)
    if converted == text:
        return CheckResult("simplified_script", True, "no Traditional characters found")

    differing = [orig for orig, conv in zip(text, converted, strict=False) if orig != conv]
    return CheckResult(
        "simplified_script",
        False,
        f"{len(differing)} Traditional character(s) present",
        evidence=sorted(set(differing)),
    )


def check_no_cantonese_markers(text: str) -> CheckResult:
    """Flag vernacular Cantonese characters in output that should be Mandarin."""
    found = sorted({ch for ch in text if ch in CANTONESE_MARKERS})
    if not found:
        return CheckResult("no_cantonese_markers", True, "no Cantonese markers found")
    return CheckResult(
        "no_cantonese_markers",
        False,
        f"{len(found)} Cantonese marker character(s) present",
        evidence=found,
    )


def check_no_residual_latin(text: str, source: str) -> CheckResult:
    """
    Flag Latin-script words left untranslated inside CJK output.

    Tokens present in the source are exempt, since a proper noun kept
    from the source is correct behavior, not leakage. Acronyms are
    exempt for the same reason. What remains is the pattern that recurs
    in real runs: an ordinary English noun sitting in a Chinese
    sentence, which a ratio-based gate passes because the output is
    still overwhelmingly CJK.
    """
    if not _CJK.search(text):
        return CheckResult(
            "no_residual_latin", True, "output is not CJK; this check does not apply"
        )

    source_tokens = {t.casefold() for t in _LATIN_TOKEN.findall(source)}
    leaked = [
        token
        for token in _LATIN_TOKEN.findall(text)
        if token.casefold() not in source_tokens
        and token.casefold() not in _ALWAYS_ALLOWED_LATIN
        and not token.isupper()  # acronyms
        and len(token) > 1
    ]
    if not leaked:
        return CheckResult("no_residual_latin", True, "no untranslated Latin words found")
    return CheckResult(
        "no_residual_latin",
        False,
        f"{len(leaked)} Latin word(s) not present in the source",
        evidence=sorted(set(leaked)),
    )


def check_not_source_copy(text: str, source: str) -> CheckResult:
    """
    Flag output that is just the source handed back.

    Checked on normalized text so that a copy with different spacing or
    punctuation still counts as a copy.
    """
    if not _normalize(text):
        return CheckResult("not_source_copy", False, "output is empty")
    if _normalize(text) == _normalize(source):
        return CheckResult(
            "not_source_copy", False, "output is identical to the source", evidence=[source[:80]]
        )
    return CheckResult("not_source_copy", True, "output differs from the source")


def check_not_empty(text: str) -> CheckResult:
    if text and text.strip():
        return CheckResult("not_empty", True, "output is non-empty")
    return CheckResult("not_empty", False, "output is empty or whitespace only")


def _digit_runs(text: str) -> list[str]:
    """
    Every number in the text, reduced to its digits.

    Grouping and decimal marks are stripped rather than interpreted,
    because they are exactly what differs between locales: 1,000 and
    1.000 are the same quantity, and 3.5 and 3,5 are too. Comparing
    digits sidesteps having to know which convention a language uses.

    A leading list or paragraph marker is removed first. Measured
    against real reference translations, a numbering prefix present on
    one side and not the other was the single largest source of false
    reports, and it is a property of the document rather than of the
    translation.
    """
    normalized = _ENUMERATION_PREFIX.sub("", unicodedata.normalize("NFKC", text).strip())
    runs = []
    for run in _NUMBER_RUN.findall(normalized):
        # Two or more separators means a date or a reference like
        # 28.10.04, not one quantity. Joining those digits invents a
        # number that appears in neither text.
        parts = [p for p in re.split(r"[.,]", run) if p]
        runs.extend(parts if len(parts) > 2 else [re.sub(r"\D", "", run)])
    return runs


def _as_chinese_numeral(digits: str) -> str | None:
    """
    Render a small integer the way Chinese writes it, or None.

    Needed because a correct translation of "12" is as likely to be
    十二 as 12, and flagging that would block a good translation. Only
    values a translator plausibly spells out are covered; anything
    larger is left to the digit comparison.
    """
    try:
        value = int(digits)
    except ValueError:
        return None
    if not 0 <= value <= 99:
        return None
    if value < 10:
        return _CHINESE_DIGITS[value]
    tens, ones = divmod(value, 10)
    return (
        ("" if tens == 1 else _CHINESE_DIGITS[tens])
        + "十"
        + ("" if ones == 0 else _CHINESE_DIGITS[ones])
    )


def check_numbers_preserved(text: str, source: str) -> CheckResult:
    """
    Flag numbers present in the source and missing from the translation.

    A wrong or dropped number is a serious error that reads as
    perfectly fluent, which is the combination a semantic judge is
    worst at. It is also one of the few meaning-level properties that
    can be settled without a model.

    Comparison is on digits only, so locale differences in grouping and
    decimal marks do not register, and a missing number is additionally
    tested against its Chinese spelled-out form.

    Only the missing direction is reported. Numbers appearing in the
    translation and not the source were measured on real reference
    translations and are dominated by document numbering, footnote
    markers and date reordering rather than by invented quantities, so
    reporting them would fail far more correct translations than it
    would catch hallucinated ones.

    A translation containing 万, 亿, 千 or 百 is passed without
    comparison. Chinese groups by ten thousand rather than by thousand,
    so 29 million is correctly written 2 900万 and 130 million as
    1亿3千万 — the same quantity carrying different digits. On real
    Mandarin references this one difference produced almost every
    remaining report, and no digit comparison can resolve it.
    """
    in_source = _digit_runs(source)
    if not in_source:
        return CheckResult("numbers_preserved", True, "no numerals in the source")

    if _CJK_SCALE.search(text):
        return CheckResult(
            "numbers_preserved",
            True,
            "translation uses Chinese scale characters; digits are not comparable",
        )

    remaining = _digit_runs(text)
    missing = []
    for number in in_source:
        if number in remaining:
            remaining.remove(number)
            continue
        spelled = _as_chinese_numeral(number)
        if spelled and spelled in text:
            continue
        missing.append(number)

    if not missing:
        return CheckResult(
            "numbers_preserved", True, f"all {len(in_source)} numeral(s) accounted for"
        )
    return CheckResult(
        "numbers_preserved",
        False,
        f"{len(missing)} numeral(s) in the source and not the translation",
        evidence=sorted(set(missing)),
    )


def check_not_truncated(text: str, source: str = "") -> CheckResult:
    """
    Flag output that stops mid-clause when the source did not.

    Only endings that cannot finish a sentence are considered: a
    trailing comma or connector, or a bracket or quote opened and never
    closed. A missing full stop is not treated as truncation, because
    models routinely drop the final mark on complete output.

    The source is compared too. A list item, a preambular clause or a
    sentence introducing a list legitimately ends on a semicolon or a
    colon, and a translation that ends the same way has preserved the
    source rather than been cut off. Measured on real references, every
    reported truncation was of this kind.
    """
    stripped = text.strip()
    if not stripped:
        return CheckResult("not_truncated", True, "empty output is covered by not_empty")

    source_stripped = source.strip()
    source_ends_open = bool(source_stripped) and source_stripped[-1] in _MID_CLAUSE_ENDINGS

    if stripped[-1] in _MID_CLAUSE_ENDINGS and not source_ends_open:
        return CheckResult(
            "not_truncated",
            False,
            f"output ends on {stripped[-1]!r} where the source does not",
            evidence=[stripped[-40:]],
        )

    unclosed = [
        f"{opener}{closer}"
        for opener, closer in _PAIRED.items()
        if stripped.count(opener) > stripped.count(closer)
        and source.count(opener) <= source.count(closer)
    ]
    if unclosed:
        return CheckResult(
            "not_truncated",
            False,
            f"{len(unclosed)} bracket or quote pair opened and not closed",
            evidence=unclosed,
        )

    return CheckResult("not_truncated", True, "output ends on a complete clause")


def check_no_degeneration(text: str) -> CheckResult:
    """
    Flag looping output: the same fragment emitted three or more times
    in a row, or the same sentence repeated.

    Small models under a constrained token budget fail this way, and
    the result is fluent enough per fragment that a judge reading for
    meaning can score it generously.

    Three repetitions is the threshold rather than two, so that Chinese
    reduplication (看看, 高高兴兴) and ordinary English doubling are not
    reported.
    """
    collapsed = re.sub(r"\s+", " ", text.strip())
    if not collapsed:
        return CheckResult("no_degeneration", True, "empty output is covered by not_empty")

    loop = _LOOP.search(collapsed)
    if loop:
        return CheckResult(
            "no_degeneration",
            False,
            f"fragment repeated {len(loop.group(0)) // len(loop.group(1))} times in a row",
            evidence=[loop.group(1)],
        )

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(collapsed) if len(s.strip()) > 3]
    repeated = {s for s in sentences if sentences.count(s) > 1}
    if repeated:
        return CheckResult(
            "no_degeneration",
            False,
            f"{len(repeated)} sentence(s) emitted more than once",
            evidence=sorted(repeated)[:3],
        )

    return CheckResult("no_degeneration", True, "no repeated fragments found")


def check_cjk_punctuation(text: str) -> CheckResult:
    """
    Flag ASCII punctuation used where CJK text requires fullwidth marks.

    A native reader sees this immediately and a judge scoring for
    meaning does not see it at all, which is what makes it worth a
    deterministic check rather than a rubric line.

    Only punctuation directly touching a CJK character is reported, so
    a decimal point inside 3.5, a kept acronym, and a URL are all left
    alone.
    """
    if not _CJK.search(text):
        return CheckResult("cjk_punctuation", True, "output is not CJK; this check does not apply")

    found = []
    for i, ch in enumerate(text):
        if ch not in _ASCII_PUNCT:
            continue
        before = text[i - 1] if i else ""
        after = text[i + 1] if i + 1 < len(text) else ""
        if _CJK.match(before) or _CJK.match(after):
            found.append(ch)

    if not found:
        return CheckResult("cjk_punctuation", True, "punctuation is fullwidth throughout")
    return CheckResult(
        "cjk_punctuation",
        False,
        f"{len(found)} halfwidth mark(s) adjacent to CJK text",
        evidence=sorted(set(found)),
    )


# Every check the pipeline knows how to run, by the name a language
# config uses to request it. Adding a check means adding an entry here
# and naming it in a config — never editing a language's code path,
# because there isn't one.
CHECK_REGISTRY = {
    "not_empty": lambda text, source: check_not_empty(text),
    "not_source_copy": lambda text, source: check_not_source_copy(text, source),
    "simplified_script": lambda text, source: check_simplified_script(text),
    "no_cantonese_markers": lambda text, source: check_no_cantonese_markers(text),
    "no_residual_latin": lambda text, source: check_no_residual_latin(text, source),
    "numbers_preserved": lambda text, source: check_numbers_preserved(text, source),
    "not_truncated": lambda text, source: check_not_truncated(text, source),
    "no_degeneration": lambda text, source: check_no_degeneration(text),
    "cjk_punctuation": lambda text, source: check_cjk_punctuation(text),
}

# Applied to every language whether or not its config asks. An empty
# output and a copied source are failures in any language.
UNIVERSAL_CHECKS = ("not_empty", "not_source_copy")


def run_checks(text: str, source: str, config: dict) -> CheckReport:
    """
    Run the checks a language's config asks for, plus the universal ones.

    Which checks apply is a property of the target language, so it lives
    in that language's config rather than in a branch here. A config
    naming an unknown check fails loudly: silently skipping a check
    someone believed was running is exactly how a detection gap goes
    unnoticed.
    """
    requested = list(config.get("output_checks") or [])
    names = list(UNIVERSAL_CHECKS) + [n for n in requested if n not in UNIVERSAL_CHECKS]

    unknown = [n for n in names if n not in CHECK_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown output_checks {unknown} in language config; "
            f"available: {sorted(CHECK_REGISTRY)}"
        )

    return CheckReport([CHECK_REGISTRY[name](text, source) for name in names])
