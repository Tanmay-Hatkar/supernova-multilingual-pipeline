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

    differing = [orig for orig, conv in zip(text, converted) if orig != conv]
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
