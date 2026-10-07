"""
Estimate how hard a sentence is to translate, without calling a model.

Why this exists. This pipeline stratified its data by sentence length,
which is not difficulty. A long, plainly-worded news sentence is easier
to translate than a six-word idiom. With the judge's blind spot fixed,
half the remaining ceiling effect is simply that the examples are easy:
short conversational lines that a current model handles well, so the
scores are high because the translations genuinely are good.

An optimizer learns nothing from examples every candidate prompt gets
right. It needs headroom, which means examples that are hard for the
*translator*, not merely long.

## This heuristic failed its own validation. Read before using it.

Measured against judge scores actually observed on 20 Spanish examples,
the features below correlate at **Spearman 0.065** — indistinguishable
from selecting at random. Idioms, clause density, named entities and
rare vocabulary are genuine sources of translation error in general,
and they do not predict what *this* model finds hard.

chrF++ against the reference, by contrast, correlates with judge scores
at 0.49 on the same pipeline. It costs one translation call per
candidate, which is not free, but it predicts roughly seven times
better than everything in this module.

So: `--stratify chrf` is the option with evidence behind it.
`--stratify difficulty` is kept because the code is written and tested
and a better feature set may yet be found, but it should not be used
for real selection until `run.py difficulty --validate N` reports a
clearly negative correlation.

Keeping this module rather than deleting it is deliberate. The
validation command is the valuable part, and a repo that silently
removes its failed hypotheses invites the next person to retry them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# English idioms and phrasal verbs whose literal translation is wrong in
# most target languages. Short and high-signal on purpose: the point is
# to find sentences worth including, not to catalogue English.
IDIOM_MARKERS = (
    "goes without saying",
    "touch base",
    "knack for",
    "getting lucky",
    "make things better",
    "hope for the best",
    "in the loop",
    "up to speed",
    "get away with",
    "look forward to",
    "run into",
    "figure out",
    "come across",
    "bring up",
    "put up with",
    "turn down",
    "call off",
    "break down",
    "once and for all",
    "by and large",
    "at the end of the day",
    "on the other hand",
    "no matter what",
    "as far as",
    "let alone",
    "kill two birds",
    "piece of cake",
    "under the weather",
    "hit the road",
)

# Words that signal a subordinate or conditional structure, which force
# clause reordering in most target languages.
SUBORDINATORS = (
    "although",
    "though",
    "whereas",
    "unless",
    "whilst",
    "despite",
    "nevertheless",
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "which",
    "whose",
    "wherein",
    "thereby",
    "insofar",
)

NEGATIONS = ("not", "no", "never", "neither", "nor", "none", "n't", "without")

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
# A capitalised word that is not sentence-initial: a reasonable proxy
# for a named entity without a tagger.
_MID_CAPITAL = re.compile(r"(?<=[a-z,;:]\s)([A-Z][a-z]{2,})")


@dataclass
class DifficultyScore:
    total: float
    features: dict[str, float] = field(default_factory=dict)

    @property
    def band(self) -> str:
        if self.total >= 6.0:
            return "hard"
        if self.total >= 3.0:
            return "medium"
        return "easy"


def score_difficulty(source: str) -> DifficultyScore:
    """
    A weighted count of features known to cause translation errors.

    Weights are judgement, not fitted parameters — there is no labelled
    difficulty data to fit them to. They are set so that a single idiom
    or a negated conditional is worth more than a merely long sentence,
    which is the ordering the length-based stratification got wrong.
    """
    text = source.strip()
    lower = text.lower()
    words = _WORD.findall(text)
    if not words:
        return DifficultyScore(0.0, {})

    features: dict[str, float] = {}

    # Idioms: the single strongest signal. A literal rendering is
    # usually wrong, so these are exactly the cases where one prompt
    # can beat another.
    idioms = sum(1 for marker in IDIOM_MARKERS if marker in lower)
    features["idioms"] = idioms * 3.0

    # Clause density. Counted as clauses rather than characters, so a
    # long simple sentence does not outrank a short tangled one.
    clauses = len(re.findall(r"[,;:]", text)) + sum(
        1 for s in SUBORDINATORS if re.search(rf"\b{s}\b", lower)
    )
    features["clause_density"] = min(clauses, 5) * 0.8

    # Negation, which is where meaning inverts if handled wrongly.
    negations = sum(1 for n in NEGATIONS if re.search(rf"\b{re.escape(n)}\b", lower))
    features["negation"] = min(negations, 3) * 0.7

    # Named entities and numbers: facts a translator can get
    # confidently wrong, and a failure type that recurs in real runs.
    entities = len(set(_MID_CAPITAL.findall(text)))
    features["named_entities"] = min(entities, 4) * 0.8
    features["numbers"] = min(len(_NUMBER.findall(text)), 3) * 0.5

    # Rare or technical vocabulary, approximated by word length since
    # there is no frequency list here.
    long_words = sum(1 for w in words if len(w) >= 10)
    features["rare_vocabulary"] = min(long_words, 5) * 0.6

    # Length still counts, but weakly, and only past the point where a
    # sentence has real structure to get wrong.
    features["length"] = min(max(len(words) - 12, 0) / 12.0, 1.5)

    return DifficultyScore(round(sum(features.values()), 3), features)


DIFFICULTY_BANDS = ("easy", "medium", "hard")


def band_of(source: str) -> str:
    return score_difficulty(source).band


def distribution(sources: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(DIFFICULTY_BANDS, 0)
    for source in sources:
        counts[band_of(source)] += 1
    return counts


def select_stratified(
    candidates: list[dict], per_band: int, rng, key: str = "source"
) -> list[dict]:
    """
    Take an equal number from each difficulty band.

    Equal rather than hardest-first on purpose. A set of only hard
    examples cannot show that a prompt has stopped handling easy ones,
    and a prompt that wins on hard cases by mangling simple ones is a
    regression the optimizer would happily select.
    """
    buckets: dict[str, list[dict]] = {band: [] for band in DIFFICULTY_BANDS}
    for candidate in candidates:
        buckets[band_of(candidate[key])].append(candidate)

    selected: list[dict] = []
    for band in DIFFICULTY_BANDS:
        available = buckets[band]
        rng.shuffle(available)
        taken = available[:per_band]
        if len(taken) < per_band:
            print(f"  note: only {len(taken)} '{band}' examples available, wanted {per_band}")
        for candidate in taken:
            candidate["difficulty_band"] = band
            candidate["difficulty_score"] = score_difficulty(candidate[key]).total
        selected.extend(taken)
    return selected
