# Findings

Measured results from this pipeline, with the reasoning that produced
them. Recorded because two of the conclusions here contradict what the
code was originally built to assume, and a repo that only records its
successes teaches the next person the wrong lessons.

Every number below came from a run in this repository. Where a sample
is too small to support a conclusion, that is stated rather than
rounded away.

---

## 1. The judge could not see wrong-language output

**Symptom.** Untranslated English handed back as the "translation"
scored 10 out of 10, six times out of six. The mean score went *up*
when the Spanish was replaced by the original English source.

**Diagnosis.** Not leniency. On the same run the judge caught every
dropped clause and every truncation, and a punctuation-only control
barely moved. It was one total blind spot with a specific cause: the
prompt sent `Source:` and `Translation:` and never said what language
the translation was supposed to be in. Handed English, the judge
correctly observed that the meaning was perfectly preserved.

**Fix.** Name the target language in the prompt and state that output
in the wrong language is a critical failure regardless of meaning.

**Result.**

| fault | before | after |
|---|---|---|
| untranslated | 0% detected, -0.80 mean drop | 100%, 9.30 |
| entity swap | 0%, 3.09 | 100%, 6.45 |
| clause omission | 100%, 8.24 | 100%, 8.52 |
| truncation | 100%, 9.21 | 100%, 9.22 |
| punctuation (control) | correctly ignored | correctly ignored |

**Why this one matters most.** The failure is invisible in the judge's
output. It produces a confident, well-reasoned, completely wrong high
score. Nothing short of showing the judge a fault you planted yourself
would have caught it.

Reproduce with `python run.py probe --language es`.

---

## 2. Severity-weighted scoring did not help

The pipeline was built expecting MQM severity weighting to beat a flat
0-10 score, on the strength of prior research. Measured on identical
translations, it did not.

| | mean | stdev | share at 9+ | agreement with chrF++ |
|---|---|---|---|---|
| flat | 8.26 | **2.17** | **50%** | **0.49** |
| severity | 8.93 | 1.84 | 75% | 0.36 |

Flat scoring separates the examples *better* and tracks the independent
metric more closely.

**Mechanism.** Severity scoring is winner-take-all: "no errors found"
is exactly 10.0, with no gradation. A flat judge saying 9.64 is
carrying information that a severity judge saying 10.0 has discarded.
Severity weighting sharpens the bottom of the scale at the cost of
flattening the top, and the top is where most examples sit.

**What this is not.** It is not a refutation of MQM. n=12, one
language, one judge model, and severity scoring should still win
wherever serious errors are common enough to separate examples. It is
a finding about this setup, and the honest reading is that the flat
score was never the problem.

Both methods remain implemented and selectable via `scoring_method`.
Flat is the default because it measured better here, not because it
was there first.

Reproduce with `python run.py compare --language es`.

---

## 3. The ceiling effect is in the data, not the scoring

The original concern was that a baseline scoring 0.983 before any
optimization leaves nothing for an optimizer to select on. That was
correct, and the cause was neither of the two things it was assumed to
be.

Fixing the judge prompt alone moved flat scoring from stdev 1.24 to
2.17, and from 75% of examples at 9-or-above down to 50%. Roughly half
the ceiling was the blind spot in finding 1.

What remains is a property of the examples. Short conversational lines
from the OPUS-100 test split are not difficult for a current model, so
the scores are high because the translations are good. The probe
confirms the judge would notice if they were not.

**Implication.** Harder data, not a different judge and not a different
scale. Stratifying by sentence length, which this repo did, is not the
same as stratifying by difficulty. See finding 6 for what happened when
that was acted on.

---

## 6. Predicting difficulty from the text does not work

Following from finding 3, a difficulty heuristic was built to select
harder examples: idioms and phrasal verbs, clause density, negation,
named entities and numbers, rare vocabulary. All plausible, all known
sources of translation error, none of them requiring a model call.

Validated against judge scores actually observed on 20 Spanish
examples, it correlates at **Spearman 0.065**. That is random.

| selection signal | correlation with judge score | cost |
|---|---|---|
| linguistic difficulty features | 0.065 | free |
| chrF++ against the reference | **0.49** | one model call per candidate |

The features describe what is hard *in general*. They do not describe
what is hard for this model, and only the second of those is useful
for selection.

**What replaced it.** `--stratify chrf` translates each candidate once
and bands it by how far the output lands from the reference. Bands are
terciles of the observed distribution rather than fixed thresholds, so
they adapt to a language whose scores sit systematically higher or
lower.

chrF++ is a flawed proxy: a low score can mean a good translation
phrased differently from the reference, a well-known weakness of it. For *selection* that is acceptable — a sentence where the
model's output diverges from a human reference is worth having in the
pool either way. It would not be acceptable as a quality verdict.

**What was kept and why.** The failed heuristic stays in the tree,
documented as failed, along with the `--validate` command that killed
it. The validation is the valuable part, and silently deleting a failed
hypothesis invites the next person to retry it.

**Selection takes an even spread, never worst-first.** A pool of only
the examples the model handles badly cannot reveal a prompt that
improves on those by regressing on simple ones, and an optimizer would
select that trade happily.

---

## 4. Things a judge should never be asked to do

An LLM judge will score an untranslated output, and a
fully-Traditional-script output, at 10 out of 10. This pipeline
reproduced the first of those directly (finding 1). Both are caught instantly by a string
comparison.

Deterministic checks (script, Chinese variety, residual Latin tokens,
source copying, empty output) run before the judge and their verdict is
not negotiable. They are the cheapest component in the pipeline and
they catch the most expensive failure class.

The corollary is a rule worth keeping: if a property can be checked
deterministically, checking it with a model is strictly worse. It costs
money, adds variance, and can be wrong.

---

## 5. Measurement defects found by reading the code

These produced numbers that looked reasonable and meant nothing.

- The final score was measured on the same pool GEPA selected against,
  so reported improvement was partly the optimizer grading its own
  work. Now measured on a sealed pool.
- A judge failure was scored 0.0 and averaged in, making a parsing bug
  indistinguishable from a terrible translation. Now counted separately
  and excluded from reported means.
- The largest data pool, 99 of 249 examples, was never read.
- The prompt-rewriting model inherited the task model, pointing the
  hardest job in the loop at the smallest model available.

None of these would have surfaced from a failing test or an error
message. All four were silent.
