# supernova-multilingual-pipeline

A reference implementation of a prompt-optimization pipeline for machine
translation, built so that adding a language is a config change rather
than a rewrite.

It translates with a frozen model, scores the output with an LLM judge,
and uses [GEPA](https://dspy.ai) to search for better instructions to
the translator. Nothing is fine tuned; the artifact a run produces is a
prompt.

**This is a template, not a copy of any production codebase.** What
lives here is the pattern, plus the measurements that shaped it.

Read [FINDINGS.md](FINDINGS.md) first if you only read one thing. It
records six measured results, including the two where the hypothesis
being tested turned out to be wrong.

## Quick start

```bash
pip install -r requirements.txt
cp gepa/.env.example gepa/.env   # then add your GROQ_API_KEY
python run.py check
```

Every command goes through `run.py`, ordered by cost:

| Command | What it does | Cost |
|---|---|---|
| `check` | Verifies the API key, every configured model, and the data pools | seconds |
| `demo --language es` | Translates a few examples and shows them beside the reference | ~1 min |
| `judge --language es` | Scores one translation, showing each run and anything excluded | ~1 min |
| `probe --language es` | Breaks good translations in known ways; reports what the judge misses | ~15 min |
| `compare --language es` | Scores identical translations flat vs severity-weighted | ~15 min |
| `difficulty --language es` | Pool difficulty, and whether the heuristic predicts anything | free |
| `data --language es` | Rebuilds the four data pools | varies |
| `optimize --language es` | The full GEPA run | hours |
| `results --language es` | Reads the last run back as a summary | free |

`probe` is the one worth running first. It is how the judge's blind spot
in finding 1 was discovered, and it costs nothing to be wrong about.

## How a run is measured

Four data pools per language, physically separate files, because the
moment a pool has been used to make a decision it can no longer measure
that decision:

| Pool | Used for |
|---|---|
| `examples.json` | GEPA's training set |
| `prompt_validation/` | GEPA's candidate selection |
| `final_test/` | Sealed. Scored once, at the end. The reported result |
| `judge_calibration/` | Validating the judge against human labels |

The headline number comes from `final_test`, which the optimizer never
sees. Reporting the score from the pool GEPA selected on measures how
well it selected, not how good the prompt is.

Two further rules the scoring follows:

- **A judge failure is not a low score.** Malformed output, an
  out-of-range value, or a zero contradicting its own feedback is
  excluded and counted separately. Above a 10% failure rate the run
  aborts rather than reporting a number that describes the judge's
  reliability.
- **Deterministic checks run before the judge**, and their verdict is
  not negotiable. Wrong script, wrong Chinese variety, leftover source
  language, source copying and empty output are all string comparisons.
  An LLM judge will score an untranslated output 10/10; a string
  comparison never will.

## Adding a language

Three additive steps, no code changes:

1. Add an entry to `configs/languages.yaml`
2. Add its judge config under `configs/judges/<code>.yaml`
3. Add its data folder under `data/<code>/`

Nothing in `gepa/` needs editing. The pipeline discovers languages from
the registry at runtime. A language starts as `status: planned`, which
scaffolds it visibly without the optimization loop picking it up, and
becomes `active` once a judge has actually been validated for it.

This matters because three assumptions that hold at two languages stop
holding at twenty: that one global judge setting works for everything,
that language quirks can live as `if language == "x"` branches, and
that flat per-direction files are manageable. All three are the same
mistake — language-specific knowledge in code rather than config.

## Layout

```
configs/
  languages.yaml        # the one place listing which languages exist
  judges/<code>.yaml    # per-language task model, judge, checks, scoring
data/<code>/            # the four pools, plus a manifest recording provenance
gepa/
  clients.py            # one model-calling path: retries, rate limits, hard failures
  judge.py              # GEMBA-style judge, flat and severity-weighted scoring
  checks.py             # deterministic output checks
  rrwa.py               # rank-reciprocal weighted aggregation across judge runs
  metrics.py            # chrF++ as an independent companion signal
  gepa_loop.py          # the optimization loop
  judge_probe.py        # fault-injection probe for judge sensitivity
  compare_scoring.py    # flat vs severity, on identical translations
  difficulty.py         # a difficulty heuristic that failed its own validation
  validate_judge.py     # judge vs human labels; refuses to pass without them
  prepare_data.py       # pool construction and stratification
  tests/
run.py                  # single entry point
FINDINGS.md             # measured results, including the negative ones
```

## Development

```bash
python -m pytest gepa/tests -q     # 86 tests, no API calls
ruff check . && ruff format .
```

Tests never hit the network: model calls go through a fake client. CI
runs the suite on every push and pull request. Dependencies are pinned,
because an unpinned minor release turning CI red teaches nothing.

Changes go through a pull request into `main`. `CODEOWNERS`
auto-requests a reviewer.

## Known gaps

Stated rather than discovered later:

- The judge is **not validated against human labels**. `validate_judge.py`
  correctly refuses to pass without them, and none exist yet. Every
  score here is a mechanics check, not a quality verdict.
- No readiness gate. A run reports a score; it does not decide whether
  to adopt the prompt.
- `yue` (Cantonese) is scaffolded as `planned` with placeholder data and
  has never been run.
- Logging is `print`, which is adequate for a CLI and would not be for a
  service.
