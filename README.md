# supernova-multilingual-pipeline

A reference architecture for a translation optimization pipeline designed to scale
from two languages to twenty (or more) without restructuring, worked out
from real lessons learned building the Cantonese/Mandarin pipeline.

**This is a template, not a copy of any production codebase.** The
actual judge models, rubric text, RRWA formula, and evaluation data
used in production live in their own project. What lives here is the
*pattern*: how to structure config, data, and code so that adding a
new language is additive, not a rewrite.

## The problem this solves

A pipeline built for one or two languages tends to make three
assumptions that stop being true at scale:

1. **One global judge/model setting for everything.** In practice,
   different languages have needed different judge models. A single
   `JUDGE_MODEL` setting doesn't survive contact with a tenth language.
2. **Language-specific logic hardcoded as `if language == "x"` branches.**
   Fine for one exception, unmanageable as a growing pile of
   conditionals.
3. **Flat, per-direction files** (`en_x.json`, `en_y.json`, ...) instead
   of a folder per language, which makes "add a new language" mean
   "edit a shared list somewhere" instead of "add a folder."

## How this repo is structured instead

```
configs/
  languages.yaml       # the one place that lists which languages exist
  judges/
    yue.yaml            # per-language judge + scoring configuration
    cmn.yaml
    es.yaml
data/
  <language_code>/
    examples.json
    judge_calibration/  # data used only to validate/select the judge
    prompt_validation/  # data used only during prompt optimization
    final_test/          # held out, touched by nothing else
gepa/
  language_registry.py  # loads + validates configs/languages.yaml
  tests/
```

**Adding a new language means:** add an entry to `configs/languages.yaml`,
add its judge config under `configs/judges/`, add its data folder under
`data/`. Nothing in `gepa/` needs to be edited — the pipeline discovers
languages from the registry at runtime, not from a hardcoded list.

A language starts as `status: planned` (scaffolded, not yet validated,
excluded from runs) and moves to `status: active` once a real judge
selection process has actually validated one for it, the same kind of
human-labeled validation process used for the languages already active.

## Repo structure

| Folder | What lives here | Status |
|---|---|---|
| `gepa/` | Language registry, optimization loop pattern | Reference implementation |
| `evaluation/` | Model comparison and judge validation pattern | Reference implementation |
| `data/` | Per-language data, four pools kept separate | See `data/README.md` |
| `configs/` | Language registry and per-language judge configs | Source of truth for what's active |
| `finetuning/` | Fine tuning readiness pattern | Reference implementation |
| `serving/` | Serving configuration pattern | Reference implementation |
| `docs/` | Reference documentation | — |

## Contributing

All changes go through a pull request into `main` with at least one
approval before merging. `CODEOWNERS` auto-requests a reviewer, and CI
runs the test suite on every PR.
