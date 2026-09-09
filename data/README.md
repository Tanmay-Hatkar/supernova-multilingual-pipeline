# data/

One folder per language, keyed by the code used in `configs/languages.yaml`.

Inside each language folder, four pools are kept physically separate,
never merged, so evaluation results stay trustworthy as the number of
languages grows:

| Pool | Purpose |
|---|---|
| `examples.json` (top level of the language folder) | The main example set used day to day, e.g. for prompt optimization |
| `judge_calibration/` | Data used only to validate and select the judge for this language — never touched by training |
| `prompt_validation/` | Data used only during optimization to score candidate prompts |
| `final_test/` | Held out, touched by nothing else, reserved for the final promotion decision |

Adding a new language means adding a new folder here with this same
shape, plus an entry in `configs/languages.yaml` and a matching file
under `configs/judges/`. Nothing in `gepa/` needs to change.
