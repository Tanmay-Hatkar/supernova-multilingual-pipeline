# finetuning/

Per-language fine tuning configs and the readiness checklist each
checkpoint must pass before promotion: language variety, safety,
memorization, and domain regression, run against a frozen baseline.

At scale, each language's checkpoint is evaluated independently using
that language's own `final_test/` pool under `data/<code>/`, never a
shared or pooled test set across languages.
