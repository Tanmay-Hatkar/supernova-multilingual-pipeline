# Architecture Notes

## Why a language registry instead of hardcoded language lists

Three real problems showed up building a two-language pipeline that
would not have survived scaling to ten or twenty languages:

1. A single global judge/model setting stopped being true once
   different languages needed different judges.
2. Language-specific rubric logic as inline conditionals in code
   became unmanageable as more exceptions accumulated.
3. Flat, per-direction data files meant "add a language" required
   editing a shared list somewhere, instead of just adding a folder.

The fix is the same in all three cases: move anything language
specific out of code and into per-language config and data, and give
the pipeline exactly one place, `configs/languages.yaml`, to discover
what languages exist and what each one needs.

## The four data pools, per language

Fine tuning data, judge calibration data, prompt validation data, and
final test data must never overlap, any example that leaks between
pools invalidates whatever conclusion was drawn from the pool it
leaked into. At two languages this is manageable to check by hand. At
twenty, it needs to be structural: each language gets its own copy of
all four pools under `data/<code>/`, so there is no shared pool for
examples to accidentally cross into.

## Status field, not a boolean

A language is `active` or `planned`, not simply present or absent from
the registry. This makes it possible to scaffold a new language's
folders and config ahead of time, visibly, without it being picked up
by the optimization loop until a real judge has actually been
validated for it.
