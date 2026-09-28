#!/usr/bin/env python
"""
One entry point for the whole pipeline.

Before this existed you had to know which of six scripts to run, in what
order, from which directory. That is fine for the person who wrote them
and hostile to everyone else, including the person who wrote them three
weeks later. Every command below is self-contained and says what it is
about to do before it does it.

    python run.py check                 # does my setup work?
    python run.py demo --language es    # show me translations
    python run.py judge --language es   # show me the judge thinking
    python run.py data --language es    # rebuild the data pools
    python run.py optimize --language es  # the full GEPA run (slow)
    python run.py results --language es   # read the last run back

Commands are ordered by cost. `check` is seconds and nearly free,
`optimize` is tens of minutes against a rate-limited tier. Run the cheap
ones first when something looks wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Every module in gepa/ imports its siblings by bare name, so that
# directory has to be importable as a package root rather than as a
# subpackage. Adding it here means this file is the only place that
# knows about the layout.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "gepa"))


def _load_language(code: str):
    from language_registry import get_active_languages

    active = {lang.code: lang for lang in get_active_languages()}
    if code not in active:
        sys.exit(f"'{code}' is not an active language. Active: {sorted(active)}")
    return active[code]


def _load_config(language):
    import yaml

    with open(language.judge_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_check(args) -> int:
    """
    Verify the things that actually break: the API key, every configured
    model, and the data pools. Deliberately cheap, so it can be the
    reflex response to "it isn't working".
    """
    from config import GROQ_API_KEY, MAX_OUTPUT_TOKENS
    from dataset import load_examples
    from language_registry import get_active_languages, load_registry
    from reliability import probe_model

    print("=== Setup check ===\n")

    print("API key:")
    if not GROQ_API_KEY:
        print("  MISSING. Put GROQ_API_KEY=... in gepa/.env")
        return 1
    print(f"  loaded, ends in ...{GROQ_API_KEY[-4:]}\n")

    registry = load_registry()
    active = get_active_languages()
    print(f"Registry: {len(registry)} languages, {len(active)} active")
    for entry in registry:
        marker = "active " if entry.is_active else "planned"
        print(f"  [{marker}] {entry.code}: {entry.name} ({entry.direction})")
    print()

    print("Data pools:")
    pools = ("", "prompt_validation", "judge_calibration", "final_test")
    labels = {"": "train (examples.json)"}
    ok = True
    for entry in active:
        print(f"  {entry.code}:")
        for pool in pools:
            try:
                count = len(load_examples(entry.data_dir, pool=pool))
                print(f"    {labels.get(pool, pool):<24} {count} examples")
            except FileNotFoundError:
                print(f"    {labels.get(pool, pool):<24} MISSING")
                ok = False
    print()

    if args.skip_models:
        print("Models: skipped (--skip-models)")
        return 0 if ok else 1

    print(f"Models (can each serve {MAX_OUTPUT_TOKENS} output tokens?):")
    for entry in active:
        print(f"  {entry.code}:")
        for role, spec in _model_roles(_load_config(entry)):
            verdict = probe_model(spec["provider"], spec["model"], MAX_OUTPUT_TOKENS)
            status = "OK  " if verdict.viable else "FAIL"
            note = "" if verdict.viable else f"  <- {verdict.reason}"
            print(f"    [{status}] {role:<11} {verdict.model}{note}")
            ok = ok and verdict.viable

    print()
    print("All good." if ok else "Problems found above.")
    return 0 if ok else 1


def _model_roles(config: dict) -> list[tuple[str, dict]]:
    """
    Every model a language depends on, labelled by the job it does.
    Roles rather than a flat list, because "one of your models is
    broken" is a much less useful thing to be told than which one.
    """
    from reliability import iter_judges

    roles = [("task", config["task_model"])]
    roles += [("judge", spec) for spec in iter_judges(config)]
    if config.get("reflection_model"):
        roles.append(("reflection", config["reflection_model"]))
    return roles


def cmd_demo(args) -> int:
    """Translate a few real examples and show them next to the reference."""
    import dspy

    from clients import DSPyClientAdapter, get_client
    from config import MAX_OUTPUT_TOKENS
    from dataset import load_examples
    from gepa_loop import Translate
    from metrics import compute_companion

    language = _load_language(args.language)
    config = _load_config(language)
    task_model = config["task_model"]

    client = get_client(task_model["provider"], task_model["model"])
    dspy.settings.configure(lm=DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS))
    translator = dspy.Predict(Translate)

    examples = load_examples(language.data_dir, pool="")[: args.n]

    print(f"=== {language.name} demo — task model: {task_model['model']} ===")
    print(f"Translating {len(examples)} examples. chrF++ is 0-100, higher is better.\n")

    for i, example in enumerate(examples, 1):
        result = translator(source=example["source"], target_language=language.name)
        companion = compute_companion(
            config.get("companion_metric"), result.translation, example.get("reference", "")
        )
        print(f"[{i}] EN:        {example['source']}")
        print(f"    Generated: {result.translation}")
        print(f"    Reference: {example.get('reference', 'n/a')}")
        if companion.score is not None:
            print(f"    {companion.metric}:    {companion.score}")
        print()

    return 0


def cmd_judge(args) -> int:
    """
    Run the full judge path over one translation and show every step:
    each individual run, anything excluded and why, and the aggregation.

    This is the part of the pipeline worth watching, because you can see
    a bad judgment get thrown out rather than quietly averaged into the
    result.
    """
    import dspy

    from checks import run_checks
    from clients import DSPyClientAdapter, get_client
    from config import MAX_OUTPUT_TOKENS
    from dataset import load_examples
    from gepa_loop import Translate
    from judge import judge_translation

    language = _load_language(args.language)
    config = _load_config(language)
    config_dir = os.path.dirname(language.judge_config_path)
    task_model = config["task_model"]

    if args.source:
        source, translation = args.source, None
    else:
        example = load_examples(language.data_dir, pool="")[args.index]
        source, translation = example["source"], None

    client = get_client(task_model["provider"], task_model["model"])
    dspy.settings.configure(lm=DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS))
    translation = dspy.Predict(Translate)(source=source, target_language=language.name).translation

    print(f"=== Judge walkthrough — {language.name} ===\n")
    print(f"Source:      {source}")
    print(f"Translation: {translation}")
    print(f"Scoring method: {config.get('scoring_method')}\n")

    # Deterministic checks first, exactly as the pipeline runs them. If
    # one fails the judge is never consulted, because the answer is
    # already certain and a judge's opinion cannot overturn it.
    report = run_checks(translation, source, config)
    print("Deterministic checks:")
    for result in report.results:
        status = "pass" if result.passed else "FAIL"
        evidence = f"  {result.evidence}" if result.evidence else ""
        print(f"  [{status}] {result.name:<22} {result.detail}{evidence}")
    print()

    if not report.passed:
        print("Failed a deterministic check, so the judge is not consulted. Score: 0")
        return 0

    verdict = judge_translation(source, translation, config, config_dir)

    for model_result in verdict.per_model:
        print(f"Judge: {model_result.model}")
        print(f"  valid runs: {model_result.valid_runs} of {model_result.total_runs}")
        for reason in model_result.invalid_reasons:
            print(f"  excluded:   {reason}")
        if model_result.score is None:
            print("  score:      none produced")
        else:
            print(f"  score:      {model_result.score} (after RRWA over its valid runs)")
        print()

    print("Final verdict:")
    if verdict.needs_review:
        print("  no valid score; this example would be excluded from results")
    else:
        print(f"  score:    {verdict.score} / 10")
        if verdict.disagreement is not None:
            print(f"  spread:   {verdict.disagreement} between judges")
        print(f"  feedback: {verdict.feedback}")
    return 0


def cmd_compare(args) -> int:
    """
    Score the same translations under flat and severity-weighted
    scoring, and report which one can actually tell them apart.
    """
    import dspy

    from clients import DSPyClientAdapter, get_client
    from compare_scoring import score_both_ways, summarize, verdict
    from config import MAX_OUTPUT_TOKENS
    from dataset import load_examples
    from gepa_loop import Translate

    language = _load_language(args.language)
    config = _load_config(language)
    config_dir = os.path.dirname(language.judge_config_path)
    task_model = config["task_model"]

    client = get_client(task_model["provider"], task_model["model"])
    dspy.settings.configure(lm=DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS))
    translator = dspy.Predict(Translate)

    examples = load_examples(language.data_dir, pool="judge_calibration")[: args.n]

    print(f"=== Scoring-method comparison — {language.name} ===")
    print(f"Generating {len(examples)} translations once, then scoring each one twice.\n")

    translations = []
    for i, example in enumerate(examples, 1):
        translations.append(
            translator(source=example["source"], target_language=language.name).translation
        )
        print(f"  translated {i}/{len(examples)}", end="\r")
    print(" " * 40, end="\r")

    print("Judging under both methods (this is the slow part)...\n")
    scored = score_both_ways(examples, translations, config, config_dir)

    print(f"{'flat':>8}  {'severity':>9}  {'chrF++':>7}   source")
    for s in scored:
        flat = f"{s.flat_score:.2f}" if s.flat_score is not None else "  -  "
        sev = f"{s.severity_score:.2f}" if s.severity_score is not None else "  -  "
        chrf = f"{s.companion_score:.1f}" if s.companion_score is not None else "  -  "
        print(f"{flat:>8}  {sev:>9}  {chrf:>7}   {s.source[:58]}")

    summary = summarize(scored)
    print("\n--- Distribution ---")
    for method in ("flat", "severity"):
        stats = summary[method]
        if not stats:
            print(f"{method}: no valid scores")
            continue
        print(
            f"{method:>9}: mean {stats['mean']:<6} stdev {stats['stdev']:<6} "
            f"range {stats['min']}-{stats['max']}   {stats['share_9_or_above']:.0%} at 9+"
        )

    print("\n--- Agreement with chrF++ (independent of both judges) ---")
    print(f"     flat: {summary['flat_vs_companion_spearman']}")
    print(f" severity: {summary['severity_vs_companion_spearman']}")

    print(f"\n--- Verdict ---\n{verdict(summary)}")
    return 0


def cmd_probe(args) -> int:
    """
    Break good translations in known ways and see whether the judge
    notices. Answers the question a scoring comparison cannot: is the
    judge lenient, or are the translations actually good?
    """
    import dspy

    from clients import DSPyClientAdapter, get_client
    from config import MAX_OUTPUT_TOKENS
    from dataset import load_examples
    from gepa_loop import Translate
    from judge_probe import probe_example, summarize, verdict

    language = _load_language(args.language)
    config = _load_config(language)
    config_dir = os.path.dirname(language.judge_config_path)
    task_model = config["task_model"]

    client = get_client(task_model["provider"], task_model["model"])
    dspy.settings.configure(lm=DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS))
    translator = dspy.Predict(Translate)

    examples = load_examples(language.data_dir, pool="judge_calibration")[: args.n]

    print(f"=== Judge sensitivity probe — {language.name} ===")
    print(f"Scoring method: {config.get('scoring_method')}")
    print(f"{len(examples)} translations, each scored intact and then deliberately broken.\n")

    results = []
    for i, example in enumerate(examples, 1):
        translation = translator(
            source=example["source"], target_language=language.name
        ).translation
        print(f"  probing {i}/{len(examples)}", end="\r")
        results.extend(probe_example(example["source"], translation, config, config_dir))
    print(" " * 40, end="\r")

    summary = summarize(results)
    print(f"{'fault':<20}{'n':>4}{'mean drop':>12}{'expected':>10}{'detected':>11}")
    for fault, stats in sorted(summary.items()):
        rate = "n/a" if stats["detection_rate"] is None else f"{stats['detection_rate']:.0%}"
        drop = "n/a" if stats["mean_drop"] is None else f"{stats['mean_drop']:.2f}"
        print(
            f"{fault:<20}{stats['n']:>4}{drop:>12}"
            f"{stats['expected_min_drop']:>10.1f}{rate:>11}"
        )

    if args.show_cases:
        print("\n--- Cases the judge missed ---")
        missed = [r for r in results if r.detected is False and r.expected_drop > 0]
        for r in missed[: args.n * 2]:
            print(f"\n  fault:    {r.degradation}  (scored {r.original_score} -> {r.degraded_score})")
            print(f"  original: {r.original[:90]}")
            print(f"  broken:   {r.degraded[:90]}")

    print(f"\n--- Verdict ---\n{verdict(summary)}")
    return 0


def cmd_data(args) -> int:
    """Rebuild a language's four data pools from its public corpus."""
    from prepare_data import collect_stratified, write_pools

    print(f"Building {args.total} stratified examples for '{args.language}'...")
    examples = collect_stratified(args.language, args.total, args.seed)
    counts = write_pools(args.language, examples)
    print(f"Collected {len(examples)} examples across length bands.")
    for pool, count in counts.items():
        print(f"  {pool}: {count}")
    return 0


def cmd_optimize(args) -> int:
    """The full GEPA run. Slow and rate-limited; expect tens of minutes."""
    from gepa_loop import run_language

    language = _load_language(args.language)
    run_language(language, args.max_examples, args.skip_preflight)
    return 0


def cmd_results(args) -> int:
    """Read the last run back as a summary rather than as raw JSON."""
    path = os.path.join(REPO_ROOT, "results", args.language, "result.json")
    if not os.path.exists(path):
        sys.exit(f"No results yet for '{args.language}'. Run: python run.py optimize --language {args.language}")

    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    if "manifest" not in result:
        # Results written before the sealed-test split measured the
        # optimized prompt on the same pool GEPA selected against, so
        # their improvement figure is not comparable to a current run.
        # Say that, rather than crashing on a missing key.
        sys.exit(
            f"results/{args.language}/result.json predates the sealed-test split, so its "
            f"numbers are not trustworthy and cannot be read here.\n"
            f"Re-run: python run.py optimize --language {args.language}"
        )

    manifest = result["manifest"]
    baseline, optimized = result["baseline_test"], result["optimized_test"]

    print(f"=== Last run: {result['language_name']} ({result['language_code']}) ===")
    print(f"Run at: {result['run_at']}")
    print(f"Config fingerprint: {manifest['config_fingerprint']}\n")

    print("Scored on the sealed test pool, which the optimizer never saw:")
    print(f"  baseline:    {baseline['mean']}  ({baseline['scored']} of {baseline['total']} scored)")
    print(f"  optimized:   {optimized['mean']}  ({optimized['scored']} of {optimized['total']} scored)")
    print(f"  improvement: {result['improvement']}\n")

    print("Run:")
    print(f"  task model:       {manifest['task_model']}")
    print(f"  judge models:     {', '.join(manifest['judge_models'])}")
    print(f"  reflection model: {manifest['reflection_model']}")
    print(f"  scoring method:   {manifest['scoring_method']}")
    print(f"  pools:            {manifest['pools']}")
    print(f"  metric calls:     {manifest['metric_calls']}")
    print(f"  judge failures:   {manifest['judge_failures']} ({manifest['judge_failure_rate']:.1%})")
    print(f"  companion mean:   {manifest['companion_mean']}\n")

    print(f"Full files in results/{args.language}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Supernova multilingual prompt-optimization pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with: python run.py check",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="verify API key, models, and data pools")
    p.add_argument("--skip-models", action="store_true", help="skip the model probes (offline)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("demo", help="translate a few examples and show them")
    p.add_argument("--language", required=True)
    p.add_argument("-n", type=int, default=3, help="how many examples")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("judge", help="show the judge scoring one translation, step by step")
    p.add_argument("--language", required=True)
    p.add_argument("--source", help="your own English sentence instead of one from the data")
    p.add_argument("--index", type=int, default=0, help="which example from the pool")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("compare", help="flat vs severity scoring on the same translations")
    p.add_argument("--language", required=True)
    p.add_argument("-n", type=int, default=12, help="how many examples")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("probe", help="can the judge detect faults shown to it on purpose?")
    p.add_argument("--language", required=True)
    p.add_argument("-n", type=int, default=6, help="how many translations to break")
    p.add_argument("--show-cases", action="store_true", help="print the faults it missed")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("data", help="rebuild a language's four data pools")
    p.add_argument("--language", required=True)
    p.add_argument("--total", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("optimize", help="the full GEPA run (slow)")
    p.add_argument("--language", required=True)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--skip-preflight", action="store_true")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("results", help="read the last run back")
    p.add_argument("--language", required=True)
    p.set_defaults(func=cmd_results)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
