"""
The GEPA optimization loop.

Generic by construction: hand it a language code, and everything else —
which models, how many judges, what data, what companion metric — comes
from that language's config. There is no language-specific logic here.

Run one language per invocation, never all of them in one process. That
keeps isolation between languages structural rather than something a
future edit has to remember to preserve.

Usage:
    python gepa_loop.py --language es
    python gepa_loop.py --language cmn --max-examples 20 --skip-preflight
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import dspy
import yaml

from clients import DSPyClientAdapter, get_client
from config import MAX_OUTPUT_TOKENS
from dataset import load_examples
from judge import judge_translation
from language_registry import LanguageEntry, get_active_languages
from metrics import compute_companion

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Translate(dspy.Signature):
    """Translate the English source text into the target language."""

    source: str = dspy.InputField(desc="English source text")
    target_language: str = dspy.InputField(desc="the language to translate into")
    translation: str = dspy.OutputField(desc="the translation, nothing else")


def _configure_task_lm(config: dict) -> dspy.BaseLM:
    """
    Route the task model through the same client the judge uses. This is
    the unification that matters: one retry strategy and one error
    taxonomy for every model call in the pipeline, rather than the task
    model quietly going through a second framework's own wrapper with
    different behavior.
    """
    task_model = config["task_model"]
    client = get_client(task_model["provider"], task_model["model"])
    lm = DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS)
    dspy.settings.configure(lm=lm)
    return lm


def _examples_to_dspy(raw: list[dict], language_name: str) -> list[dspy.Example]:
    return [
        dspy.Example(
            source=ex["source"],
            target_language=language_name,
            reference=ex.get("reference", ""),
        ).with_inputs("source", "target_language")
        for ex in raw
    ]


def _build_metric(config: dict, config_dir: str, companion_log: list):
    """
    Wrap the judge into the shape the optimizer expects: a score plus
    written feedback, which is what lets GEPA propose a meaningful
    revision rather than searching blind.

    The companion metric is computed alongside but deliberately does not
    influence the optimization score — it's an independent second signal
    recorded for later comparison, not a second opinion that gets voted
    into the result.
    """
    companion_name = config.get("companion_metric")

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        translation = (getattr(pred, "translation", "") or "").strip()
        if not translation:
            return dspy.Prediction(score=0.0, feedback="Empty translation produced.")

        verdict = judge_translation(gold.source, translation, config, config_dir)

        companion = compute_companion(companion_name, translation, getattr(gold, "reference", ""))
        companion_log.append(
            {
                "source": gold.source,
                "translation": translation,
                "judge_score": verdict.score,
                "companion_metric": companion.metric,
                "companion_score": companion.score,
                "judge_disagreement": verdict.disagreement,
                "needs_review": verdict.needs_review,
            }
        )

        if verdict.needs_review:
            # No valid judgment. Report it as a failure of measurement,
            # not as evidence the translation was bad.
            return dspy.Prediction(score=0.0, feedback="Judge produced no valid score; needs review.")

        return dspy.Prediction(score=verdict.score / 10.0, feedback=verdict.feedback)

    return metric


def _score_program(program, examples, metric) -> float:
    scores = []
    for example in examples:
        pred = program(source=example.source, target_language=example.target_language)
        scores.append(metric(example, pred).score)
    return sum(scores) / len(scores) if scores else 0.0


def _extract_prompt(program) -> dict:
    """
    Pull the actual instruction text out of the optimized program. Saving
    only a score means a good result can't be reused — the prompt is the
    artifact, the score is just its report card.
    """
    prompts = {}
    for name, predictor in program.named_predictors():
        signature = getattr(predictor, "signature", None)
        if signature is not None:
            prompts[name] = getattr(signature, "instructions", "")
    return prompts


def run_language(
    language: LanguageEntry, max_examples: int | None = None, skip_preflight: bool = False
) -> dict:
    print(f"=== GEPA run: {language.name} ({language.code}) ===")

    if not skip_preflight:
        from reliability import assert_config_viable

        print("Pre-flight: checking every configured model can serve this token budget...")
        assert_config_viable(language.code, MAX_OUTPUT_TOKENS)
        print("Pre-flight: all configured models viable.\n")

    with open(language.judge_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(language.judge_config_path)

    _configure_task_lm(config)
    task_program = dspy.Predict(Translate)

    train_raw = load_examples(language.data_dir, pool="prompt_validation")
    val_raw = load_examples(language.data_dir, pool="judge_calibration")
    if max_examples:
        train_raw, val_raw = train_raw[:max_examples], val_raw[:max_examples]

    trainset = _examples_to_dspy(train_raw, language.name)
    valset = _examples_to_dspy(val_raw, language.name)

    companion_log: list = []
    metric = _build_metric(config, config_dir, companion_log)

    print(f"Scoring baseline on {len(valset)} validation examples...")
    baseline_prompt = _extract_prompt(task_program)
    baseline_avg = _score_program(task_program, valset, metric)
    print(f"Baseline average score: {baseline_avg:.3f}")

    optimizer = dspy.GEPA(
        metric=metric,
        max_metric_calls=len(trainset) * 4 + len(valset) * 4,
        reflection_lm=dspy.settings.lm,
    )

    print(f"Optimizing on {len(trainset)} training examples...")
    optimized_program = optimizer.compile(task_program, trainset=trainset, valset=valset)

    print("Re-scoring the optimized program on the same validation set...")
    optimized_avg = _score_program(optimized_program, valset, metric)
    print(f"Optimized average score: {optimized_avg:.3f}")

    optimized_prompt = _extract_prompt(optimized_program)
    companion_scores = [c["companion_score"] for c in companion_log if c["companion_score"] is not None]

    results = {
        "language_code": language.code,
        "language_name": language.name,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "baseline_avg_score": baseline_avg,
        "optimized_avg_score": optimized_avg,
        "improvement": optimized_avg - baseline_avg,
        "n_train": len(trainset),
        "n_val": len(valset),
        "task_model": config["task_model"]["model"],
        "companion_metric": config.get("companion_metric"),
        "companion_mean": (sum(companion_scores) / len(companion_scores)) if companion_scores else None,
        "judge_needs_review_count": sum(1 for c in companion_log if c["needs_review"]),
        "baseline_prompt": baseline_prompt,
        "optimized_prompt": optimized_prompt,
    }

    results_dir = os.path.join(REPO_ROOT, "results", language.code)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # The prompt also gets its own file, so it can be picked up and reused
    # without parsing a results blob for it.
    with open(os.path.join(results_dir, "optimized_prompt.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"language_code": language.code, "run_at": results["run_at"], "prompt": optimized_prompt},
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(results_dir, "per_example.json"), "w", encoding="utf-8") as f:
        json.dump(companion_log, f, indent=2, ensure_ascii=False)

    print(f"Saved results, optimized prompt, and per-example log to results/{language.code}/")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, help="language code from configs/languages.yaml")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the model viability check (not recommended)",
    )
    args = parser.parse_args()

    active = {lang.code: lang for lang in get_active_languages()}
    if args.language not in active:
        print(f"'{args.language}' is not active. Active: {sorted(active)}", file=sys.stderr)
        sys.exit(1)

    run_language(active[args.language], args.max_examples, args.skip_preflight)


if __name__ == "__main__":
    main()
