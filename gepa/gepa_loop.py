"""
The GEPA optimization loop.

Generic by construction: hand it a language code, and everything else —
which models, how many judges, what data, what companion metric — comes
from that language's config. There is no language-specific logic here.

Three things here exist because getting them wrong produces a number
that looks fine and means nothing:

  * The headline result is scored on `final_test`, a pool GEPA never
    sees. Scoring on the set the optimizer selected against measures
    how well the optimizer selected, not how good the prompt is.
  * A judge failure is never scored as zero. It is excluded from the
    reported averages and counted separately, because a parsing bug and
    a terrible translation are different things and only one of them
    should steer the optimizer.
  * The reflection model is configured separately from the task model.
    The model rewriting the prompt is doing the hardest job in the loop
    and should not inherit the smallest model in the pipeline by default.

Run one language per invocation, never all of them in one process. That
keeps isolation between languages structural rather than something a
future edit has to remember to preserve.

Usage:
    python gepa_loop.py --language es
    python gepa_loop.py --language cmn --max-examples 20 --skip-preflight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
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

# Above this share of examples failing to produce a valid judgment, the
# run is measuring the judge's reliability rather than the prompt's
# quality. Reporting a score built on that would be worse than failing.
MAX_JUDGE_FAILURE_RATE = 0.10

# Neutral score handed to the optimizer when an example could not be
# judged at all, used only until enough valid scores exist to average.
NEUTRAL_SCORE = 0.5


class Translate(dspy.Signature):
    """Translate the English source text into the target language."""

    source: str = dspy.InputField(desc="English source text")
    target_language: str = dspy.InputField(desc="the language to translate into")
    translation: str = dspy.OutputField(desc="the translation, nothing else")


@dataclass
class RunState:
    """
    Everything the metric accumulates while the loop runs.

    Kept as explicit state rather than a closure variable so the failure
    counts end up in the results file instead of being lost when the run
    finishes.
    """

    per_example: list[dict] = field(default_factory=list)
    valid_scores: list[float] = field(default_factory=list)
    judge_failures: int = 0
    empty_translations: int = 0
    metric_calls: int = 0

    @property
    def running_mean(self) -> float:
        """
        The neutral value for an unjudgeable example: the average of
        everything successfully judged so far. It pulls the optimizer
        neither toward nor away from that candidate, which is the
        closest thing to "no opinion" a metric that must return a
        number can express.
        """
        if not self.valid_scores:
            return NEUTRAL_SCORE
        return sum(self.valid_scores) / len(self.valid_scores)

    @property
    def failure_rate(self) -> float:
        if not self.metric_calls:
            return 0.0
        return (self.judge_failures + self.empty_translations) / self.metric_calls


@dataclass
class PoolScore:
    """A score over one pool, with the unjudgeable examples kept out of it."""

    mean: float | None
    scored: int
    unjudged: int
    total: int

    def as_dict(self) -> dict:
        return {
            "mean": self.mean,
            "scored": self.scored,
            "unjudged": self.unjudged,
            "total": self.total,
        }


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


def _build_reflection_lm(config: dict, fallback: dspy.BaseLM) -> dspy.BaseLM:
    """
    The model GEPA uses to rewrite prompts. It reads a failing example
    plus the judge's written feedback and proposes a better instruction,
    which is a harder job than the translation itself — so it gets its
    own config entry rather than silently inheriting the task model.
    """
    spec = config.get("reflection_model")
    if not spec:
        return fallback
    client = get_client(spec["provider"], spec["model"])
    return DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS)


def _examples_to_dspy(raw: list[dict], language_name: str) -> list[dspy.Example]:
    return [
        dspy.Example(
            source=ex["source"],
            target_language=language_name,
            reference=ex.get("reference", ""),
        ).with_inputs("source", "target_language")
        for ex in raw
    ]


def _build_metric(config: dict, config_dir: str, state: RunState):
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
        state.metric_calls += 1
        translation = (getattr(pred, "translation", "") or "").strip()

        if not translation:
            # An empty output is a real failure of the program, not a
            # failure of measurement, so zero is the honest score here.
            state.empty_translations += 1
            state.per_example.append(
                {
                    "source": gold.source,
                    "translation": "",
                    "judge_score": 0.0,
                    "outcome": "empty_translation",
                }
            )
            return dspy.Prediction(score=0.0, feedback="Empty translation produced.")

        verdict = judge_translation(gold.source, translation, config, config_dir)
        companion = compute_companion(companion_name, translation, getattr(gold, "reference", ""))

        record = {
            "source": gold.source,
            "translation": translation,
            "judge_score": verdict.score,
            "companion_metric": companion.metric,
            "companion_score": companion.score,
            "judge_disagreement": verdict.disagreement,
            "outcome": "judged",
        }

        if verdict.needs_review:
            # No valid judgment exists for this example. Scoring it zero
            # would tell the optimizer the translation was terrible,
            # which is a claim nothing here supports. Hand back the
            # running mean instead and count it as a measurement
            # failure, not a quality signal.
            state.judge_failures += 1
            record["outcome"] = "judge_failed"
            record["judge_score"] = None
            state.per_example.append(record)
            return dspy.Prediction(
                score=state.running_mean,
                feedback="Judge produced no valid score; example excluded from reported results.",
            )

        normalized = verdict.score / 10.0
        state.valid_scores.append(normalized)
        state.per_example.append(record)
        return dspy.Prediction(score=normalized, feedback=verdict.feedback)

    return metric


def _score_pool(program, examples, metric, state: RunState, label: str) -> PoolScore:
    """
    Score a program over one pool, reporting only what was actually
    judged. Unjudgeable examples are counted and excluded rather than
    averaged in, so the mean answers "how good is this prompt" and not
    "how good is this prompt plus how often did the judge break".
    """
    before = len(state.per_example)
    for example in examples:
        pred = program(source=example.source, target_language=example.target_language)
        metric(example, pred)

    produced = state.per_example[before:]
    for record in produced:
        record["pool"] = label

    usable = [r["judge_score"] for r in produced if r["outcome"] == "judged"]
    empties = [r for r in produced if r["outcome"] == "empty_translation"]

    # An empty translation is a genuine zero, so it belongs in the mean.
    scores = [s / 10.0 for s in usable] + [0.0] * len(empties)
    unjudged = sum(1 for r in produced if r["outcome"] == "judge_failed")

    return PoolScore(
        mean=(sum(scores) / len(scores)) if scores else None,
        scored=len(scores),
        unjudged=unjudged,
        total=len(examples),
    )


def _extract_prompt(program) -> dict:
    """
    Pull the optimized program's instruction text *and* its few-shot
    demonstrations. Saving only the instructions means the saved artifact
    can't reproduce the score it was saved for, since the demos are part
    of what GEPA tuned.
    """
    prompts = {}
    for name, predictor in program.named_predictors():
        signature = getattr(predictor, "signature", None)
        demos = getattr(predictor, "demos", []) or []
        prompts[name] = {
            "instructions": getattr(signature, "instructions", "") if signature else "",
            "demos": [dict(d.toDict() if hasattr(d, "toDict") else d) for d in demos],
        }
    return prompts


def _config_fingerprint(config: dict) -> str:
    """
    A short hash of the exact configuration a run used, so a result can
    be tied back to the settings that produced it rather than to
    whatever the config file happens to say today.
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _judge_models(config: dict) -> list[dict]:
    from reliability import iter_judges

    return list(iter_judges(config))


def _abort_if_judge_unreliable(state: RunState) -> None:
    """
    Stop rather than report. Past a certain failure rate the number this
    run would produce describes the judge's reliability, not the
    prompt's quality, and publishing it would be worse than publishing
    nothing.
    """
    if state.metric_calls and state.failure_rate > MAX_JUDGE_FAILURE_RATE:
        raise RuntimeError(
            f"Judge failed on {state.failure_rate:.1%} of examples "
            f"({state.judge_failures} of {state.metric_calls}), above the "
            f"{MAX_JUDGE_FAILURE_RATE:.0%} limit. Fix the judge before trusting a score."
        )


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

    task_lm = _configure_task_lm(config)
    reflection_lm = _build_reflection_lm(config, fallback=task_lm)
    task_program = dspy.Predict(Translate)

    # Pool roles, stated once here so they cannot drift:
    #   examples          -> GEPA's training set (the largest pool)
    #   prompt_validation -> GEPA's validation set, used for candidate selection
    #   judge_calibration -> reserved for validating the judge, never read here
    #   final_test        -> sealed; scored once, after optimization is finished
    train_raw = load_examples(language.data_dir, pool="")
    dev_raw = load_examples(language.data_dir, pool="prompt_validation")
    test_raw = load_examples(language.data_dir, pool="final_test")
    if max_examples:
        train_raw = train_raw[:max_examples]
        dev_raw = dev_raw[:max_examples]
        test_raw = test_raw[:max_examples]

    trainset = _examples_to_dspy(train_raw, language.name)
    devset = _examples_to_dspy(dev_raw, language.name)
    testset = _examples_to_dspy(test_raw, language.name)

    state = RunState()
    metric = _build_metric(config, config_dir, state)

    print(f"Scoring baseline on the sealed test pool ({len(testset)} examples)...")
    baseline_prompt = _extract_prompt(task_program)
    baseline_test = _score_pool(task_program, testset, metric, state, "baseline_test")
    print(f"Baseline test score: {baseline_test.mean}")

    _abort_if_judge_unreliable(state)

    optimizer = dspy.GEPA(
        metric=metric,
        max_metric_calls=len(trainset) * 4 + len(devset) * 4,
        reflection_lm=reflection_lm,
    )

    print(f"Optimizing on {len(trainset)} training examples, selecting on {len(devset)}...")
    optimized_program = optimizer.compile(task_program, trainset=trainset, valset=devset)

    print(f"Scoring the optimized prompt on the sealed test pool ({len(testset)} examples)...")
    optimized_test = _score_pool(optimized_program, testset, metric, state, "optimized_test")
    print(f"Optimized test score: {optimized_test.mean}")

    _abort_if_judge_unreliable(state)

    improvement = (
        optimized_test.mean - baseline_test.mean
        if optimized_test.mean is not None and baseline_test.mean is not None
        else None
    )

    companion_scores = [
        r["companion_score"] for r in state.per_example if r.get("companion_score") is not None
    ]

    results = {
        "language_code": language.code,
        "language_name": language.name,
        "run_at": datetime.now(timezone.utc).isoformat(),
        # The headline numbers, both measured on the sealed pool GEPA
        # never saw. This is the only honest before-and-after available.
        "baseline_test": baseline_test.as_dict(),
        "optimized_test": optimized_test.as_dict(),
        "improvement": improvement,
        "manifest": {
            "config_fingerprint": _config_fingerprint(config),
            "task_model": config["task_model"]["model"],
            "judge_models": [j["model"] for j in _judge_models(config)],
            "reflection_model": (config.get("reflection_model") or config["task_model"])["model"],
            "scoring_method": config.get("scoring_method"),
            "companion_metric": config.get("companion_metric"),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "pools": {
                "train": len(trainset),
                "dev": len(devset),
                "sealed_test": len(testset),
            },
            "metric_calls": state.metric_calls,
            "judge_failures": state.judge_failures,
            "empty_translations": state.empty_translations,
            "judge_failure_rate": round(state.failure_rate, 4),
            "companion_mean": (
                sum(companion_scores) / len(companion_scores) if companion_scores else None
            ),
        },
        "baseline_prompt": baseline_prompt,
        "optimized_prompt": _extract_prompt(optimized_program),
    }

    _write_results(language.code, results, state)
    return results


def _write_results(language_code: str, results: dict, state: RunState) -> None:
    results_dir = os.path.join(REPO_ROOT, "results", language_code)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # The prompt also gets its own file, so it can be picked up and reused
    # without parsing a results blob for it.
    with open(os.path.join(results_dir, "optimized_prompt.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "language_code": language_code,
                "run_at": results["run_at"],
                "config_fingerprint": results["manifest"]["config_fingerprint"],
                "prompt": results["optimized_prompt"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(results_dir, "per_example.json"), "w", encoding="utf-8") as f:
        json.dump(state.per_example, f, indent=2, ensure_ascii=False)

    print(f"Saved results, optimized prompt, and per-example log to results/{language_code}/")


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
