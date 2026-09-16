"""
Judge validation: check a judge against human labels, then freeze it.

The premise, learned the expensive way: a judge that has never been
checked against human judgment is an opinion, not a measurement. GEPA
optimizes entirely against whatever the judge says, so an uncalibrated
judge means optimizing toward that judge's quirks rather than toward
translation quality.

This module is deliberately generic. Validating a new language's judge
is running one command, not reimplementing a process. It expects the
language's judge_calibration pool to contain examples carrying a
`human_score` (0-10). Where those labels don't exist yet, it says so
plainly and refuses to emit a passing verdict — an unvalidated judge
should look unvalidated, not quietly default to approved.

Usage:
    python validate_judge.py --language es
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import yaml

from dataset import load_examples
from judge import judge_translation
from language_registry import get_active_languages

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Agreement below this is too weak to trust for optimization. Chosen to
# match the gate used in the project's earlier judge validation work.
MIN_SPEARMAN = 0.45
MIN_COVERAGE = 0.90


@dataclass
class ValidationResult:
    language_code: str
    validated_at: str
    n_examples: int
    n_scored: int
    coverage: float
    spearman: float | None
    pearson: float | None
    mae: float | None
    passed: bool
    reason: str


def _rank(values: list[float]) -> list[float]:
    """Average ranks, so ties don't distort the correlation."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = average_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = (
        sum((x - mean_x) ** 2 for x in xs) ** 0.5 * sum((y - mean_y) ** 2 for y in ys) ** 0.5
    )
    return None if denominator == 0 else numerator / denominator


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def validate(language_code: str) -> ValidationResult:
    languages = {lang.code: lang for lang in get_active_languages()}
    if language_code not in languages:
        raise KeyError(f"'{language_code}' is not an active language.")
    language = languages[language_code]

    with open(language.judge_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(language.judge_config_path)

    examples = load_examples(language.data_dir, pool="judge_calibration")
    labeled = [ex for ex in examples if ex.get("human_score") is not None]

    now = datetime.now(timezone.utc).isoformat()

    if not labeled:
        return ValidationResult(
            language_code=language_code,
            validated_at=now,
            n_examples=len(examples),
            n_scored=0,
            coverage=0.0,
            spearman=None,
            pearson=None,
            mae=None,
            passed=False,
            reason=(
                "No human-labeled examples found in the judge_calibration pool. "
                "This judge is UNVALIDATED — add examples carrying a 'human_score' "
                "field (0-10) and re-run. Not treating absence of labels as a pass."
            ),
        )

    human_scores, judge_scores = [], []
    for example in labeled:
        # The pool stores a translation to be graded; fall back to the
        # reference where a dedicated hypothesis isn't provided.
        hypothesis = example.get("translation") or example.get("reference", "")
        verdict = judge_translation(example["source"], hypothesis, config, config_dir)
        if verdict.score is None:
            continue
        human_scores.append(float(example["human_score"]))
        judge_scores.append(verdict.score)

    coverage = len(judge_scores) / len(labeled) if labeled else 0.0
    spearman = _spearman(human_scores, judge_scores)
    pearson = _pearson(human_scores, judge_scores)
    mae = (
        statistics.fmean(abs(h - j) for h, j in zip(human_scores, judge_scores))
        if judge_scores
        else None
    )

    passed = bool(
        coverage >= MIN_COVERAGE and spearman is not None and spearman >= MIN_SPEARMAN
    )
    reason = (
        f"coverage {coverage:.2f} and Spearman {spearman:.3f} meet the gates"
        if passed
        else f"coverage {coverage:.2f} / Spearman {spearman} below gates "
        f"(need >= {MIN_COVERAGE} and >= {MIN_SPEARMAN})"
    )

    return ValidationResult(
        language_code=language_code,
        validated_at=now,
        n_examples=len(examples),
        n_scored=len(judge_scores),
        coverage=coverage,
        spearman=spearman,
        pearson=pearson,
        mae=mae,
        passed=passed,
        reason=reason,
    )


def freeze_registry(result: ValidationResult, config: dict) -> str:
    """
    Record the exact judge configuration that passed validation, so every
    later run is scored under identical conditions. Changing a judge
    means a new validation run producing a new registry entry, never an
    edit to an existing one.
    """
    from reliability import iter_judges

    registry_dir = os.path.join(REPO_ROOT, "results", result.language_code)
    os.makedirs(registry_dir, exist_ok=True)
    path = os.path.join(registry_dir, "judge_registry.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "frozen_at": result.validated_at,
                "language_code": result.language_code,
                "judges": iter_judges(config),
                "validation": asdict(result),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    args = parser.parse_args()

    result = validate(args.language)
    print(json.dumps(asdict(result), indent=2))

    languages = {lang.code: lang for lang in get_active_languages()}
    with open(languages[args.language].judge_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if result.passed:
        print(f"\nPASSED. Frozen registry written to {freeze_registry(result, config)}")
    else:
        print(f"\nNOT VALIDATED: {result.reason}")


if __name__ == "__main__":
    main()
