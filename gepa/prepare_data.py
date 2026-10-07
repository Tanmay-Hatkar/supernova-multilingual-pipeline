"""
Build a language's four data pools from a public parallel corpus.

Two things this does that a naive "grab the first N rows" does not:

  * Stratifies by length. An earlier run scored 0.991 on its baseline
    before optimization even started — not because the prompt was
    excellent, but because every example was a short, easy sentence.
    With no headroom, no prompt can look better than any other, and the
    whole optimization becomes unmeasurable. Sampling evenly across
    short, medium, and long examples restores the range needed to tell
    prompts apart.
  * Records provenance per example, so a result can be traced back to
    where its data came from rather than being an anonymous pile of text.

Deterministic: the same seed and the same source produce the same pools.

Usage:
    python prepare_data.py --language es
    python prepare_data.py --language cmn --total 250
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import UTC, datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which public corpus config backs each language. Kept here rather than
# in the language config, since this is a data-sourcing concern, not a
# runtime one — the pipeline itself never reads this.
SOURCES = {
    "es": {"dataset": "Helsinki-NLP/opus-100", "config": "en-es", "target_key": "es"},
    "cmn": {"dataset": "Helsinki-NLP/opus-100", "config": "en-zh", "target_key": "zh"},
}

# Length bands, in source characters. Sampling evenly across these is
# what gives the evaluation set genuine range instead of a wall of
# equally-easy short sentences.
LENGTH_BANDS = [("short", 20, 60), ("medium", 61, 140), ("long", 141, 400)]

# Pool proportions. prompt_validation feeds GEPA's training, and
# judge_calibration is its held-out validation set, so both need real
# size. final_test stays sealed and untouched by the loop.
POOL_SPLIT = {
    "examples": 0.40,
    "prompt_validation": 0.20,
    "judge_calibration": 0.20,
    "final_test": 0.20,
}


def _band_of(text: str) -> str | None:
    length = len(text)
    for name, low, high in LENGTH_BANDS:
        if low <= length <= high:
            return name
    return None


def collect_stratified(
    language_code: str, total: int, seed: int, stratify: str = "length"
) -> list[dict]:
    """
    Gather candidates, then take an even spread across bands.

    Three axes, with very different amounts of evidence behind them:

      * "length" — what the original pools were built with. Free, and
        the wrong axis: a long plainly-worded sentence is easier than a
        six-word idiom, and length-stratified pools came out 58% easy.
        Still the default only because it costs nothing and surprises
        nobody with an API bill during data preparation.
      * "difficulty" — linguistic features. Free, and measured at
        Spearman 0.065 against observed judge scores, which is random.
        Do not use for real selection; see difficulty.py.
      * "chrf" — translate each candidate and band it by chrF++ against
        the reference. Costs one model call per candidate, and is the
        only axis with evidence: chrF++ tracks judge scores at 0.49 on
        this pipeline. Use this when you can afford it.

    chrF++ is a proxy and a flawed one — a low score can mean a good
    translation phrased differently from the reference, which is a
    well-known weakness of it. For *selection* that is acceptable: a sentence
    where the model's output diverges from a human reference is worth
    having in the pool either way.
    """
    from datasets import load_dataset
    from difficulty import DIFFICULTY_BANDS, band_of, select_stratified

    source = SOURCES[language_code]
    dataset = load_dataset(source["dataset"], source["config"])["test"]

    candidates: list[dict] = []
    seen: set[str] = set()

    for row in dataset:
        english = row["translation"]["en"].strip()
        target = row["translation"][source["target_key"]].strip()
        if not english or not target or english in seen:
            continue

        length_band = _band_of(english)
        if length_band is None:
            continue  # still used as a sanity filter on absurdly short or long text

        seen.add(english)
        candidates.append(
            {
                "source": english,
                "reference": target,
                "length_band": length_band,
                "provenance": f"{source['dataset']}:{source['config']}:test",
            }
        )

    rng = random.Random(seed)

    if stratify == "chrf":
        selected = _select_by_chrf(candidates, total, seed, language_code)
    elif stratify == "difficulty":
        per_band = total // len(DIFFICULTY_BANDS)
        selected = select_stratified(candidates, per_band, rng)
    else:
        buckets: dict[str, list[dict]] = {name: [] for name, _, _ in LENGTH_BANDS}
        for candidate in candidates:
            buckets[candidate["length_band"]].append(candidate)
        per_band = total // len(LENGTH_BANDS)
        selected = []
        for band, _, _ in LENGTH_BANDS:
            available = buckets[band]
            rng.shuffle(available)
            take = available[:per_band]
            if len(take) < per_band:
                print(f"  note: only {len(take)} '{band}' examples available, wanted {per_band}")
            for candidate in take:
                candidate["difficulty_band"] = band_of(candidate["source"])
            selected.extend(take)

    rng.shuffle(selected)  # so pools don't end up band-segregated
    return selected


def _select_by_chrf(
    candidates: list[dict], total: int, seed: int, language_code: str
) -> list[dict]:
    """
    Band candidates by how far the model's own output lands from the
    reference, then take an even spread across those bands.

    This is the only selection axis on this pipeline with measured
    predictive power. It is also the only one that costs money, so the
    candidate pool is sampled down first: scoring every row of a corpus
    to build 250 examples would be absurd.

    An even spread again, not worst-first. A pool of only the examples
    the model handles badly cannot show a prompt regressing on the ones
    it handles well.
    """
    import dspy
    import yaml
    from clients import DSPyClientAdapter, get_client
    from config import MAX_OUTPUT_TOKENS
    from gepa_loop import Translate
    from language_registry import get_active_languages
    from metrics import compute_chrf_plus_plus

    language = {lang.code: lang for lang in get_active_languages()}[language_code]
    with open(language.judge_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task_model = config["task_model"]
    client = get_client(task_model["provider"], task_model["model"])
    dspy.settings.configure(lm=DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS))
    translator = dspy.Predict(Translate)

    rng = random.Random(seed)
    # Three times the target, so each band has something to choose from
    # without translating the entire corpus.
    pool = candidates[:]
    rng.shuffle(pool)
    pool = pool[: total * 3]

    print(f"  translating {len(pool)} candidates to measure difficulty (this costs API calls)...")
    for i, candidate in enumerate(pool, 1):
        translation = translator(
            source=candidate["source"], target_language=language.name
        ).translation
        score = compute_chrf_plus_plus(translation, candidate["reference"]).score
        candidate["baseline_chrf"] = score
        candidate["baseline_translation"] = translation
        if i % 25 == 0:
            print(f"    {i}/{len(pool)}")

    scored = [c for c in pool if c.get("baseline_chrf") is not None]

    # Band chrF++ *within* each length band, not across the whole pool.
    #
    # Banding globally does not measure difficulty, it measures length.
    # chrF++ is an n-gram overlap score, and on a six-word fragment a
    # single different word choice destroys it even when the
    # translation is perfect. A global tercile split therefore fills
    # the "hard" band with short sentences: measured on a real run, the
    # hard band averaged 42 source characters against 115 and 91 for
    # medium and easy, and 18 of the 32 shortest examples in the pool
    # landed in it. Those are not hard sentences. They are sentences
    # where the metric is noisy.
    #
    # Splitting within length bands holds length roughly constant, so
    # what remains is the part of the chrF++ signal that is about the
    # translation rather than about the string.
    for length_band, _, _ in LENGTH_BANDS:
        group = [c for c in scored if c["length_band"] == length_band]
        group.sort(key=lambda c: c["baseline_chrf"])
        third = len(group) // 3
        for band, members in (
            ("hard", group[:third]),
            ("medium", group[third : 2 * third]),
            ("easy", group[2 * third :]),
        ):
            for candidate in members:
                candidate["difficulty_band"] = band

    # Sample evenly across the full length x difficulty grid, so each
    # difficulty band ends up with the same length profile and the two
    # axes cannot be confused for one another later.
    per_cell = total // (3 * len(LENGTH_BANDS))
    selected: list[dict] = []
    for band in ("easy", "medium", "hard"):
        for length_band, _, _ in LENGTH_BANDS:
            available = [
                c
                for c in scored
                if c.get("difficulty_band") == band and c["length_band"] == length_band
            ]
            rng.shuffle(available)
            take = available[:per_cell]
            if len(take) < per_cell:
                print(
                    f"  note: only {len(take)} '{band}/{length_band}' examples "
                    f"available, wanted {per_cell}"
                )
            selected.extend(take)
    return selected


def write_pools(language_code: str, examples: list[dict]) -> dict[str, int]:
    data_dir = os.path.join(REPO_ROOT, "data", language_code)
    counts: dict[str, int] = {}
    cursor = 0

    for pool, proportion in POOL_SPLIT.items():
        size = int(len(examples) * proportion)
        chunk = examples[cursor : cursor + size]
        cursor += size

        path = (
            os.path.join(data_dir, "examples.json")
            if pool == "examples"
            else os.path.join(data_dir, pool, "examples.json")
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        counts[pool] = len(chunk)

    manifest = {
        "language_code": language_code,
        "built_at": datetime.now(UTC).isoformat(),
        "source": SOURCES[language_code],
        "total_examples": len(examples),
        "pool_counts": counts,
        "length_bands": {name: [low, high] for name, low, high in LENGTH_BANDS},
        "length_distribution": {
            band: sum(1 for e in examples if e.get("length_band") == band)
            for band, _, _ in LENGTH_BANDS
        },
        # The axis that actually matters for headroom. Length-stratified
        # pools came out 58% easy, which is why difficulty is now the
        # default and why this distribution is recorded alongside it.
        "difficulty_distribution": {
            band: sum(1 for e in examples if e.get("difficulty_band") == band)
            for band in ("easy", "medium", "hard")
        },
    }
    with open(os.path.join(data_dir, "data_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=sorted(SOURCES))
    parser.add_argument("--total", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify",
        choices=["length", "difficulty", "chrf"],
        default="length",
        help="chrf is the only axis with measured predictive power (0.49 vs 0.065), "
        "but costs one model call per candidate",
    )
    args = parser.parse_args()

    print(f"Building {args.total} examples for '{args.language}', stratified by {args.stratify}...")
    examples = collect_stratified(args.language, args.total, args.seed, args.stratify)
    counts = write_pools(args.language, examples)

    print(f"Collected {len(examples)} examples across length bands.")
    for pool, count in counts.items():
        print(f"  {pool}: {count}")
    print(f"Wrote data/{args.language}/ including data_manifest.json")


if __name__ == "__main__":
    main()
