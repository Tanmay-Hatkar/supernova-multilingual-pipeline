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
from datetime import datetime, timezone

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


def collect_stratified(language_code: str, total: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    source = SOURCES[language_code]
    dataset = load_dataset(source["dataset"], source["config"])["test"]

    buckets: dict[str, list[dict]] = {name: [] for name, _, _ in LENGTH_BANDS}
    seen: set[str] = set()

    for row in dataset:
        english = row["translation"]["en"].strip()
        target = row["translation"][source["target_key"]].strip()
        if not english or not target or english in seen:
            continue

        band = _band_of(english)
        if band is None:
            continue

        seen.add(english)
        buckets[band].append(
            {
                "source": english,
                "reference": target,
                "length_band": band,
                "provenance": f"{source['dataset']}:{source['config']}:test",
            }
        )

    rng = random.Random(seed)
    per_band = total // len(LENGTH_BANDS)
    selected: list[dict] = []

    for band, _, _ in LENGTH_BANDS:
        available = buckets[band]
        rng.shuffle(available)
        take = available[:per_band]
        if len(take) < per_band:
            print(f"  note: only {len(take)} '{band}' examples available, wanted {per_band}")
        selected.extend(take)

    rng.shuffle(selected)  # so pools don't end up band-segregated
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
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCES[language_code],
        "total_examples": len(examples),
        "pool_counts": counts,
        "length_bands": {name: [low, high] for name, low, high in LENGTH_BANDS},
        "band_distribution": {
            band: sum(1 for e in examples if e["length_band"] == band)
            for band, _, _ in LENGTH_BANDS
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
    args = parser.parse_args()

    print(f"Building {args.total} stratified examples for '{args.language}'...")
    examples = collect_stratified(args.language, args.total, args.seed)
    counts = write_pools(args.language, examples)

    print(f"Collected {len(examples)} examples across length bands.")
    for pool, count in counts.items():
        print(f"  {pool}: {count}")
    print(f"Wrote data/{args.language}/ including data_manifest.json")


if __name__ == "__main__":
    main()
