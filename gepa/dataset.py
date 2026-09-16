"""
Loads a language's example set. Generic by construction: it just reads
whichever data_dir the registry points at for a given language, it
never hardcodes a path or a language name.
"""

from __future__ import annotations

import json
import os


def load_examples(data_dir: str, pool: str = "") -> list[dict]:
    """
    Load one pool of examples for a language.

    pool="" loads the main working set at <data_dir>/examples.json.
    pool="judge_calibration" / "prompt_validation" / "final_test" loads
    that pool's own separated examples.json, never the main one.
    """
    path = (
        os.path.join(data_dir, "examples.json")
        if not pool
        else os.path.join(data_dir, pool, "examples.json")
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No example set found at {path}")

    with open(path, encoding="utf-8") as f:
        examples = json.load(f)

    for i, ex in enumerate(examples):
        if "source" not in ex:
            raise ValueError(f"Example {i} in {path} is missing a 'source' field.")

    return examples
