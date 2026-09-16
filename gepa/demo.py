"""
Quick, human-readable demo: translate a handful of real examples live
and print source, generated translation, reference, and the companion
metric side by side. Not part of the pipeline, just a way to actually
look at output instead of only reading an aggregate score.

Usage:
    python demo.py es 3
"""

import json
import sys

import dspy
import yaml

from clients import DSPyClientAdapter, get_client
from config import MAX_OUTPUT_TOKENS
from gepa_loop import Translate
from language_registry import get_active_languages
from metrics import compute_companion

language_code = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

languages = {lang.code: lang for lang in get_active_languages()}
if language_code not in languages:
    sys.exit(f"'{language_code}' is not active. Active: {sorted(languages)}")
language = languages[language_code]

with open(language.judge_config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

task_model = config["task_model"]
client = get_client(task_model["provider"], task_model["model"])
dspy.settings.configure(lm=DSPyClientAdapter(client, max_tokens=MAX_OUTPUT_TOKENS))
translator = dspy.Predict(Translate)

with open(f"../data/{language_code}/examples.json", encoding="utf-8") as f:
    examples = json.load(f)[:n]

print(f"=== {language.name} demo — task model: {task_model['model']} ===\n")
for example in examples:
    result = translator(source=example["source"], target_language=language.name)
    companion = compute_companion(
        config.get("companion_metric"), result.translation, example.get("reference", "")
    )
    print(f"EN:        {example['source']}")
    print(f"Generated: {result.translation}")
    print(f"Reference: {example.get('reference', 'n/a')}")
    if companion.score is not None:
        print(f"{companion.metric}:    {companion.score}")
    print()
