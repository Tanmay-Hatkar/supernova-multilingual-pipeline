"""
Environment configuration. Only the things that must not be hardcoded
live here: API keys and a couple of run-time knobs. Everything
language-specific comes from the registry and per-language configs,
not from here.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Kept small on purpose for a first smoke test on a free-tier budget.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
RATE_LIMIT_MAX_RETRIES = int(os.getenv("RATE_LIMIT_MAX_RETRIES", "3"))

# Task-model response cache. On by default because an optimization run
# re-translates the same examples for every candidate prompt, and that
# repetition is most of its cost. The judge is never cached: see
# cache.py for why.
TASK_CACHE_ENABLED = os.getenv("TASK_CACHE", "1") not in ("0", "false", "False")
TASK_CACHE_DIR = os.getenv("TASK_CACHE_DIR", ".cache/task_model")
