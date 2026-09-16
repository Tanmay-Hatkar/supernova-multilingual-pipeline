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
