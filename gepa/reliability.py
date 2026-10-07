"""
Pre-flight model reliability checks.

The pipeline hit the same class of failure three separate times —
picking a model, running, and only then discovering the provider
wouldn't serve it at the token budget we need. Each discovery cost a
full failed run. This module turns that into a cheap check that runs
before any real work starts.

A model is "viable" here if the provider will accept a realistically
sized request at the exact max_tokens budget this pipeline uses. That's
a narrower, more useful question than "does the model exist" — a model
can be listed, respond to a tiny request, and still be structurally
unable to serve the budget we need.

KNOWN LIMIT OF THIS CHECK, stated plainly because an over-trusted check
is worse than no check: some providers enforce output-token ceilings
per minute, computed from an estimate of a request's expected output
rather than from max_tokens alone. A single probe run while idle can
therefore pass at a budget that fails under sustained load — this was
observed directly, with one model passing a trivial probe at a budget
that had already broken a real run. The probe below deliberately sends
a realistically sized prompt to narrow that gap, but it cannot fully
predict behavior under load. Runtime handling in clients.py, which
distinguishes structural rejections from transient limits, remains the
real safety net; this check is an early warning, not a guarantee.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import requests
from clients import StructuralLimitError, get_client
from config import GROQ_API_KEY, MAX_OUTPUT_TOKENS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(REPO_ROOT, "results", "model_reliability.json")


@dataclass
class ModelVerdict:
    provider: str
    model: str
    max_tokens_tested: int
    viable: bool
    reason: str
    limit_requests_per_min: str | None = None
    limit_tokens_per_min: str | None = None


# A prompt shaped like the pipeline's real work, not a toy. A trivial
# probe understates the provider's expected-output estimate and can pass
# at a budget that fails in practice — which is exactly what happened
# before this was widened.
_REPRESENTATIVE_PROMPT = (
    "Translate the following English text into Spanish, preserving meaning, "
    "register, and factual accuracy. Return only the translation.\n\n"
    "Source: The forum produced business contracts amounting to more than "
    "$24 million between Asian and African private companies, and organizers "
    "expect participation to grow substantially over the next three years."
)


def probe_model(provider: str, model: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> ModelVerdict:
    """
    Send one realistically sized request at the budget the pipeline
    actually uses, and report whether the provider accepts it. See the
    module docstring for what this check can and cannot tell you.
    """
    try:
        client = get_client(provider, model)
        client.chat(
            [{"role": "user", "content": _REPRESENTATIVE_PROMPT}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except StructuralLimitError as e:
        return ModelVerdict(
            provider=provider,
            model=model,
            max_tokens_tested=max_tokens,
            viable=False,
            reason=f"structural limit: cannot serve max_tokens={max_tokens}. {str(e)[:200]}",
        )
    except Exception as e:  # model missing, auth failure, anything else
        return ModelVerdict(
            provider=provider,
            model=model,
            max_tokens_tested=max_tokens,
            viable=False,
            reason=f"{type(e).__name__}: {str(e)[:200]}",
        )

    limits = _fetch_rate_limits(provider, model)
    return ModelVerdict(
        provider=provider,
        model=model,
        max_tokens_tested=max_tokens,
        viable=True,
        reason="accepted a real request at the configured token budget",
        **limits,
    )


def _fetch_rate_limits(provider: str, model: str) -> dict:
    """Best-effort: read published rate limits from response headers."""
    if provider != "groq" or not GROQ_API_KEY:
        return {}
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            timeout=30,
        )
        return {
            "limit_requests_per_min": response.headers.get("x-ratelimit-limit-requests"),
            "limit_tokens_per_min": response.headers.get("x-ratelimit-limit-tokens"),
        }
    except Exception:
        return {}


def check_language_config(
    language_code: str, max_tokens: int = MAX_OUTPUT_TOKENS
) -> list[ModelVerdict]:
    """
    Check every model a language's config depends on — task model and
    every judge — before a run commits to them.
    """
    from language_registry import get_judge_config

    config = get_judge_config(language_code)
    verdicts = []

    task = config["task_model"]
    verdicts.append(probe_model(task["provider"], task["model"], max_tokens))

    for judge in iter_judges(config):
        verdicts.append(probe_model(judge["provider"], judge["model"], max_tokens))

    return verdicts


def iter_judges(config: dict) -> list[dict]:
    """
    Normalizes single-judge and multi-judge config shapes into one list,
    so callers never branch on which form a language happens to use.
    """
    if "judges" in config:
        return list(config["judges"])
    if "judge" in config:
        return [config["judge"]]
    raise ValueError("Config defines neither 'judge' nor 'judges'.")


def assert_config_viable(language_code: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> None:
    """
    Fail loudly before a run rather than halfway through it. This is the
    check that would have saved three separate failed runs.
    """
    verdicts = check_language_config(language_code, max_tokens)
    broken = [v for v in verdicts if not v.viable]
    if broken:
        details = "\n".join(f"  - {v.model}: {v.reason}" for v in broken)
        raise RuntimeError(
            f"Pre-flight check failed for '{language_code}' at max_tokens={max_tokens}:\n"
            f"{details}\n"
            f"Fix the config or lower MAX_OUTPUT_TOKENS before running."
        )


def write_report(verdicts: list[ModelVerdict], path: str = REPORT_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(v) for v in verdicts], f, indent=2)
    return path


def main():
    """Probe every model referenced by every active language, and report."""
    from language_registry import get_active_languages, get_judge_config

    seen, verdicts = set(), []
    for language in get_active_languages():
        config = get_judge_config(language.code)
        candidates = [config["task_model"]] + iter_judges(config)
        for candidate in candidates:
            key = (candidate["provider"], candidate["model"])
            if key in seen:
                continue
            seen.add(key)
            verdict = probe_model(*key)
            verdicts.append(verdict)
            status = "OK " if verdict.viable else "BAD"
            print(f"[{status}] {verdict.model}: {verdict.reason}")

    print(f"\nWrote {write_report(verdicts)}")


if __name__ == "__main__":
    main()
