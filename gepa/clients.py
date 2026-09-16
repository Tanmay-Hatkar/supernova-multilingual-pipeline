"""
The single model-calling path for this pipeline.

Every model call — task model and judge alike — goes through a
ModelClient here. Nothing calls a provider SDK directly, and nothing
uses a second framework's built-in LM wrapper, which is what the
earlier version did and why rate-limit failures behaved differently
depending on which role hit them. One client means one retry
strategy, one error taxonomy, and one place to fix a provider quirk.

Two failure modes are deliberately distinguished, because conflating
them cost us real debugging time:

  * TRANSIENT: "rate limit reached, try again in 12.4s" — the request
    is valid, the budget is momentarily spent. Wait and retry.
  * STRUCTURAL: "request too large, requested 4096, limit 1000" — the
    request can never succeed as written, no matter how long you wait.
    Retrying is pure waste and hides the real problem. Fail fast with
    an actionable message instead.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any

import dspy
from groq import APIStatusError, Groq, RateLimitError

from config import GROQ_API_KEY, MAX_OUTPUT_TOKENS, RATE_LIMIT_MAX_RETRIES

_RETRY_AFTER_PATTERN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")
# Groq's phrasing when a request's own max_tokens exceeds a per-minute
# ceiling. This is not a wait-and-retry condition.
_STRUCTURAL_MARKERS = ("request too large", "reduce max_tokens", "expected output tokens exceed")


class StructuralLimitError(RuntimeError):
    """
    The request cannot succeed as written — typically max_tokens is
    above the model's own per-minute output ceiling. Raised instead of
    retrying, since no amount of waiting changes the outcome.
    """


def _parse_retry_after_seconds(message: str) -> float:
    match = _RETRY_AFTER_PATTERN.search(message)
    if not match:
        return 5.0
    minutes = float(match.group(1)) if match.group(1) else 0.0
    return minutes * 60 + float(match.group(2))


def _is_structural(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _STRUCTURAL_MARKERS)


class ModelClient(ABC):
    """One model, one provider, one consistent call surface."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> Any:
        """Provider-native chat call returning an OpenAI-shaped response."""

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Convenience wrapper returning just the text of the first choice."""
        response = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens or MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content


class GroqClient(ModelClient):
    def __init__(self, model: str):
        super().__init__(model)
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to gepa/.env yourself — this is "
                "never something to hardcode or have written on your behalf."
            )
        self._client = Groq(api_key=GROQ_API_KEY)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> Any:
        kwargs.setdefault("max_tokens", MAX_OUTPUT_TOKENS)
        kwargs.setdefault("temperature", 0.3)

        last_transient: Exception | None = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return self._client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )
            except (RateLimitError, APIStatusError) as e:
                message = str(e)
                if _is_structural(message):
                    raise StructuralLimitError(
                        f"Model '{self.model}' cannot serve a request with "
                        f"max_tokens={kwargs.get('max_tokens')}: its per-minute output "
                        f"ceiling is lower than the request itself. Lower "
                        f"MAX_OUTPUT_TOKENS or pick a different model — retrying "
                        f"cannot fix this.\nProvider said: {message}"
                    ) from e
                if not isinstance(e, RateLimitError):
                    raise
                last_transient = e
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    break
                time.sleep(_parse_retry_after_seconds(message))

        raise RuntimeError(
            f"Groq rate limit not cleared after {RATE_LIMIT_MAX_RETRIES} retries "
            f"for model '{self.model}'. Last error: {last_transient}"
        )


_PROVIDERS: dict[str, type[ModelClient]] = {"groq": GroqClient}


def get_client(provider: str, model: str) -> ModelClient:
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {sorted(_PROVIDERS)}. "
            f"Add a ModelClient subclass and register it here to support another."
        )
    return _PROVIDERS[provider](model)


class DSPyClientAdapter(dspy.BaseLM):
    """
    Lets DSPy drive a ModelClient.

    This is the piece that unifies the two paths: DSPy still gets an LM
    object with the interface it expects, but the actual HTTP call goes
    through the same client, the same retry logic, and the same error
    taxonomy as every judge call. Previously the task model went
    through DSPy's own provider wrapper instead, which is why identical
    provider limits surfaced as two different failures.
    """

    def __init__(self, client: ModelClient, max_tokens: int | None = None, **kwargs):
        super().__init__(
            model=client.model,
            max_tokens=max_tokens or MAX_OUTPUT_TOKENS,
            **kwargs,
        )
        self.client = client

    def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        call_kwargs = {
            "max_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", MAX_OUTPUT_TOKENS)),
            "temperature": kwargs.get("temperature", self.kwargs.get("temperature", 0.3)),
        }
        return self.client.chat(messages, **call_kwargs)
