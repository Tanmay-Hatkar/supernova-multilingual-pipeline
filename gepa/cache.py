"""
On-disk cache for task-model responses.

An optimization run re-translates the same examples many times: every
candidate prompt is evaluated against the same pool, and a candidate
that survives selection is evaluated again on the next round. The
translations themselves are not the interesting part of that work, and
paying for them repeatedly is what makes a full run cost roughly 1.5M
tokens against a rate-limited tier.

A response is reused only when the model, the full message list, the
temperature and the max token count all match, so a cache hit is a
request that would have produced an answer drawn from the same
distribution as the one stored.

## This cache must never wrap the judge

The judge is called three times at temperature 0.7 and the three
answers are aggregated by rank-reciprocal weighting. The spread between
those answers is itself a measurement: it is how the pipeline knows its
own noise floor. A cache would return one stored answer three times,
collapse the spread to zero, and make every judgment look perfectly
reliable while measuring nothing.

Wrap the task model. Never the judge.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

CACHE_FORMAT = 1


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cache_key(
    model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
) -> str:
    """
    A stable identity for one request.

    Temperature and max_tokens are part of the key because they change
    the distribution the answer is drawn from, so a response sampled
    under different settings is a different response, not the same one.
    """
    return hashlib.sha256(
        _canonical(
            {
                "format": CACHE_FORMAT,
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
    ).hexdigest()


@dataclass
class _Message:
    content: str
    reasoning_content: str | None = None
    tool_calls: None = None


@dataclass
class _Choice:
    message: _Message


@dataclass
class CachedResponse:
    """
    The subset of an OpenAI-shaped response that anything downstream
    reads: the text, the model name, and a token usage mapping. Built
    from stored text rather than re-parsed from a provider payload, so
    a provider changing an unrelated field cannot invalidate the cache.
    """

    choices: list[_Choice]
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    from_cache: bool = True


def _response_text(response: Any) -> str | None:
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return text if isinstance(text, str) else None


class ResponseCache:
    """
    One directory of JSON files, one per request.

    A directory rather than a single file so that a run interrupted
    mid-write loses at most one entry, and so two processes working on
    different languages never rewrite each other's results.
    """

    def __init__(self, directory: str, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.enabled:
            os.makedirs(self.directory, exist_ok=True)

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.json")

    def get(self, key: str) -> CachedResponse | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
        except FileNotFoundError:
            self.misses += 1
            return None
        except (OSError, json.JSONDecodeError):
            # A truncated or unreadable entry is treated as absent. The
            # request is simply made again, which is always correct.
            self.misses += 1
            return None

        if entry.get("format") != CACHE_FORMAT or not isinstance(entry.get("text"), str):
            self.misses += 1
            return None

        self.hits += 1
        return CachedResponse(
            choices=[_Choice(_Message(entry["text"]))],
            model=entry.get("model", ""),
            usage=entry.get("usage") or {},
        )

    def put(self, key: str, response: Any) -> None:
        if not self.enabled:
            return
        text = _response_text(response)
        if text is None:
            # Nothing useful to replay. Storing a malformed response
            # would turn one bad call into a permanent one.
            return

        usage = getattr(response, "usage", None)
        if usage is not None and not isinstance(usage, dict):
            usage = getattr(usage, "model_dump", lambda: {})() or {}

        entry = {
            "format": CACHE_FORMAT,
            "model": getattr(response, "model", "") or "",
            "text": text,
            "usage": usage or {},
        }

        # Written to a temporary file and moved into place so a reader
        # never sees a half-written entry.
        try:
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
            os.replace(tmp, self._path(key))
            self.writes += 1
        except OSError:
            # A cache that cannot write is a slow cache, not a failure.
            return

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "writes": self.writes,
            "directory": self.directory if self.enabled else None,
        }
