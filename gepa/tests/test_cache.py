"""
Tests for the task-model response cache.

A cache is only safe if a hit is indistinguishable from the call it
replaces. Two properties carry that, and both are silent when wrong:
the key must change whenever anything that changes the answer changes,
and a stored entry must rebuild into the same shape a live response
has, or the first cache hit of a real run fails somewhere far away from
here.

The third group pins the rule the cache exists under: it wraps the task
model only. Caching the judge would replay one answer for all three of
its sampled runs and collapse the spread the pipeline measures its own
noise floor with.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache import CACHE_FORMAT, ResponseCache, cache_key
from clients import CachedClient, ModelClient

MESSAGES = [
    {"role": "system", "content": "Translate into Spanish."},
    {"role": "user", "content": "Where are the keys?"},
]


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    """The shape a provider returns, reduced to what anything reads."""

    def __init__(self, content, model="fake/model", usage=None):
        self.choices = [FakeChoice(content)]
        self.model = model
        self.usage = usage if usage is not None else {"total_tokens": 11}


class CountingClient(ModelClient):
    """Records every call so a replay can be told from a real request."""

    def __init__(self, replies=None):
        super().__init__("fake/model")
        self.calls = []
        self.replies = list(replies) if replies else None

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.replies:
            return FakeResponse(self.replies.pop(0))
        return FakeResponse(f"reply {len(self.calls)}")


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(str(tmp_path / "task"), enabled=True)


# --- the key ----------------------------------------------------------


def test_identical_requests_share_a_key():
    a = cache_key("m", MESSAGES, 0.3, 900)
    b = cache_key("m", list(MESSAGES), 0.3, 900)
    assert a == b


@pytest.mark.parametrize(
    "model,messages,temperature,max_tokens",
    [
        ("other/model", MESSAGES, 0.3, 900),
        ("m", [{"role": "user", "content": "different"}], 0.3, 900),
        ("m", MESSAGES, 0.7, 900),
        ("m", MESSAGES, 0.3, 400),
    ],
)
def test_every_field_that_changes_the_answer_changes_the_key(
    model, messages, temperature, max_tokens
):
    # Temperature and max_tokens are in the key because they change the
    # distribution sampled from, so a response produced under different
    # settings is a different response rather than the same one.
    assert cache_key(model, messages, temperature, max_tokens) != cache_key("m", MESSAGES, 0.3, 900)


def test_message_order_is_part_of_the_identity():
    swapped = list(reversed(MESSAGES))
    assert cache_key("m", swapped, 0.3, 900) != cache_key("m", MESSAGES, 0.3, 900)


# --- storing and replaying -------------------------------------------


def test_a_stored_response_replays_its_text(cache):
    key = cache_key("m", MESSAGES, 0.3, 900)
    cache.put(key, FakeResponse("¿Dónde están las llaves?"))

    replayed = cache.get(key)
    assert replayed is not None
    assert replayed.choices[0].message.content == "¿Dónde están las llaves?"


def test_a_replayed_response_has_the_shape_a_live_one_has(cache):
    # dspy reads choices[0].message.content, response.model, and calls
    # dict() on response.usage. A hit that breaks any of those fails at
    # the first real cache hit rather than here.
    key = cache_key("m", MESSAGES, 0.3, 900)
    cache.put(key, FakeResponse("hola", model="groq/x", usage={"total_tokens": 7}))

    replayed = cache.get(key)
    assert replayed.model == "groq/x"
    assert dict(replayed.usage) == {"total_tokens": 7}
    assert replayed.choices[0].message.reasoning_content is None
    assert replayed.choices[0].message.tool_calls is None


def test_a_missing_entry_is_a_miss_not_an_error(cache):
    assert cache.get(cache_key("m", MESSAGES, 0.3, 900)) is None
    assert (cache.hits, cache.misses) == (0, 1)


def test_hits_and_misses_are_counted(cache):
    key = cache_key("m", MESSAGES, 0.3, 900)
    cache.get(key)
    cache.put(key, FakeResponse("hola"))
    cache.get(key)
    cache.get(key)

    assert (cache.hits, cache.misses) == (2, 1)
    assert cache.hit_rate == pytest.approx(2 / 3)


# --- refusing to store or trust junk ----------------------------------


def test_a_response_with_no_text_is_not_stored(cache):
    # Storing a malformed response would turn one bad call into a
    # permanent one, replayed for the rest of the cache's life.
    key = cache_key("m", MESSAGES, 0.3, 900)
    cache.put(key, FakeResponse(None))
    assert cache.get(key) is None
    assert cache.writes == 0


def test_an_empty_choices_list_is_not_stored(cache):
    class NoChoices:
        choices: list = []
        model = "m"
        usage: dict = {}

    key = cache_key("m", MESSAGES, 0.3, 900)
    cache.put(key, NoChoices())
    assert cache.get(key) is None


def test_a_corrupt_entry_is_treated_as_absent(cache):
    key = cache_key("m", MESSAGES, 0.3, 900)
    cache.put(key, FakeResponse("hola"))
    Path(cache.directory, f"{key}.json").write_text("{ truncated", encoding="utf-8")

    assert cache.get(key) is None


def test_an_entry_from_another_cache_format_is_ignored(cache):
    # The stored shape is versioned so a future change to what is kept
    # cannot be read back under today's assumptions.
    key = cache_key("m", MESSAGES, 0.3, 900)
    Path(cache.directory, f"{key}.json").write_text(
        json.dumps({"format": CACHE_FORMAT + 1, "text": "stale"}), encoding="utf-8"
    )
    assert cache.get(key) is None


def test_a_disabled_cache_stores_nothing_and_never_hits(tmp_path):
    off = ResponseCache(str(tmp_path / "off"), enabled=False)
    key = cache_key("m", MESSAGES, 0.3, 900)
    off.put(key, FakeResponse("hola"))

    assert off.get(key) is None
    assert (off.hits, off.misses, off.writes) == (0, 0, 0)
    assert off.stats()["directory"] is None


# --- the client wrapper ----------------------------------------------


def test_a_repeated_request_reaches_the_provider_once(cache):
    inner = CountingClient()
    client = CachedClient(inner, cache)

    first = client.chat(MESSAGES, temperature=0.3, max_tokens=900)
    second = client.chat(MESSAGES, temperature=0.3, max_tokens=900)

    assert len(inner.calls) == 1
    assert first.choices[0].message.content == second.choices[0].message.content


def test_a_different_temperature_is_a_different_request(cache):
    # The guard that keeps sampled calls honest: same prompt, different
    # temperature, must not replay.
    inner = CountingClient()
    client = CachedClient(inner, cache)

    client.chat(MESSAGES, temperature=0.3, max_tokens=900)
    client.chat(MESSAGES, temperature=0.7, max_tokens=900)

    assert len(inner.calls) == 2


def test_a_different_prompt_is_a_different_request(cache):
    inner = CountingClient()
    client = CachedClient(inner, cache)

    client.chat(MESSAGES, temperature=0.3, max_tokens=900)
    client.chat(
        [{"role": "user", "content": "Something else entirely."}],
        temperature=0.3,
        max_tokens=900,
    )

    assert len(inner.calls) == 2


def test_an_unwrapped_client_repeats_every_call(cache):
    # States the property the judge depends on. The judge calls its
    # model three times at temperature 0.7 and aggregates the spread;
    # if that path were ever wrapped, the three answers would collapse
    # into one and the pipeline's noise floor would read as zero.
    inner = CountingClient(replies=["8.1", "7.4", "9.0"])

    answers = [inner.chat(MESSAGES, temperature=0.7).choices[0].message.content for _ in range(3)]

    assert len(inner.calls) == 3
    assert answers == ["8.1", "7.4", "9.0"]
