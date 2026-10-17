import asyncio
import random

import groq
import httpx
import pytest
from pydantic import BaseModel

from app.config import ConfigError, Settings
from llm.groq_client import ClientStats, GroqClient, LLMUnavailableError, call_with_retry, classify_error

S = Settings(groq_api_key="x", llm_max_attempts=4, llm_base_delay=1.0, llm_max_delay=20.0)


def status_error(code, headers=None):
    req = httpx.Request("POST", "https://api.groq.com/x")
    resp = httpx.Response(code, headers=headers or {}, request=req)
    cls = {429: groq.RateLimitError, 500: groq.InternalServerError, 503: groq.InternalServerError, 400: groq.BadRequestError}[code]
    return cls("boom", response=resp, body=None)


class Flaky:
    def __init__(self, errors, result="ok"):
        self.errors, self.result, self.calls = list(errors), result, 0

    async def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def run(coro):
    return asyncio.run(coro)


def test_retry_on_429_then_success_counts_one_call():
    sleeps = []
    async def fake_sleep(s): sleeps.append(s)
    fn, stats = Flaky([status_error(429)]), ClientStats()
    result, attempts = run(call_with_retry(fn, S, stats, fake_sleep, random.Random(0)))
    assert result == "ok" and attempts == 2 and fn.calls == 2
    assert stats.failed_attempts == 1 and len(sleeps) == 1 and 1.0 <= sleeps[0] <= 1.25


def test_exponential_backoff_between_attempts():
    sleeps = []
    async def fake_sleep(s): sleeps.append(s)
    run(call_with_retry(Flaky([status_error(503)] * 3), S, ClientStats(), fake_sleep, random.Random(0)))
    assert [round(s) for s in sleeps] == [1, 2, 4]


def test_gives_up_after_max_attempts():
    fn, stats = Flaky([status_error(429)] * 10), ClientStats()
    async def fake_sleep(s): pass
    with pytest.raises(LLMUnavailableError) as e:
        run(call_with_retry(fn, S, stats, fake_sleep))
    assert fn.calls == 4 and e.value.attempts == 4 and e.value.reason == "rate_limit" and stats.failed_attempts == 4


def test_retry_after_header_is_honoured():
    sleeps = []
    async def fake_sleep(s): sleeps.append(s)
    run(call_with_retry(Flaky([status_error(429, {"retry-after": "7"})]), S, ClientStats(), fake_sleep, random.Random(0)))
    assert sleeps[0] >= 7


def test_huge_retry_after_fails_fast_instead_of_hanging():
    fn = Flaky([status_error(429, {"retry-after": "3600"})])
    async def fake_sleep(s): raise AssertionError("must not wait")
    with pytest.raises(LLMUnavailableError) as e:
        run(call_with_retry(fn, S, ClientStats(), fake_sleep))
    assert fn.calls == 1 and e.value.retry_after == 3600


def test_non_retryable_error_propagates_immediately():
    fn = Flaky([status_error(400)])
    with pytest.raises(groq.BadRequestError):
        run(call_with_retry(fn, S, ClientStats()))
    assert fn.calls == 1


def test_timeouts_and_connection_errors_are_retryable():
    req = httpx.Request("POST", "https://x")
    assert classify_error(groq.APITimeoutError(request=req))[:2] == (True, "timeout")
    assert classify_error(groq.APIConnectionError(request=req))[:2] == (True, "server_error")
    assert classify_error(ValueError("x"))[0] is False


class Route(BaseModel):
    route: str


class FakeStructured:
    def __init__(self, outputs): self.outputs, self.n = list(outputs), 0
    async def ainvoke(self, messages):
        self.n += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception): raise out
        return out


class FakeChat:
    def __init__(self, outputs): self.s = FakeStructured(outputs); self.kwargs = None
    def with_structured_output(self, schema, **kw):
        self.kwargs = kw
        return self.s


class Raw:
    content = '{"route": "greeting"}'


def test_structured_uses_json_schema_strict_and_counts_calls():
    chat = FakeChat([{"raw": Raw(), "parsed": Route(route="greeting"), "parsing_error": None}])
    client = GroqClient(S, chat_model=chat)
    res = run(client.structured(Route, ["hi"]))
    assert chat.kwargs == {"method": "json_schema", "strict": True, "include_raw": True}
    assert res.parsed == Route(route="greeting") and res.llm_calls == 1 and res.attempts == 1
    assert client.stats.successes == 1


def test_structured_invalid_output_is_returned_not_raised():
    chat = FakeChat([{"raw": Raw(), "parsed": None, "parsing_error": ValueError("bad json")}])
    res = run(GroqClient(S, chat_model=chat).structured(Route, ["hi"]))
    assert res.parsed is None and "bad json" in res.parsing_error and res.raw_text


def test_structured_retries_then_counts_a_single_call():
    async def fake_sleep(s): pass
    ok = {"raw": Raw(), "parsed": Route(route="greeting"), "parsing_error": None}
    client = GroqClient(S, chat_model=FakeChat([status_error(429), ok]), sleep=fake_sleep)
    res = run(client.structured(Route, ["hi"]))
    assert res.attempts == 2 and res.llm_calls == 1 and client.stats.failed_attempts == 1 and client.stats.successes == 1


def test_missing_key_fails_with_clear_message():
    with pytest.raises(ConfigError, match="GROQ_API_KEY is not set"):
        Settings.from_env({}).require_key()
    assert Settings.from_env({"GROQ_API_KEY": ' "abc" '}).groq_api_key == "abc"
    assert "abc" not in repr(Settings.from_env({"GROQ_API_KEY": "abc"}))


def test_settings_defaults_match_decisions_and_overrides():
    s = Settings.from_env({})
    assert (s.router_model, s.history_turns, s.max_message_chars, s.host, s.port, s.llm_max_attempts) == \
        ("openai/gpt-oss-120b", 2, 2000, "127.0.0.1", 8000, 4)
    assert Settings.from_env({"APP_PORT": "9000", "MAX_MESSAGE_CHARS": "500"}).port == 9000
    with pytest.raises(ConfigError):
        Settings.from_env({"APP_PORT": "abc"})
