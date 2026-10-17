"""Groq client: ChatGroq setup, retries with exponential backoff, call counting (Stage R1).

Retries only rate limits (429), timeouts/connection errors and 5xx. A `retry-after` header is honoured, but if
it asks for longer than `llm_max_delay` (e.g. a daily quota) we fail fast instead of hanging the request.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from app.config import Settings

log = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """The LLM could not be reached after retries (or retrying would take too long)."""

    def __init__(self, message: str, *, reason: str = "unavailable", retry_after: float | None = None, attempts: int = 0):
        super().__init__(message)
        self.reason = reason            # rate_limit | server_error | timeout | unavailable
        self.retry_after = retry_after  # seconds the caller may suggest to the client
        self.attempts = attempts


@dataclass
class StructuredResult:
    parsed: BaseModel | None
    raw_text: str
    parsing_error: str | None
    attempts: int                       # HTTP attempts used (1 = no retry)
    llm_calls: int = 1                  # a successful call counts once, however many retries it took
    total_tokens: int = 0               # as reported by Groq; used to pace the evaluation under the tokens/minute limit


@dataclass
class ClientStats:
    successes: int = 0
    failed_attempts: int = 0            # retried / exhausted attempts, logged separately from llm_calls
    tokens: int = 0


def classify_error(exc: BaseException) -> tuple[bool, str, float | None]:
    """(retryable, reason, retry_after_seconds) for an exception raised by the groq SDK."""
    import groq
    retry_after = None
    response = getattr(exc, "response", None)
    if response is not None:
        raw = response.headers.get("retry-after")
        try:
            retry_after = float(raw) if raw is not None else None
        except ValueError:
            retry_after = None
    if isinstance(exc, groq.RateLimitError):
        return True, "rate_limit", retry_after
    if isinstance(exc, groq.APITimeoutError):
        return True, "timeout", retry_after
    if isinstance(exc, groq.APIConnectionError):
        return True, "server_error", retry_after
    if isinstance(exc, groq.APIStatusError):
        code = exc.status_code
        if code == 429:
            return True, "rate_limit", retry_after
        if code == 408 or code >= 500:
            return True, "server_error", retry_after
    return False, "other", None


async def call_with_retry(fn: Callable[[], Awaitable[Any]], settings: Settings, stats: ClientStats | None = None,
                          sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                          rng: random.Random | None = None) -> tuple[Any, int]:
    """Run `fn` with backoff. Returns (result, attempts). Raises LLMUnavailableError when retries are exhausted."""
    rng = rng or random.Random()
    last_reason, last_after = "unavailable", None
    for attempt in range(1, settings.llm_max_attempts + 1):
        try:
            return await fn(), attempt
        except Exception as exc:                          # noqa: BLE001 - classified below, others re-raised
            retryable, reason, retry_after = classify_error(exc)
            if not retryable:
                raise
            last_reason, last_after = reason, retry_after
            if stats:
                stats.failed_attempts += 1
            log.warning("llm attempt %d/%d failed (%s, retry-after=%s)", attempt, settings.llm_max_attempts, reason, retry_after)
            if attempt == settings.llm_max_attempts:
                break
            delay = min(settings.llm_max_delay, settings.llm_base_delay * 2 ** (attempt - 1))
            if retry_after is not None:
                if retry_after > settings.llm_max_delay:       # e.g. daily quota: waiting would hang the request
                    raise LLMUnavailableError(f"Groq asked to wait {retry_after:.0f}s ({reason})", reason=reason,
                                              retry_after=retry_after, attempts=attempt) from exc
                delay = max(delay, retry_after)
            await sleep(delay + rng.uniform(0, 0.25 * delay))
    raise LLMUnavailableError(f"Groq unavailable after {settings.llm_max_attempts} attempts ({last_reason})",
                              reason=last_reason, retry_after=last_after, attempts=settings.llm_max_attempts)


class GroqClient:
    """Structured-output calls to Groq. `chat_model` can be injected (tests)."""

    def __init__(self, settings: Settings, chat_model: Any = None, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.settings = settings
        self.stats = ClientStats()
        self._sleep = sleep
        self._chat_model = chat_model

    @property
    def chat_model(self):
        if self._chat_model is None:
            from langchain_groq import ChatGroq
            s = self.settings.require_key()
            kwargs: dict[str, Any] = {}
            if s.router_model.startswith("openai/gpt-oss"):
                kwargs["reasoning_effort"] = s.router_reasoning_effort
            self._chat_model = ChatGroq(model=s.router_model, api_key=s.groq_api_key, temperature=s.router_temperature,
                                        max_retries=0,                      # our backoff is the only retry layer
                                        request_timeout=s.llm_timeout, **kwargs)
        return self._chat_model

    async def structured(self, schema: type[BaseModel], messages: list) -> StructuredResult:
        """One structured-output call (json_schema, strict). Invalid model output is NOT an exception: it comes
        back as parsed=None with parsing_error and the raw text, so the caller can retry once and fall back."""
        runnable = self.chat_model.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)
        out, attempts = await call_with_retry(lambda: runnable.ainvoke(messages), self.settings, self.stats, self._sleep)
        self.stats.successes += 1
        raw = out.get("raw")
        err = out.get("parsing_error")
        tokens = int((getattr(raw, "usage_metadata", None) or {}).get("total_tokens", 0))
        self.stats.tokens += tokens
        return StructuredResult(out.get("parsed"), getattr(raw, "content", "") or "", str(err) if err else None, attempts,
                                total_tokens=tokens)

    async def ping(self) -> tuple[str, float]:
        """Tiny live call for diagnostics: returns (reply, seconds)."""
        started = time.perf_counter()
        out, _ = await call_with_retry(lambda: self.chat_model.ainvoke("Reply with the single word: ok"), self.settings,
                                       self.stats, self._sleep)
        self.stats.successes += 1
        return str(out.content).strip(), time.perf_counter() - started
