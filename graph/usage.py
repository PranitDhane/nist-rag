"""Token usage per message (GRAPH_INTEGRATION_PLAN Section 7: meta.usage = total tokens across the message's LLM calls).

The router's tokens are not part of its node's return value, so a tiny client proxy records them into a per-request
ContextVar that the classify wrapper reads. Generation's tokens come from its GenerationResult."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_current: ContextVar[list | None] = ContextVar("router_token_usage", default=None)


@contextmanager
def collect() -> Iterator[list]:
    tokens: list[int] = []
    reset = _current.set(tokens)
    try:
        yield tokens
    finally:
        _current.reset(reset)


def record(tokens: int) -> None:
    bucket = _current.get()
    if bucket is not None:
        bucket.append(int(tokens or 0))


class TokenTrackingClient:
    """Wraps the router's Groq client: every successful structured() call records its reported total tokens."""

    def __init__(self, inner: Any):
        self._inner = inner

    async def structured(self, schema, messages):
        result = await self._inner.structured(schema, messages)
        record(getattr(result, "total_tokens", 0))
        return result

    def __getattr__(self, name: str):             # stats, settings, ping ... pass through
        return getattr(self._inner, name)


def total(router_tokens: int = 0, generation_tokens: int = 0) -> dict:
    return {"router": router_tokens, "generation": generation_tokens, "total": router_tokens + generation_tokens}
