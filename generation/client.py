"""Generation client (Stage N1/N3): a ChatGroq instance separate from the router's, built through the router's
retry helpers (llm.groq_client.call_with_retry / ClientStats), so backoff and counting are shared.

Streaming rules: retries happen only before the first CONTENT token; reasoning chunks are never yielded;
a failure after the first token raises MidStreamError (the caller emits an error event and discards the text)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from app.config import Settings
from generation import config
from generation.models import GenerationError, LLMOutput
from llm.groq_client import ClientStats, LLMUnavailableError, call_with_retry, classify_error

log = logging.getLogger(__name__)


class MidStreamError(GenerationError):
    def __init__(self, message: str):
        super().__init__("midstream", message)


def usage_from(meta: dict | None) -> dict:
    """langchain usage_metadata -> {prompt, completion, reasoning, total}. Reasoning tokens are INCLUDED in completion."""
    meta = meta or {}
    reasoning = ((meta.get("output_token_details") or {}).get("reasoning")) or 0
    return {"prompt": int(meta.get("input_tokens", 0)), "completion": int(meta.get("output_tokens", 0)),
            "reasoning": int(reasoning), "total": int(meta.get("total_tokens", 0))}


def _reasoning_in(chunk) -> bool:
    return bool((getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content"))


def _text(chunk) -> str:
    c = chunk.content
    return c if isinstance(c, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


class GenerationClient:
    def __init__(self, settings: Settings, chat_model: Any = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, *, max_tokens: int | None = None,
                 ttft_timeout: float | None = None, total_timeout: float | None = None):
        self.settings = settings
        self.stats = ClientStats()
        self._sleep = sleep
        self._chat_model = chat_model
        self.max_tokens = max_tokens
        self.ttft_timeout = config.TTFT_TIMEOUT_S if ttft_timeout is None else ttft_timeout
        self.total_timeout = config.TOTAL_TIMEOUT_S if total_timeout is None else total_timeout

    @property
    def chat_model(self):
        if self._chat_model is None:
            from langchain_groq import ChatGroq
            s = self.settings.require_key()
            self._chat_model = ChatGroq(
                model=config.GEN_MODEL, api_key=s.groq_api_key, temperature=config.TEMPERATURE,
                reasoning_effort=config.REASONING_EFFORT, reasoning_format=config.REASONING_FORMAT,
                max_tokens=self.max_tokens, max_retries=0,          # our backoff is the only retry layer
                request_timeout=self.total_timeout, tags=[config.ANSWER_TAG])
        return self._chat_model

    # ---------------------------------------------------------------- non-streaming
    async def complete(self, messages: list) -> LLMOutput:
        started = time.perf_counter()
        try:
            msg, attempts = await call_with_retry(lambda: asyncio.wait_for(self.chat_model.ainvoke(messages), self.total_timeout),
                                                  self.settings, self.stats, self._sleep)
        except LLMUnavailableError as e:
            raise GenerationError(e.reason if e.reason in ("rate_limit", "server_error", "timeout") else "unavailable",
                                  str(e)) from e
        except asyncio.TimeoutError as e:
            raise GenerationError("total_timeout", f"generation exceeded {self.total_timeout:.0f}s") from e
        self.stats.successes += 1
        usage = usage_from(getattr(msg, "usage_metadata", None))
        self.stats.tokens += usage["total"]
        return LLMOutput(text=_text(msg), usage=usage, finish_reason=(msg.response_metadata or {}).get("finish_reason"),
                         ttft_ms=None, total_ms=round((time.perf_counter() - started) * 1000, 1), attempts=attempts)

    # ---------------------------------------------------------------- streaming
    async def stream(self, messages: list) -> AsyncIterator[tuple[str, Any]]:
        """Yields ("token", text) for content only, then ("final", LLMOutput). Raises GenerationError / MidStreamError."""
        started = time.perf_counter()
        usage_meta: dict | None = None
        finish: str | None = None

        async def open_stream():
            agen = self.chat_model.astream(messages).__aiter__()
            pending_meta: list = []
            try:
                while True:
                    chunk = await asyncio.wait_for(agen.__anext__(), self.ttft_timeout)
                    pending_meta.append(chunk)
                    if not _reasoning_in(chunk) and _text(chunk):
                        return agen, pending_meta
            except StopAsyncIteration:
                return agen, pending_meta                  # ended without content (e.g. all tokens used by reasoning)

        try:
            (agen, first_chunks), attempts = await call_with_retry(open_stream, self.settings, self.stats, self._sleep)
        except LLMUnavailableError as e:
            raise GenerationError(e.reason if e.reason in ("rate_limit", "server_error", "timeout") else "unavailable",
                                  str(e)) from e
        except asyncio.TimeoutError as e:
            raise GenerationError("ttft_timeout", f"no content token within {self.ttft_timeout:.0f}s") from e

        parts: list[str] = []
        ttft_ms = None

        def absorb(chunk):
            nonlocal usage_meta, finish
            if getattr(chunk, "usage_metadata", None):
                usage_meta = chunk.usage_metadata
            fr = (getattr(chunk, "response_metadata", None) or {}).get("finish_reason")
            if fr:
                finish = fr

        try:
            for chunk in first_chunks:
                absorb(chunk)
                if not _reasoning_in(chunk) and _text(chunk):
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started) * 1000, 1)
                    parts.append(_text(chunk))
                    yield "token", _text(chunk)
            remaining = self.total_timeout - (time.perf_counter() - started)
            async with asyncio.timeout(max(remaining, 0.001)):
                async for chunk in agen:
                    absorb(chunk)
                    if _reasoning_in(chunk) or not _text(chunk):
                        continue                               # reasoning (or empty) chunks never reach the client
                    parts.append(_text(chunk))
                    yield "token", _text(chunk)
        except asyncio.TimeoutError as e:
            raise MidStreamError(f"generation exceeded {self.total_timeout:.0f}s") from e
        except GenerationError:
            raise
        except Exception as e:                                 # noqa: BLE001 - any failure after the first token
            retryable, reason, _ = classify_error(e)
            raise MidStreamError(f"stream failed after {len(parts)} chunks ({reason if retryable else type(e).__name__})") from e
        self.stats.successes += 1
        usage = usage_from(usage_meta)
        self.stats.tokens += usage["total"]
        yield "final", LLMOutput(text="".join(parts), usage=usage, finish_reason=finish, ttft_ms=ttft_ms,
                                 total_ms=round((time.perf_counter() - started) * 1000, 1), attempts=attempts)
