"""Stage N2/N3: generate() and generate_stream(): build the prompt, call Groq, parse, fix, return a GenerationResult."""
from __future__ import annotations

import logging
import math
from typing import AsyncIterator

from app.config import Settings
from augmentation import config as aug_config
from augmentation.messages import build_messages
from generation import config
from generation.client import GenerationClient, MidStreamError
from generation.fixer import CASE_D, apply_fixes
from generation.models import (ANSWERED, ERROR, INSUFFICIENT, GenerationError, GenerationResult, LLMOutput)
from retrieval.models import RetrievalResult

log = logging.getLogger(__name__)


def estimate_tokens(messages: list) -> int:
    chars = sum(len(m.content) for m in messages)
    return math.ceil(chars / config.CHARS_PER_TOKEN * config.ESTIMATE_MARGIN)


def resolve_max_tokens(measurement_mode: bool, override: int | None = None) -> int:
    if override is not None:
        return override
    if config.MAX_COMPLETION_TOKENS is not None:
        return config.MAX_COMPLETION_TOKENS
    if measurement_mode:
        return config.MEASUREMENT_MAX_COMPLETION_TOKENS
    raise GenerationError("config", "MAX_COMPLETION_TOKENS is not set (Stage N4); live generation refuses to run "
                                    "without it except in measurement mode")


def precheck_budget(messages: list, max_tokens: int, tpm_limit: int | None = None) -> int:
    """Fail fast when prompt + maximum output cannot fit the per-minute token budget. Returns the estimate."""
    limit = config.TPM_LIMIT if tpm_limit is None else tpm_limit
    est = estimate_tokens(messages)
    if est + max_tokens > limit:
        raise GenerationError("budget", f"prompt (~{est} tokens) + max output ({max_tokens}) exceeds the "
                                        f"{limit} tokens/minute limit")
    return est


def _meta(result: GenerationResult, client: GenerationClient) -> GenerationResult:
    result.model, result.prompt_version = config.GEN_MODEL, aug_config.PROMPT_VERSION
    result.reasoning_effort, result.temperature = config.REASONING_EFFORT, config.TEMPERATURE
    return result


def error_result(err: GenerationError, key_map: dict, client: GenerationClient, streamed: str | None = None) -> GenerationResult:
    return _meta(GenerationResult(outcome=ERROR, final_text=None, streamed_text=None, key_map=key_map,
                                  error={"code": err.code, "message": err.message}), client)


def finalize(out: LLMOutput, key_map: dict, client: GenerationClient, *, streamed_text: str | None = None,
             measurement_mode: bool = False) -> GenerationResult:
    """LLM output -> GenerationResult: length policy, apply_fixes, bookkeeping. Raises GenerationError on policy 'error'."""
    extra_fixes: list[str] = []
    extra_warnings: list[str] = []
    text = out.text
    if out.finish_reason == "length":
        policy = config.LENGTH_POLICY or ("keep" if measurement_mode else None)
        if policy is None:
            raise GenerationError("config", "LENGTH_POLICY is not set (Stage N4)")
        log.warning("generation hit the output limit (finish_reason=length, usage=%s)", out.usage)
        if policy == "error":
            raise GenerationError("length", "the answer was cut off at the output limit")
        if policy == "insufficient":
            text, extra_fixes = CASE_D, ["LENGTH"]
        else:
            extra_warnings = ["TRUNCATED"]
    fx = apply_fixes(text, key_map)
    result = GenerationResult(
        outcome=fx.outcome, final_text=fx.final_text, streamed_text=streamed_text,
        corrected=streamed_text is not None and fx.final_text != streamed_text,
        cited_keys=list(fx.parsed.cited_keys) if fx.parsed else [], key_map=key_map,
        is_ambiguous_format=bool(fx.parsed and fx.parsed.is_ambiguous_format),
        fixes=extra_fixes + fx.fixes, warnings=extra_warnings + fx.warnings, usage=out.usage,
        finish_reason=out.finish_reason, timings_ms={"ttft": out.ttft_ms, "total": out.total_ms})
    for code in result.fixes:
        log.info("generation fix applied: %s", code)
    for code in result.warnings:
        log.warning("generation warning: %s", code)
    return _meta(result, client)


async def generate(standalone_query: str, retrieval_result: RetrievalResult, *, client: GenerationClient | None = None,
                   settings: Settings | None = None, measurement_mode: bool = False,
                   max_tokens: int | None = None) -> GenerationResult:
    """Non-streaming generation. Empty context and missing config raise GenerationError; runtime failures
    (budget, rate limit, timeout, length policy 'error') come back as outcome ERROR with error={code, message}."""
    if not retrieval_result.chunks:
        raise GenerationError("empty_context", "generation needs at least one retrieved chunk")
    messages, key_map = build_messages(standalone_query, retrieval_result)
    limit = resolve_max_tokens(measurement_mode, max_tokens)
    client = client or GenerationClient(settings or Settings.from_env(), max_tokens=limit)
    try:
        if not measurement_mode:
            precheck_budget(messages, limit)
        out = await client.complete(messages)
        return finalize(out, key_map, client, measurement_mode=measurement_mode)
    except GenerationError as e:
        if e.code == "config":
            raise
        log.error("generation failed: %s (%s)", e.code, e.message)
        return error_result(e, key_map, client)


async def generate_stream(standalone_query: str, retrieval_result: RetrievalResult, *,
                          client: GenerationClient | None = None, settings: Settings | None = None,
                          measurement_mode: bool = False, max_tokens: int | None = None) -> AsyncIterator[dict]:
    """Yields {"type": "token", "text"}, then optionally {"type": "correction", "text", "reasons"}, then
    {"type": "result", "result": GenerationResult}; or {"type": "error", "code", "message"} (then it stops).
    A failure after the first token discards the partial text: no correction, no result (GN-D9)."""
    if not retrieval_result.chunks:
        raise GenerationError("empty_context", "generation needs at least one retrieved chunk")
    messages, key_map = build_messages(standalone_query, retrieval_result)
    limit = resolve_max_tokens(measurement_mode, max_tokens)
    client = client or GenerationClient(settings or Settings.from_env(), max_tokens=limit)
    try:
        if not measurement_mode:
            precheck_budget(messages, limit)
        out: LLMOutput | None = None
        async for kind, value in client.stream(messages):
            if kind == "token":
                yield {"type": "token", "text": value}
            else:
                out = value
        result = finalize(out, key_map, client, streamed_text=out.text, measurement_mode=measurement_mode)
    except GenerationError as e:
        if e.code == "config":
            raise
        log.error("streaming generation failed: %s (%s)", e.code, e.message)
        yield {"type": "error", "code": e.code, "message": e.message}
        return
    if result.corrected:
        yield {"type": "correction", "text": result.final_text,
               "reasons": list(result.fixes)}
    yield {"type": "result", "result": result}
