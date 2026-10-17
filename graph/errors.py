"""Error codes, generic messages and the node error helper (GRAPH_INTEGRATION_PLAN Section 5, GI-D4).

Every node catches its own failure and writes `error = {code, message, node}` into state. The message is GENERIC: it never
carries a stack trace or an internal detail (those are logged at ERROR with the node name). The HTTP status mapping used
by the API is HTTP_STATUS (approved in Stage G6, Q4)."""
from __future__ import annotations

LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
RERANK_FAILED = "RERANK_FAILED"
GENERATION_FAILED = "GENERATION_FAILED"
INTERNAL = "INTERNAL"

GENERIC_MESSAGES = {
    LLM_UNAVAILABLE: "The language model is temporarily unavailable. Please retry shortly.",
    RETRIEVAL_FAILED: "The document search failed. Please try again.",
    RERANK_FAILED: "The document search failed. Please try again.",
    GENERATION_FAILED: "The answer could not be generated. Please try again.",
    INTERNAL: "Internal server error.",
}

# Generation phase error codes (generation.models.GenerationError.code) -> graph error code.
# Groq retries exhausted (rate limit / 5xx / API timeout / unreachable) and generation timeouts are LLM_UNAVAILABLE (Q4);
# everything else that fails while generating is GENERATION_FAILED.
GENERATION_CODE_MAP = {
    "rate_limit": LLM_UNAVAILABLE, "server_error": LLM_UNAVAILABLE, "timeout": LLM_UNAVAILABLE, "unavailable": LLM_UNAVAILABLE,
    "ttft_timeout": LLM_UNAVAILABLE, "total_timeout": LLM_UNAVAILABLE,       # Q4: an unresponsive model is 'unavailable' (503)
    "budget": GENERATION_FAILED, "midstream": GENERATION_FAILED, "length": GENERATION_FAILED, "config": INTERNAL,
    "empty_context": INTERNAL,
}

# HTTP status for /chat (approved in Stage G6, Q4). LLM_UNAVAILABLE keeps the API phase's 503 + Retry-After.
HTTP_STATUS = {LLM_UNAVAILABLE: 503, RETRIEVAL_FAILED: 500, RERANK_FAILED: 500, GENERATION_FAILED: 500, INTERNAL: 500}


def make_error(code: str, node: str) -> dict:
    return {"code": code, "message": GENERIC_MESSAGES[code], "node": node}


def from_generation_code(gen_code: str, node: str) -> dict:
    return make_error(GENERATION_CODE_MAP.get(gen_code, GENERATION_FAILED), node)
