"""Request / response models (API_ROUTING_PLAN Stage R6)."""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    debug: bool = False                    # GI-D6: adds meta.debug (ids, labels, scores, timings, versions; never text)

    @field_validator("message")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v

    @field_validator("thread_id")
    @classmethod
    def _uuid(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        try:
            return str(uuid.UUID(v))
        except ValueError:
            raise ValueError("thread_id must be a UUID returned by an earlier response") from None


class Meta(BaseModel):
    llm_calls: int
    route_source: Literal["rule", "llm", "fallback"] | None
    latency_ms: int
    # GRAPH_INTEGRATION_PLAN Section 7 (Q9: on every route). Fields below are always present...
    outcome: Literal["answered", "insufficient", "greeting", "out_of_scope"] | None = None
    insufficient_reason: str | None = None       # INSUFFICIENT_CONTEXT_PLAN Section 3 codes; null otherwise
    usage: dict | None = None                    # {router, generation, total} tokens across the message's LLM calls
    # ...and these only when generation ran:
    model: str | None = None
    prompt_version: str | None = None
    fixes: list[str] | None = None
    warnings: list[str] | None = None
    debug: dict | None = None                    # only with debug=true


class RelatedSection(BaseModel):
    """A closest section found for an insufficient answer. No chunk text, no score (IC-D3, Q3)."""
    doc: str
    section: str
    pages: str
    citation_label: str
    citation_url: str
    kind: Literal["prose", "table", "figure-ocr", "footnote"]


class Source(BaseModel):
    """One Sources list entry (ATTRIBUTION_PLAN Section 4). `keys` are the [C#] markers in the answer that point here."""
    keys: list[str]
    label: str
    url: str
    kind: Literal["prose", "table", "figure-ocr", "footnote"]


class ChatResponse(BaseModel):
    thread_id: str
    route: Literal["greeting", "nist_question", "out_of_scope"]
    answer: str
    sources: list[Source] = []
    standalone_query: str | None = None
    related_sections: list[RelatedSection] | None = None   # NIST route: always present (empty when answered)
    meta: Meta


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
