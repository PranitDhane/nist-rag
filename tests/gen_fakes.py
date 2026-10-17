"""Shared fakes for the generation tests: a fake ChatGroq (invoke + stream), retrieval fixtures, error builders."""
from __future__ import annotations

import asyncio

import groq
import httpx
from langchain_core.messages import AIMessage, AIMessageChunk

from retrieval.models import RetrievalResult, RetrievedChunk

KEY_MAP = {"C1": "ai:1", "C2": "ai:2", "C3": "gen:1"}
USAGE = {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500, "output_token_details": {"reasoning": 220}}


def chunk(cid, text, **meta):
    m = {"chunk_id": cid, "display_name": "NIST AI 100-1 (AI RMF 1.0)", "chunk_type": "prose",
         "section_path": ["Part 2", "5 AI RMF Core"], "part": 1, "parts_total": 1, "subcategory": None, "category": None, **meta}
    return RetrievedChunk(chunk_id=cid, display_text=text, metadata=m, rerank_score=0.9, rerank_rank=1)


def result(n=3, relevant=True):
    chunks = [chunk("ai:1", "The AI RMF Core has four functions: GOVERN, MAP, MEASURE and MANAGE."),
              chunk("ai:2", "GOVERN is a cross-cutting function.", section_path=["Part 2", "5.1 Govern"]),
              chunk("gen:1", "GV-1.7-001 - Suggested action: decommission.", chunk_type="genai_action_table",
                    subcategory="GOVERN 1.7", display_name="NIST AI 600-1 (Generative AI Profile)")][:n]
    return RetrievalResult(query="q", chunks=chunks, relevant=relevant, best_score=0.9, threshold=0.15, candidate_count=10)


def status_error(code):
    req = httpx.Request("POST", "https://api.groq.com/x")
    resp = httpx.Response(code, request=req)
    cls = {429: groq.RateLimitError, 503: groq.InternalServerError}[code]
    return cls("boom", response=resp, body=None)


class FakeChat:
    """ainvoke() returns `text`; astream() yields `chunks` (strings, or ("reasoning", str), or AIMessageChunk).
    `errors` are raised one per call before anything is produced; `fail_after` raises after that many yielded chunks;
    `hang` blocks the stream start (TTFT timeout tests)."""

    def __init__(self, text="", chunks=None, errors=(), fail_after=None, finish="stop", usage=USAGE, hang=False):
        self.text, self.chunks, self.errors = text, list(chunks if chunks is not None else [text]), list(errors)
        self.fail_after, self.finish, self.usage, self.hang = fail_after, finish, usage, hang
        self.calls = 0
        self.last_messages = None

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        if self.errors:
            raise self.errors.pop(0)
        return AIMessage(content=self.text, usage_metadata=self.usage, response_metadata={"finish_reason": self.finish})

    async def astream(self, messages):
        self.calls += 1
        self.last_messages = messages
        if self.errors:
            raise self.errors.pop(0)
        if self.hang:
            await asyncio.sleep(3600)
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise status_error(503)
            if isinstance(c, tuple):
                yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": c[1]})
            elif isinstance(c, AIMessageChunk):
                yield c
            else:
                yield AIMessageChunk(content=c)
        yield AIMessageChunk(content="", usage_metadata=self.usage, response_metadata={"finish_reason": self.finish})
