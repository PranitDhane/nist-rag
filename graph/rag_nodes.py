"""Nodes of the RAG subgraph (GRAPH_INTEGRATION_PLAN Stage G3). Each is a THIN wrapper around an existing module function:
no module internals are changed here. Every node: appends its name to `path`, records its duration in `timings_ms`, and
turns any failure into `error = {code, message, node}` (logged at ERROR with the stack trace; never silently swallowed).
"""
from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from langchain_core.runnables import RunnableConfig

from attribution import format as fmt
from attribution.build import build_sources
from graph import errors
from graph.components import Components, run_blocking
from graph.serial import chunk_from_dict, chunk_to_dict, gate_result_from_state, retrieval_result_from_state
from indexing.metadata import to_chroma_metadata
from insufficient.reply import finalize_model_refusal, gate_failure_reply, log_insufficient
from retrieval import config as retrieval_config, ordering, siblings
from retrieval.fusion import fuse
from retrieval.gate import apply_gate
from retrieval.rerank import Scored, score_candidates, select_top

log = logging.getLogger(__name__)

NodeFn = Callable[[dict, RunnableConfig], Awaitable[dict]]


def instrumented(name: str, error_code: str):
    """Decorator: path + timing + error capture. The wrapped function takes (state, config) and returns a state update
    (or raises)."""
    def deco(fn: NodeFn) -> NodeFn:
        async def node(state: dict, config: RunnableConfig) -> dict:
            started = time.perf_counter()
            try:
                update = dict(await fn(state, config))
            except Exception:                                  # noqa: BLE001 - recorded in state, never propagated (GI-D4)
                log.exception("node %s failed", name)
                update = {"error": errors.make_error(error_code, name)}
            update["path"] = [*(state.get("path") or []), name]
            update["timings_ms"] = {**(state.get("timings_ms") or {}), name: round((time.perf_counter() - started) * 1000, 1)}
            return update
        node.__name__ = name
        return node
    return deco


def _thread_id(config: RunnableConfig) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id", "-")


def make_rag_nodes(comps: Components) -> dict[str, NodeFn]:
    lookup = comps.chunk_lookup

    @instrumented("hybrid_search", errors.RETRIEVAL_FAILED)
    async def hybrid_search(state: dict, config: RunnableConfig) -> dict:
        query = state["standalone_query"]

        def work():
            r = comps.retrieval
            t = time.perf_counter()
            vector_docs = r.vector.invoke(query)
            t_vec = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            bm25_docs = r.bm25.invoke(query)
            t_bm25 = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            fused = fuse(r.ensemble, vector_docs, bm25_docs)
            return fused, {"vector": round(t_vec, 2), "bm25": round(t_bm25, 2), "fusion": round((time.perf_counter() - t) * 1000, 2)}

        fused, timings = await run_blocking(work)
        candidates = []
        for pos, doc in enumerate(fused, start=1):
            rec = lookup[doc.metadata["chunk_id"]]
            candidates.append({"chunk_id": rec["chunk_id"], "citation_label": fmt.citation_label(rec), "fused_position": pos,
                               "display_text": rec["display_text"], "rerank_score": None, "rerank_rank": None})
        return {"retrieval": {"query": query, "candidate_count": len(candidates), "config_version": retrieval_config.config_version(),
                              "timings_ms": timings, "candidates": candidates, "top": [], "best_score": None, "threshold": None,
                              "relevant": None, "chunks": [], "siblings_added": []}}

    @instrumented("rerank", errors.RERANK_FAILED)
    async def rerank(state: dict, config: RunnableConfig) -> dict:
        retrieval = state["retrieval"]
        query, ids = retrieval["query"], [c["chunk_id"] for c in retrieval["candidates"]]

        def work():
            with comps.rerank_lock:
                t = time.perf_counter()
                scored = score_candidates(comps.reranker, query, ids, lookup)
                return scored, round((time.perf_counter() - t) * 1000, 2)

        scored, ms = await run_blocking(work)
        by_id = {s.chunk_id: s for s in scored}
        candidates = [{**c, "rerank_score": by_id[c["chunk_id"]].score, "rerank_rank": by_id[c["chunk_id"]].rank}
                      for c in retrieval["candidates"]]
        top = [chunk_to_dict(siblings.make_chunk(s.chunk_id, to_chroma_metadata(lookup[s.chunk_id]), lookup[s.chunk_id]["embed_text"],
                                                 score=s.score, rank=s.rank)) for s in select_top(scored)]
        return {"retrieval": {**retrieval, "candidates": candidates, "top": top,
                              "timings_ms": {**retrieval["timings_ms"], "rerank": ms}}}

    @instrumented("gate", errors.INTERNAL)
    async def gate(state: dict, config: RunnableConfig) -> dict:
        retrieval = state["retrieval"]
        top = [Scored(c["chunk_id"], c["rerank_score"], 0, c["rerank_rank"]) for c in retrieval["top"]]
        result = apply_gate(top)
        kept = {s.chunk_id for s in result.kept}
        return {"retrieval": {**retrieval, "best_score": result.best_score, "threshold": result.threshold,
                              "relevant": result.relevant, "top": [c for c in retrieval["top"] if c["chunk_id"] in kept]}}

    @instrumented("expand_context", errors.RETRIEVAL_FAILED)
    async def expand_context(state: dict, config: RunnableConfig) -> dict:
        retrieval = state["retrieval"]
        selected = [chunk_from_dict(c) for c in retrieval["top"]]

        def work():
            return ordering.final_order(siblings.expand(comps.retrieval.store, selected))

        ordered = await run_blocking(work)
        return {"retrieval": {**retrieval, "chunks": [chunk_to_dict(c) for c in ordered],
                              "siblings_added": [c.chunk_id for c in ordered if c.added_as_sibling]}}

    @instrumented("gate_failure_reply", errors.INTERNAL)
    async def gate_failure_node(state: dict, config: RunnableConfig) -> dict:
        reply = gate_failure_reply(gate_result_from_state(state["retrieval"]))
        log_insufficient(reply, _thread_id(config), state.get("llm_calls", 0))
        return _insufficient_update(reply)

    @instrumented("finalize_model_refusal", errors.INTERNAL)
    async def model_refusal_node(state: dict, config: RunnableConfig) -> dict:
        reply = finalize_model_refusal(state["generation"], retrieval_result_from_state(state["retrieval"]))
        log_insufficient(reply, _thread_id(config), state.get("llm_calls", 0))
        return _insufficient_update(reply)

    @instrumented("attribute", errors.INTERNAL)
    async def attribute(state: dict, config: RunnableConfig) -> dict:
        gen = state["generation"]
        entries = build_sources(gen["final_text"], list(gen["cited_keys"]), dict(gen["key_map"]), lookup)
        return {"sources": [e.to_dict() for e in entries], "outcome": "answered", "insufficient": None,
                "related_sections": [], "insufficient_reason": None}

    return {"hybrid_search": hybrid_search, "rerank": rerank, "gate": gate, "expand_context": expand_context,
            "gate_failure_reply": gate_failure_node, "finalize_model_refusal": model_refusal_node,
            "attribute": attribute}


def _insufficient_update(reply) -> dict:
    d = reply.to_dict()
    return {"answer": reply.answer, "sources": [], "outcome": "insufficient", "insufficient": d,
            "related_sections": d["related_sections"], "insufficient_reason": d["reason"]}
