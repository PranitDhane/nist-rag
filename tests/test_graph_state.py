"""Stage G1: state definition, plain-data rule, checkpoint round trip, per-message reset."""
from __future__ import annotations

import asyncio
import warnings

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from graph.serial import (chunk_from_dict, chunk_to_dict, gate_result_from_state, plain_data_violations,
                          retrieval_result_from_state)
from graph.state import PER_MESSAGE_DEFAULTS, GraphState
from tests.gen_fakes import result


def sample_retrieval() -> dict:
    res = result()
    chunks = [chunk_to_dict(c) for c in res.chunks]
    return {"query": "Which function is cross-cutting?", "candidate_count": 12, "config_version": "abc123",
            "timings_ms": {"vector": 20.5, "bm25": 1.0},
            "candidates": [{"chunk_id": c["chunk_id"], "citation_label": "NIST AI 100-1, §5, p. 20", "fused_position": i,
                            "display_text": c["display_text"], "rerank_score": 0.9 - i / 10, "rerank_rank": i}
                           for i, c in enumerate(chunks, start=1)],
            "top": chunks, "best_score": 0.9, "threshold": 0.15, "relevant": True, "chunks": chunks, "siblings_added": []}


def sample_generation() -> dict:
    return {"outcome": "ANSWERED", "final_text": "GOVERN [C2].", "streamed_text": "GOVERN [C2].", "corrected": False,
            "cited_keys": ["C2"], "key_map": {"C1": "ai:1", "C2": "ai:2"}, "is_ambiguous_format": False, "fixes": [],
            "warnings": [], "usage": {"prompt": 1200, "completion": 300, "reasoning": 220, "total": 1500},
            "finish_reason": "stop", "timings_ms": {"ttft": 900.0, "total": 1200.0}, "model": "openai/gpt-oss-20b",
            "prompt_version": "v1", "reasoning_effort": "medium", "temperature": 0.2, "error": None}


def full_state() -> dict:
    return {"user_message": "q", "route": "nist_question", "route_source": "llm", "standalone_query": "q?",
            "route_reason": "r", "answer": "GOVERN [C2].", "sources": [{"keys": ["C2"], "label": "l", "url": "u", "kind": "prose"}],
            "llm_calls": 2, "error": None, "retry_after": None, "retrieval": sample_retrieval(), "generation": sample_generation(),
            "cited_keys": ["C2"], "key_map": {"C2": "ai:2"}, "insufficient": None, "related_sections": [],
            "insufficient_reason": None, "outcome": "answered", "path": ["hybrid_search", "rerank"], "timings_ms": {"rerank": 700.0},
            "usage": {"total": 2800}}


def test_per_message_defaults_cover_every_new_field():
    new = {"retrieval", "generation", "cited_keys", "key_map", "insufficient", "related_sections", "insufficient_reason",
           "outcome", "path", "timings_ms", "usage", "error", "answer", "sources", "llm_calls"}
    assert new <= set(PER_MESSAGE_DEFAULTS)
    assert set(PER_MESSAGE_DEFAULTS) <= set(GraphState.__annotations__)        # no default for an undeclared field
    assert plain_data_violations(PER_MESSAGE_DEFAULTS) == []


def test_full_state_is_plain_data_and_the_walker_catches_live_objects():
    assert plain_data_violations(full_state()) == []

    class Client:
        pass
    bad = {**full_state(), "retrieval": {**sample_retrieval(), "retriever": Client()}, "x": {1: "int key"}, "t": result()}
    found = plain_data_violations(bad)
    assert any("retriever" in f and "Client" in f for f in found)
    assert any("non-string key" in f for f in found) and any("RetrievalResult" in f for f in found)


def test_checkpoint_round_trip_of_full_retrieval_and_generation_without_warnings():
    async def node(state):
        return {"retrieval": sample_retrieval(), "generation": sample_generation(), "path": ["n"], "sources": full_state()["sources"]}

    g = StateGraph(GraphState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}

    async def go():
        await app.ainvoke({"user_message": "q"}, cfg)
        return (await app.aget_state(cfg)).values

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = asyncio.run(go())
    assert values["retrieval"] == sample_retrieval() and values["generation"] == sample_generation()
    assert values["sources"] == full_state()["sources"]
    assert not [w for w in caught if "unregistered type" in str(w.message)]      # plain dicts: no future-blocked types


def test_chunk_and_result_conversions_are_lossless():
    res = result()
    for c in res.chunks:
        assert chunk_from_dict(chunk_to_dict(c)) == c
    back = retrieval_result_from_state(sample_retrieval())
    assert [c.chunk_id for c in back.chunks] == [c.chunk_id for c in res.chunks] and back.relevant and back.threshold == 0.15
    gate = gate_result_from_state({**sample_retrieval(), "relevant": False})
    assert gate.relevant is False and len(gate.chunks) == 3


def test_reset_turn_defaults_leave_no_value_from_the_previous_message():
    """Simulates the reset node's update applied on top of a fully populated previous turn."""
    previous = full_state()
    after = {**previous, **PER_MESSAGE_DEFAULTS}
    for key in PER_MESSAGE_DEFAULTS:
        assert after[key] == PER_MESSAGE_DEFAULTS[key], key
    assert plain_data_violations({k: v for k, v in after.items() if k != "messages"}) == []
