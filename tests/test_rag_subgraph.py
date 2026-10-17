"""Stage G3: the RAG subgraph with fake components (no network, no models)."""
from __future__ import annotations

import asyncio

import pytest

from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation import config as gen_config
from graph.rag_subgraph import build_rag_subgraph
from graph.serial import plain_data_violations
from retrieval import config as retrieval_config
from tests.gen_fakes import FakeChat, status_error
from tests.graph_fakes import GOOD, make_components

ANSWERED_PATH = ["hybrid_search", "rerank", "gate", "expand_context", "generate", "attribute"]


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(gen_config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(gen_config, "LENGTH_POLICY", "error")
    monkeypatch.setattr(retrieval_config, "THRESHOLD", 0.15)


def run(comps, query="Which function is cross-cutting?", extra=None):
    graph = build_rag_subgraph(comps)
    return asyncio.run(graph.ainvoke({"standalone_query": query, "llm_calls": 1, **(extra or {})}))


def chat_calls(comps) -> int:
    return comps.generation_client.chat_model.calls


# ------------------------------------------------------------------ the four success paths
def test_answered_path():
    comps = make_components()
    out = run(comps)
    assert out["path"] == ANSWERED_PATH and out["outcome"] == "answered" and out.get("error") is None
    assert out["answer"] == GOOD and out["cited_keys"] == ["C1", "C2"] and out["llm_calls"] == 2
    assert out["insufficient"] is None and out["related_sections"] == [] and out["insufficient_reason"] is None
    # prompt order: C1 = ai:govern (best), C2 = ai:core; Sources are ordered document -> PDF page, so the Core entry is first
    assert [(s["keys"], s["label"], s["kind"]) for s in out["sources"]] == [
        (["C2"], "NIST AI 100-1, §5 AI RMF Core, p. 21", "prose"), (["C1"], "NIST AI 100-1, §5.1 Govern, p. 22", "prose")]
    assert [s["url"] for s in out["sources"]] == ["https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf#page=26",
                                                  "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf#page=27"]
    r = out["retrieval"]
    assert r["relevant"] is True and r["best_score"] == 0.95 and r["threshold"] == 0.15
    assert [c["rerank_rank"] for c in r["top"]] == [1, 2, 3, 4] and r["chunks"][0]["chunk_id"] == "ai:govern"
    assert all(c["rerank_score"] is not None for c in r["candidates"])
    assert plain_data_violations({k: v for k, v in out.items() if k != "messages"}) == []
    assert out["generation"]["outcome"] == "ANSWERED" and set(out["timings_ms"]) == set(ANSWERED_PATH)


def test_gate_failure_path_makes_no_generation_call():
    comps = make_components(scores={"ai:govern": 0.05, "ai:core": 0.04}, chat=FakeChat(chunks=["MUST NOT BE CALLED"]))
    out = run(comps)
    assert out["path"] == ["hybrid_search", "rerank", "gate", "gate_failure_reply"]
    assert out["outcome"] == "insufficient" and out["insufficient_reason"] == "gate_low_score" and out["answer"] == D
    assert chat_calls(comps) == 0 and out["llm_calls"] == 1 and out.get("generation") is None and out["sources"] == []
    # 4 reranked chunks, but ai:other has the same label as ai:core (same section and page): deduplicated by label (insufficient Q1)
    assert len(out["related_sections"]) == 3 and out["retrieval"]["relevant"] is False and out["retrieval"]["chunks"] == []
    assert out["insufficient"]["failure_source"] == "gate"
    assert [s["citation_label"] for s in out["related_sections"]][0].startswith("NIST AI 100-1")


def test_no_candidates_gate_failure():
    comps = make_components(vector_ids=(), bm25_ids=())
    out = run(comps)
    assert out["path"] == ["hybrid_search", "rerank", "gate", "gate_failure_reply"]
    assert out["insufficient_reason"] == "no_candidates" and out["related_sections"] == [] and out["retrieval"]["best_score"] is None


def test_model_refusal_path():
    comps = make_components(chat=FakeChat(chunks=[D]))
    out = run(comps)
    assert out["path"] == ["hybrid_search", "rerank", "gate", "expand_context", "generate", "finalize_model_refusal"]
    assert out["outcome"] == "insufficient" and out["insufficient_reason"] == "model_case_d" and out["answer"] == D
    assert out["llm_calls"] == 2 and out["sources"] == [] and out["insufficient"]["failure_source"] == "model"
    assert len(out["related_sections"]) >= 1


def test_fixer_forced_refusal_path():
    comps = make_components(chat=FakeChat(chunks=["GOVERN is cross-cutting [C1]. ", D]))
    out = run(comps)
    assert out["path"][-1] == "finalize_model_refusal" and out["insufficient_reason"] == "fix_partial_answer"
    assert out["answer"] == D and out["generation"]["fixes"] == ["D_WITH_EXTRA_TEXT"]
    out2 = run(make_components(chat=FakeChat(chunks=["GOVERN is cross-cutting [C9]."])))
    assert out2["insufficient_reason"] == "fix_no_citations" and out2["path"][-1] == "finalize_model_refusal"


def test_siblings_are_expanded_and_listed_for_attribution():
    comps = make_components(vector_ids=("gen:t1", "ai:core"), bm25_ids=("gen:t1",), scores={"gen:t1": 0.9, "ai:core": 0.8},
                            chat=FakeChat(chunks=["Part one [C1]. Part two [C2]."]))
    out = run(comps)
    ids = [c["chunk_id"] for c in out["retrieval"]["chunks"]]
    assert ids == ["gen:t1", "gen:t2", "ai:core"] and out["retrieval"]["siblings_added"] == ["gen:t2"]
    assert out["outcome"] == "answered"
    assert [(s["keys"], s["label"]) for s in out["sources"]] == [
        (["C1"], "NIST AI 600-1, §3 Suggested Actions, GOVERN 1.7, p. 17"), (["C2"], "NIST AI 600-1, §3 Suggested Actions, GOVERN 1.7, p. 18")]


def test_subgraph_can_be_invoked_alone_with_only_a_standalone_query():
    out = asyncio.run(build_rag_subgraph(make_components()).ainvoke({"standalone_query": "q"}))
    assert out["outcome"] == "answered" and out["path"] == ANSWERED_PATH


# ------------------------------------------------------------------ errors at each node
def error_of(out):
    return out["error"]


@pytest.mark.parametrize("name,kwargs,code,path", [
    ("hybrid_search", {"boom_vector": True}, "RETRIEVAL_FAILED", ["hybrid_search"]),
    ("rerank", {"reranker_boom": True}, "RERANK_FAILED", ["hybrid_search", "rerank"]),
    ("expand_context", {"store_boom": True, "vector_ids": ("gen:t1",), "bm25_ids": ("gen:t1",), "scores": {"gen:t1": 0.9}},
     "RETRIEVAL_FAILED", ["hybrid_search", "rerank", "gate", "expand_context"]),
])
def test_error_in_retrieval_nodes_exits_with_a_generic_error(name, kwargs, code, path):
    comps = make_components(chat=FakeChat(chunks=["MUST NOT BE CALLED"]), **kwargs)
    out = run(comps)
    assert out["error"] == {"code": code, "message": out["error"]["message"], "node": name}
    assert out["path"] == path and chat_calls(comps) == 0
    assert "SECRET" not in repr(out["error"]) and out.get("outcome") is None and out.get("answer") is None


def test_error_in_gate_when_threshold_missing(monkeypatch):
    monkeypatch.setattr(retrieval_config, "THRESHOLD", None)
    out = run(make_components())
    assert out["error"]["code"] == "INTERNAL" and out["error"]["node"] == "gate" and out["path"][-1] == "gate"


def test_generation_errors_map_to_codes():
    out = run(make_components(chat=FakeChat(chunks=[GOOD], errors=[status_error(429)] * 5)))
    assert out["error"]["code"] == "LLM_UNAVAILABLE" and out["error"]["node"] == "generate" and out["path"][-1] == "generate"
    assert out["llm_calls"] == 1 and out["generation"]["error"]["code"] == "rate_limit"
    out = run(make_components(chat=FakeChat(chunks=["GOVERN is the ", "x [C1]. ", "more"], fail_after=2)))
    assert out["error"]["code"] == "GENERATION_FAILED" and out["generation"]["final_text"] is None
    assert out.get("sources", []) == [] and out.get("outcome") is None


def test_error_in_attribute_and_reply_nodes(monkeypatch):
    import graph.rag_nodes as nodes

    def boom(*a, **k):
        raise RuntimeError("SECRET attribution failure")
    monkeypatch.setattr(nodes, "build_sources", boom)
    out = run(make_components())
    assert out["error"]["code"] == "INTERNAL" and out["error"]["node"] == "attribute" and out["path"][-1] == "attribute"
    monkeypatch.setattr(nodes, "gate_failure_reply", boom)
    out = run(make_components(scores={"ai:govern": 0.01}))
    assert out["error"]["node"] == "gate_failure_reply" and "SECRET" not in repr(out["error"])
    monkeypatch.setattr(nodes, "finalize_model_refusal", boom)
    out = run(make_components(chat=FakeChat(chunks=[D])))
    assert out["error"]["node"] == "finalize_model_refusal"


def test_every_caught_error_is_logged_with_node_and_stack_trace(caplog):
    with caplog.at_level("ERROR", logger="graph.rag_nodes"):
        run(make_components(reranker_boom=True))
    rec = [r for r in caplog.records if "node rerank failed" in r.getMessage()]
    assert rec and rec[0].exc_info and "SECRET reranker failure" in str(rec[0].exc_info[1])


def test_error_in_state_is_plain_data():
    out = run(make_components(boom_vector=True))
    assert plain_data_violations({k: v for k, v in out.items() if k != "messages"}) == []
