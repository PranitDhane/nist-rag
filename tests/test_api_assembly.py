"""Stage G6: response assembly (meta, debug) and error mapping through the real graph + FastAPI app (fake components)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.sse import parse_events
from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation import config as gen_config
from generation.client import GenerationClient
from graph.build import build_app_graph
from graph.errors import GENERATION_CODE_MAP
from llm.groq_client import LLMUnavailableError, StructuredResult
from retrieval import config as retrieval_config
from tests.gen_fakes import FakeChat, status_error
from tests.graph_fakes import SETTINGS, ScriptedChat, make_components
from tests.test_main_graph import d

TAG = ["answer_generation"]
REASONING_MARKER = "PRIVATEREASONINGMARKER"
CHUNK_MARKER = "CHUNKTEXTMARKER"
ANSWER = ["GOVERN is the cross-cutting function [C1]. ", "The Core has four functions [C2]."]


class Router:
    """Fake Groq client for the real router node; reports 50 tokens per call (for meta.usage)."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)

    async def structured(self, schema, messages):
        item = self.decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return StructuredResult(item, "{}", None, 1, total_tokens=50)


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(gen_config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(gen_config, "LENGTH_POLICY", "error")
    monkeypatch.setattr(retrieval_config, "THRESHOLD", 0.15)


def app_for(router, hang=False, **kw):
    comps = make_components(**kw)
    if hang:      # a model that never produces a token: the first-token timeout fires (the node captures its client at build time)
        comps.generation_client = GenerationClient(SETTINGS, chat_model=FakeChat(hang=True), ttft_timeout=0.05, max_tokens=2000)
    for rec in comps.chunk_lookup.values():
        rec["display_text"] = f"{CHUNK_MARKER} {rec['chunk_id']}"
    return TestClient(create_app(SETTINGS, build_app_graph(SETTINGS, router, components=comps)), raise_server_exceptions=False), comps


def chat(c, text="Which function is cross-cutting?", **extra):
    with c:
        return c.post("/chat", json={"message": text, **extra})


def stream(c, text="Which function is cross-cutting?", **extra):
    with c:
        r = c.post("/chat/stream", json={"message": text, **extra})
    return r.text, parse_events(r.text)


# ------------------------------------------------------------------ meta on every route (Q9)
def test_meta_on_every_route():
    c, _ = app_for(Router(d("nist_question", "q?"), d("out_of_scope"), d("nist_question", "q2?")),
                   chat=ScriptedChat(script=ANSWER, tags=TAG))
    with c:
        answered = c.post("/chat", json={"message": "q"}).json()
        oos = c.post("/chat", json={"message": "capital?"}).json()
        greet = c.post("/chat", json={"message": "hello!"}).json()
    a = answered["meta"]
    assert (a["outcome"], a["insufficient_reason"], a["llm_calls"]) == ("answered", None, 2)
    assert (a["model"], a["prompt_version"], a["fixes"], a["warnings"]) == ("openai/gpt-oss-20b", "v1", [], [])
    assert a["usage"] == {"router": 50, "generation": 1500, "total": 1550}              # tokens across both LLM calls
    assert oos["meta"]["outcome"] == "out_of_scope" and oos["meta"]["usage"] == {"router": 50, "generation": 0, "total": 50}
    assert "model" not in oos["meta"] and "fixes" not in oos["meta"]                    # generation did not run
    assert greet["meta"]["outcome"] == "greeting" and greet["meta"]["llm_calls"] == 0 and greet["meta"]["usage"]["total"] == 0


def test_meta_insufficient_has_reason_and_no_generation_fields_for_a_gate_failure():
    c, _ = app_for(Router(d("nist_question", "q?")), scores={"ai:govern": 0.05})
    m = chat(c).json()["meta"]
    assert m["outcome"] == "insufficient" and m["insufficient_reason"] == "gate_low_score" and "model" not in m


def test_meta_fixes_are_reported_when_the_fixer_changed_the_text():
    c, _ = app_for(Router(d("nist_question", "q?")), chat=ScriptedChat(script=["GOVERN [C1][C9]. Core [C2]."], tags=TAG))
    m = chat(c).json()["meta"]
    assert m["fixes"] == ["UNKNOWN_CITATION"] and m["outcome"] == "answered"


def test_stream_done_carries_the_same_fields():
    c, _ = app_for(Router(d("nist_question", "q?")), chat=ScriptedChat(script=ANSWER, tags=TAG))
    raw, ev = stream(c)
    done = dict(ev)["done"]
    assert done["outcome"] == "answered" and done["usage"]["total"] == 1550 and done["model"] == "openai/gpt-oss-20b"
    assert "debug" not in done and "route_source" not in done


# ------------------------------------------------------------------ debug
def test_debug_is_absent_by_default_and_present_with_the_flag():
    c, _ = app_for(Router(d("nist_question", "q?"), d("nist_question", "q?")), chat=ScriptedChat(script=ANSWER, tags=TAG))
    with c:
        plain = c.post("/chat", json={"message": "q"}).json()
        dbg = c.post("/chat", json={"message": "q", "debug": True}).json()
    assert "debug" not in plain["meta"] and "debug" in dbg["meta"]
    assert dbg["meta"]["debug"]["path"][:3] == ["reset_turn", "rule_greeting_check", "classify_and_rewrite"]


def test_debug_contents_answered():
    c, comps = app_for(Router(d("nist_question", "Which AI RMF function is cross-cutting?")), chat=ScriptedChat(script=ANSWER, tags=TAG))
    dbg = chat(c, debug=True).json()["meta"]["debug"]
    assert dbg["path"] == ["reset_turn", "rule_greeting_check", "classify_and_rewrite", "hybrid_search", "rerank", "gate",
                           "expand_context", "generate", "attribute", "finalize"]
    assert set(dbg["timings_ms"]) == set(dbg["path"])
    assert (dbg["route"], dbg["route_source"], dbg["standalone_query"]) == ("nist_question", "llm", "Which AI RMF function is cross-cutting?")
    r = dbg["retrieval"]
    assert r["candidate_count"] == 4 and r["best_score"] == 0.95 and r["threshold"] == 0.15 and r["relevant"] is True
    assert r["siblings_added"] == [] and len(r["candidates"]) <= 20
    assert r["candidates"][0] == {"chunk_id": "ai:govern", "label": "NIST AI 100-1, §5.1 Govern, p. 22", "rerank_score": 0.95}
    assert all(set(x) == {"chunk_id", "label", "rerank_score"} for x in r["candidates"])
    g = dbg["generation"]
    assert g["model"] == "openai/gpt-oss-20b" and g["prompt_version"] == "v1" and g["reasoning_effort"] == "medium"
    assert g["temperature"] == 0.2 and g["usage"]["reasoning"] == 220 and g["finish_reason"] == "stop" and g["corrected"] is False
    assert dbg["config_version"]["retrieval"] and "embedding_model" in dbg["config_version"]["indexing"]


def test_debug_candidates_are_capped_at_twenty_best_first():
    from graph.debug import build_debug
    cands = [{"chunk_id": f"c{i}", "citation_label": f"L{i}", "rerank_score": 1 - i / 100, "rerank_rank": i} for i in range(1, 31)]
    dbg = build_debug({"retrieval": {"candidates": list(reversed(cands)), "candidate_count": 30}})
    assert [x["chunk_id"] for x in dbg["retrieval"]["candidates"]] == [f"c{i}" for i in range(1, 21)]


def test_debug_for_a_gate_failure_has_retrieval_but_no_generation():
    c, _ = app_for(Router(d("nist_question", "q?")), scores={"ai:govern": 0.05})
    dbg = chat(c, debug=True).json()["meta"]["debug"]
    assert dbg["retrieval"]["relevant"] is False and "generation" not in dbg and dbg["path"][-2:] == ["gate_failure_reply", "finalize"]


def test_debug_never_contains_chunk_prompt_or_reasoning_text():
    chat_model = ScriptedChat(script=[("reasoning", REASONING_MARKER), *ANSWER], tags=TAG)
    c, _ = app_for(Router(d("nist_question", "q?"), d("nist_question", "q?")), chat=chat_model)
    with c:
        j = c.post("/chat", json={"message": "q", "debug": True}).json()
        raw, ev = stream(c, debug=True)
    blobs = [json.dumps(j["meta"]["debug"]), json.dumps(dict(ev)["done"]["debug"])]
    for blob in blobs:
        for forbidden in (CHUNK_MARKER, REASONING_MARKER, "<source", "<question>", "# PURPOSE", "GOVERN is the cross-cutting",
                          "display ", "emb "):
            assert forbidden not in blob, forbidden
    assert REASONING_MARKER not in json.dumps(j) and REASONING_MARKER not in raw


def test_stream_done_carries_debug_when_requested():
    c, _ = app_for(Router(d("nist_question", "q?")), chat=ScriptedChat(script=ANSWER, tags=TAG))
    raw, ev = stream(c, debug=True)
    assert dict(ev)["done"]["debug"]["path"][-1] == "finalize" and dict(ev)["done"]["debug"]["retrieval"]["relevant"] is True


# ------------------------------------------------------------------ error mapping (Q4)
@pytest.mark.parametrize("scenario,status,code", [
    ("router_outage", 503, "llm_unavailable"),
    ("generation_rate_limit", 503, "llm_unavailable"),
    ("generation_ttft_timeout", 503, "llm_unavailable"),
    ("retrieval", 500, "retrieval_failed"),
    ("rerank", 500, "rerank_failed"),
    ("generation_midstream", 500, "generation_failed"),
    ("generation_length", 500, "generation_failed"),
    ("internal_gate", 500, "internal_error"),
])
def test_each_error_code_gives_the_agreed_status_and_a_generic_body(scenario, status, code, monkeypatch):
    kw, router = {}, Router(d("nist_question", "q?"))
    if scenario == "router_outage":
        router = Router(LLMUnavailableError("x", reason="rate_limit", retry_after=7.0, attempts=4))
    elif scenario == "generation_rate_limit":
        kw["chat"] = FakeChat(chunks=["x"], errors=[status_error(429)] * 5)
    elif scenario == "retrieval":
        kw["boom_vector"] = True
    elif scenario == "rerank":
        kw["reranker_boom"] = True
    elif scenario == "generation_midstream":
        kw["chat"] = FakeChat(chunks=["GOVERN ", "is ", "x [C1]. ", "more"], fail_after=2)
    elif scenario == "generation_length":
        kw["chat"] = FakeChat(chunks=["GOVERN [C1]."], finish="length")
    elif scenario == "internal_gate":
        monkeypatch.setattr(retrieval_config, "THRESHOLD", None)
    hang = scenario == "generation_ttft_timeout"
    c, _ = app_for(router, hang=hang, **kw)
    r = chat(c)
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message"} and body["error"]["code"] == code
    for leak in ("Traceback", "File \"", "SECRET", ".py", "Exception", "groq"):
        assert leak not in r.text, leak
    if status == 503:
        assert int(r.headers["Retry-After"]) >= 1
    again = [LLMUnavailableError("x", reason="rate_limit", attempts=4)] if scenario == "router_outage" else [d("nist_question", "q?")]
    c2, _ = app_for(Router(*again), hang=hang, **kw)
    raw, ev = stream(c2)                                                   # the stream reports the same code, nothing after it
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == code and not {"sources", "related", "done"} & {e for e, _ in ev}
    assert "Traceback" not in raw and "SECRET" not in raw


def test_the_error_table_is_the_approved_one():
    from graph.errors import HTTP_STATUS
    assert HTTP_STATUS == {"LLM_UNAVAILABLE": 503, "RETRIEVAL_FAILED": 500, "RERANK_FAILED": 500, "GENERATION_FAILED": 500, "INTERNAL": 500}
    assert GENERATION_CODE_MAP["ttft_timeout"] == GENERATION_CODE_MAP["total_timeout"] == "LLM_UNAVAILABLE"      # Q4


def test_a_failed_request_leaves_a_usable_conversation():
    c, _ = app_for(Router(d("nist_question", "q?"), d("out_of_scope")), boom_vector=True)
    with c:
        r1 = c.post("/chat", json={"message": "q"})
        r2 = c.post("/chat", json={"message": "capital of France?"})
    assert r1.status_code == 500 and r2.status_code == 200 and r2.json()["meta"]["outcome"] == "out_of_scope"
