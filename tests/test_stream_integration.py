"""Stage G5: /chat/stream through the real main graph + RAG subgraph (fake components, scripted chat model, real FastAPI app)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.main import create_app
from app.sse import parse_events
from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation import config as gen_config
from graph.build import build_app_graph
from llm.groq_client import LLMUnavailableError, StructuredResult
from retrieval import config as retrieval_config
from tests.gen_fakes import status_error
from tests.graph_fakes import SETTINGS, ScriptedChat, make_components
from tests.test_main_graph import d

TAG = ["answer_generation"]
ROUTER_MARKER = "ROUTERJSONMARKER"
REASONING_MARKER = "PRIVATEREASONINGMARKER"
CHUNK_MARKER = "CHUNKTEXTMARKER"


class LeakyRouter:
    """Router client whose (untagged) chat model really streams a JSON marker while it 'classifies'."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)

    async def structured(self, schema, messages):
        item = self.decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        model = GenericFakeChatModel(messages=iter([AIMessage(content=f'{{"route":"x","reason":"{ROUTER_MARKER}"}}')]))
        await model.ainvoke("route this")
        return StructuredResult(item, "{}", None, 1)


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(gen_config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(gen_config, "LENGTH_POLICY", "error")
    monkeypatch.setattr(retrieval_config, "THRESHOLD", 0.15)


def client_for(router, **comps_kw):
    comps = make_components(**comps_kw)
    graph = build_app_graph(SETTINGS, router, components=comps)
    return TestClient(create_app(SETTINGS, graph), raise_server_exceptions=False), comps


def stream(c, text="Which function is cross-cutting?", **extra):
    with c:
        r = c.post("/chat/stream", json={"message": text, **extra})
    assert r.status_code == 200, r.text
    return r.text, parse_events(r.text)


def names(ev):
    return [e for e, _ in ev]


def tokens(ev):
    return "".join(p["text"] for e, p in ev if e == "token")


ANSWER = ["GOVERN is the cross-cutting function [C1]. ", "The Core has four functions [C2]."]


# ------------------------------------------------------------------ order per path
def test_answered_stream_order_and_payloads():
    c, _ = client_for(LeakyRouter(d("nist_question", "q?")), chat=ScriptedChat(script=ANSWER, tags=TAG))
    raw, ev = stream(c)
    assert names(ev) == ["meta"] + ["token"] * 2 + ["sources", "done"]
    assert dict(ev)["meta"]["route"] == "nist_question" and tokens(ev) == "".join(ANSWER)
    sources = dict(ev)["sources"]["sources"]
    assert [s["keys"] for s in sources] == [["C2"], ["C1"]] and all(set(s) == {"keys", "label", "url", "kind"} for s in sources)
    done = dict(ev)["done"]
    assert done["outcome"] == "answered" and done["insufficient_reason"] is None and done["llm_calls"] == 2


def test_corrected_stream_invented_citation_removed_by_a_correction_before_sources():
    c, _ = client_for(LeakyRouter(d("nist_question", "q?")),
                      chat=ScriptedChat(script=["GOVERN is cross-cutting [C1][C9]. ", "Four functions [C2]."], tags=TAG))
    raw, ev = stream(c)
    assert names(ev) == ["meta", "token", "token", "correction", "sources", "done"]
    corr = dict(ev)["correction"]
    assert corr == {"text": "GOVERN is cross-cutting [C1]. Four functions [C2].", "reasons": ["UNKNOWN_CITATION"]}
    assert [s["keys"] for s in dict(ev)["sources"]["sources"]] == [["C2"], ["C1"]]          # built from the CORRECTED text


def test_fixer_forced_refusal_stream():
    c, _ = client_for(LeakyRouter(d("nist_question", "q?")), chat=ScriptedChat(script=["GOVERN is cross-cutting [C1]. ", D], tags=TAG))
    raw, ev = stream(c)
    n = names(ev)
    assert n == ["meta", "token", "token", "correction", "sources", "related", "done"]
    assert dict(ev)["correction"]["text"] == D and dict(ev)["correction"]["reasons"] == ["D_WITH_EXTRA_TEXT"]
    assert dict(ev)["sources"] == {"sources": []} and dict(ev)["done"]["insufficient_reason"] == "fix_partial_answer"
    assert len(dict(ev)["related"]["sections"]) >= 1 and "text" not in str(dict(ev)["related"]["sections"])


def test_model_refusal_stream_has_no_correction():
    c, _ = client_for(LeakyRouter(d("nist_question", "q?")), chat=ScriptedChat(script=[D[:30], D[30:]], tags=TAG))
    raw, ev = stream(c)
    assert names(ev) == ["meta", "token", "token", "sources", "related", "done"] and tokens(ev) == D
    assert dict(ev)["done"]["insufficient_reason"] == "model_case_d"


def test_gate_failure_stream_is_one_token_then_related_and_makes_no_generation_call():
    chat = ScriptedChat(script=["MUST NOT RUN"], tags=TAG)
    c, comps = client_for(LeakyRouter(d("nist_question", "q?")), scores={"ai:govern": 0.05}, chat=chat)
    raw, ev = stream(c, "What penalties does the EU AI Act set?")
    assert names(ev) == ["meta", "token", "sources", "related", "done"]
    assert dict(ev)["token"] == {"text": D} and dict(ev)["done"]["insufficient_reason"] == "gate_low_score"
    assert dict(ev)["done"]["llm_calls"] == 1 and "MUST NOT RUN" not in raw


def test_template_replies_are_one_token_event():
    c, _ = client_for(LeakyRouter(d("out_of_scope")))
    raw, ev = stream(c, "capital of France?")
    assert names(ev) == ["meta", "token", "sources", "done"] and dict(ev)["token"]["text"] == "Sorry, we don't have this answer."


# ------------------------------------------------------------------ errors: nothing after `error`
def test_mid_stream_generation_failure_sends_error_and_nothing_after_it():
    chat = ScriptedChat(script=["GOVERN is ", "cross-cutting ", "[C1]. ", ("fail",)], fail_with=status_error(503), tags=TAG)
    c, _ = client_for(LeakyRouter(d("nist_question", "q?")), chat=chat)
    raw, ev = stream(c)
    n = names(ev)
    assert n[0] == "meta" and n[-1] == "error" and set(n[1:-1]) == {"token"} and len(n) == 5
    assert not {"correction", "sources", "related", "done"} & set(n)
    err = dict(ev)["error"]
    assert err["code"] == "generation_failed" and "503" not in err["message"] and "Traceback" not in raw


def test_router_outage_stream_is_a_single_error_event():
    c, _ = client_for(LeakyRouter(LLMUnavailableError("x", reason="rate_limit", attempts=4)))
    raw, ev = stream(c)
    assert names(ev) == ["error"] and dict(ev)["error"]["code"] == "llm_unavailable"


def test_retrieval_failure_stream_error_event_and_history_rolled_back():
    c, comps = client_for(LeakyRouter(d("nist_question", "q?"), d("out_of_scope")), boom_vector=True)
    with c:
        r = c.post("/chat/stream", json={"message": "q"})
        ev = parse_events(r.text)
        assert names(ev) == ["meta", "error"] and dict(ev)["error"]["code"] == "retrieval_failed"
        assert "SECRET" not in r.text


# ------------------------------------------------------------------ leaks
def test_nothing_internal_appears_in_any_event():
    """Markers planted in: the router's streamed JSON, the reasoning chunks, the chunk texts, the rerank internals."""
    chat = ScriptedChat(script=[("reasoning", REASONING_MARKER), "GOVERN is cross-cutting [C1]. ",
                                ("reasoning", REASONING_MARKER), "Four functions [C2]."], tags=TAG)
    comps = make_components(chat=chat)
    for rec in comps.chunk_lookup.values():
        rec["display_text"] = f"{CHUNK_MARKER} {rec['chunk_id']}"
    graph = build_app_graph(SETTINGS, LeakyRouter(d("nist_question", "q?")), components=comps)
    c = TestClient(create_app(SETTINGS, graph), raise_server_exceptions=False)
    raw, ev = stream(c)
    assert names(ev) == ["meta", "token", "token", "sources", "done"]
    for marker in (ROUTER_MARKER, REASONING_MARKER, CHUNK_MARKER, "rerank", "emb ", "display "):
        assert marker not in raw, marker
    assert tokens(ev) == "GOVERN is cross-cutting [C1]. Four functions [C2]."          # exactly once: no duplicate of finalize's text
    assert "q" not in [p.get("text") for e, p in ev if e == "token"]                    # the stored user message is never a token


def test_finalize_stripped_history_text_is_never_streamed():
    c, _ = client_for(LeakyRouter(d("nist_question", "q?")), chat=ScriptedChat(script=ANSWER, tags=TAG))
    raw, ev = stream(c)
    assert "GOVERN is the cross-cutting function. The Core has four functions." not in raw          # markers stripped version
    assert "[C1]" in tokens(ev)


def test_untagged_model_tokens_never_stream():
    """The same answer from a model WITHOUT the answer_generation tag is not forwarded as tokens (one fallback token event)."""
    c, _ = client_for(LeakyRouter(d("nist_question", "q?")), chat=ScriptedChat(script=ANSWER))
    raw, ev = stream(c)
    assert names(ev) == ["meta", "token", "sources", "done"]                  # the validated answer, sent once as a whole
    assert dict(ev)["token"]["text"] == "".join(ANSWER)
