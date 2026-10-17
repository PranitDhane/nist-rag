import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import api
from app.config import Settings
from app.main import create_app
from app.sse import parse_events
from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation.models import INSUFFICIENT, GenerationResult
from insufficient.reply import RELATED_NOTE
from tests.insufficient_harness import answered_node, build_harness_graph, gate_failure_node, model_refusal_node
from tests.test_api import FakeClient, d
from tests.test_insufficient_reply import gen, retrieval

S = Settings(groq_api_key="x", max_message_chars=200)
ENTRY_KEYS = {"doc", "section", "pages", "citation_label", "citation_url", "kind"}
QUESTION = "What penalties does the EU AI Act set?"


def make(rag_node, *decisions):
    client = FakeClient(*(decisions or (d("nist_question", QUESTION),)))
    graph = build_harness_graph(_classify(client), rag_node)
    return TestClient(create_app(S, graph), raise_server_exceptions=False), client


def _classify(client):
    from graph.router import make_classify_node
    return make_classify_node(client, S)


def chat(c, text=QUESTION, thread_id=None):
    return c.post("/chat", json={"message": text, **({"thread_id": thread_id} if thread_id else {})})


def stream(c, text=QUESTION):
    r = c.post("/chat/stream", json={"message": text})
    assert r.status_code == 200, r.text
    return parse_events(r.text)


def fake_model(text):
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]), tags=["answer_generation"])   # only tagged calls stream


# ======================================================================== /chat
def test_chat_gate_failure_contract():
    c, client = make(gate_failure_node(retrieval()))
    with c:
        j = chat(c).json()
    assert j["route"] == "nist_question" and j["answer"] == D and j["sources"] == []
    assert len(j["related_sections"]) == 5 and all(set(e) == ENTRY_KEYS for e in j["related_sections"])
    assert j["meta"]["outcome"] == "insufficient" and j["meta"]["insufficient_reason"] == "gate_low_score"
    assert j["meta"]["llm_calls"] == 1 and client.calls == 1                       # routing only: no generation call
    assert "SECRET" not in str(j)                                                   # no chunk text anywhere in the response


def test_chat_no_candidates_reason():
    from retrieval.models import RetrievalResult
    empty = RetrievalResult(query="q", chunks=[], relevant=False, best_score=None, threshold=0.15, candidate_count=0)
    c, _ = make(gate_failure_node(empty))
    with c:
        j = chat(c).json()
    assert j["related_sections"] == [] and j["meta"]["insufficient_reason"] == "no_candidates" and j["answer"] == D


@pytest.mark.parametrize("fixes,reason", [([], "model_case_d"), (["D_WITH_EXTRA_TEXT"], "fix_partial_answer"),
                                          (["NO_CITATIONS"], "fix_no_citations"), (["EMPTY"], "fix_empty_output"),
                                          (["LENGTH"], "fix_truncated")])
def test_chat_model_refusal_and_fixer_forced_contract(fixes, reason):
    c, client = make(model_refusal_node(gen(fixes=fixes), retrieval(relevant=True)))
    with c:
        j = chat(c).json()
    assert j["answer"] == D and j["sources"] == [] and len(j["related_sections"]) == 5
    assert j["meta"]["outcome"] == "insufficient" and j["meta"]["insufficient_reason"] == reason
    assert j["meta"]["llm_calls"] == 2                                              # routing + generation


def test_chat_answered_nist_response_has_stable_schema_with_empty_related_sections():
    c, _ = make(answered_node(sources=[{"keys": ["C1"], "label": "NIST AI 100-1, §5.1 Govern, p. 21", "url": "u", "kind": "prose"}]))
    with c:
        j = chat(c).json()
    assert j["related_sections"] == [] and j["meta"]["outcome"] == "answered" and j["meta"]["insufficient_reason"] is None
    assert j["sources"][0]["keys"] == ["C1"]                                         # sources untouched when answered


def test_insufficient_sources_are_empty_even_if_the_node_leaked_some():
    async def leaky(state):
        from insufficient.reply import gate_failure_reply
        r = gate_failure_reply(retrieval())
        return {"answer": D, "sources": [{"leak": True}], "outcome": "insufficient", "insufficient": r.to_dict()}
    c, _ = make(leaky)
    with c:
        j = chat(c).json()
    assert j["sources"] == []


def test_greeting_and_out_of_scope_keep_their_top_level_shape():
    c, _ = make(answered_node(), d("out_of_scope"))
    with c:
        j = chat(c, "What's the capital of France?").json()
    assert set(j) == {"thread_id", "route", "answer", "sources", "standalone_query", "meta"}
    assert set(j["meta"]) == {"llm_calls", "route_source", "latency_ms", "outcome", "insufficient_reason", "usage"} and j["meta"]["outcome"] == "out_of_scope"      # new meta on every route (Q9)


# ======================================================================== history
def test_history_stores_only_the_sentence_and_related_sections_never_reach_the_routing_window():
    c, client = make(gate_failure_node(retrieval()), d("nist_question", QUESTION), d("nist_question", "second question"))
    with c:
        tid = chat(c).json()["thread_id"]
        msgs = c.get(f"/threads/{tid}").json()["messages"]
        assert msgs[1] == {"role": "assistant", "content": D, "route": "nist_question"}
        chat(c, "And a second question about GOVERN?", tid)
    second_prompt = client.prompts[1]
    assert f"Assistant: {D}" in second_prompt
    for leaked in ("closest sections", "Section 5.1", "§5.", "citation", "pdf#page", "#page="):
        assert leaked not in second_prompt


# ======================================================================== /chat/stream
def names(events):
    return [e for e, _ in events]


def test_stream_gate_failure_order_and_payloads():
    c, client = make(gate_failure_node(retrieval()))
    with c:
        ev = stream(c)
    assert names(ev) == ["meta", "token", "sources", "related", "done"]
    assert ev[0][1]["route"] == "nist_question" and ev[1][1] == {"text": D} and ev[2][1] == {"sources": []}
    assert ev[3][1]["note"] == RELATED_NOTE and len(ev[3][1]["sections"]) == 5
    assert all(set(s) == ENTRY_KEYS for s in ev[3][1]["sections"])
    assert ev[4][1]["outcome"] == "insufficient" and ev[4][1]["insufficient_reason"] == "gate_low_score" and ev[4][1]["llm_calls"] == 1
    assert client.calls == 1


def test_stream_model_refusal_sentence_written_by_the_model_has_no_correction(monkeypatch):
    c, _ = make(model_refusal_node(gen(), retrieval(True), streaming_model=fake_model(D)))
    with c:
        ev = stream(c)
    assert names(ev)[0] == "meta" and names(ev)[-3:] == ["sources", "related", "done"]
    tokens = [x["text"] for e, x in ev if e == "token"]
    assert len(tokens) > 1 and "".join(tokens) == D and "correction" not in names(ev)
    assert ev[-1][1]["insufficient_reason"] == "model_case_d" and ev[-1][1]["llm_calls"] == 2


def test_stream_fixer_forced_refusal_streams_tokens_then_a_correction_with_the_sentence(monkeypatch):
    partial = f"GOVERN is cross-cutting [C1]. {D}"
    c, _ = make(model_refusal_node(gen(fixes=["D_WITH_EXTRA_TEXT"]), retrieval(True), streaming_model=fake_model(partial)))
    with c:
        ev = stream(c)
    n = names(ev)
    assert n[0] == "meta" and n[-4:] == ["correction", "sources", "related", "done"] and n.count("token") > 1
    assert "".join(x["text"] for e, x in ev if e == "token") == partial
    assert dict(ev)["correction"] == {"text": D}
    assert ev[-1][1]["insufficient_reason"] == "fix_partial_answer" and dict(ev)["sources"] == {"sources": []}


def test_stream_answered_nist_has_outcome_in_done_and_no_related_event():
    c, _ = make(answered_node())
    with c:
        ev = stream(c)
    assert names(ev) == ["meta", "token", "sources", "done"]
    assert ev[-1][1]["outcome"] == "answered" and ev[-1][1]["insufficient_reason"] is None


def test_stream_greeting_and_out_of_scope_done_events_carry_outcome_but_no_related():
    c, _ = make(answered_node(), d("out_of_scope"))
    with c:
        ev = stream(c, "What's the capital of France?")
    assert names(ev) == ["meta", "token", "sources", "done"] and set(ev[-1][1]) == {"llm_calls", "latency_ms", "outcome", "insufficient_reason", "usage"}
    assert ev[-1][1]["outcome"] == "out_of_scope" and "related" not in names(ev)


def test_related_sections_are_never_inside_the_sources_event():
    c, _ = make(gate_failure_node(retrieval()))
    with c:
        ev = stream(c)
    assert dict(ev)["sources"] == {"sources": []} and "citation_label" not in str(dict(ev)["sources"])
