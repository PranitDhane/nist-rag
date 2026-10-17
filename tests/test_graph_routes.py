"""Real graph + real in-memory checkpointer, mocked classify node (no network)."""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from graph.build import build_graph
from tests.stub_rag import STUB_PREFIX, stub_rag
from graph.replies import GREETING_REPLY, OUT_OF_SCOPE_REPLY


class FakeClassify:
    """Stands in for the router: returns whatever the test queued, and records what it saw."""

    def __init__(self, *results):
        self.results, self.seen = list(results), []

    async def __call__(self, state):
        self.seen.append({k: state.get(k) for k in ("user_message", "route", "answer", "standalone_query", "llm_calls", "error")})
        r = dict(self.results.pop(0))
        r["llm_calls"] = state.get("llm_calls", 0) + 1
        return {"route_source": "llm", "route_reason": "test", **r}


def cfg(tid):
    return {"configurable": {"thread_id": tid}}


def send(graph, tid, text):
    return asyncio.run(graph.ainvoke({"user_message": text}, cfg(tid)))


def make(*results, rule=lambda t: False):
    fake = FakeClassify(*results)
    return build_graph(fake, rule, stub_rag), fake


def test_nist_route_reaches_rag_stub():
    graph, _ = make({"route": "nist_question", "standalone_query": "What are the four functions?"})
    out = send(graph, "t", "what are the functions")
    assert out["route"] == "nist_question" and out["answer"] == STUB_PREFIX + "What are the four functions?"
    assert out["sources"] == [] and out["llm_calls"] == 1


def test_out_of_scope_route_gives_exact_template_and_no_extra_calls():
    graph, _ = make({"route": "out_of_scope", "standalone_query": None})
    out = send(graph, "t", "capital of France?")
    assert out["answer"] == OUT_OF_SCOPE_REPLY and out["llm_calls"] == 1 and out["route"] == "out_of_scope"


def test_llm_greeting_route_gives_greeting_template():
    graph, _ = make({"route": "greeting", "standalone_query": None})
    out = send(graph, "t", "good morning, hope you are well")
    assert out["answer"] == GREETING_REPLY and out["route_source"] == "llm" and out["llm_calls"] == 1


def test_rule_greeting_skips_the_classifier_entirely():
    graph, fake = make(rule=lambda t: t == "hi")
    out = send(graph, "t", "hi")
    assert out["answer"] == GREETING_REPLY and out["route_source"] == "rule" and out["llm_calls"] == 0
    assert fake.seen == []


def test_every_reply_including_templates_is_stored_with_route_metadata():
    graph, _ = make({"route": "out_of_scope"}, {"route": "greeting"}, rule=lambda t: t == "hi")
    send(graph, "t", "hi")
    send(graph, "t", "capital of France?")
    send(graph, "t", "good morning!")
    msgs = graph.get_state(cfg("t")).values["messages"]
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage] * 3
    assert [m.additional_kwargs["route"] for m in msgs if isinstance(m, AIMessage)] == ["greeting", "out_of_scope", "greeting"]
    assert msgs[1].content == GREETING_REPLY and msgs[3].content == OUT_OF_SCOPE_REPLY


def test_same_thread_shares_history_and_other_thread_does_not():
    graph, _ = make({"route": "nist_question", "standalone_query": "q1"}, {"route": "nist_question", "standalone_query": "q2"},
                    {"route": "nist_question", "standalone_query": "q3"})
    send(graph, "A", "first")
    send(graph, "A", "second")
    send(graph, "B", "other")
    a = [m.content for m in graph.get_state(cfg("A")).values["messages"]]
    b = [m.content for m in graph.get_state(cfg("B")).values["messages"]]
    assert a[0] == "first" and a[2] == "second" and len(a) == 4
    assert b[0] == "other" and len(b) == 2 and "first" not in b


def test_per_message_fields_do_not_leak_into_the_next_message():
    graph, fake = make({"route": "nist_question", "standalone_query": "SQ1"}, {"route": "out_of_scope"})
    send(graph, "t", "first")
    send(graph, "t", "second")
    seen2 = fake.seen[1]
    assert seen2["user_message"] == "second"
    assert seen2["route"] is None and seen2["answer"] is None and seen2["standalone_query"] is None
    assert seen2["llm_calls"] == 0 and seen2["error"] is None


def test_llm_calls_are_per_message_not_cumulative():
    graph, _ = make({"route": "nist_question", "standalone_query": "a"}, {"route": "nist_question", "standalone_query": "b"})
    assert send(graph, "t", "one")["llm_calls"] == 1
    assert send(graph, "t", "two")["llm_calls"] == 1


def test_error_stops_the_graph_and_removes_the_unanswered_message():
    graph, _ = make({"route": "nist_question", "standalone_query": "a"}, {"error": "llm_unavailable", "retry_after": 5.0})
    send(graph, "t", "one")
    out = send(graph, "t", "two")
    assert out["error"]["code"] == "LLM_UNAVAILABLE" and out["error"]["node"] == "classify_and_rewrite" and out["answer"] is None
    assert out["outcome"] == "error"
    msgs = graph.get_state(cfg("t")).values["messages"]
    assert [m.content for m in msgs][0] == "one" and len(msgs) == 2          # "two" was removed


def test_graph_is_compiled_with_an_in_memory_checkpointer_and_exports_mermaid():
    graph, _ = make()
    assert type(graph.checkpointer).__name__ in ("InMemorySaver", "MemorySaver")
    mermaid = graph.get_graph().draw_mermaid()
    for node in ("rule_greeting_check", "classify_and_rewrite", "greeting_reply", "out_of_scope_reply", "rag", "finalize", "error_finalize", "reset_turn"):
        assert node in mermaid
