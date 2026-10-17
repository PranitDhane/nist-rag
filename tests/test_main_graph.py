"""Stage G4: the full main graph (real router node + real RAG subgraph + real checkpointer) with fake components and a
mocked LLM. No network, no models."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation import config as gen_config
from graph.build import build_app_graph
from graph.replies import GREETING_REPLY, OUT_OF_SCOPE_REPLY
from graph.router import RouteDecision
from graph.serial import plain_data_violations
from llm.groq_client import LLMUnavailableError, StructuredResult
from retrieval import config as retrieval_config
from tests.gen_fakes import FakeChat, status_error
from tests.graph_fakes import GOOD, make_components

S = Settings(groq_api_key="x", llm_max_attempts=2, llm_base_delay=0.0, llm_max_delay=1.0)
HEAD = ["reset_turn", "rule_greeting_check"]
RAG_ANSWERED = ["hybrid_search", "rerank", "gate", "expand_context", "generate", "attribute"]
GOOD_STRIPPED = "GOVERN is the cross-cutting function. The Core has four functions."


class RouterClient:
    """Fake Groq client for the real router node: returns queued decisions (or raises), records every user prompt."""

    def __init__(self, *decisions):
        self.decisions, self.prompts, self.calls = list(decisions), [], 0

    async def structured(self, schema, messages):
        self.calls += 1
        self.prompts.append(messages[-1].content)
        item = self.decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return StructuredResult(item, "{}", None, 1)


def d(route, query="", reason="r"):
    return RouteDecision(route=route, standalone_query=query, reason=reason)


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(gen_config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(gen_config, "LENGTH_POLICY", "error")
    monkeypatch.setattr(retrieval_config, "THRESHOLD", 0.15)


def make(router, **comps_kw):
    comps = make_components(**comps_kw)
    return build_app_graph(S, router, components=comps), comps


def send(graph, text, tid="t"):
    return asyncio.run(graph.ainvoke({"user_message": text}, {"configurable": {"thread_id": tid}}))


def history(graph, tid="t"):
    return [(type(m).__name__, m.content) for m in graph.get_state({"configurable": {"thread_id": tid}}).values["messages"]]


def gen_calls(comps):
    return comps.generation_client.chat_model.calls


# ------------------------------------------------------------------ every route
def test_pure_greeting_by_rule_makes_no_llm_call():
    router = RouterClient()
    graph, comps = make(router)
    out = send(graph, "hello!")
    assert out["path"] == HEAD + ["greeting_reply", "finalize"] and out["outcome"] == "greeting"
    assert out["answer"] == GREETING_REPLY and out["llm_calls"] == 0 and router.calls == 0 and gen_calls(comps) == 0
    assert history(graph) == [("HumanMessage", "hello!"), ("AIMessage", GREETING_REPLY)]


def test_greeting_by_llm():
    graph, comps = make(RouterClient(d("greeting")))
    out = send(graph, "Good morning, hope you're doing well")
    assert out["path"] == HEAD + ["classify_and_rewrite", "greeting_reply", "finalize"] and out["llm_calls"] == 1
    assert out["outcome"] == "greeting" and out["route_source"] == "llm" and gen_calls(comps) == 0


def test_out_of_scope_never_reaches_retrieval_or_generation():
    graph, comps = make(RouterClient(d("out_of_scope")))
    out = send(graph, "capital of France?")
    assert out["path"] == HEAD + ["classify_and_rewrite", "out_of_scope_reply", "finalize"]
    assert out["answer"] == OUT_OF_SCOPE_REPLY and out["outcome"] == "out_of_scope" and out["llm_calls"] == 1
    assert gen_calls(comps) == 0 and comps.reranker.calls == 0 and out.get("retrieval") is None


def test_answered():
    graph, comps = make(RouterClient(d("nist_question", "Which AI RMF function is cross-cutting?")))
    out = send(graph, "which one is cross-cutting?")
    assert out["path"] == HEAD + ["classify_and_rewrite"] + RAG_ANSWERED + ["finalize"]
    assert out["outcome"] == "answered" and out["answer"] == GOOD and out["llm_calls"] == 2 and out["route"] == "nist_question"
    assert out["cited_keys"] == ["C1", "C2"] and len(out["sources"]) == 2 and out["related_sections"] == []
    assert out["retrieval"]["relevant"] is True and out["generation"]["outcome"] == "ANSWERED" and out.get("error") is None
    assert history(graph)[-1] == ("AIMessage", GOOD_STRIPPED)                    # markers stripped (attribution Q3)
    assert plain_data_violations({k: v for k, v in out.items() if k != "messages"}) == []


def test_gate_failure_is_final_with_no_generation_call():
    graph, comps = make(RouterClient(d("nist_question", "What penalties does the EU AI Act set?")),
                        scores={"ai:govern": 0.05, "ai:core": 0.04}, chat=FakeChat(chunks=["MUST NOT RUN"]))
    out = send(graph, "EU AI Act penalties?")
    assert out["path"] == HEAD + ["classify_and_rewrite", "hybrid_search", "rerank", "gate", "gate_failure_reply", "finalize"]
    assert out["outcome"] == "insufficient" and out["insufficient_reason"] == "gate_low_score" and out["answer"] == D
    assert out["llm_calls"] == 1 and gen_calls(comps) == 0 and out["sources"] == [] and out["related_sections"]
    assert history(graph)[-1] == ("AIMessage", D)                                   # related_sections never enter history
    assert "related" not in repr(history(graph)).lower() and "closest" not in repr(history(graph)).lower()


def test_model_refusal():
    graph, comps = make(RouterClient(d("nist_question", "q?")), chat=FakeChat(chunks=[D]))
    out = send(graph, "q")
    assert out["path"] == HEAD + ["classify_and_rewrite", "hybrid_search", "rerank", "gate", "expand_context", "generate",
                                  "finalize_model_refusal", "finalize"]
    assert out["outcome"] == "insufficient" and out["insufficient_reason"] == "model_case_d" and out["llm_calls"] == 2
    assert out["sources"] == [] and history(graph)[-1] == ("AIMessage", D)


def test_fixer_forced_refusal():
    graph, _ = make(RouterClient(d("nist_question", "q?")), chat=FakeChat(chunks=["Partly known [C1]. ", D]))
    out = send(graph, "q")
    assert out["outcome"] == "insufficient" and out["insufficient_reason"] == "fix_partial_answer" and out["answer"] == D


# ------------------------------------------------------------------ errors: one error node, history rolled back (Q5)
def test_router_outage_goes_to_error_finalize_and_rolls_the_turn_back():
    graph, comps = make(RouterClient(LLMUnavailableError("x", reason="rate_limit", retry_after=5.0, attempts=4)))
    out = send(graph, "what is GOVERN?")
    assert out["path"] == HEAD + ["classify_and_rewrite", "error_finalize"] and out["outcome"] == "error"
    assert out["error"] == {"code": "LLM_UNAVAILABLE", "message": out["error"]["message"], "node": "classify_and_rewrite"}
    assert out["retry_after"] == 5.0 and out["answer"] is None and history(graph) == [] and gen_calls(comps) == 0


def test_generation_outage_goes_to_error_finalize_and_rolls_the_turn_back():
    graph, _ = make(RouterClient(d("nist_question", "q?")), chat=FakeChat(chunks=[GOOD], errors=[status_error(429)] * 5))
    out = send(graph, "q")
    assert out["path"][-2:] == ["generate", "error_finalize"] and out["outcome"] == "error"
    assert out["error"]["code"] == "LLM_UNAVAILABLE" and out["error"]["node"] == "generate" and out["answer"] is None
    assert history(graph) == [] and out["sources"] == [] and "SECRET" not in repr(out["error"])


def test_retrieval_failure_goes_to_error_finalize():
    graph, comps = make(RouterClient(d("nist_question", "q?")), boom_vector=True)
    out = send(graph, "q")
    assert out["path"][-2:] == ["hybrid_search", "error_finalize"] and out["error"]["code"] == "RETRIEVAL_FAILED"
    assert "SECRET" not in repr(out["error"]) and history(graph) == [] and gen_calls(comps) == 0


def test_failed_turn_does_not_poison_the_next_one():
    graph, _ = make(RouterClient(d("nist_question", "q?"), d("out_of_scope")), boom_vector=True)
    send(graph, "first")
    out = send(graph, "second")
    assert out["outcome"] == "out_of_scope" and out.get("error") is None
    assert history(graph) == [("HumanMessage", "second"), ("AIMessage", OUT_OF_SCOPE_REPLY)]


# ------------------------------------------------------------------ state hygiene across turns
def test_nothing_from_an_answered_turn_leaks_into_the_next_message():
    graph2, _ = make(RouterClient(d("nist_question", "q?")))
    first = send(graph2, "q")                                           # fully populated turn 1
    assert first["retrieval"] and first["generation"] and first["sources"] and first["key_map"]
    out = send(graph2, "thanks!")                                       # same thread: turn 2 is a rule greeting
    assert out["outcome"] == "greeting" and out["path"] == HEAD + ["greeting_reply", "finalize"]
    for key, empty in (("retrieval", None), ("generation", None), ("sources", []), ("cited_keys", []), ("key_map", {}),
                       ("related_sections", []), ("insufficient", None), ("insufficient_reason", None), ("error", None)):
        assert out[key] == empty, key
    assert out["llm_calls"] == 0 and set(out["timings_ms"]) == {"reset_turn", "rule_greeting_check", "greeting_reply", "finalize"}


def test_one_assistant_message_per_turn_and_extras_never_in_messages():
    graph, _ = make(RouterClient(d("nist_question", "a?"), d("nist_question", "b?")),
                    chat=FakeChat(chunks=[GOOD]))
    send(graph, "one")
    send(graph, "two")
    kinds = [k for k, _ in history(graph)]
    assert kinds == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    blob = repr(history(graph))
    assert "https://" not in blob and "label" not in blob and "[C" not in blob


# ------------------------------------------------------------------ a follow-up that depends on turn 1
def test_three_turn_conversation_follow_up_uses_history_and_the_standalone_query():
    router = RouterClient(
        d("nist_question", "What are the four core functions of the AI RMF?"),
        d("out_of_scope"),
        d("nist_question", "Which of the four AI RMF functions is cross-cutting?"))
    graph, comps = make(router, chat=FakeChat(chunks=[GOOD]))
    t1 = send(graph, "What are the four core functions of the AI RMF?")
    t2 = send(graph, "what's the weather?")
    t3 = send(graph, "Which one is cross-cutting?")
    assert [t["outcome"] for t in (t1, t2, t3)] == ["answered", "out_of_scope", "answered"]
    # the router saw turn 1 (question + the stored answer, markers stripped) when it classified turn 3
    p3 = router.prompts[2]
    assert "What are the four core functions of the AI RMF?" in p3 and GOOD_STRIPPED in p3 and "[C1]" not in p3
    # the RAG subgraph searched with the rewritten, self-contained query, not the raw follow-up
    assert t3["standalone_query"] == "Which of the four AI RMF functions is cross-cutting?"
    assert t3["retrieval"]["query"] == t3["standalone_query"]
    kinds = [k for k, _ in history(graph)]
    assert kinds == ["HumanMessage", "AIMessage"] * 3
