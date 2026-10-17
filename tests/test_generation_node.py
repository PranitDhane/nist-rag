"""Stage N6: generate_node in a mini graph that mirrors how item 5 will wire it (extended state + the existing `fail`/`finalize`)."""
from __future__ import annotations

import asyncio
import uuid
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation import config
from generation.client import GenerationClient
from generation.models import GenerationError
from generation.node import make_generate_node
from graph.build import error_finalize, finalize
from graph.state import GraphState
from tests.gen_fakes import FakeChat, result, status_error

S = Settings(groq_api_key="x", llm_max_attempts=2, llm_base_delay=0.0, llm_max_delay=1.0)


class WiredState(GraphState, total=False):
    retrieval_result: object
    generation: dict | None
    cited_keys: list
    key_map: dict


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(config, "LENGTH_POLICY", "insufficient")


async def _nosleep(_):
    pass


def node_for(fake):
    return make_generate_node(GenerationClient(S, chat_model=fake, sleep=_nosleep, max_tokens=2000))


def run_node(fake, state=None):
    base = {"standalone_query": "Which function is cross-cutting?", "retrieval_result": result(), "llm_calls": 1}
    return asyncio.run(node_for(fake)({**base, **(state or {})}))


def wired_graph(fake):
    """user message is stored first (like rule_greeting_check), then generate, then finalize or fail (Q7 roll-back)."""
    async def store(state):
        return {"messages": [HumanMessage(content=state["user_message"], id=str(uuid.uuid4()))], "answer": None, "error": None}

    async def retrieve(state):
        return {"standalone_query": state["user_message"], "retrieval_result": result()}

    g = StateGraph(WiredState)
    g.add_node("store", store)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", node_for(fake))
    g.add_node("finalize", finalize)
    g.add_node("fail", error_finalize)
    g.add_edge(START, "store")
    g.add_edge("store", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", lambda s: "fail" if s.get("error") else "finalize", ["finalize", "fail"])
    g.add_edge("finalize", END)
    g.add_edge("fail", END)
    return g.compile(checkpointer=InMemorySaver())


def converse(graph, text, tid="t1"):
    return asyncio.run(graph.ainvoke({"user_message": text, "route": "nist_question"}, {"configurable": {"thread_id": tid}}))


GOOD = ["GOVERN is the cross-cutting function [C2]. ", "The Core has four functions [C1]."]


def test_answered_case_writes_the_expected_fields():
    out = run_node(FakeChat(chunks=GOOD))
    assert out["answer"] == "".join(GOOD) and out["cited_keys"] == ["C2", "C1"] and out["llm_calls"] == 2
    assert out["key_map"] == {"C1": "ai:1", "C2": "ai:2", "C3": "gen:1"} and "error" not in out
    g = out["generation"]
    assert g["outcome"] == "ANSWERED" and g["corrected"] is False and g["usage"]["reasoning"] == 220


def test_corrected_case_answer_is_final_text_not_streamed_text():
    out = run_node(FakeChat(chunks=["GOVERN is cross-cutting [C2][C9]."]))
    assert out["answer"] == "GOVERN is cross-cutting [C2]." and out["generation"]["corrected"] is True
    assert out["generation"]["streamed_text"] == "GOVERN is cross-cutting [C2][C9]." and out["cited_keys"] == ["C2"]


def test_history_holds_final_text_after_a_correction():
    graph = wired_graph(FakeChat(chunks=["GOVERN is cross-cutting [C2][C9]."]))
    out = converse(graph, "Which function is cross-cutting?")
    ai = [m for m in out["messages"] if isinstance(m, AIMessage)]
    # finalize stores the FINAL text (never the streamed one) with [C#] markers stripped (attribution Q3)
    assert [m.content for m in ai] == ["GOVERN is cross-cutting."]
    assert "[C9]" not in repr(out["messages"]) and "[C2]" not in repr(out["messages"])


def test_error_case_sets_error_and_rolls_the_turn_back_q7():
    fake = FakeChat(chunks=GOOD, errors=[status_error(429)] * 5)
    out = run_node(FakeChat(chunks=GOOD, errors=[status_error(429)] * 5))
    assert out["answer"] is None and out["error"] == "generation_failed" and out["generation"]["error"]["code"] == "rate_limit"
    assert "llm_calls" not in out
    graph = wired_graph(fake)
    final = converse(graph, "Which function is cross-cutting?")
    assert final["messages"] == [] and final["answer"] is None          # the unanswered user message is gone


def test_mid_stream_failure_leaves_no_partial_text_anywhere():
    out = run_node(FakeChat(chunks=GOOD + ["more"], fail_after=2))
    assert out["answer"] is None and out["error"] == "generation_failed"
    assert out["generation"]["final_text"] is None and out["generation"]["streamed_text"] is None
    assert "GOVERN is the" not in repr(out)


def test_nothing_from_reasoning_appears_in_the_state():
    chunks = [("reasoning", "PRIVATE CHAIN OF THOUGHT"), "GOVERN [C2]."]
    out = run_node(FakeChat(chunks=chunks))
    assert "PRIVATE" not in repr(out)


def test_follow_up_turn_does_not_leak_history_into_the_prompt():
    fake = FakeChat(chunks=["GOVERN [C2]."])
    graph = wired_graph(fake)
    converse(graph, "first question about GOVERN")
    converse(graph, "second question")
    assert len(fake.last_messages) == 2 and "first question" not in repr(fake.last_messages)


def test_missing_or_empty_retrieval_result_raises():
    with pytest.raises(GenerationError):
        run_node(FakeChat(chunks=GOOD), {"retrieval_result": None})
    with pytest.raises(GenerationError):
        run_node(FakeChat(chunks=GOOD), {"retrieval_result": result(n=0)})


def test_exact_case_d_is_written_as_the_answer():
    out = run_node(FakeChat(chunks=[D]))
    assert out["answer"] == D and out["generation"]["outcome"] == "INSUFFICIENT" and out["cited_keys"] == []
