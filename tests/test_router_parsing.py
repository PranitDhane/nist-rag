"""Router node with a mocked LLM client: valid, invalid-then-valid, invalid twice -> fallback, outage -> error."""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from graph.router import RouteDecision, build_router_messages, make_classify_node, validate_decision
from llm.groq_client import LLMUnavailableError, StructuredResult

S = Settings(groq_api_key="x")


class FakeClient:
    def __init__(self, *outputs):
        self.outputs, self.calls, self.last_messages = list(outputs), 0, None

    async def structured(self, schema, messages):
        self.calls += 1
        self.last_messages = messages
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def ok(route, query="", reason="r"):
    return StructuredResult(RouteDecision(route=route, standalone_query=query, reason=reason), "{}", None, 1)


BAD = StructuredResult(None, "not json", "bad json", 1)


def state(text="Which one is cross-cutting?", history=()):
    return {"user_message": text, "llm_calls": 0, "messages": [*history, HumanMessage(content=text)]}


def run(client, st=None):
    return asyncio.run(make_classify_node(client, S)(st or state()))


def test_valid_nist_output():
    out = run(FakeClient(ok("nist_question", " Which of the four AI RMF functions is cross-cutting? ", " follow-up ")))
    assert out["route"] == "nist_question" and out["route_source"] == "llm" and out["llm_calls"] == 1
    assert out["standalone_query"] == "Which of the four AI RMF functions is cross-cutting?" and out["route_reason"] == "follow-up"


def test_out_of_scope_and_greeting_carry_no_query_even_if_the_model_adds_one():
    for route in ("out_of_scope", "greeting"):
        out = run(FakeClient(ok(route, "What's the capital of France?")))
        assert out["route"] == route and out["standalone_query"] is None and out["llm_calls"] == 1


def test_invalid_then_valid_retries_once_and_counts_both_calls():
    client = FakeClient(BAD, ok("out_of_scope"))
    out = run(client)
    assert client.calls == 2 and out["route"] == "out_of_scope" and out["route_source"] == "llm" and out["llm_calls"] == 2


def test_empty_standalone_query_for_nist_counts_as_invalid():
    client = FakeClient(ok("nist_question", "  "), ok("nist_question", "Real query?"))
    out = run(client)
    assert client.calls == 2 and out["standalone_query"] == "Real query?"


def test_invalid_twice_uses_the_configured_fallback_route():
    client = FakeClient(BAD, BAD)
    out = run(client)
    assert client.calls == 2 and out["route"] == "out_of_scope" and out["route_source"] == "fallback"
    assert out["llm_calls"] == 2 and out.get("error") is None and out["standalone_query"] is None


def test_outage_sets_error_and_never_guesses_a_route():
    out = run(FakeClient(LLMUnavailableError("down", reason="rate_limit", retry_after=12.0, attempts=4)))
    assert out["error"] == "llm_unavailable" and out["retry_after"] == 12.0 and "route" not in out and out["llm_calls"] == 0


def test_outage_on_the_retry_also_errors():
    out = run(FakeClient(BAD, LLMUnavailableError("down")))
    assert out["error"] == "llm_unavailable" and "route" not in out and out["llm_calls"] == 1


def test_history_window_and_current_message_reach_the_prompt():
    hist = [HumanMessage(content="What are the four core functions of the AI RMF?"), AIMessage(content="GOVERN, MAP, MEASURE, MANAGE")]
    client = FakeClient(ok("nist_question", "q"))
    run(client, state(history=hist))
    system, human = client.last_messages
    assert "routing step" in system.content
    assert "User: What are the four core functions of the AI RMF?" in human.content
    assert "Assistant: GOVERN, MAP, MEASURE, MANAGE" in human.content
    assert human.content.rstrip().endswith("Which one is cross-cutting?")
    assert human.content.count("Which one is cross-cutting?") == 1            # current message not duplicated in history


def test_only_two_turns_are_sent():
    hist = []
    for i in range(1, 5):
        hist += [HumanMessage(content=f"question{i}"), AIMessage(content=f"answer{i}")]
    client = FakeClient(ok("greeting"))
    run(client, state("hello there friend?", hist))
    text = client.last_messages[1].content
    assert "question3" in text and "question4" in text and "question2" not in text and "question1" not in text


def test_validate_decision_unit():
    assert validate_decision(None) is None
    assert validate_decision(RouteDecision(route="nist_question", standalone_query="", reason="r")) is None
    assert validate_decision(RouteDecision(route="greeting", standalone_query="x", reason=" r ")).standalone_query == ""


def test_prompt_contains_the_required_parts():
    from graph.prompts import ROUTER_SYSTEM_PROMPT as P
    for needle in ("NIST AI 100-1", "NIST AI 600-1", "greeting", "nist_question", "out_of_scope", "Hi again! Which one is cross-cutting?",
                   "GV-1.7-001", "Add no facts", "Respond only in the required structure", "the assistant itself"):
        assert needle in P, needle
    assert P.count("[current message]") >= 3
