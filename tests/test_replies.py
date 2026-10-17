"""Stage R5: every path through the real production graph (real rule + real router node, counting fake LLM)."""
import asyncio

from app.config import Settings
from graph.build import build_app_graph
from tests.stub_rag import STUB_PREFIX, stub_rag
from graph.replies import GREETING_REPLY, OUT_OF_SCOPE_REPLY
from graph.router import RouteDecision
from llm.groq_client import StructuredResult

S = Settings(groq_api_key="x")


class CountingClient:
    def __init__(self, *decisions):
        self.decisions, self.calls = list(decisions), 0

    async def structured(self, schema, messages):
        self.calls += 1
        return StructuredResult(self.decisions.pop(0), "{}", None, 1)


def d(route, query="", reason="r"):
    return RouteDecision(route=route, standalone_query=query, reason=reason)


def conversation(client, texts):
    graph = build_app_graph(S, client, rag_node=stub_rag)
    async def main():
        return [await graph.ainvoke({"user_message": t}, {"configurable": {"thread_id": "t"}}) for t in texts]
    return asyncio.run(main())


def test_exact_template_texts_are_the_approved_ones():
    assert GREETING_REPLY == ("Hello! I can answer questions about the NIST AI Risk Management Framework and its "
                              "Generative AI Profile. What would you like to know?")
    assert OUT_OF_SCOPE_REPLY == "Sorry, we don't have this answer."


def test_pure_greeting_makes_zero_llm_calls():
    client = CountingClient()
    (out,) = conversation(client, ["Hello!"])
    assert client.calls == 0 and out["llm_calls"] == 0 and out["route_source"] == "rule" and out["answer"] == GREETING_REPLY


def test_out_of_scope_is_exactly_one_call_and_exactly_the_template():
    client = CountingClient(d("out_of_scope"))
    (out,) = conversation(client, ["What's the capital of France?"])
    assert client.calls == 1 and out["llm_calls"] == 1 and out["answer"] == OUT_OF_SCOPE_REPLY and out["sources"] == []


def test_llm_greeting_is_one_call_and_the_template():
    client = CountingClient(d("greeting"))
    (out,) = conversation(client, ["Good morning, hope you're doing well"])
    assert client.calls == 1 and out["llm_calls"] == 1 and out["answer"] == GREETING_REPLY and out["route_source"] == "llm"


def test_nist_route_is_one_call_and_the_obvious_placeholder():
    client = CountingClient(d("nist_question", "What are the four core functions of the AI RMF?"))
    (out,) = conversation(client, ["What are the four core functions of the AI RMF?"])
    assert client.calls == 1 and out["answer"].startswith("[RAG pipeline not implemented yet]")
    assert out["answer"] == STUB_PREFIX + "What are the four core functions of the AI RMF?" and out["sources"] == []


def test_four_message_conversation_through_all_routes():
    client = CountingClient(d("nist_question", "What are the four core functions of the AI RMF?"),
                            d("nist_question", "Which of the four AI RMF core functions is cross-cutting?"),
                            d("out_of_scope"))
    outs = conversation(client, ["hi", "What are the four core functions of the AI RMF?", "Which one is cross-cutting?",
                                 "What's the capital of France?"])
    assert [o["route"] for o in outs] == ["greeting", "nist_question", "nist_question", "out_of_scope"]
    assert [o["llm_calls"] for o in outs] == [0, 1, 1, 1] and client.calls == 3
