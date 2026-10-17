"""Live Groq calls: skipped without GROQ_API_KEY; kept tiny for the free tier."""
import asyncio

import pytest

from app.config import Settings, get_settings

pytestmark = pytest.mark.skipif(not get_settings().groq_api_key, reason="GROQ_API_KEY not set")


def test_live_ping_and_structured_output():
    from pydantic import BaseModel

    from llm.groq_client import GroqClient

    class Answer(BaseModel):
        word: str

    async def main():       # one event loop: the Groq HTTP client is bound to the loop that first used it
        client = GroqClient(get_settings())
        reply, seconds = await client.ping()
        res = await client.structured(Answer, [("system", "Return the requested word."), ("user", "word: hello")])
        return reply, seconds, res, client.stats.failed_attempts

    reply, seconds, res, failed = asyncio.run(main())
    assert reply and seconds < 30 and failed == 0
    assert res.parsed is not None and res.llm_calls == 1


# ---- Stage R4 live routing cases (6 LLM calls in total) -------------------------------------------
def _live_graph():
    from graph.build import build_app_graph
    from tests.stub_rag import stub_rag
    settings = get_settings().require_key()
    from llm.groq_client import GroqClient
    # these cases test ROUTING only: a stub stands in for the RAG subgraph so no retrieval/generation tokens are spent
    return build_app_graph(settings, GroqClient(settings), rag_node=stub_rag)


async def _send(graph, tid, text):
    return await graph.ainvoke({"user_message": text}, {"configurable": {"thread_id": tid}})


FIRST_Q = "What are the four core functions of the AI RMF?"


def test_live_routing_cases():
    async def main():
        graph = _live_graph()
        return {
            "r1": await _send(graph, "A", FIRST_Q),
            "r2": await _send(graph, "A", "Which one is cross-cutting?"),            # follow-up, uses history
            "b1": await _send(graph, "B", FIRST_Q),
            "r3": await _send(graph, "B", "Hi again! Which one is cross-cutting?"),  # mixed greeting + follow-up
            "r4": await _send(graph, "C", "What's the capital of France?"),
            "r5": await _send(graph, "D", "Good morning, hope you're doing well"),
        }
    r = asyncio.run(main())
    assert r["r1"]["route"] == "nist_question" and r["r1"]["llm_calls"] == 1
    q2 = r["r2"]["standalone_query"].lower()
    assert r["r2"]["route"] == "nist_question" and "function" in q2 and ("ai rmf" in q2 or "govern" in q2)
    assert r["r3"]["route"] == "nist_question" and "function" in r["r3"]["standalone_query"].lower()
    assert r["r4"]["route"] == "out_of_scope" and r["r4"]["answer"] == "Sorry, we don't have this answer." and r["r4"]["llm_calls"] == 1
    assert r["r5"]["route"] == "greeting" and r["r5"]["llm_calls"] == 1
