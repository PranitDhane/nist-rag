import asyncio
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import api
from app.config import Settings
from app.main import create_app
from app.sse import format_event, parse_events
from graph.build import build_app_graph
from tests.stub_rag import stub_rag
from graph.replies import GREETING_REPLY, OUT_OF_SCOPE_REPLY
from graph.router import RouteDecision
from llm.groq_client import GroqClient, LLMUnavailableError, StructuredResult
from tests.test_api import FakeClient, d

S = Settings(groq_api_key="x", max_message_chars=200)


def stream(c, message, thread_id=None):
    body = {"message": message, **({"thread_id": thread_id} if thread_id else {})}
    r = c.post("/chat/stream", json=body)
    return r, (parse_events(r.text) if r.status_code == 200 else None)


def make(*decisions, delay=0.0):
    client = FakeClient(*decisions, delay=delay)
    return TestClient(create_app(S, build_app_graph(S, client, rag_node=stub_rag)), raise_server_exceptions=False), client


def test_format_event_is_single_line_json_frame():
    assert format_event("token", {"text": "a\nb “q”"}) == 'event: token\ndata: {"text":"a\\nb “q”"}\n\n'
    assert parse_events(format_event("meta", {"x": 1}) + format_event("done", {})) == [("meta", {"x": 1}), ("done", {})]


def test_greeting_rule_stream_has_exact_event_order_and_zero_calls():
    c, client = make()
    with c:
        r, ev = stream(c, "hello!")
    assert r.headers["content-type"].startswith("text/event-stream") and r.headers["cache-control"] == "no-cache"
    assert [e for e, _ in ev] == ["meta", "token", "sources", "done"]
    assert ev[0][1]["route"] == "greeting" and len(ev[0][1]["thread_id"]) == 36
    assert ev[1][1] == {"text": GREETING_REPLY} and ev[2][1] == {"sources": []}
    assert ev[3][1]["llm_calls"] == 0 and ev[3][1]["latency_ms"] >= 0 and client.calls == 0


def test_out_of_scope_template_is_one_token_event():
    c, _ = make(d("out_of_scope"))
    with c:
        _, ev = stream(c, "What's the capital of France?")
    assert [e for e, _ in ev] == ["meta", "token", "sources", "done"]
    assert ev[0][1]["route"] == "out_of_scope" and ev[1][1]["text"] == OUT_OF_SCOPE_REPLY and ev[3][1]["llm_calls"] == 1


def test_nist_route_streams_the_stub_text_as_one_token_event():
    c, _ = make(d("nist_question", "What are the four core functions of the AI RMF?"))
    with c:
        _, ev = stream(c, "What are the four core functions of the AI RMF?")
    assert [e for e, _ in ev] == ["meta", "token", "sources", "done"]
    assert ev[0][1]["route"] == "nist_question"
    assert ev[1][1]["text"] == "[RAG pipeline not implemented yet] Standalone query: What are the four core functions of the AI RMF?"


def test_stream_continues_a_thread_and_uses_history():
    c, client = make(d("nist_question", "q1"), d("nist_question", "q2"))
    with c:
        _, ev = stream(c, "What are the four core functions of the AI RMF?")
        tid = ev[0][1]["thread_id"]
        _, ev2 = stream(c, "Which one is cross-cutting?", tid)
    assert ev2[0][1]["thread_id"] == tid and "User: What are the four core functions" in client.prompts[1]


def test_http_errors_happen_before_the_stream_starts():
    c, _ = make()
    with c:
        assert c.post("/chat/stream", json={"message": ""}).status_code == 422
        assert c.post("/chat/stream", json={"message": "x" * 201}).status_code == 422
        r = c.post("/chat/stream", json={"message": "hi", "thread_id": "3f2b8f64-5717-4562-b3fc-2c963f66afa6"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "thread_not_found"


def test_llm_outage_gives_an_error_event_and_cleans_the_history():
    c, _ = make(d("nist_question", "q1"), LLMUnavailableError("down", reason="rate_limit", retry_after=3, attempts=4), d("out_of_scope"))
    with c:
        _, ev = stream(c, "first question about AI RMF")
        tid = ev[0][1]["thread_id"]
        r, ev2 = stream(c, "What is GOVERN 1.7?", tid)
        assert [e for e, _ in ev2] == ["error"] and ev2[0][1]["code"] == "llm_unavailable" and "temporarily" in ev2[0][1]["message"]
        assert len(c.get(f"/threads/{tid}").json()["messages"]) == 2           # failed message removed
        _, ev3 = stream(c, "capital of France?", tid)
        assert [e for e, _ in ev3][-1] == "done"


def test_second_request_on_a_streaming_thread_is_409_and_lock_is_released_afterwards():
    client = FakeClient(d("nist_question", "q1"))
    app = create_app(S, build_app_graph(S, client, rag_node=stub_rag))

    async def main():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
                first = await http.post("/chat", json={"message": "first question"})
                tid = first.json()["thread_id"]
                client.delay, client.decisions = 0.3, [d("nist_question", "slow"), d("nist_question", "again")]
                slow = asyncio.create_task(http.post("/chat/stream", json={"message": "slow one", "thread_id": tid}))
                await asyncio.sleep(0.05)
                busy = await http.post("/chat/stream", json={"message": "impatient", "thread_id": tid})
                done = await slow
                after = await http.post("/chat/stream", json={"message": "after", "thread_id": tid})
                return busy, done, after
    busy, done, after = asyncio.run(main())
    assert busy.status_code == 409 and done.status_code == 200 and after.status_code == 200


# ---- the router's output must never reach the client ------------------------------------------------
ROUTER_JSON = '{"route": "nist_question", "standalone_query": "ROUTERSECRET query", "reason": "ROUTERSECRET reason"}'


class StreamingChat:
    """A chat model whose structured-output runnable calls a REAL (fake-token-streaming) LangChain chat model inside the
    router node, so LangGraph's message stream sees router tokens exactly as it would with Groq."""

    def with_structured_output(self, schema, **kw):
        async def run(messages):
            inner = GenericFakeChatModel(messages=iter([AIMessage(content=ROUTER_JSON)]))
            raw = await inner.ainvoke(messages)
            return {"raw": raw, "parsed": RouteDecision(route="nist_question", standalone_query="ROUTERSECRET query",
                                                        reason="ROUTERSECRET reason"), "parsing_error": None}
        return RunnableLambda(run)


def test_router_output_is_not_streamed():
    client = GroqClient(S, chat_model=StreamingChat())
    graph = build_app_graph(S, client, rag_node=stub_rag)

    async def raw_stream_nodes():                           # prove the leak is real without the filter
        seen = []
        async for mode, chunk in graph.astream({"user_message": "What is GOVERN 1.7?"}, {"configurable": {"thread_id": "probe"}},
                                               stream_mode=["messages"]):
            token, meta = chunk
            seen.append((meta["langgraph_node"], getattr(token, "content", "")))
        return seen
    seen = asyncio.run(raw_stream_nodes())
    assert any(node == "classify_and_rewrite" and "ROUTERSECRET" in text for node, text in seen), \
        "test premise: LangGraph does expose router tokens in the raw message stream"

    c = TestClient(create_app(S, graph), raise_server_exceptions=False)
    with c:
        r, ev = stream(c, "What is GOVERN 1.7?")
    assert "ROUTERSECRET reason" not in r.text and '"route": "nist_question"' not in r.text and "{" not in "".join(
        d_["text"] for e, d_ in ev if e == "token")
    assert [e for e, _ in ev] == ["meta", "token", "sources", "done"]
    assert ev[1][1]["text"] == "[RAG pipeline not implemented yet] Standalone query: ROUTERSECRET query"   # only the stub's text


def test_only_streaming_nodes_stream_model_tokens(monkeypatch):
    """The design the RAG phase will use: tokens from the allowed node stream individually; nothing is duplicated."""
    async def generating_node(state):
        model = GenericFakeChatModel(messages=iter([AIMessage(content="alpha beta gamma")]), tags=["answer_generation"])
        msg = await model.ainvoke("q")
        return {"answer": msg.content, "sources": [{"id": "s1"}]}
    client = FakeClient(d("nist_question", "q"))
    graph = build_app_graph(S, client, rag_node=generating_node)
    c = TestClient(create_app(S, graph), raise_server_exceptions=False)
    with c:
        _, ev = stream(c, "What is GOVERN 1.7?")
    tokens = [d_["text"] for e, d_ in ev if e == "token"]
    assert len(tokens) > 1 and "".join(tokens).replace(" ", "") == "alphabetagamma"
    assert [e for e, _ in ev][0] == "meta" and [e for e, _ in ev][-2:] == ["sources", "done"]
    assert dict(ev)["sources"] == {"sources": [{"id": "s1"}]}
    assert not any("ROUTER" in t for t in tokens)


# ---- disconnects -------------------------------------------------------------------------------------
def test_client_disconnect_stops_cleanly_cleans_history_and_releases_the_lock(caplog):
    client = FakeClient(d("nist_question", "q1"), d("nist_question", "q2"))
    graph = build_app_graph(S, client, rag_node=stub_rag)
    released = []

    async def main():
        cfg = {"configurable": {"thread_id": "t1"}}
        first = [f async for f in api.chat_events(graph, "t1", "first question about AI", release=lambda: released.append(1))]
        gen = api.chat_events(graph, "t1", "second question about GOVERN", release=lambda: released.append(2))
        frame = await gen.__anext__()                       # client reads the meta event ...
        await gen.aclose()                                  # ... then disconnects
        msgs = (await graph.aget_state(cfg)).values["messages"]
        return first, frame, msgs
    with caplog.at_level(logging.INFO, logger="app.api"):
        first, frame, msgs = asyncio.run(main())
    assert frame.startswith("event: meta")
    assert [m.content for m in msgs][0] == "first question about AI" and len(msgs) == 2          # no dangling second message
    assert released == [1, 2] and "interrupted before completion" in caplog.text
