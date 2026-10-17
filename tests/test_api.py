"""FastAPI app with the real production graph, a mocked LLM client and the real in-memory checkpointer."""
import asyncio
import logging
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from graph.build import build_app_graph
from tests.stub_rag import stub_rag
from graph.replies import GREETING_REPLY, OUT_OF_SCOPE_REPLY
from graph.router import RouteDecision
from llm.groq_client import LLMUnavailableError, StructuredResult

S = Settings(groq_api_key="x", max_message_chars=200)


class FakeClient:
    """decisions: RouteDecision | Exception | callable(messages) -> RouteDecision"""

    def __init__(self, *decisions, delay=0.0):
        self.decisions, self.calls, self.prompts, self.delay = list(decisions), 0, [], delay

    async def structured(self, schema, messages):
        self.calls += 1
        self.prompts.append(messages[1].content)
        if self.delay:
            await asyncio.sleep(self.delay)
        d = self.decisions.pop(0)
        if isinstance(d, Exception):
            raise d
        return StructuredResult(d, "{}", None, 1)


def d(route, query="", reason="r"):
    return RouteDecision(route=route, standalone_query=query, reason=reason)


def make(*decisions, delay=0.0, settings=S):
    client = FakeClient(*decisions, delay=delay)
    app = create_app(settings, build_app_graph(settings, client, rag_node=stub_rag))
    return TestClient(app, raise_server_exceptions=False), client


def post(c, message, thread_id=None, **kw):
    body = {"message": message, **({"thread_id": thread_id} if thread_id else {})}
    return c.post("/chat", json=body, **kw)


def test_first_message_creates_a_thread_id_and_returns_the_documented_shape():
    c, _ = make(d("nist_question", "What are the four core functions of the AI RMF?"))
    with c:
        r = post(c, "What are the four core functions of the AI RMF?")
    assert r.status_code == 200
    j = r.json()
    assert uuid.UUID(j["thread_id"]).version == 4
    assert j["route"] == "nist_question" and j["sources"] == [] and j["answer"].startswith("[RAG pipeline not implemented yet]")
    assert j["standalone_query"] == "What are the four core functions of the AI RMF?"
    # NIST route (INSUFFICIENT_CONTEXT_PLAN Section 5): stable schema, related_sections always present (empty when answered)
    assert set(j["meta"]) == {"llm_calls", "route_source", "latency_ms", "outcome", "insufficient_reason", "usage"}          # + generation fields only when generation ran (not here)
    assert j["meta"]["llm_calls"] == 1 and j["meta"]["route_source"] == "llm"
    assert j["meta"]["outcome"] == "answered" and j["meta"]["insufficient_reason"] is None and j["related_sections"] == []


def test_follow_up_with_thread_id_is_routed_using_history():
    c, client = make(d("nist_question", "What are the four core functions of the AI RMF?"),
                     d("nist_question", "Which of the four AI RMF core functions is cross-cutting?"))
    with c:
        tid = post(c, "What are the four core functions of the AI RMF?").json()["thread_id"]
        r = post(c, "Which one is cross-cutting?", tid)
    assert r.json()["thread_id"] == tid and r.json()["standalone_query"].startswith("Which of the four")
    assert "User: What are the four core functions of the AI RMF?" in client.prompts[1]
    assert "Assistant: [RAG pipeline not implemented yet]" in client.prompts[1]


def test_threads_are_isolated():
    c, client = make(d("nist_question", "q1"), d("out_of_scope"))
    with c:
        a = post(c, "secret question one").json()["thread_id"]
        post(c, "another conversation")
    assert "secret question one" not in client.prompts[1]
    assert a


def test_out_of_scope_reply_is_exactly_the_template_and_pure_greeting_costs_zero_calls():
    c, client = make(d("out_of_scope"))
    with c:
        r1 = post(c, "What's the capital of France?").json()
        r2 = post(c, "hello!").json()
    assert r1["answer"] == OUT_OF_SCOPE_REPLY and r1["route"] == "out_of_scope" and r1["standalone_query"] is None
    assert r2["answer"] == GREETING_REPLY and r2["meta"] == {**r2["meta"], "llm_calls": 0, "route_source": "rule"}
    assert client.calls == 1


def test_greeting_and_out_of_scope_responses_keep_their_top_level_shape_and_get_the_new_meta():
    c, _ = make(d("out_of_scope"))
    with c:
        for text in ("hello!", "What's the capital of France?"):
            j = post(c, text).json()
            assert set(j) == {"thread_id", "route", "answer", "sources", "standalone_query", "meta"}
            # top level unchanged (no related_sections); meta carries outcome on EVERY route (GRAPH_INTEGRATION_PLAN, Q9)
            assert set(j["meta"]) == {"llm_calls", "route_source", "latency_ms", "outcome", "insufficient_reason", "usage"}
            assert j["meta"]["outcome"] in ("greeting", "out_of_scope") and j["meta"]["insufficient_reason"] is None


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "   "}, {}, {"message": 5}, {"message": "hi", "thread_id": "not-a-uuid"}])
def test_validation_errors_are_422_in_fastapi_format(body):
    c, client = make()
    with c:
        r = c.post("/chat", json=body)
    assert r.status_code == 422 and isinstance(r.json()["detail"], list) and client.calls == 0


def test_too_long_message_is_422():
    c, _ = make()
    with c:
        r = post(c, "x" * 201)
    assert r.status_code == 422 and "at most 200 characters" in r.text
    with c:
        assert post(c, "hello").status_code == 200


def test_message_is_trimmed():
    c, client = make(d("out_of_scope"))
    with c:
        post(c, "   What's the capital of France?   ")
    assert client.prompts[0].rstrip().endswith("What's the capital of France?") and "   What" not in client.prompts[0]


def test_unknown_thread_id_is_404_with_clear_error():
    c, client = make()
    with c:
        r = post(c, "hello", str(uuid.uuid4()))
    assert r.status_code == 404 and r.json() == {"error": {"code": "thread_not_found",
        "message": "Unknown or expired thread_id. Omit thread_id to start a new conversation."}}


def test_llm_outage_is_503_with_retry_after_and_history_is_clean():
    c, _ = make(d("nist_question", "q1"), LLMUnavailableError("down", reason="rate_limit", retry_after=7.2, attempts=4), d("out_of_scope"))
    with c:
        tid = post(c, "first").json()["thread_id"]
        r = post(c, "second", tid)
        assert r.status_code == 503 and r.headers["retry-after"] == "8"
        assert r.json()["error"]["code"] == "llm_unavailable"
        assert [m["content"] for m in c.get(f"/threads/{tid}").json()["messages"]][0] == "first"
        assert len(c.get(f"/threads/{tid}").json()["messages"]) == 2           # the failed message was not kept
        assert post(c, "third", tid).status_code == 200                         # thread still usable


def test_outage_without_retry_after_still_sets_the_header():
    c, _ = make(LLMUnavailableError("down"))
    with c:
        r = post(c, "What is GOVERN 1.7?")
    assert r.status_code == 503 and int(r.headers["retry-after"]) >= 1


def test_unexpected_error_is_500_generic_and_never_leaks_details(caplog):
    class Boom:
        async def aget_state(self, cfg): raise RuntimeError("secret internal detail")
        async def ainvoke(self, *a, **k): raise RuntimeError("secret internal detail")
    c = TestClient(create_app(S, Boom()), raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR), c:
        r = post(c, "hello")
    assert r.status_code == 500 and r.json() == {"error": {"code": "internal_error", "message": "Internal server error."}}
    assert "secret internal detail" not in r.text and "Traceback" not in r.text
    assert "secret internal detail" in caplog.text                              # details are logged


def test_unknown_route_uses_the_error_shape():
    c, _ = make()
    with c:
        r = c.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_error"


def test_concurrent_requests_on_the_same_thread_get_409():
    client = FakeClient(d("nist_question", "q1"), d("nist_question", "q2"), delay=0.0)
    app = create_app(S, build_app_graph(S, client, rag_node=stub_rag))

    async def main():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
                tid = (await http.post("/chat", json={"message": "first question"})).json()["thread_id"]
                client.delay = 0.3
                client.decisions = [d("nist_question", "slow"), d("nist_question", "second")]
                slow = asyncio.create_task(http.post("/chat", json={"message": "slow one", "thread_id": tid}))
                await asyncio.sleep(0.05)
                busy = await http.post("/chat", json={"message": "impatient", "thread_id": tid})
                done = await slow
                after = await http.post("/chat", json={"message": "after", "thread_id": tid})
                return busy, done, after
    busy, done, after = asyncio.run(main())
    assert busy.status_code == 409 and busy.json()["error"]["code"] == "thread_busy"
    assert done.status_code == 200 and after.status_code == 200


def test_health_threads_get_and_delete():
    c, _ = make(d("out_of_scope"))
    with c:
        h = c.get("/health").json()
        assert h == {"status": "ok", "model": "openai/gpt-oss-120b", "key_configured": True}
        tid = post(c, "What's the capital of France?").json()["thread_id"]
        t = c.get(f"/threads/{tid}").json()
        assert t["messages"] == [{"role": "user", "content": "What's the capital of France?"},
                                 {"role": "assistant", "content": OUT_OF_SCOPE_REPLY, "route": "out_of_scope"}]
        assert c.delete(f"/threads/{tid}").status_code == 204
        assert c.get(f"/threads/{tid}").status_code == 404 and c.delete(f"/threads/{tid}").status_code == 404
        assert post(c, "hi", tid).status_code == 404


def test_log_line_has_structure_and_no_message_text_at_info(caplog):
    c, _ = make(d("out_of_scope"))
    with caplog.at_level(logging.INFO, logger="app.api"), c:
        tid = post(c, "What is my super secret question about France?").json()["thread_id"]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("chat "))
    for part in (f"thread_id={tid}", "route=out_of_scope", "route_source=llm", "llm_calls=1", "latency_ms="):
        assert part in line
    assert "super secret" not in caplog.text


def test_startup_without_key_fails_with_a_clear_message():
    app = create_app(Settings(groq_api_key=""))
    with pytest.raises(Exception, match="GROQ_API_KEY is not set"):
        with TestClient(app):
            pass
