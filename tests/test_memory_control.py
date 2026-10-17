"""Stage G8 / Q8: the API persists ONE checkpoint per request (durability="exit"); history and follow-ups still work."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from generation import config as gen_config
from graph.build import build_app_graph
from retrieval import config as retrieval_config
from tests.graph_fakes import SETTINGS, ScriptedChat, make_components
from tests.test_api_assembly import ANSWER, TAG, Router
from tests.test_main_graph import d


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(gen_config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(gen_config, "LENGTH_POLICY", "error")
    monkeypatch.setattr(retrieval_config, "THRESHOLD", 0.15)


def n_checkpoints(graph) -> int:
    return sum(len(ns) for t in graph.checkpointer.storage.values() for ns in t.values())


def test_one_checkpoint_per_request_on_both_endpoints_and_the_conversation_still_works():
    router = Router(d("nist_question", "first?"), d("nist_question", "second?"), d("nist_question", "third?"))
    comps = make_components(chat=ScriptedChat(script=ANSWER, tags=TAG))
    graph = build_app_graph(SETTINGS, router, components=comps)
    c = TestClient(create_app(SETTINGS, graph), raise_server_exceptions=False)
    with c:
        r1 = c.post("/chat", json={"message": "one"}).json()
        tid = r1["thread_id"]
        assert n_checkpoints(graph) == 1                                   # was ~15 per NIST turn with per-step durability
        c.post("/chat", json={"message": "two", "thread_id": tid})
        assert n_checkpoints(graph) == 2
        c.post("/chat/stream", json={"message": "three", "thread_id": tid})
        assert n_checkpoints(graph) == 3
        msgs = c.get(f"/threads/{tid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3 and msgs[0]["content"] == "one"
    assert "[C" not in " ".join(m["content"] for m in msgs)                # history is stored without markers


def test_a_follow_up_still_sees_the_previous_turn_in_the_router_window():
    seen = []

    class Spy(Router):
        async def structured(self, schema, messages):
            seen.append(messages[-1].content)
            return await super().structured(schema, messages)

    graph = build_app_graph(SETTINGS, Spy(d("nist_question", "first?"), d("nist_question", "second?")),
                            components=make_components(chat=ScriptedChat(script=ANSWER, tags=TAG)))
    c = TestClient(create_app(SETTINGS, graph), raise_server_exceptions=False)
    with c:
        tid = c.post("/chat", json={"message": "What are the functions?"}).json()["thread_id"]
        c.post("/chat", json={"message": "Which one is cross-cutting?", "thread_id": tid})
    assert "What are the functions?" in seen[1] and "GOVERN is the cross-cutting function." in seen[1]
