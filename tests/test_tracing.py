"""Stage G7: LangSmith is optional and off by default. Nothing here touches the network."""
from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

import graph.tracing as tracing
from app.main import create_app
from generation import config as gen_config
from graph.build import build_app_graph
from retrieval import config as retrieval_config
from tests.graph_fakes import SETTINGS, ScriptedChat, make_components
from tests.test_api_assembly import ANSWER, TAG, Router
from tests.test_main_graph import d

LANGSMITH_ENV = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_API_KEY",
                 "LANGCHAIN_API_KEY", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", "LANGSMITH_ENDPOINT")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No LangSmith variables, and any outbound socket fails the test: nothing in this file may reach the network."""
    for name in LANGSMITH_ENV:
        monkeypatch.delenv(name, raising=False)
    real_connect = socket.socket.connect

    def guarded(self, address, *a, **k):
        if isinstance(address, tuple) and address and address[0] not in ("127.0.0.1", "localhost", "::1"):
            raise AssertionError(f"outbound network call attempted to {address!r}")
        return real_connect(self, address, *a, **k)
    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(gen_config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(gen_config, "LENGTH_POLICY", "error")
    monkeypatch.setattr(retrieval_config, "THRESHOLD", 0.15)


def test_off_by_default_and_run_config_is_unchanged():
    assert tracing.tracing_enabled() is False
    assert tracing.run_config("t1") == {"configurable": {"thread_id": "t1"}}


def test_the_app_runs_with_no_langsmith_variables_and_creates_no_client_and_no_network_call(monkeypatch):
    import langsmith

    created = []
    real_init = langsmith.Client.__init__
    monkeypatch.setattr(langsmith.Client, "__init__", lambda self, *a, **k: (created.append(1), real_init(self, *a, **k))[1])
    connections = []

    def no_network(self, address, *a, **k):
        connections.append(address)
        raise OSError("network is blocked in this test")
    monkeypatch.setattr(socket.socket, "connect", no_network)

    comps = make_components(chat=ScriptedChat(script=ANSWER, tags=TAG))
    c = TestClient(create_app(SETTINGS, build_app_graph(SETTINGS, Router(d("nist_question", "q?"), d("out_of_scope")), components=comps)),
                   raise_server_exceptions=False)
    with c:
        j = c.post("/chat", json={"message": "q"})
        s = c.post("/chat/stream", json={"message": "capital?"})
    assert j.status_code == 200 and j.json()["meta"]["outcome"] == "answered" and s.status_code == 200
    assert created == [] and not [a for a in connections if a not in ("localhost", "127.0.0.1")]    # no LangSmith client, no outbound socket


@pytest.mark.parametrize("flag", ["LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"])
def test_enabled_only_by_the_env_flag_and_the_run_carries_the_agreed_tags_and_metadata(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "nist-rag")
    assert tracing.tracing_enabled() is True
    cfg = tracing.run_config("thread-9")
    assert cfg["configurable"] == {"thread_id": "thread-9"} and cfg["tags"] == ["nist-rag", "prompt:v1"]
    md = cfg["metadata"]
    assert md["thread_id"] == "thread-9" and md["prompt_version"] == "v1"
    assert len(md["retrieval_config_version"]) == 12 and md["chunks_sha256"]


def test_flag_must_be_exactly_true(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tracing.tracing_enabled() is False
    monkeypatch.setenv("LANGSMITH_TRACING", "")
    assert tracing.tracing_enabled() is False


def test_the_api_passes_the_traced_config_to_the_graph(monkeypatch):
    """With tracing enabled the /chat and /chat/stream handlers hand the tagged config to the graph (spy, no network).
    The enabled-check is patched (NOT the env var), so LangChain's own tracer is never switched on."""
    monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
    seen = []
    comps = make_components(chat=ScriptedChat(script=ANSWER, tags=TAG))
    graph = build_app_graph(SETTINGS, Router(d("nist_question", "q?"), d("nist_question", "q?")), components=comps)

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        async def ainvoke(self, inp, config=None, **kw):
            seen.append(config)
            return await self.inner.ainvoke(inp, {"configurable": config["configurable"]}, **kw)

        def astream(self, inp, config=None, **kw):
            seen.append(config)
            return self.inner.astream(inp, {"configurable": config["configurable"]}, **kw)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    c = TestClient(create_app(SETTINGS, Spy(graph)), raise_server_exceptions=False)
    with c:
        c.post("/chat", json={"message": "q"})
        c.post("/chat/stream", json={"message": "q"})
    assert len(seen) == 2 and all(cfg["tags"] == ["nist-rag", "prompt:v1"] and "thread_id" in cfg["metadata"] for cfg in seen)


@pytest.mark.skipif(not __import__("os").environ.get("LANGSMITH_API_KEY"), reason="live LangSmith check needs LANGSMITH_API_KEY")
def test_live_trace_carries_the_tags(monkeypatch):    # pragma: no cover - needs the user's key and sends data to LangSmith
    import asyncio

    from langchain_core.tracers.langchain import wait_for_all_tracers
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "nist-rag-test")
    comps = make_components(chat=ScriptedChat(script=ANSWER, tags=TAG))
    graph = build_app_graph(SETTINGS, Router(d("nist_question", "q?")), components=comps)
    asyncio.run(graph.ainvoke({"user_message": "q"}, tracing.run_config("live-thread")))
    wait_for_all_tracers()
    from langsmith import Client
    runs = list(Client().list_runs(project_name="nist-rag-test", limit=3))
    assert runs and any("nist-rag" in (r.tags or []) for r in runs)
