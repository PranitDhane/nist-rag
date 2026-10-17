"""Stage G2: components load once; CPU-heavy work runs in worker threads so concurrent requests progress."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from generation import config as gen_config
from graph.components import load_components, run_blocking
from graph.rag_subgraph import build_rag_subgraph
from retrieval import config as retrieval_config
from tests.graph_fakes import SETTINGS, make_components


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(gen_config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(gen_config, "LENGTH_POLICY", "error")
    monkeypatch.setattr(retrieval_config, "THRESHOLD", 0.15)


def test_components_load_once_and_requests_never_reload_them():
    counts = {"retrieval": 0, "reranker": 0}
    base = make_components()

    def retrieval_loader():
        counts["retrieval"] += 1
        return base.retrieval

    def reranker_loader():
        counts["reranker"] += 1
        return base.reranker

    comps = load_components(SETTINGS, retrieval_loader=retrieval_loader, reranker_loader=reranker_loader,
                            generation_client=base.generation_client)
    graph = build_rag_subgraph(comps)

    async def three_requests():
        for _ in range(3):
            await graph.ainvoke({"standalone_query": "q"})
    asyncio.run(three_requests())
    assert counts == {"retrieval": 1, "reranker": 1} and comps.load_seconds >= 0


def test_run_blocking_uses_a_worker_thread():
    main = threading.get_ident()
    assert asyncio.run(run_blocking(threading.get_ident)) != main


def test_event_loop_stays_responsive_during_blocking_search_and_rerank():
    comps = make_components(delay=0.3, rerank_delay=0.3)
    graph = build_rag_subgraph(comps)
    ticks = []

    async def ticker():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.02)

    async def go():
        t = asyncio.create_task(ticker())
        await graph.ainvoke({"standalone_query": "q"})
        t.cancel()
    asyncio.run(go())
    assert len(ticks) >= 20                         # ~0.6 s of blocking work at 20 ms ticks: the loop kept running
    assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.15


def test_two_concurrent_requests_on_different_threads_both_progress():
    comps = make_components(delay=0.4, rerank_delay=0.2)
    graph = build_rag_subgraph(comps)

    async def go():
        started = time.monotonic()
        a, b = await asyncio.gather(graph.ainvoke({"standalone_query": "request A"}, {"configurable": {"thread_id": "A"}}),
                                    graph.ainvoke({"standalone_query": "request B"}, {"configurable": {"thread_id": "B"}}))
        return a, b, time.monotonic() - started

    a, b, elapsed = asyncio.run(go())
    assert a["outcome"] == b["outcome"] == "answered"
    starts = sorted(comps.retrieval.vector.starts)
    assert len(starts) == 2 and starts[1] - starts[0] < 0.15       # B's search started without waiting for A's search or rerank
    # searches overlap (0.4 s each); only the model call is serialized by the lock (0.2 s each): ~0.4 + 0.2 + 0.2 < 2 x 0.6
    assert elapsed < 1.1
