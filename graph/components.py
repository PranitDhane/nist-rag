"""Shared components, loaded ONCE at startup and handed to the node factories as closures (Q2). Never stored in state."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import Settings

log = logging.getLogger(__name__)


@dataclass
class Components:
    retrieval: Any                         # retrieval.components.Components: vector, bm25, ensemble, chunk_lookup, store
    reranker: Any                          # sentence-transformers CrossEncoder
    generation_client: Any                 # generation.client.GenerationClient
    settings: Settings
    # One model call at a time: the reranker shares a single (MPS/GPU or CPU) device, so concurrent torch calls from
    # several threads would only contend. Embedding/BM25/Chroma and everything else still overlap across requests.
    rerank_lock: threading.Lock = field(default_factory=threading.Lock)
    load_seconds: float = 0.0

    @property
    def chunk_lookup(self) -> dict:
        return self.retrieval.chunk_lookup


def load_components(settings: Settings, *, retrieval_loader: Callable[[], Any] | None = None,
                    reranker_loader: Callable[[], Any] | None = None, generation_client: Any = None) -> Components:
    """Load every heavy component once. The loaders are injectable so tests can count calls and avoid real models."""
    started = time.perf_counter()
    if retrieval_loader is None:
        from retrieval.components import get_components as retrieval_loader
    if reranker_loader is None:
        from retrieval.rerank import load_reranker as reranker_loader
    if generation_client is None:
        from generation.client import GenerationClient
        from generation.generate import resolve_max_tokens
        generation_client = GenerationClient(settings, max_tokens=resolve_max_tokens(False))
    comps = Components(retrieval=retrieval_loader(), reranker=reranker_loader(), generation_client=generation_client,
                       settings=settings)
    comps.load_seconds = round(time.perf_counter() - started, 2)
    log.info("graph components loaded in %.2fs", comps.load_seconds)
    return comps


async def run_blocking(fn: Callable[..., Any], *args: Any) -> Any:
    """Run CPU-heavy / blocking work in a worker thread so the event loop is never blocked (async rule, Section 7)."""
    return await asyncio.to_thread(fn, *args)
