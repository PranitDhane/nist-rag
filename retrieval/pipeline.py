"""Stage T7: retrieve(query) -> RetrievalResult. Everything is built once (components.get_components)."""
from __future__ import annotations

import logging
import time

from retrieval import config, ordering, siblings
from retrieval.components import get_components
from retrieval.fusion import fuse
from retrieval.gate import apply_gate
from retrieval.models import RetrievalResult, RetrievedChunk
from retrieval.rerank import load_reranker, score_candidates, select_top

log = logging.getLogger(__name__)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def retrieve(query: str, *, evaluation_mode: bool = False, comps=None, reranker=None) -> RetrievalResult:
    """Hybrid retrieval for one standalone query.

    vector (prefixed query) + BM25 -> weighted RRF (EnsembleRetriever) -> cross-encoder over ALL fused
    candidates -> top N -> relevance gate -> sibling parts of split tables -> best-first final order.

    `evaluation_mode=True` lets the gate run without a threshold. Always returns the chunks and scores,
    also when `relevant` is False (the next phase decides the reply)."""
    comps = comps or get_components()
    reranker = reranker or load_reranker()
    t_total = time.perf_counter()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    vector_docs = comps.vector.invoke(query)
    timings["vector"] = _ms(t)
    t = time.perf_counter()
    bm25_docs = comps.bm25.invoke(query)
    timings["bm25"] = _ms(t)
    t = time.perf_counter()
    fused = fuse(comps.ensemble, vector_docs, bm25_docs)
    timings["fusion"] = _ms(t)
    by_id = {d.metadata["chunk_id"]: d for d in fused}

    t = time.perf_counter()
    scored = score_candidates(reranker, query, [d.metadata["chunk_id"] for d in fused], comps.chunk_lookup)
    top = select_top(scored)
    timings["rerank"] = _ms(t)

    gate = apply_gate(top, evaluation_mode=evaluation_mode)

    t = time.perf_counter()
    selected = [siblings.make_chunk(s.chunk_id, by_id[s.chunk_id].metadata, by_id[s.chunk_id].page_content,
                                    score=s.score, rank=s.rank) for s in gate.kept]
    chunks: list[RetrievedChunk] = ordering.final_order(siblings.expand(comps.store, selected))
    timings["siblings"] = _ms(t)
    timings["total"] = _ms(t_total)

    return RetrievalResult(query=query, chunks=chunks, relevant=gate.relevant, best_score=gate.best_score,
                           threshold=gate.threshold, candidate_count=len(fused), timings_ms=timings,
                           config_version=config.config_version())


def warmup() -> None:
    """Load every model and index now (call at app startup) so the first request is not slow."""
    retrieve("warm up", evaluation_mode=True)
