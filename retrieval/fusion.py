"""Stage T3: weighted-RRF fusion through LangChain's EnsembleRetriever, plus the chunk_id dedupe guard."""
from __future__ import annotations

import logging
from collections import Counter

from langchain_core.documents import Document

from retrieval import config

log = logging.getLogger(__name__)


class DuplicateChunkError(Exception):
    """The same chunk_id appeared twice after fusion (never silently dropped)."""


def check_unique(docs: list[Document], *, strict: bool = False) -> None:
    """strict=True (tests): raise. strict=False (production): log ERROR and keep the list unchanged."""
    dupes = {cid: n for cid, n in Counter(d.metadata[config.ID_KEY] for d in docs).items() if n > 1}
    if not dupes:
        return
    msg = f"chunk_id appears more than once after fusion: {dupes}"
    if strict:
        raise DuplicateChunkError(msg)
    log.error(msg)


def fuse(ensemble, vector_docs: list[Document], bm25_docs: list[Document], *, strict: bool = False) -> list[Document]:
    """Fuse already-retrieved lists with the ensemble's own weighted-RRF method.

    Retrieving first (instead of ensemble.invoke) lets the pipeline time each retriever separately without
    running a retriever twice; tests prove the result equals ensemble.invoke(query)."""
    fused = ensemble.weighted_reciprocal_rank([vector_docs, bm25_docs])
    check_unique(fused, strict=strict)
    return fused


def ensemble_invoke(ensemble, query: str, *, strict: bool = False) -> list[Document]:
    """The plain EnsembleRetriever call (retrieves with both retrievers itself)."""
    fused = ensemble.invoke(query)
    check_unique(fused, strict=strict)
    return fused
