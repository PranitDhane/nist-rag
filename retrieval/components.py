"""Builds the retrievers once (RETRIEVAL_PLAN Stages T2/T3). Nothing here rebuilds an index."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document

from indexing import bm25_index, config as index_config
from indexing.embedder import get_embeddings
from indexing.loaders import load_bm25, load_chunk_lookup, load_vector_store
from indexing.metadata import to_chroma_metadata
from retrieval import config

log = logging.getLogger(__name__)


def get_query_embeddings(device: str | None = None):
    """Same model object as the document embedder, but embed_query() adds the bge instruction prefix.

    model_copy shares the loaded SentenceTransformer, so the model is not loaded twice, and
    embed_documents() keeps using the unchanged `encode_kwargs` (no prefix)."""
    base = get_embeddings(device)
    query_kwargs = {**base.encode_kwargs, "prompt": config.QUERY_PREFIX}
    return base.model_copy(update={"query_encode_kwargs": query_kwargs})


def build_vector_retriever(store=None, k: int | None = None):
    store = store or load_vector_store(get_query_embeddings())
    return store.as_retriever(search_kwargs={"k": k or config.VECTOR_K})


def build_bm25_retriever(k: int | None = None, chunk_lookup: dict[str, dict] | None = None):
    """langchain-community BM25Retriever around the BM25Okapi rebuilt from the saved token lists.

    `vectorizer` is a plain field, so the prebuilt object is passed in and nothing is re-tokenized.
    `preprocess_func` (our versioned tokenize) is used only on queries. Each Document's page_content is
    embed_text, the same text Chroma stores, and its metadata carries chunk_id."""
    from langchain_community.retrievers import BM25Retriever

    bm25, chunk_ids = load_bm25()
    lookup = chunk_lookup or load_chunk_lookup()
    docs = [Document(page_content=lookup[cid]["embed_text"], metadata=to_chroma_metadata(lookup[cid]), id=cid)
            for cid in chunk_ids]
    return BM25Retriever(vectorizer=bm25, docs=docs, k=k or config.BM25_K, preprocess_func=bm25_index.tokenize)


def build_ensemble(vector, bm25):
    from langchain_classic.retrievers import EnsembleRetriever
    return EnsembleRetriever(retrievers=[vector, bm25], weights=list(config.WEIGHTS), c=config.RRF_C,
                             id_key=config.ID_KEY)


@dataclass
class Components:
    vector: object
    bm25: object
    ensemble: object
    chunk_lookup: dict
    store: object
    reranker: object = None


@lru_cache(maxsize=1)
def get_components() -> Components:
    """Build everything once per process (startup check first)."""
    config.startup_check()
    store = load_vector_store(get_query_embeddings())
    lookup = load_chunk_lookup()
    vector = build_vector_retriever(store)
    bm25 = build_bm25_retriever(chunk_lookup=lookup)
    return Components(vector=vector, bm25=bm25, ensemble=build_ensemble(vector, bm25),
                      chunk_lookup=lookup, store=store)
