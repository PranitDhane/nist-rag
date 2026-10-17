"""Loaders the retrieval phase reuses. Every loader reads the manifest first and refuses to load on a mismatch."""
from __future__ import annotations

import json
from pathlib import Path

from indexing import config
from indexing.manifest import check_compatibility


def read_chunks(path: Path | None = None) -> list[dict]:
    path = path or config.CHUNKS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run `python -m chunking` first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunk_lookup(path: Path | None = None) -> dict[str, dict]:
    """chunk_id -> full chunk record (display_text, citation fields, lists as real lists)."""
    return {c["chunk_id"]: c for c in read_chunks(path)}


def load_vector_store(embeddings=None):
    """The persisted Chroma collection as a langchain-chroma store, after the manifest compatibility check."""
    from indexing.embedder import get_embeddings
    from indexing.vector_store import open_collection
    check_compatibility(config)
    return open_collection(embeddings or get_embeddings())


def load_bm25(corpus_path: Path | None = None):
    """(BM25Okapi, ordered chunk_ids). The BM25 object is rebuilt from the saved token lists; no pickle."""
    from indexing import bm25_index
    from indexing.manifest import IndexIncompatible, check_bm25_compatibility
    path = corpus_path or config.BM25_CORPUS_PATH
    if not path.is_file():
        raise IndexIncompatible(f"{path} is missing; build it with `python -m indexing --rebuild`")
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus["tokenizer_version"] != bm25_index.TOKENIZER_VERSION:
        raise IndexIncompatible(f"BM25 corpus was tokenized with {corpus['tokenizer_version']!r}, code is "
                                f"{bm25_index.TOKENIZER_VERSION!r}; rebuild with `python -m indexing --rebuild`")
    check_bm25_compatibility(bm25_index.TOKENIZER_VERSION)
    return bm25_index.make_bm25(corpus), corpus["chunk_ids"]
