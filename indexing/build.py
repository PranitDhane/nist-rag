"""Orchestrates the build: chunks.jsonl -> Chroma + BM25 + manifest (Stages E4, E5 and the manifest of E1)."""
from __future__ import annotations

import logging
import shutil
import time

from indexing import bm25_index, config
from indexing.loaders import read_chunks
from indexing.manifest import merge_manifest, sha256_file, sha256_ids

log = logging.getLogger(__name__)


def _rel(path) -> str:
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_chroma(chunks: list[dict], embeddings=None) -> dict:
    from indexing.embedder import get_embeddings, model_info
    from indexing.vector_store import build_vector_store
    emb = embeddings or get_embeddings()
    stats = build_vector_store(chunks, emb)
    info = model_info(emb)
    merge_manifest({
        "embedding_model": config.MODEL_NAME, "model_revision": config.MODEL_REVISION, "dimension": info["dimension"],
        "normalized": config.NORMALIZE, "max_seq_length": info["max_seq_length"], "document_prefix": config.DOCUMENT_PREFIX,
        "vector_store": "chroma", "chroma_version": stats["chroma_version"], "collection_name": config.COLLECTION_NAME,
        "distance": stats["distance"], "persist_dir": _rel(config.CHROMA_DIR), "batch_size": config.BATCH_SIZE,
        "device": stats["device"], "chroma_build_seconds": stats["build_seconds"],
    })
    return stats


def build_bm25(chunks: list[dict]) -> dict:
    started = time.time()
    corpus = bm25_index.build_corpus(chunks)
    path = bm25_index.save_corpus(corpus)
    stats = bm25_index.corpus_stats(corpus)
    merge_manifest({"bm25": {
        "library": corpus["library"], "version": corpus["version"], "tokenizer_version": corpus["tokenizer_version"],
        "k1": corpus["k1"], "b": corpus["b"], "epsilon": corpus["epsilon"], "stopwords": corpus["stopwords"],
        "stemming": corpus["stemming"], "indexed_text": corpus["indexed_text"], "corpus_file": _rel(path),
        "vocab_size": stats["vocab_size"], "avg_doc_len": stats["avg_doc_len"],
    }, "bm25_build_seconds": round(time.time() - started, 2)})
    return stats


def record_chunk_hashes(chunks: list[dict]) -> None:
    merge_manifest({"chunks_file_sha256": sha256_file(config.CHUNKS_PATH), "chunk_count": len(chunks),
                    "chunk_ids_sha256": sha256_ids([c["chunk_id"] for c in chunks])})


def build(only: str | None = None, rebuild: bool = False) -> dict:
    """Full rebuild of both indexes (Q1) or of one of them. `rebuild` first wipes both persisted outputs."""
    started = time.time()
    chunks = read_chunks()
    if rebuild:
        for d in (config.CHROMA_DIR, config.INDEX_DIR):
            if d.exists():
                shutil.rmtree(d)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    if only in (None, "chroma"):
        out["chroma"] = build_chroma(chunks)
    if only in (None, "bm25"):
        out["bm25"] = build_bm25(chunks)
    record_chunk_hashes(chunks)
    seconds = round(time.time() - started, 2)
    merge_manifest({"build_seconds": seconds})
    out["build_seconds"] = seconds
    return out
