"""Stage E4: embed all chunks and persist them in ONE cosine-distance Chroma collection (via langchain-chroma)."""
from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path

from indexing import config
from indexing.embedder import count_tokens, get_embeddings, model_info
from indexing.metadata import to_chroma_metadata

log = logging.getLogger(__name__)


class TokenLimitExceeded(Exception):
    """A chunk's embed_text is longer than the model's input limit (never truncated silently)."""


def check_token_limits(chunks: list[dict], embeddings=None) -> int:
    """Recount every chunk with the model's own tokenizer; stop (listing IDs) on any violation. Returns the max."""
    emb = embeddings or get_embeddings()
    limit = model_info(emb)["max_seq_length"]
    counts = {c["chunk_id"]: count_tokens(c["embed_text"], emb) for c in chunks}
    over = {cid: n for cid, n in counts.items() if n > limit}
    if over:
        raise TokenLimitExceeded(f"{len(over)} chunk(s) exceed the model's {limit}-token limit: {over}")
    return max(counts.values())


def _client(persist_dir: Path):
    import chromadb
    return chromadb.PersistentClient(path=str(persist_dir))


def open_collection(embeddings=None, persist_dir: Path | None = None, name: str | None = None):
    """langchain-chroma wrapper around the existing collection (does not create one with other settings)."""
    from langchain_chroma import Chroma
    return Chroma(collection_name=name or config.COLLECTION_NAME,
                  embedding_function=embeddings or get_embeddings(),
                  persist_directory=str(persist_dir or config.CHROMA_DIR),
                  create_collection_if_not_exists=False)


def read_back_settings(collection) -> dict:
    """What the collection itself says (not what we asked for)."""
    cfg = collection.configuration
    hnsw = cfg["hnsw"] if isinstance(cfg, dict) else cfg.hnsw
    space = hnsw["space"] if isinstance(hnsw, dict) else hnsw.space
    return {"distance": space, "count": collection.count()}


def sample_norms(collection, k: int = 5, seed: int = 0) -> list[float]:
    ids = collection.get(include=[])["ids"]
    pick = random.Random(seed).sample(ids, min(k, len(ids)))
    vectors = collection.get(ids=pick, include=["embeddings"])["embeddings"]
    return [math.sqrt(sum(float(x) * float(x) for x in v)) for v in vectors]


def build_vector_store(chunks: list[dict], embeddings=None, *, persist_dir: Path | None = None,
                       name: str | None = None, batch_size: int | None = None) -> dict:
    """Full rebuild (Q1): delete the collection if it exists, recreate it with cosine distance, add every chunk.

    Cosine is set through `collection_configuration={"hnsw": {"space": "cosine"}}` (Chroma >= 1.x; the
    older `hnsw:space` collection-metadata key is only a legacy alias for the same setting).
    """
    from langchain_chroma import Chroma

    started = time.time()
    emb = embeddings or get_embeddings()
    persist_dir = persist_dir or config.CHROMA_DIR
    name = name or config.COLLECTION_NAME
    batch_size = batch_size or config.BATCH_SIZE
    max_tokens = check_token_limits(chunks, emb)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = _client(persist_dir)
    try:
        client.delete_collection(name)
        log.info("deleted existing collection %r", name)
    except Exception:                      # collection did not exist yet
        pass
    store = Chroma(collection_name=name, embedding_function=emb, persist_directory=str(persist_dir),
                   collection_configuration={"hnsw": {"space": config.DISTANCE}})
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk_ids in chunks.jsonl")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        store.add_texts(texts=[c["embed_text"] for c in batch], metadatas=[to_chroma_metadata(c) for c in batch],
                        ids=[c["chunk_id"] for c in batch])
        log.info("embedded %d/%d chunks", min(i + batch_size, len(chunks)), len(chunks))

    col = store._collection
    back = read_back_settings(col)
    return {"count": back["count"], "distance": back["distance"], "ids": set(col.get(include=[])["ids"]),
            "sample_norms": sample_norms(col), "max_embed_tokens": max_tokens,
            "build_seconds": round(time.time() - started, 2), "device": model_info(emb)["device"],
            "chroma_version": __import__("chromadb").__version__}
