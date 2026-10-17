"""Stage E2: the embedding function, built exactly as specified (EMBEDDING_INDEXING_PLAN E-D1, E-D2, Section 3).

Query-time code must call get_embeddings() too, so documents and queries share model, revision and normalization.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from indexing import config

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_embeddings(device: str | None = None):
    """HuggingFaceEmbeddings for bge-small-en-v1.5 at the pinned revision.

    langchain-huggingface passes `model_kwargs` to SentenceTransformer (device, revision) and
    `encode_kwargs` to .encode() (normalize_embeddings, batch_size). The local cache is tried first so
    that, after the first download, nothing touches the network.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    device = device or config.resolve_device()
    encode_kwargs = {"normalize_embeddings": config.NORMALIZE, "batch_size": config.BATCH_SIZE}
    for local_only in (True, False):
        try:
            return HuggingFaceEmbeddings(
                model_name=config.MODEL_NAME,
                model_kwargs={"device": device, "revision": config.MODEL_REVISION, "local_files_only": local_only},
                encode_kwargs=encode_kwargs,
            )
        except OSError:
            if not local_only:
                raise
            log.info("model not in the local cache yet; downloading %s@%s", config.MODEL_NAME, config.MODEL_REVISION[:8])
    raise AssertionError("unreachable")


def model_info(embeddings=None) -> dict:
    """Facts about the loaded model, read back from the SentenceTransformer itself."""
    emb = embeddings or get_embeddings()
    st = emb._client
    return {"max_seq_length": st.max_seq_length, "dimension": st.get_embedding_dimension(),
            "device": str(st.device), "tokenizer_class": type(st.tokenizer).__name__,
            "model_name": config.MODEL_NAME, "model_revision": config.MODEL_REVISION}


def count_tokens(text: str, embeddings=None) -> int:
    """Token count with the model's own tokenizer, special tokens included, no truncation."""
    emb = embeddings or get_embeddings()
    return len(emb._client.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def cache_location() -> str:
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(config.MODEL_NAME, "config.json", revision=config.MODEL_REVISION)
    if not isinstance(cached, str):
        raise FileNotFoundError(f"{config.MODEL_NAME}@{config.MODEL_REVISION[:8]} is not in the local cache")
    return str(Path(cached).parent)
