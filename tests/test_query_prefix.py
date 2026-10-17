"""Stage T1/T2: query prefix on queries only, embedding compatibility check, retrievers on the real indexes."""
from __future__ import annotations

import numpy as np
import pytest

from indexing import config as index_config
from indexing.manifest import IndexIncompatible, config_with
from retrieval import config


def _indexes_present() -> bool:
    return (index_config.MANIFEST_PATH.is_file() and index_config.BM25_CORPUS_PATH.is_file()
            and index_config.CHROMA_DIR.is_dir())


need_indexes = pytest.mark.skipif(not _indexes_present(), reason="indexes not built (python -m indexing --rebuild)")


def test_config_version_changes_with_settings(monkeypatch):
    v = config.config_version()
    assert len(v) == 12 and v == config.config_version()
    monkeypatch.setattr(config, "TOP_N", 7)
    assert config.config_version() != v


def test_prefix_text_and_weights_order():
    assert config.QUERY_PREFIX == "Represent this sentence for searching relevant passages: "
    assert config.WEIGHTS == [0.6, 0.4] and config.VECTOR_K == config.BM25_K == 10


@need_indexes
def test_startup_check_passes():
    config.startup_check()


@need_indexes
def test_mismatched_embedding_model_raises():
    with pytest.raises(IndexIncompatible):
        config.startup_check(config_with(MODEL_NAME="BAAI/bge-base-en-v1.5"))


@need_indexes
def test_query_prefix_changes_query_but_not_documents():
    from indexing.embedder import get_embeddings
    from indexing.loaders import load_chunk_lookup
    from indexing.vector_store import open_collection
    from retrieval.components import get_query_embeddings

    docs_emb, q_emb = get_embeddings(), get_query_embeddings()
    text = "What are the four core functions of the AI RMF?"
    with_prefix = np.array(q_emb.embed_query(text))
    without = np.array(docs_emb.embed_query(text))
    assert not np.allclose(with_prefix, without, atol=1e-4)
    # the prefix embedding equals encoding the prefixed text by hand
    manual = np.array(docs_emb.embed_query(config.QUERY_PREFIX + text))
    assert np.allclose(with_prefix, manual, atol=1e-4)

    # document side: the query-embeddings object still embeds documents exactly like the indexing one,
    # and equals what Chroma stored for that chunk
    chunk = next(iter(load_chunk_lookup().values()))
    stored = open_collection().get(ids=[chunk["chunk_id"]], include=["embeddings"])["embeddings"][0]
    assert np.allclose(q_emb.embed_documents([chunk["embed_text"]])[0], np.array(stored), atol=1e-3)
    assert np.allclose(docs_emb.embed_documents([chunk["embed_text"]])[0], np.array(stored), atol=1e-3)
