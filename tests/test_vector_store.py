import math

import pytest

pytest.importorskip("chromadb")
from indexing import config  # noqa: E402
from indexing.embedder import get_embeddings  # noqa: E402
from indexing.metadata import from_chroma_metadata  # noqa: E402
from indexing.vector_store import (TokenLimitExceeded, build_vector_store, check_token_limits,  # noqa: E402
                                   open_collection, read_back_settings)


def chunk(i, text, **over):
    base = {"chunk_id": f"d:prose:{i:04d}:1", "doc_id": "d", "display_name": "D", "chunk_type": "prose",
            "section_path": ["1 A"], "section_path_is_page_level": False, "merged_sections": [], "sequence": i,
            "part": 1, "parts_total": 1, "char_start": 0, "char_end": 1, "char_spans": [[0, 1]], "pdf_page_start": 1,
            "pdf_page_end": 1, "printed_page_start": "1", "printed_page_end": "1", "source_url": "u",
            "citation_url": "u#page=1", "citation_label": "L", "display_text": text, "embed_text": "D > 1 A\n\n" + text,
            "token_count_embed": 10, "function": None, "category": None, "subcategory": None, "subcategories": [],
            "action_ids": [], "gai_risks": [], "ai_actor_tasks": [], "trustworthy_characteristics": [],
            "figure_number": None, "ocr_confidence": None, "footnote_number": None, "links": []}
    base.update(over)
    return base


CHUNKS = [chunk(1, "Confabulation is confidently stated erroneous content."),
          chunk(2, "Systems must be safely decommissioned and deactivated."),
          chunk(3, "Bananas grow in warm climates."),
          chunk(4, "Governance is a cross-cutting function.", doc_id="e")]


@pytest.fixture(scope="module")
def emb():
    return get_embeddings("cpu")


def test_builds_cosine_collection_with_all_chunks(tmp_path, emb):
    stats = build_vector_store(CHUNKS, emb, persist_dir=tmp_path, name="test_coll")
    assert stats["count"] == 4 and stats["distance"] == "cosine"
    assert stats["ids"] == {c["chunk_id"] for c in CHUNKS}
    assert all(abs(n - 1.0) < 1e-3 for n in stats["sample_norms"])
    store = open_collection(emb, tmp_path, "test_coll")
    assert read_back_settings(store._collection) == {"distance": "cosine", "count": 4}


def test_second_run_creates_no_duplicates(tmp_path, emb):
    build_vector_store(CHUNKS, emb, persist_dir=tmp_path, name="test_coll")
    stats = build_vector_store(CHUNKS, emb, persist_dir=tmp_path, name="test_coll")
    assert stats["count"] == 4


def test_removed_chunks_do_not_survive_a_rebuild(tmp_path, emb):
    build_vector_store(CHUNKS, emb, persist_dir=tmp_path, name="test_coll")
    stats = build_vector_store(CHUNKS[:2], emb, persist_dir=tmp_path, name="test_coll")
    assert stats["count"] == 2 and stats["ids"] == {c["chunk_id"] for c in CHUNKS[:2]}


def test_stored_document_is_embed_text_and_metadata_round_trips(tmp_path, emb):
    build_vector_store(CHUNKS, emb, persist_dir=tmp_path, name="test_coll")
    col = open_collection(emb, tmp_path, "test_coll")._collection
    got = col.get(ids=[CHUNKS[0]["chunk_id"]])
    assert got["documents"][0] == CHUNKS[0]["embed_text"]
    assert from_chroma_metadata(got["metadatas"][0], got["documents"][0]) == CHUNKS[0]


def test_doc_id_filter_and_similarity_order(tmp_path, emb):
    build_vector_store(CHUNKS, emb, persist_dir=tmp_path, name="test_coll")
    store = open_collection(emb, tmp_path, "test_coll")
    hits = store.similarity_search_with_score("what is confabulation?", k=3, filter={"doc_id": "d"})
    assert hits[0][0].metadata["chunk_id"] == "d:prose:0001:1" and all(h[0].metadata["doc_id"] == "d" for h in hits)
    assert hits[0][1] < hits[-1][1]                      # cosine distance: smaller is closer
    assert 0.0 <= hits[0][1] <= 2.0


def test_over_long_chunk_stops_the_build_and_names_it(tmp_path, emb):
    long = chunk(9, "word " * 700)
    with pytest.raises(TokenLimitExceeded, match="d:prose:0009:1"):
        check_token_limits(CHUNKS + [long], emb)
    with pytest.raises(TokenLimitExceeded):
        build_vector_store(CHUNKS + [long], emb, persist_dir=tmp_path, name="test_coll")


def test_duplicate_ids_rejected(tmp_path, emb):
    with pytest.raises(ValueError):
        build_vector_store([CHUNKS[0], CHUNKS[0]], emb, persist_dir=tmp_path, name="test_coll")
