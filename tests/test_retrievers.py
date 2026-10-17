"""Stage T2: each retriever on its own, real indexes (skipped if absent)."""
from __future__ import annotations

import pytest

from tests.test_query_prefix import need_indexes

pytestmark = need_indexes


@pytest.fixture(scope="module")
def comps():
    from retrieval.components import get_components
    return get_components()


def ids(docs):
    return [d.metadata["chunk_id"] for d in docs]


def test_bm25_exact_action_id_rank_one(comps):
    top = comps.bm25.invoke("GV-1.7-001")
    assert len(top) == 10
    assert top[0].metadata["subcategory"] == "GOVERN 1.7" and top[0].metadata["doc_id"] == "genai_profile"


def test_bm25_confabulation_in_top3(comps):
    top = comps.bm25.invoke("confabulation")[:3]
    assert any(d.metadata.get("section_leaf", "").startswith("2.2 Confabulation") for d in top)


def test_vector_four_functions_in_top5(comps):
    full = comps.vector.invoke("What are the four core functions of the AI RMF?")
    assert len(full) == 10
    top = full[:5]
    assert any(d.metadata["doc_id"] == "ai_rmf_1_0" and d.metadata.get("section_leaf", "").startswith("5 AI RMF Core")
               for d in top)


def test_documents_carry_chunk_id_and_same_text_in_both_retrievers(comps):
    q = "decommissioning GAI systems"
    v = {d.metadata["chunk_id"]: d for d in comps.vector.invoke(q)}
    b = {d.metadata["chunk_id"]: d for d in comps.bm25.invoke(q)}
    both = set(v) & set(b)
    assert both, "expected overlap for this query"
    for cid in both:
        assert v[cid].page_content == b[cid].page_content
