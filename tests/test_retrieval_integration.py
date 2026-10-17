"""Stage T7: retrieve() end to end on the real indexes and reranker (skipped if the indexes are absent)."""
from __future__ import annotations

import pytest

from eval.retrieval_eval import is_hit, load_cases
from tests.test_query_prefix import need_indexes

pytestmark = need_indexes

CASE_IDS = ["c01", "c03", "c15", "c23", "c32"]       # four-functions, exact ID, confabulation, table, both docs


@pytest.fixture(scope="module")
def lookup():
    from indexing.loaders import load_chunk_lookup
    return load_chunk_lookup()


@pytest.mark.parametrize("case", [c for c in load_cases() if c["id"] in CASE_IDS], ids=lambda c: c["id"])
def test_answerable_cases_are_relevant_and_hit(case, lookup):
    from retrieval.pipeline import retrieve
    res = retrieve(case["query"])
    assert res.relevant and res.best_score >= res.threshold == 0.15
    assert any(is_hit(lookup[c.chunk_id], case["gold"]) for c in res.chunks)
    assert 10 <= res.candidate_count <= 20 and res.config_version
    assert set(res.timings_ms) == {"vector", "bm25", "fusion", "rerank", "siblings", "total"}
    ids = [c.chunk_id for c in res.chunks]
    assert len(ids) == len(set(ids))                                   # no duplicates ever
    selected = [c for c in res.chunks if not c.added_as_sibling]
    assert len(selected) == 5 and all(c.rerank_score is not None for c in selected)
    for c in res.chunks:
        assert c.metadata["citation_label"] and c.display_text and c.metadata["chunk_id"] == c.chunk_id


def test_unanswerable_is_not_relevant_but_still_returns_chunks():
    from retrieval.pipeline import retrieve
    res = retrieve("What penalties does the EU AI Act set for non-compliance?")
    assert res.relevant is False and res.best_score < res.threshold
    assert len(res.chunks) >= 5                                         # next phase can log/explain


def test_best_first_by_score_for_selected_chunks():
    from retrieval.pipeline import retrieve
    res = retrieve("What is confabulation?")
    scores = [c.rerank_score for c in res.chunks if not c.added_as_sibling]
    # group placement can move a lower part before a higher one, but only inside one split table
    assert scores[0] == max(scores)


def test_deterministic_two_runs():
    from retrieval.pipeline import retrieve
    a, b = retrieve("GV-1.7-001"), retrieve("GV-1.7-001")
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]


def test_sibling_end_to_end_through_retrieve():
    from retrieval.pipeline import retrieve
    res = retrieve("how to decommission a GAI system")
    sibs = [c for c in res.chunks if c.added_as_sibling]
    assert sibs and all(c.rerank_score is None and c.sibling_of for c in sibs)
