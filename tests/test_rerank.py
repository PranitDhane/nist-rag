"""Stage T4: reranker scoring, ordering, determinism."""
from __future__ import annotations

import pytest

from retrieval import config
from retrieval.rerank import Scored, score_candidates, select_top


class FakeReranker:
    """predict(pairs, activation_fn, show_progress_bar) -> a score per pair taken from a lookup by text."""
    def __init__(self, by_text):
        self.by_text = by_text

    def predict(self, pairs, activation_fn=None, show_progress_bar=False):
        return [self.by_text[t] for _, t in pairs]


LOOKUP = {cid: {"embed_text": f"t-{cid}", "display_text": f"d-{cid}"} for cid in "abcdefg"}


def test_sorted_desc_with_ranks_and_scores_attached():
    rr = FakeReranker({"t-a": 0.2, "t-b": 0.9, "t-c": 0.5})
    out = score_candidates(rr, "q", ["a", "b", "c"], LOOKUP)
    assert [(s.chunk_id, s.rank) for s in out] == [("b", 1), ("c", 2), ("a", 3)]
    assert all(isinstance(s.score, float) for s in out)


def test_ties_broken_by_fused_position_then_chunk_id():
    rr = FakeReranker({"t-a": 0.5, "t-b": 0.5, "t-c": 0.5})
    out = score_candidates(rr, "q", ["c", "a", "b"], LOOKUP)
    assert [s.chunk_id for s in out] == ["c", "a", "b"]          # equal scores: fused position decides


def test_top_n_default_is_five():
    rr = FakeReranker({f"t-{c}": i / 10 for i, c in enumerate("abcdefg")})
    top = select_top(score_candidates(rr, "q", list("abcdefg"), LOOKUP))
    assert len(top) == config.TOP_N == 5 and top[0].chunk_id == "g"


def test_reads_the_configured_text(monkeypatch):
    monkeypatch.setattr(config, "RERANK_TEXT", "display_text")
    rr = FakeReranker({"d-a": 0.1, "d-b": 0.8})
    assert score_candidates(rr, "q", ["a", "b"], LOOKUP)[0].chunk_id == "b"


def test_empty_candidates():
    assert score_candidates(FakeReranker({}), "q", [], LOOKUP) == []


# ---------------------------------------------------------------- real model
from tests.test_query_prefix import need_indexes  # noqa: E402


@need_indexes
def test_real_reranker_scores_in_unit_range_and_deterministic():
    from retrieval.components import get_components
    from retrieval.fusion import fuse
    from retrieval.rerank import load_reranker
    c = get_components()
    q = "GV-1.7-001"
    ids = [d.metadata["chunk_id"] for d in fuse(c.ensemble, c.vector.invoke(q), c.bm25.invoke(q))]
    rr = load_reranker()
    a = score_candidates(rr, q, ids, c.chunk_lookup)
    b = score_candidates(rr, q, ids, c.chunk_lookup)
    assert len(a) == len(ids) and all(0.0 <= s.score <= 1.0 for s in a)
    assert [s.chunk_id for s in a] == [s.chunk_id for s in b]
    assert a[0].chunk_id == "genai_profile:genai_action_table:0034:1"      # GOVERN 1.7 beats the vector-only hits
