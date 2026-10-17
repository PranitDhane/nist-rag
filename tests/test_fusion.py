"""Stage T3: fusion through EnsembleRetriever, dedupe guard, weights order, diagnostics."""
from __future__ import annotations

import pytest
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from retrieval import config
from retrieval.diagnostics import independent_rrf
from retrieval.fusion import DuplicateChunkError, check_unique, fuse


class Fixed(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query, *, run_manager=None):
        return list(self.docs)


def doc(cid, text=None):
    return Document(page_content=text or f"text {cid}", metadata={"chunk_id": cid})


def ensemble(v, b):
    return EnsembleRetriever(retrievers=[Fixed(docs=v), Fixed(docs=b)], weights=list(config.WEIGHTS),
                             c=config.RRF_C, id_key=config.ID_KEY)


def test_chunk_found_by_both_appears_once():
    e = ensemble([doc("a"), doc("b")], [doc("b"), doc("c")])
    out = [d.metadata["chunk_id"] for d in e.invoke("q")]
    assert sorted(out) == ["a", "b", "c"]
    assert out[0] == "b"                       # in both lists: its two RRF terms add up


def test_same_chunk_with_different_page_content_still_deduped_by_id():
    e = ensemble([doc("a", "vector text")], [doc("a", "bm25 text")])
    assert len(e.invoke("q")) == 1             # id_key makes page_content irrelevant


def test_guard_raises_in_strict_and_logs_in_production(caplog):
    with pytest.raises(DuplicateChunkError):
        check_unique([doc("a"), doc("a")], strict=True)
    with caplog.at_level("ERROR"):
        check_unique([doc("a"), doc("a")], strict=False)
    assert "more than once" in caplog.text


def test_weights_order_vector_preferred_wins_on_symmetric_ranks():
    v, b = [doc("v1"), doc("b1")], [doc("b1x"), doc("v1x")]      # v1 is vector rank 1; b1x is bm25 rank 1
    out = [d.metadata["chunk_id"] for d in fuse(ensemble(v, b), v, b)]
    assert out.index("v1") < out.index("b1x")


def test_fuse_equals_invoke_and_independent_rrf_matches():
    v = [doc(f"v{i}") for i in range(10)]
    b = [doc("v3")] + [doc(f"b{i}") for i in range(9)]
    e = ensemble(v, b)
    assert [d.metadata["chunk_id"] for d in fuse(e, v, b)] == [d.metadata["chunk_id"] for d in e.invoke("q")]
    rrf = independent_rrf([[d.metadata["chunk_id"] for d in v], [d.metadata["chunk_id"] for d in b]],
                          config.WEIGHTS, config.RRF_C)
    assert rrf["v3"] == pytest.approx(0.6 / (4 + 60) + 0.4 / (1 + 60))


# ---------------------------------------------------------------- real indexes
from tests.test_query_prefix import need_indexes  # noqa: E402


@need_indexes
@pytest.mark.parametrize("q", ["Which function is cross-cutting?", "GV-1.7-001",
                               "What are the four core functions of the AI RMF?"])
def test_real_fused_size_unique_and_diagnostics(q):
    from retrieval.components import get_components
    from retrieval.diagnostics import diagnose
    comps = get_components()
    fused = fuse(comps.ensemble, comps.vector.invoke(q), comps.bm25.invoke(q), strict=True)
    # 10 + 10 minus overlap; the plan expected 10-20 unique candidates
    assert 10 <= len(fused) <= 20
    d = diagnose(comps, q)
    assert d["n_fused"] == len(fused) and d["order_matches"]
