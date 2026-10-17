"""Stage T5: sibling expansion (fake store for units, real index for the end-to-end case)."""
from __future__ import annotations

import pytest

from indexing.metadata import to_chroma_metadata
from retrieval import config
from retrieval.siblings import expand, group_key, make_chunk, missing_siblings


def rec(cid, *, chunk_type="genai_action_table", sub="GOVERN 1.3", part=1, total=2, doc="genai_profile"):
    return {"chunk_id": cid, "doc_id": doc, "display_name": "x", "chunk_type": chunk_type, "sequence": 0,
            "part": part, "parts_total": total, "subcategory": sub if chunk_type == "genai_action_table" else None,
            "category": None if chunk_type == "genai_action_table" else sub, "display_text": f"disp {cid}",
            "embed_text": f"emb {cid}", "citation_label": f"label {cid}", "section_path": ["s"]}


class FakeStore:
    def __init__(self, recs):
        self.recs = recs

    def get(self, where, include=None):
        conds = where["$and"]
        hit = [r for r in self.recs if all(to_chroma_metadata(r).get(k) == v for c in conds for k, v in c.items())]
        return {"ids": [r["chunk_id"] for r in hit], "metadatas": [to_chroma_metadata(r) for r in hit],
                "documents": [r["embed_text"] for r in hit]}


def chunk(r, rank, score=0.9):
    return make_chunk(r["chunk_id"], to_chroma_metadata(r), r["embed_text"], score=score, rank=rank)


P1, P2 = rec("g:13:1", part=1), rec("g:13:2", part=2)
SINGLE = rec("g:17:1", sub="GOVERN 1.7", part=1, total=1)
PROSE = rec("p:1", chunk_type="prose", sub="x", part=1, total=3)
STORE = FakeStore([P1, P2, SINGLE, PROSE, rec("p:2", chunk_type="prose", sub="x", part=2, total=3)])


def test_selected_part1_brings_part2():
    out = expand(STORE, [chunk(P1, 1)])
    sib = [c for c in out if c.added_as_sibling]
    assert [c.chunk_id for c in sib] == ["g:13:2"] and sib[0].sibling_of == "g:13:1"
    assert sib[0].rerank_score is None and sib[0].display_text == "disp g:13:2"


def test_both_parts_selected_adds_nothing():
    out = expand(STORE, [chunk(P1, 1), chunk(P2, 2)])
    assert [c.chunk_id for c in out] == ["g:13:1", "g:13:2"] and not any(c.added_as_sibling for c in out)


def test_non_table_and_single_part_never_expanded():
    assert group_key(to_chroma_metadata(PROSE)) is None and group_key(to_chroma_metadata(SINGLE)) is None
    out = expand(STORE, [chunk(PROSE, 1), chunk(SINGLE, 2)])
    assert [c.chunk_id for c in out] == ["p:1", "g:17:1"]


def test_no_duplicates_and_attached_to_best_part():
    out = missing_siblings(STORE, [chunk(P1, 1)])
    assert list(out) == ["g:13:1"] and len(out["g:13:1"]) == 1


def test_missing_part_in_store_is_an_error_not_silent():
    with pytest.raises(LookupError):
        expand(FakeStore([P1]), [chunk(P1, 1)])


def test_count_toward_top_n_caps_total():
    a, b = rec("a:1", sub="A", part=1), rec("a:2", sub="A", part=2)
    c1, c2 = rec("c:1", sub="C", part=1), rec("c:2", sub="C", part=2)
    d = rec("d:1", sub="D", part=1, total=1)
    store = FakeStore([a, b, c1, c2, d])
    sel = [chunk(a, 1), chunk(c1, 2), chunk(d, 3)]
    assert len(expand(store, sel, top_n=5, count_toward_top_n=False)) == 5
    capped = expand(store, sel, top_n=4, count_toward_top_n=True)
    assert [c.chunk_id for c in capped] == ["a:1", "a:2", "c:1", "c:2"]      # d does not fit


# ---------------------------------------------------------------- real index
from tests.test_query_prefix import need_indexes  # noqa: E402


@need_indexes
def test_real_store_parts_match_chunks_jsonl():
    from retrieval.components import get_components
    from retrieval.siblings import fetch_parts
    c = get_components()
    parts = fetch_parts(c.store, ("genai_profile", "genai_action_table", "MEASURE 2.11"))
    assert sorted(parts) == [1, 2]
    expected = sorted(r["chunk_id"] for r in c.chunk_lookup.values()
                      if r["subcategory"] == "MEASURE 2.11" and r["chunk_type"] == "genai_action_table")
    assert sorted(p["id"] for p in parts.values()) == expected


@need_indexes
def test_real_split_table_sibling_added_end_to_end():
    from retrieval.components import get_components
    from retrieval.fusion import fuse
    from retrieval.rerank import load_reranker, score_candidates, select_top
    from retrieval.ordering import final_order
    c = get_components()
    q = "how to decommission a GAI system"
    ids = [d.metadata["chunk_id"] for d in fuse(c.ensemble, c.vector.invoke(q), c.bm25.invoke(q))]
    top = select_top(score_candidates(load_reranker(), q, ids, c.chunk_lookup))
    sel = [make_chunk(s.chunk_id, to_chroma_metadata(c.chunk_lookup[s.chunk_id]), c.chunk_lookup[s.chunk_id]["embed_text"],
                      score=s.score, rank=s.rank) for s in top]
    out = final_order(expand(c.store, sel))
    sibs = [x for x in out if x.added_as_sibling]
    assert len(sel) == 5 and len(sibs) >= 1 and len({x.chunk_id for x in out}) == len(out)
    for s in sibs:                                    # sibling sits directly next to the part that caused it
        i = [x.chunk_id for x in out].index(s.chunk_id)
        assert out[i - 1].chunk_id == s.sibling_of or out[i + 1].chunk_id == s.sibling_of
