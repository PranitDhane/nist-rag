"""Stage T5: final ordering rules (Q5 = group at best part's position; Q6 = siblings are extra)."""
from __future__ import annotations

import pytest

from indexing.metadata import to_chroma_metadata
from retrieval.ordering import final_order
from retrieval.siblings import make_chunk
from tests.test_siblings import rec


def mk(r, rank=None, score=None, sib_of=None):
    return make_chunk(r["chunk_id"], to_chroma_metadata(r), r["embed_text"], score=score, rank=rank,
                      added_as_sibling=sib_of is not None, sibling_of=sib_of)


A1, A2 = rec("A:1", sub="A", part=1), rec("A:2", sub="A", part=2)
B1, B2 = rec("B:1", sub="B", part=1), rec("B:2", sub="B", part=2)
S = rec("S:1", sub="S", part=1, total=1)
PR = rec("P:1", chunk_type="prose", sub="x", part=1, total=3)


def ids(cs):
    return [c.chunk_id for c in cs]


def test_best_first_without_siblings():
    cs = [mk(S, 3, .3), mk(PR, 1, .9), mk(A1, 2, .5), mk(B1, 4, .2)]
    assert ids(final_order(cs, "group")) == ["P:1", "A:1", "S:1", "B:1"]


def test_group_keeps_table_together_in_part_order_at_best_parts_position():
    # A2 (rank 2) is selected, A1 added; prose P at rank 1; S at rank 3
    cs = [mk(PR, 1, .9), mk(A2, 2, .5), mk(S, 3, .3), mk(A1, sib_of="A:2")]
    assert ids(final_order(cs, "group")) == ["P:1", "A:1", "A:2", "S:1"]


def test_group_with_both_parts_selected_takes_best_position_once():
    cs = [mk(A1, 1, .9), mk(S, 2, .6), mk(A2, 3, .5)]
    assert ids(final_order(cs, "group")) == ["A:1", "A:2", "S:1"]


def test_append_puts_siblings_last_grouped_by_cause():
    cs = [mk(B1, 1, .9), mk(A1, 2, .5), mk(S, 3, .3), mk(A2, sib_of="A:1"), mk(B2, sib_of="B:1")]
    assert ids(final_order(cs, "append")) == ["B:1", "A:1", "S:1", "B:2", "A:2"]


def test_selected_order_never_depends_on_input_order():
    cs = [mk(PR, 1, .9), mk(A2, 2, .5), mk(S, 3, .3), mk(A1, sib_of="A:2")]
    assert ids(final_order(list(reversed(cs)), "group")) == ids(final_order(cs, "group"))


def test_unknown_placement_rejected():
    with pytest.raises(ValueError):
        final_order([mk(S, 1, .5)], "nope")
