"""Stage T7: relevance gate."""
from __future__ import annotations

import pytest

from retrieval import config
from retrieval.gate import ThresholdNotSet, apply_gate
from retrieval.rerank import Scored


def top(*scores):
    return [Scored(f"c{i}", s, i + 1, i + 1) for i, s in enumerate(scores)]


def test_config_threshold_is_the_user_choice():
    assert config.THRESHOLD == 0.15 and config.GATE_REMOVES_CHUNKS is False


def test_above_and_below_threshold():
    assert apply_gate(top(0.9, 0.2), threshold=0.15).relevant is True
    assert apply_gate(top(0.1, 0.05), threshold=0.15).relevant is False


def test_equal_to_threshold_is_relevant():
    assert apply_gate(top(0.15), threshold=0.15).relevant is True


def test_gate_only_keeps_all_chunks_even_when_not_relevant():
    g = apply_gate(top(0.1, 0.05), threshold=0.15, removes_chunks=False)
    assert not g.relevant and len(g.kept) == 2 and g.best_score == 0.1 and g.threshold == 0.15


def test_removal_mode_drops_chunks_below_threshold():
    g = apply_gate(top(0.9, 0.2, 0.1, 0.05), threshold=0.15, removes_chunks=True)
    assert [s.chunk_id for s in g.kept] == ["c0", "c1"] and g.relevant


def test_removal_mode_can_leave_nothing():
    g = apply_gate(top(0.1), threshold=0.15, removes_chunks=True)
    assert g.kept == [] and not g.relevant and g.best_score == 0.1


def test_empty_result_is_not_relevant():
    g = apply_gate([], threshold=0.15)
    assert g.relevant is False and g.best_score is None and g.kept == []


def test_refuses_without_threshold_except_in_evaluation_mode(monkeypatch):
    monkeypatch.setattr(config, "THRESHOLD", None)
    with pytest.raises(ThresholdNotSet):
        apply_gate(top(0.9))
    g = apply_gate(top(0.9), evaluation_mode=True)
    assert g.threshold is None and g.best_score == 0.9
