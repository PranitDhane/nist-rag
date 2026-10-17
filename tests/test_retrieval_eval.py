"""Eval helpers: gold matching, metrics, threshold table, and that the shipped cases are well-formed."""
from __future__ import annotations

from eval.retrieval_eval import first_hit_rank, load_cases, matches, summarize, threshold_table, validate_gold
from tests.test_query_prefix import need_indexes

CH = {"doc_id": "genai_profile", "chunk_type": "genai_action_table", "subcategory": "GOVERN 1.7", "subcategories": [],
      "action_ids": ["GV-1.7-001"], "function": "GOVERN", "section_path": ["3 Suggested Actions", "GOVERN 1.7"],
      "figure_number": None, "footnote_number": None, "embed_text": "Decommissioning GAI"}


def test_matches_requires_all_keys_of_a_target():
    assert matches(CH, {"doc_id": "genai_profile", "subcategory": "GOVERN 1.7"})
    assert not matches(CH, {"doc_id": "ai_rmf_1_0", "subcategory": "GOVERN 1.7"})
    assert matches(CH, {"doc_id": "genai_profile", "action_id": "GV-1.7-001", "text_contains": "decommission"})
    assert not matches(CH, {"doc_id": "genai_profile", "section": "2.2"})
    assert matches({**CH, "subcategory": None, "subcategories": ["GOVERN 1.7"]},
                   {"doc_id": "genai_profile", "subcategory": "GOVERN 1.7"})


def test_first_hit_rank_and_summary():
    lookup = {"a": {**CH, "subcategory": "X"}, "b": CH}
    gold = [{"doc_id": "genai_profile", "subcategory": "GOVERN 1.7"}]
    assert first_hit_rank(["a", "b"], lookup, gold) == 2 and first_hit_rank(["a"], lookup, gold) is None
    per = [{"answerable": True, "configs": {"x": {"first_hit_rank": 2}}},
           {"answerable": True, "configs": {"x": {"first_hit_rank": None}}},
           {"answerable": False, "configs": {}}]
    assert summarize(per, "x") == {"hit_at_5": 0.5, "mrr": 0.25, "hits": 1, "answerable_cases": 2}


def test_threshold_table():
    t = threshold_table([0.9, 0.8, 0.3], [0.1, 0.4], steps=3)
    assert [r["threshold"] for r in t["rows"]] == [0.1, 0.5, 0.9]
    assert t["rows"][1] == {"threshold": 0.5, "answerable_kept_pct": 66.7, "unanswerable_rejected_pct": 100.0}
    assert t["answerable_scores"] == {"best": 0.9, "worst": 0.3}


def test_shipped_cases_wellformed():
    cases = load_cases()
    ans = [c for c in cases if c["answerable"]]
    assert len(cases) >= 30 and len(cases) - len(ans) >= 8


@need_indexes
def test_every_gold_target_exists_in_corpus():
    from indexing.loaders import load_chunk_lookup
    assert validate_gold(load_cases(), load_chunk_lookup()) == []
