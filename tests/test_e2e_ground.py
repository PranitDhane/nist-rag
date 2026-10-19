"""Stage V4: groundedness scoring and calibration, with a fake scorer (no model, no network)."""
import random

import pytest

from eval import e2e_checks as ck
from eval import e2e_ground as g

CHUNKS = {
    "a": {"chunk_id": "a", "doc_id": "d1", "section_path": ["S1"], "display_text": "GOVERN is a cross-cutting function that is infused throughout the other functions."},
    "b": {"chunk_id": "b", "doc_id": "d1", "section_path": ["S2"], "display_text": "| Action | Text |\n| GV-1.7-001 | Protocols are put in place to ensure GAI systems are able to be deactivated when necessary |"},
    "c": {"chunk_id": "c", "doc_id": "d1", "section_path": ["S3"], "display_text": "Short."},
    "d": {"chunk_id": "d", "doc_id": "d2", "section_path": ["S1"], "display_text": "Confabulation is the production of confidently stated but erroneous content by systems."},
    "e": {"chunk_id": "e", "doc_id": "d2", "section_path": ["S9"], "display_text": "Incident disclosure means sharing information about harmful events with the public."},
}


def word_overlap_scorer(sentence, texts):
    """Deterministic stand-in for the reranker: share of the sentence's words found in the chunk text."""
    words = {w.lower().strip(".,[]") for w in sentence.split()}
    return [round(len(words & {w.lower().strip(".,|") for w in t.split()}) / len(words), 4) for t in texts]


# ---------------------------------------------------------------- sentence extraction
def test_units_with_their_citation_keys_and_markers_removed():
    units, uncited = g.extract_units("GOVERN is cross-cutting [C1][C2]. It is infused throughout [C1].")
    assert [(u.text, u.keys) for u in units] == [("GOVERN is cross-cutting.", ["C1", "C2"]), ("It is infused throughout.", ["C1"])]
    assert uncited == []


def test_uncited_sentences_are_listed_separately():
    units, uncited = g.extract_units("GOVERN is cross-cutting [C1]. This sentence has no citation.")
    assert len(units) == 1 and uncited == ["This sentence has no citation."]


def test_a_marker_after_the_full_stop_belongs_to_the_previous_sentence():
    units, uncited = g.extract_units("GOVERN is cross-cutting. [C1] It is infused throughout. [C2]")
    assert [(u.text, u.keys) for u in units] == [("GOVERN is cross-cutting.", ["C1"]), ("It is infused throughout.", ["C2"])] and uncited == []


def test_bullets_lead_in_and_case_b_first_line_are_handled_like_the_parser():
    text = "The AI RMF has four functions:\n- GOVERN sets policy [C1].\n- MAP frames context [C2]."
    units, uncited = g.extract_units(text)
    assert [u.keys for u in units] == [["C1"], ["C2"]] and uncited == []          # the lead-in line is exempt
    b = "This question can be read in more than one way:\n- Reading one: it is X [C1].\n- Reading two: not covered by the provided documents."
    units, uncited = g.extract_units(b)
    assert len(units) == 1 and uncited == []


def test_abbreviation_and_quoted_period_do_not_split_a_sentence():
    units, _ = g.extract_units('It covers risks, e.g. bias and privacy [C1]. NIST calls it "a voluntary framework. Not a rule" [C2].')
    assert len(units) == 2


# ---------------------------------------------------------------- scoring
def test_score_is_the_max_over_cited_chunks_and_unresolved_keys_are_skipped():
    units, _ = g.extract_units("GOVERN is a cross-cutting function [C1][C2][C9].")
    scored = g.score_units(units, {"C1": CHUNKS["a"]["display_text"], "C2": CHUNKS["c"]["display_text"]}, word_overlap_scorer)
    assert set(scored[0].scores) == {"C1", "C2"} and scored[0].best == scored[0].scores["C1"] > scored[0].scores["C2"]


def test_a_unit_whose_keys_all_fail_to_resolve_has_no_score():
    units, _ = g.extract_units("Something [C9].")
    assert g.score_units(units, {}, word_overlap_scorer)[0].best is None


# ---------------------------------------------------------------- distributions
def test_dist_nearest_rank_percentiles():
    d = g.dist([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert (d["n"], d["min"], d["max"], d["median"]) == (10, 0.1, 1.0, 0.55) and d["p10"] == 0.1 and d["p90"] == 0.9
    assert g.dist([]) == {"n": 0} and g.dist([0.4]) == {"n": 1, "min": 0.4, "p10": 0.4, "median": 0.4, "p90": 0.4, "max": 0.4}


def test_threshold_table_is_monotonic_and_handles_empty():
    rows = g.threshold_table([0.9, 0.8, 0.2], [0.1, 0.05, 0.6])
    pos = [r["positives_supported_pct"] for r in rows]
    assert pos == sorted(pos, reverse=True) and rows[0]["positives_supported_pct"] == 100.0 and rows[0]["threshold"] == 0.0
    assert next(r for r in rows if r["threshold"] == 0.5)["negatives_supported_pct"] == 33.3
    assert g.threshold_table([], [])[0]["positives_supported_pct"] is None


def test_negative_pool_excludes_cited_sections_and_is_sorted():
    assert g.negative_pool(CHUNKS, ["a"]) == ["b", "c", "d", "e"]          # d (other doc, same section name) stays: sections are per document
    assert g.negative_pool(CHUNKS, ["a", "d", "e"]) == ["b", "c"]
    assert g.negative_pool(CHUNKS, []) == sorted(CHUNKS)


def test_sanity_sentence_is_a_real_sentence_of_the_chunk_or_none():
    s = g.first_sentence(CHUNKS["a"]["display_text"])
    assert s and s in CHUNKS["a"]["display_text"]
    t = g.first_sentence(CHUNKS["b"]["display_text"])
    assert "deactivated" in t and "|" not in t
    assert g.first_sentence(CHUNKS["c"]["display_text"]) is None


# ---------------------------------------------------------------- calibration
TURNS = [{"id": "T1", "answer": "GOVERN is a cross-cutting function that is infused throughout the other functions [C1]. "
                                 "Protocols are put in place to ensure GAI systems are able to be deactivated [C2].",
          "key_map": {"C1": "a", "C2": "b"}}]


def test_calibration_positives_beat_negatives_for_a_sensible_scorer():
    cal = g.calibrate(TURNS, CHUNKS, word_overlap_scorer)
    assert cal["positives"]["n"] == 2 and cal["negatives"]["n"] == 6 and cal["negatives_best_of_3"]["n"] == 2
    assert cal["positives"]["min"] > cal["negatives"]["max"] and cal["overlap"]["separable"] is True
    assert cal["sanity"]["n"] == 2 and cal["sanity"]["min"] >= 0.99                # a verbatim sentence vs its own chunk
    assert cal["unresolved_keys"] == [] and len(cal["sentences"]) == 2


def test_calibration_negatives_never_come_from_the_cited_sections():
    cal = g.calibrate(TURNS, CHUNKS, word_overlap_scorer)
    for row in cal["sentences"]:
        assert not {n["chunk_id"] for n in row["negatives"]} & {"a", "b"}


def test_calibration_is_deterministic_and_seed_dependent():
    one = g.calibrate(TURNS, CHUNKS, word_overlap_scorer, seed=1)
    assert one == g.calibrate(TURNS, CHUNKS, word_overlap_scorer, seed=1)
    many = {tuple(n["chunk_id"] for r in g.calibrate(TURNS, CHUNKS, word_overlap_scorer, seed=s)["sentences"] for n in r["negatives"]) for s in range(8)}
    assert len(many) > 1


def test_calibration_reports_overlap_when_a_scorer_cannot_separate():
    flat = lambda sentence, texts: [0.5] * len(texts)                                # noqa: E731
    cal = g.calibrate(TURNS, CHUNKS, flat)
    assert cal["overlap"]["separable"] is False and cal["overlap"]["positives_at_or_below_negative_p90_pct"] == 100.0


def test_calibration_lists_unresolved_keys_and_skips_them():
    turns = [{"id": "T9", "answer": "Something about GOVERN [C1]. Something else [C5].", "key_map": {"C1": "a"}}]
    cal = g.calibrate(turns, CHUNKS, word_overlap_scorer)
    assert cal["unresolved_keys"] == ["T9:C5"] and cal["positives"]["n"] == 1


def test_calibration_with_no_turns_is_empty_not_a_crash():
    cal = g.calibrate([], CHUNKS, word_overlap_scorer)
    assert cal["positives"] == {"n": 0} and cal["overlap"] == {} and "n=0" in g.format_calibration(cal)


def test_format_calibration_prints_the_threshold_table():
    out = g.format_calibration(g.calibrate(TURNS, CHUNKS, word_overlap_scorer))
    assert "positives (own chunk)" in out and "threshold" in out and "0.95" in out


# ---------------------------------------------------------------- checks D1-D3
CTX = ck.Context(texts={}, chunks=CHUNKS, llm_calls={}, scorer=word_overlap_scorer, support_threshold=0.5)


def answered(text, key_map):
    return {"id": "T", "status": 200, "elapsed_ms": 1, "retried": False, "waited_s": 0, "body": {
        "route": "nist_question", "answer": text, "sources": [{"keys": sorted(key_map)}], "related_sections": [],
        "meta": {"outcome": "answered", "llm_calls": 2, "latency_ms": 1, "usage": {}, "debug": {"key_map": key_map}}}}


CASE = {"id": "T", "message": "q", "expected_route": "nist_question", "expected_outcome": "answered"}


def d_checks(text, key_map, ctx=CTX, case=CASE):
    return {c.id: c for c in ck.check_d(case, ck.view(answered(text, key_map)), ctx)}


def test_d_supported_sentence_passes_and_unsupported_fails():
    ok = d_checks("GOVERN is a cross-cutting function that is infused throughout the other functions [C1].", {"C1": "a"})
    assert [ok[k].status for k in ("D1", "D2", "D3")] == ["pass"] * 3 and ok["D1"].data["sentences"][0]["score"] >= 0.9
    bad = d_checks("Quantum computers break encryption quickly [C1].", {"C1": "a"})
    assert bad["D1"].status == "pass" and bad["D2"].status == "fail" and bad["D2"].data["unsupported"][0]["score"] < 0.5


def test_d3_counts_uncited_sentences_even_without_a_scorer():
    ctx = ck.Context(texts={}, chunks=CHUNKS, llm_calls={})
    r = d_checks("GOVERN is cross-cutting [C1]. This one has no citation.", {"C1": "a"}, ctx)
    assert r["D3"].status == "fail" and r["D3"].data["count"] == 1 and r["D1"].status == r["D2"].status == "na"


def test_d2_is_na_until_a_threshold_is_chosen():
    ctx = ck.Context(texts={}, chunks=CHUNKS, llm_calls={}, scorer=word_overlap_scorer, support_threshold=None)
    r = d_checks("GOVERN is a cross-cutting function [C1].", {"C1": "a"}, ctx)
    assert r["D1"].status == "pass" and r["D2"].status == "na" and "Q2" in r["D2"].detail


def test_d_is_na_for_non_answered_cases_and_missing_key_map():
    assert all(c.status == "na" for c in ck.check_d({**CASE, "expected_outcome": "insufficient"}, ck.view(answered("x", {})), CTX))
    r = d_checks("GOVERN is cross-cutting [C1].", {}, CTX)
    assert r["D1"].status == "na" and "key_map" in r["D1"].detail


def test_d_runs_inside_the_normal_check_pass():
    case = {**CASE, "must_contain": [{"fact": "g", "variants": ["govern"]}], "min_facts": 1, "expected_sections": [{"doc": "d1", "section": "S1"}]}
    got = {c.id: c.status for c in ck.run_turn_checks(case, answered("GOVERN is a cross-cutting function [C1].", {"C1": "a"}), CTX)}
    assert got["D1"] == got["D2"] == got["D3"] == "pass"


# ---------------------------------------------------------------- the real reranker (skipped without the cached model / data)
@pytest.mark.skipif(not (g.Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.jsonl").exists(), reason="needs data/processed")
def test_real_reranker_scores_a_supported_sentence_far_above_an_unrelated_chunk():
    import json
    root = g.Path(__file__).resolve().parents[1]
    chunks = {c["chunk_id"]: c for c in (json.loads(x) for x in (root / "data/processed/chunks.jsonl").read_text().splitlines())}
    own, other = chunks["ai_rmf_1_0:prose:0022:1"], chunks["genai_profile:genai_action_table:0034:1"]
    try:
        scorer = g.reranker_scorer()
    except Exception as exc:                                  # noqa: BLE001 - no cached model on this machine
        pytest.skip(f"reranker not available: {exc}")
    s_own, s_other = scorer("GOVERN is a cross-cutting function that is infused throughout AI risk management.", [own["display_text"], other["display_text"]])
    assert s_own > 0.5 > s_other and 0.0 <= s_other <= s_own <= 1.0
