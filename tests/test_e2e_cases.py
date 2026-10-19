"""Stage V1: the case-file validator. Offline; the last test checks the real case file against the real PDFs."""
import copy
from pathlib import Path

import pytest

from eval import e2e_cases as ec

PAGES = {"d": ["Intro page. Characteristics of trustworthy AI systems include: valid and reliable, safe, secure and resili- ent.",
               "Second page: GOVERN is a cross-cutting function. Nothing about fines."]}


def turn(**over) -> dict:
    t = {"id": "T1", "message": "q", "expected_route": "nist_question", "expected_outcome": "answered",
         "must_contain": [{"fact": "govern", "variants": ["govern"]}], "min_facts": 1, "must_not_contain": [],
         "expected_sections": [{"doc": "ai_rmf_1_0", "section": "5"}],
         "gold_locations": [{"fact": "g", "doc": "d", "pdf_page": 2, "printed": "1", "quote": "GOVERN is a cross-cutting function"}]}
    t.update(over)
    return t


def data(*turns) -> dict:
    return {"single_turn": list(turns), "conversations": []}


# ---- matching semantics
@pytest.mark.parametrize("text,needle,ok", [
    ("It is a cross-cutting function", "cross cutting", True),
    ("It is a cross‑cutting function", "cross-cutting", True),
    ("Cross Cutting", "cross-cutting", True),
    ("the system was deactivated", "deactivat", True),
    ("Management of risk", "manage", True),
    ("a train of thought", "rain", False),           # must start at a word boundary
    ("costs 5 $", "$", True),                        # symbol-only needle: plain substring
    ("costs five", "$", False),
    ("anything", "", False),
])
def test_contains(text, needle, ok):
    assert ec.contains(text, needle) is ok


def test_fact_found_uses_any_variant():
    assert ec.fact_found("valid & reliable", ["valid and reliable", "valid & reliable"])
    assert not ec.fact_found("nothing here", ["valid and reliable"])


# ---- schema
def test_valid_turn_has_no_schema_errors():
    assert ec.validate_schema(data(turn())).errors == []


@pytest.mark.parametrize("mutate,fragment", [
    (lambda t: t.pop("message"), "missing 'message'"),
    (lambda t: t.update(expected_route="bogus"), "unknown expected_route"),
    (lambda t: t.update(expected_outcome="bogus"), "unknown expected_outcome"),
    (lambda t: t.update(must_contain=[]), "needs must_contain"),
    (lambda t: t.update(expected_sections=[]), "needs expected_sections"),
    (lambda t: t.update(gold_locations=[]), "needs gold_locations"),
    (lambda t: t.update(min_facts=3), "min_facts"),
    (lambda t: t.update(must_contain=[{"fact": "x", "variants": []}]), "non-empty 'variants'"),
    (lambda t: t.update(expected_sections=[{"doc": "nope", "section": "5"}]), "section entries need doc"),
    (lambda t: t.update(expected_text_key="MADE_UP"), "unknown expected_text_key"),
])
def test_schema_negative(mutate, fragment):
    t = copy.deepcopy(turn())
    mutate(t)
    assert any(fragment in e for e in ec.validate_schema(data(t)).errors)


def test_unknown_insufficient_reason_is_rejected():
    ok = turn(expected_outcome="insufficient", expected_insufficient_reason_any=["gate_low_score", "model_case_d"])
    bad = turn(expected_outcome="insufficient", expected_insufficient_reason_any=["gate_low_score", "made_up"])
    assert not any("reason" in e for e in ec.validate_schema(data(ok)).errors)
    assert any("unknown insufficient reason" in e for e in ec.validate_schema(data(bad)).errors)


def test_duplicate_ids_and_fixed_text_outcome_without_key():
    rep = ec.validate_schema(data(turn(), turn()))
    assert any("duplicate id" in e for e in rep.errors)
    t = {"id": "G", "message": "Hi", "expected_route": "greeting", "expected_outcome": "greeting"}
    assert any("needs expected_text_key" in e for e in ec.validate_schema(data(t)).errors)


# ---- gold quotes against page text
def test_quote_found_even_when_the_pdf_breaks_a_word():
    d = data(turn(gold_locations=[{"fact": "f", "doc": "d", "pdf_page": 1, "printed": "1",
                                   "quote": "secure and resilient"}]))
    assert ec.verify_gold(d, PAGES).errors == []


def test_quote_with_ellipsis_must_match_in_order():
    ok = {"fact": "f", "doc": "d", "pdf_page": 1, "printed": "1", "quote": "Characteristics of trustworthy ... valid and reliable"}
    bad = {**ok, "quote": "valid and reliable ... Characteristics of trustworthy"}
    assert ec.verify_gold(data(turn(gold_locations=[ok])), PAGES).errors == []
    assert ec.verify_gold(data(turn(gold_locations=[bad])), PAGES).errors


def test_quote_on_the_wrong_page_is_reported_with_the_right_one():
    loc = {"fact": "f", "doc": "d", "pdf_page": 1, "printed": "1", "quote": "GOVERN is a cross-cutting function"}
    errs = ec.verify_gold(data(turn(gold_locations=[loc])), PAGES).errors
    assert errs and "pdf page(s) [2]" in errs[0]


def test_made_up_quote_and_bad_page_number_fail():
    fake = {"fact": "f", "doc": "d", "pdf_page": 2, "printed": "1", "quote": "GOVERN is the least important function"}
    assert "found on no page" in ec.verify_gold(data(turn(gold_locations=[fake])), PAGES).errors[0]
    out = {**fake, "pdf_page": 9}
    assert "bad doc/pdf_page" in ec.verify_gold(data(turn(gold_locations=[out])), PAGES).errors[0]


def test_absence_claims():
    absent = {"fact": "f", "doc": "d", "absent_terms": ["penalt", "EU AI Act"]}
    present = {"fact": "f", "doc": "d", "absent_terms": ["fines"]}
    assert ec.verify_gold(data(turn(gold_locations=[absent])), PAGES).errors == []
    assert "NOT absent" in ec.verify_gold(data(turn(gold_locations=[present])), PAGES).errors[0]


# ---- printed pages, traps, facts, sections
def test_printed_page_must_match_the_page_records():
    recs = [{"doc_id": "d", "pdf_page": 2, "printed_page": 1}]
    assert ec.verify_printed_pages(data(turn()), recs).errors == []
    assert ec.verify_printed_pages(data(turn()), [{"doc_id": "d", "pdf_page": 2, "printed_page": 9}]).errors


def test_trap_that_appears_in_the_gold_quote_is_an_error():
    bad = turn(must_not_contain=[{"fact": "t", "variants": ["cross-cutting"]}])
    assert ec.verify_traps_not_in_gold(data(bad)).errors
    assert ec.verify_traps_not_in_gold(data(turn(must_not_contain=[{"fact": "t", "variants": ["paris"]}]))).errors == []


def test_fact_without_any_variant_in_the_expected_document_is_unreachable():
    pages = {"ai_rmf_1_0": ["GOVERN is cross-cutting."]}
    ok = turn(must_contain=[{"fact": "f", "variants": ["cross cutting"]}])
    bad = turn(must_contain=[{"fact": "f", "variants": ["zebra"]}])
    assert ec.verify_facts_reachable(data(ok), pages).errors == []
    assert ec.verify_facts_reachable(data(bad), pages).errors


def test_section_matching_and_missing_section():
    chunks = [{"doc_id": "ai_rmf_1_0", "section_path": ["Part 2", "5 AI RMF Core", "5.1 Govern"], "category": "GOVERN 1",
               "subcategories": ["GOVERN 1.1"]}]
    for sec in ("5", "5.1", "GOVERN 1", "GOVERN 1.1", "5 AI RMF Core"):
        assert ec.section_matches(chunks[0], sec), sec
    assert not ec.section_matches(chunks[0], "5.2")
    assert not ec.section_matches(chunks[0], "1")           # '1' must not match '5.1 Govern'
    assert ec.verify_sections(data(turn()), chunks).errors == []
    missing = turn(expected_sections=[{"doc": "ai_rmf_1_0", "section": "9.9"}])
    assert ec.verify_sections(data(missing), chunks).errors
    trap = turn(must_not_cite=[{"doc": "ai_rmf_1_0", "section": "1.3"}])
    rep = ec.verify_sections(data(trap), chunks)
    assert rep.errors == [] and rep.warnings                 # a trap on a non-existent section only warns


def test_all_turns_flattens_conversations_in_order():
    d = {"single_turn": [{"id": "S1"}], "conversations": [{"id": "C1", "turns": [{"id": "C1-T1"}, {"id": "C1-T2"}]}]}
    assert [t["id"] for t in ec.all_turns(d)] == ["S1", "C1-T1", "C1-T2"]
    assert ec.all_turns(d)[1]["conversation"] == "C1"


def test_fixed_texts_resolve_from_the_app_config():
    consts = ec.text_constants()
    assert set(consts) == ec.TEXT_KEYS and consts["INSUFFICIENT_TEXT"].endswith("this information.")


# ---- the real case file
REAL_DATA = ec.ROOT / "data" / "processed" / "chunks.jsonl"


@pytest.mark.skipif(not (REAL_DATA.exists() and (ec.ROOT / "data" / "raw").exists()), reason="needs data/raw and data/processed")
def test_the_real_case_file_is_verified_against_the_real_pdfs():
    result = ec.run()
    assert result["turns"] == 18
    for name, rep in result["reports"].items():
        assert rep.errors == [], (name, rep.errors)
    assert result["reports"]["gold_locations"].checked["quotes"] >= 15


def test_case_file_does_not_copy_fixed_texts():
    raw = (ec.DEFAULT_CASES).read_text(encoding="utf-8")
    for text in ec.text_constants().values():
        assert text not in raw
