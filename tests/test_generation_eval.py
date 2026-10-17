"""Helpers of the live generation evaluation: check evaluators, fixtures, pacer, reports (no network)."""
from __future__ import annotations

import asyncio

import yaml

from augmentation.config import INSUFFICIENT_CONTEXT as D
from eval.generation_checks import actual_case, run_check
from eval.generation_eval import CASES_PATH, Pacer, behaviour_report, fixture_result, usage_report
from tests.gen_fakes import KEY_MAP

GOOD = "GOVERN is a \"cross-cutting\" function [C2].\n- Map [C1]\n- Measure [C1]"


def ok(check, text=GOOD):
    return run_check(check, text, KEY_MAP)[0]


def test_check_types():
    assert ok({"type": "quote_exact", "value": "cross-cutting"}) and not ok({"type": "quote_exact", "value": "GOVERN is"})
    assert ok({"type": "bullets", "min": 2}) and not ok({"type": "bullets", "min": 3})
    assert ok({"type": "contains_all", "values": ["govern", "MAP"]}) and not ok({"type": "contains_all", "values": ["MANAGE"]})
    assert ok({"type": "contains_any", "values": ["x", "map"]}) and ok({"type": "not_contains", "values": ["pirate"]})
    assert ok({"type": "cites_key", "value": "C2"}) and not ok({"type": "cites_key", "value": "C3"})
    assert ok({"type": "no_pages"}) and not ok({"type": "no_pages"}, "See page 5 [C1].")
    assert not ok({"type": "no_pages"}, "On p. 12 [C1].")
    assert ok({"type": "exact_d"}, D) and not ok({"type": "exact_d"}, D + " More.")
    assert ok({"type": "only_english"}) and not ok({"type": "only_english"}, "GOVERN के चार [C1]")
    assert not ok({"type": "no_chunk_ids"}, "see genai_profile:genai_action_table:0034:1 [C1]")
    assert ok({"type": "max_words", "value": 20}) and not ok({"type": "max_words", "value": 3})
    assert ok({"type": "parser_ok"}) and not ok({"type": "parser_ok"}, "No citation here.")


def test_b_format_checks():
    b = "This question can be read in more than one way:\n- Reading one: a [C1]\n- Reading two: not covered by the provided documents."
    assert ok({"type": "ambiguous_format"}, b) and ok({"type": "has_not_covered"}, b) and not ok({"type": "ambiguous_format"})
    assert actual_case("ANSWERED", b, KEY_MAP) == "B" and actual_case("ANSWERED", GOOD, KEY_MAP) == "A"
    assert actual_case("INSUFFICIENT", D, KEY_MAP) == "D" and actual_case("ERROR", None, KEY_MAP) == "ERROR"


def test_fixture_sources_become_ordered_chunks():
    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    fx = next(c for c in cases if c["id"] == "b01-both-readings-covered")
    res = fixture_result(fx["context"]["sources"], fx["query"])
    from augmentation.context_builder import build_context
    block, km = build_context(res)
    assert list(km) == ["C1", "C2"] and 'doc="NIST AI 100-1 (AI RMF 1.0)"' in block and 'part="1 of 2"' in block


def test_pacer_waits_for_the_window_to_clear():
    t = [0.0]
    sleeps = []

    async def sleep(s):
        sleeps.append(s)
        t[0] += s
    p = Pacer(8000, clock=lambda: t[0], sleep=sleep)
    p.record(6000)
    waited = asyncio.run(p.wait_for(3000))
    assert waited > 0 and p.used() + 3000 <= 8000 * 0.9
    p2 = Pacer(8000, clock=lambda: t[0], sleep=sleep)
    assert asyncio.run(p2.wait_for(3000)) == 0              # empty window: no wait


def test_reports():
    base = {"expect_case": "A", "actual_case": "A", "outcome": "ANSWERED", "corrected": False, "fixes": [], "warnings": [],
            "raw_violations": [], "raw_units": 4, "raw_uncited_units": 1, "usage": {"prompt": 3000, "completion": 400,
            "reasoning": 300, "total": 3400}, "finish_reason": "stop", "ttft_ms": 900.0, "total_ms": 2000.0, "checks": []}
    rows = [{"id": "a", **base}, {"id": "b", **base, "expect_case": "D", "raw_violations": ["OVER_LENGTH"]}]
    u = usage_report(rows)
    assert u["total_tokens"]["max"] == 3400 and u["visible_answer_tokens"]["median"] == 100
    b = behaviour_report(rows)
    assert b["case_accuracy"]["correct"] == 1 and b["over_length_rate_pct"] == 50.0 and b["uncited_sentence_rate"]["uncited"] == 2
    assert b["case_mismatches"][0]["id"] == "b"


def test_quote_exact_allows_the_quote_to_continue_past_the_value():
    text = 'EO 14110 defines it as “the class of AI models that emulate input. This can include images.” [C1]'
    assert ok({"type": "quote_exact", "value": "the class of AI models that emulate input"}, text)
    assert not ok({"type": "quote_exact", "value": "the class of AI"}, "the class of AI models (unquoted) [C1]")
    assert not ok({"type": "quote_exact", "value": "emulate input"}, 'He said "other words" and emulate input [C1]')


def test_typographic_variants_fail_strict_but_pass_normalized():
    from eval.generation_checks import run_checks
    text = "Suggested action GV\u20111.7\u2011001 says so [C1]; see Figure\u202f5."
    res = run_checks([{"type": "contains_all", "values": ["GV-1.7-001", "Figure 5"]}], text, KEY_MAP)[0]
    assert res["passed"] is False and res["passed_normalized"] is True
