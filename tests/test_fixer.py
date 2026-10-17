"""Stage N2: apply_fixes() - one test per row of the violation policy (GENERATION_PLAN Section 5)."""
from __future__ import annotations

import re

import pytest

from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation.fixer import apply_fixes
from generation.models import ANSWERED, INSUFFICIENT
from tests.gen_fakes import KEY_MAP

GOOD = "GOVERN is the cross-cutting function [C2]. The Core has four functions [C1]."


def fx(text):
    return apply_fixes(text, KEY_MAP)


def test_correct_answer_passes_unchanged():
    r = fx(GOOD)
    assert (r.final_text, r.outcome, r.fixes, r.warnings) == (GOOD, ANSWERED, [], [])
    assert r.parsed.cited_keys == ["C2", "C1"]


def test_exact_case_d_needs_no_fix():
    r = fx(D)
    assert (r.final_text, r.outcome, r.fixes) == (D, INSUFFICIENT, [])


def test_case_d_without_final_period_is_still_insufficient_and_unchanged():
    r = fx(D.rstrip("."))
    assert r.outcome == INSUFFICIENT and r.fixes == [] and r.final_text == D.rstrip(".")


def test_d_with_extra_text_replaced_by_the_sentence():
    r = fx(f"GOVERN is cross-cutting [C2]. {D}")
    assert (r.final_text, r.outcome, r.fixes) == (D, INSUFFICIENT, ["D_WITH_EXTRA_TEXT"])


@pytest.mark.parametrize("raw", ["", "   \n "])
def test_empty_becomes_case_d(raw):
    r = fx(raw)
    assert (r.final_text, r.outcome, r.fixes) == (D, INSUFFICIENT, ["EMPTY"])


def test_malformed_citations_normalized():
    r = fx("GOVERN is cross-cutting [C2, C1]. The Core has four functions (C1). Also [C1,C3].")
    assert r.final_text == "GOVERN is cross-cutting [C2][C1]. The Core has four functions [C1]. Also [C1][C3]."
    assert r.fixes == ["MALFORMED_CITATION"] and r.outcome == ANSWERED and r.warnings == []


def test_unrecognised_malformed_form_left_untouched_with_warning():
    r = fx("GOVERN is cross-cutting [C1 and C2]. Four functions [C1].")
    assert "[C1 and C2]" in r.final_text and "MALFORMED_CITATION_UNFIXED" in r.warnings


def test_unknown_citation_removed():
    r = fx("GOVERN is cross-cutting [C2][C9]. Four functions [C1].")
    assert r.final_text == "GOVERN is cross-cutting [C2]. Four functions [C1]." and r.fixes == ["UNKNOWN_CITATION"]


def test_all_citations_invented_becomes_case_d_after_fixes():
    r = fx("GOVERN is cross-cutting [C8]. Four functions [C9].")
    assert (r.final_text, r.outcome) == (D, INSUFFICIENT) and r.fixes == ["UNKNOWN_CITATION", "NO_CITATIONS"]


def test_no_citations_at_all_is_case_d():
    r = fx("GOVERN is cross-cutting.")
    assert (r.final_text, r.outcome, r.fixes) == (D, INSUFFICIENT, ["NO_CITATIONS"])


def test_sources_list_removed():
    r = fx(GOOD + "\n\nSources:\n- C1: NIST AI 100-1\n- C2: https://x.example/a")
    assert r.final_text == GOOD and r.fixes == ["HAS_SOURCES_LIST"]


def test_url_removed_without_leaving_empty_brackets():
    r = fx("GOVERN is cross-cutting (https://nvlpubs.nist.gov/x.pdf) [C2]. Four functions [C1].")
    assert "http" not in r.final_text and "()" not in r.final_text and r.fixes == ["HAS_URL"]
    assert r.final_text == "GOVERN is cross-cutting [C2]. Four functions [C1]."


def test_markdown_link_keeps_its_text():
    r = fx("See [the AI RMF](https://nist.gov/x) for GOVERN [C2].")
    assert r.final_text == "See the AI RMF for GOVERN [C2]." and r.fixes == ["HAS_URL"]


def test_heading_and_code_fence_markers_removed_text_kept():
    r = fx("## Answer\nGOVERN is cross-cutting [C2].\n```\nfour functions [C1]\n```")
    assert r.final_text == "Answer\nGOVERN is cross-cutting [C2].\nfour functions [C1]" and r.fixes == ["HAS_HEADING_OR_CODE"]


def test_uncited_sentence_kept_with_warning():
    r = fx("GOVERN is cross-cutting [C2]. It matters a lot.")
    assert r.outcome == ANSWERED and r.final_text.endswith("It matters a lot.") and r.warnings == ["UNCITED_SENTENCE"]


def test_over_length_kept_with_warning():
    text = " ".join(["word"] * 160) + " [C1]."
    r = fx(text)
    assert r.outcome == ANSWERED and "OVER_LENGTH" in r.warnings and r.final_text == text


def test_fix_order_several_violations_at_once():
    raw = "# Answer\nGOVERN is cross-cutting [C2, C9] https://x.org.\n\nReferences:\n[C1]"
    r = fx(raw)
    assert r.outcome == ANSWERED and r.fixes == ["MALFORMED_CITATION", "UNKNOWN_CITATION", "HAS_SOURCES_LIST", "HAS_URL",
                                                 "HAS_HEADING_OR_CODE"]
    # the heading text ("Answer") survives as a plain line with no citation: kept, reported as a warning (GN-D6)
    assert r.parsed.cited_keys == ["C2"] and r.parsed.violations == ["UNCITED_SENTENCE"]


@pytest.mark.parametrize("raw", [
    "GOVERN is cross-cutting [C2, C1]. Four [C9].", GOOD + "\nSources:\n- C1", "## H\nGOVERN [C2] https://a.b/c now.",
    "GOVERN [C2]. (see https://a.b)", "```\nx [C1]\n```", "Text [C1]. Sources: here"])
def test_fixer_never_adds_words(raw):
    r = fx(raw)
    words = lambda t: set(re.findall(r"[A-Za-z0-9']+", t))  # noqa: E731
    assert words(r.final_text) <= words(raw) | words(D)
