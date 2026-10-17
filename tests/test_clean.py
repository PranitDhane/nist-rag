"""Stage 7 tests: every rule with positive AND negative cases (inline fixtures, no PDFs),
plus checks on the real cleaned output (skipped without PDFs)."""
from __future__ import annotations

import json

import pytest

from ingest import config
from ingest.clean import (RULES, append_rotated_text, clean_page, collapse_whitespace, extract_footnotes, fix_line_hyphenation,
                          init_page, load_protect_list, normalize_unicode, strip_footnote_markers,
                          strip_running_headers_footers, strip_table_leftovers, unwrap_markup_tags)

PROTECT = frozenset({"third-party", "ai-generated"})


def page(md: str, strategy: str = "page_label", printed: str | None = None, **kw) -> dict:
    base = {"doc_id": "d", "pdf_page": 1, "markdown": md, "printed_page": printed, "body_font_size": 11.0,
            "marker_candidates": [], "bottom_blocks": []}
    return init_page({**base, **kw}, strategy)


# ---- 1 normalize_unicode
def test_normalize_unicode_ligature_and_spaces():
    p = normalize_unicode(page("ﬁnd it now​!"))
    assert p["markdown"] == "find it now!" and p["cleaning_log"] == ["normalize_unicode"]


def test_normalize_unicode_noop_not_logged():
    assert normalize_unicode(page("plain text"))["cleaning_log"] == []


# ---- 2 strip_running_headers_footers
@pytest.mark.parametrize("head", ["AI RMF 1.0 \n\nNIST AI 100-1 \n\n", "NIST AI 100-1\n\n**AI RMF 1.0**\n\n"])
def test_header_either_order_and_footer(head):
    p = strip_running_headers_footers(page(head + "# **Section**\n\nbody\n\nPage 23 \n"))
    assert "NIST AI 100-1" not in p["markdown"] and "AI RMF 1.0" not in p["markdown"]
    assert "Page 23" not in p["markdown"] and "body" in p["markdown"]


def test_header_text_in_body_is_kept():
    md = "# T\n\nintro\n\nsee NIST AI 100-1\n\nAI RMF 1.0\n\nPage 7 is referenced here\n\nend"
    assert strip_running_headers_footers(page(md))["markdown"] == md


def test_footer_bare_roman_equal_to_printed_page():
    p = strip_running_headers_footers(page("# **Table of Contents**\n\nrow\n\ni \n", printed="i"))
    assert p["markdown"].split() == ["#", "**Table", "of", "Contents**", "row"]


def test_genai_page_number_last_line_only_if_equal_to_printed():
    assert "6" not in strip_running_headers_footers(page("text\n\n6 \n", "first_line_offset", "6"))["markdown"].split()
    kept = page("text\n\n7 \n", "first_line_offset", "6")
    assert strip_running_headers_footers(kept)["markdown"] == kept["markdown"]
    first = strip_running_headers_footers(page("6\n\ntext", "first_line_offset", "6"))
    assert first["markdown"].strip() == "text"


def test_genai_no_printed_page_removes_nothing():
    p = page("12\n\ntext\n\n4", "first_line_offset", None)
    assert strip_running_headers_footers(p)["markdown"] == p["markdown"]


# ---- 3 extract_footnotes
def fn_page(md, blocks):
    return page(md, "first_line_offset", "6", bottom_blocks=blocks)


def block(num, text, size=6.6):
    return {"bbox": [72, 600, 500, 620], "first_span_text": num, "first_span_size": size, "text": f"{num}  {text}"}


def test_extract_footnotes_blockquotes_and_tags():
    md = ("Body text.\n\n> 9 Confabulations of falsehoods are common.\n\n"
          "> 10 For example, <u>shown to be pervasive</u> in LLMs.\n\n6\n")
    p = extract_footnotes(fn_page(md, [block("9", "Confabulations of falsehoods are common."),
                                      block("10", "For example, shown to be pervasive in LLMs.")]))
    assert [f["number"] for f in p["footnotes"]] == ["9", "10"]
    assert "Confabulations" not in p["markdown"] and ">" not in p["markdown"] and "Body text." in p["markdown"]
    assert p["warnings"] == []


def test_extract_footnotes_two_in_one_paragraph():
    md = "Body.\n\n3 First note text. 4 Second note text.\n"
    p = extract_footnotes(fn_page(md, [block("3", "First note text."), block("4", "Second note text.")]))
    assert len(p["footnotes"]) == 2 and p["markdown"].strip() == "Body."


def test_extract_footnotes_not_found_keeps_text_and_warns():
    md = "Body.\n\n> 5 Text that differs in Markdown.\n"
    p = extract_footnotes(fn_page(md, [block("5", "Totally different raw text.")]))
    assert p["footnotes"] == [] and "Text that differs" in p["markdown"]
    assert p["warnings"] and "footnote 5" in p["warnings"][0]


def test_extract_footnotes_ignores_page_number_and_big_digits():
    md = "Body.\n\n6\n"
    p = extract_footnotes(fn_page(md, [block("6", "", size=10.0)]))   # a 10pt '6' is the page number
    assert p["footnotes"] == [] and p["markdown"] == md


# ---- 4 strip_footnote_markers
def test_markers_removed_and_cross_check_ok():
    p = page("contexts.<sup>9,10</sup> More<sup>3</sup> text", marker_candidates=[
        {"text": "9,10", "bbox": [1, 1, 2, 2], "size": 7}, {"text": "3", "bbox": [3, 3, 4, 4], "size": 7}])
    out = strip_footnote_markers(p)
    assert out["markdown"] == "contexts. More text" and out["warnings"] == []


def test_markers_keep_text_sup_and_untouched_numbers():
    md = "ISO 31000, Act of 2020, GPT-4 and 2.12 <sup>Protocols are put in place</sup>"
    out = strip_footnote_markers(page(md))
    assert out["markdown"] == md and out["cleaning_log"] == []


def test_marker_count_mismatch_warns():
    assert strip_footnote_markers(page("a<sup>1</sup>"))["warnings"]


def test_marker_cross_check_excludes_footnote_numbers_at_page_bottom():
    p = page("a<sup>1</sup>", bottom_blocks=[block("1", "note")], marker_candidates=[
        {"text": "1", "bbox": [10, 10, 12, 12], "size": 7}, {"text": "1", "bbox": [72, 601, 75, 610], "size": 6.6}])
    p["bottom_blocks"][0]["bbox"] = [72, 598, 500, 620]
    assert strip_footnote_markers(p)["warnings"] == []


# ---- 5 unwrap_markup_tags
def test_unwrap_keeps_inner_text_and_br():
    md = "<sup>Protocols are put in place</sup> and <u>link</u>|a<br>b|"
    assert unwrap_markup_tags(page(md))["markdown"] == "Protocols are put in place and link|a<br>b|"


# ---- 6 fix_line_hyphenation
@pytest.mark.parametrize("src,expected", [
    ("or-<br>ganizational", "organizational"),
    ("or-\nganizational", "organizational"),
    ("GV-\n1.1-001", "GV-1.1-001"),
    ("GV-<br>1.1-001", "GV-1.1-001"),
    ("human-<br>AI", "human-AI"),
    ("third-<br>party vendors", "third-party vendors"),
    ("AI-<br>generated", "AI-generated"),
    ("ISO 31000, Act of 2020, GPT-4, 2.12 and state-of-the-art", "ISO 31000, Act of 2020, GPT-4, 2.12 and state-of-the-art"),
    ("dash - <br>space", "dash - <br>space"),
])
def test_hyphenation(src, expected):
    assert fix_line_hyphenation(page(src), PROTECT)["markdown"] == expected


def test_real_protect_list_has_approved_words():
    pl = load_protect_list()
    assert len(pl) == 10 and {"third-party", "red-teaming", "pre-trained"} <= pl


# ---- 7 strip_table_leftovers
def test_table_leftovers():
    md = ("Table 1: Categories and subcategories for the **GOVERN** function. (Continued) \n\n"
          "|**Categories**|**Subcategories**|\n|---|---|\n|a|b|\n\nContinued on next page \n\n"
          "Categories Subcategories\n")
    out = strip_table_leftovers(page(md))["markdown"]
    assert "Continued" not in out and out.count("Categories") == 1 and "|**Categories**|**Subcategories**|" in out


def test_table_caption_without_continued_kept():
    md = "Table 1: Categories and subcategories for the **GOVERN** function."
    assert strip_table_leftovers(page(md))["markdown"] == md


@pytest.mark.parametrize("src,expected", [
    ("|**AI Actor Tasks   i**|**: AI Deployment**||", "|**AI Actor Tasks**|**: AI Deployment**||"),
    ("|**AI Actor Tasks        f   i**<br>**x**|**: AI**||", "|**AI Actor Tasks**<br>**x**|**: AI**||"),
    ("**AI Actor Tasks    i** AI Design", "**AI Actor Tasks:** AI Design"),
    ("|**AI Actor Tasks**|**: Governance**||", "|**AI Actor Tasks**|**: Governance**||"),
    ("Descriptions of AI Actor Tasks from Figures 2 and 3", "Descriptions of AI Actor Tasks from Figures 2 and 3"),
    ("|MP-3.4-004   i      i|Delineate tests.|Human-AI Configuration|", "|MP-3.4-004|Delineate tests.|Human-AI Configuration|"),
    ("|MS-1.1-001        ii|Employ methods.|x|", "|MS-1.1-001|Employ methods.|x|"),
    ("|MS-2.3-002|Evaluate claims, see MS-2.3-002 i here|x|", "|MS-2.3-002|Evaluate claims, see MS-2.3-002 i here|x|"),
])
def test_garbled_actor_tasks(src, expected):
    assert strip_table_leftovers(page(src))["markdown"] == expected


# ---- 8 collapse_whitespace
def test_collapse_whitespace():
    assert collapse_whitespace(page("\n\na  \n\n\n\n\nb \n\n"))["markdown"] == "a\n\nb"


# ---- 9 append_rotated_text (added after the plan's eight rules)
def test_rotated_text_appended_when_markdown_lacks_it():
    p = append_rotated_text(page("", rotated_text=["Fig. 3. AI actors  across\nstages."]))
    assert p["markdown"] == "Fig. 3. AI actors across stages." and p["cleaning_log"] == ["append_rotated_text"]
    q = append_rotated_text(page("Body.", rotated_text=["Caption text here."]))
    assert q["markdown"] == "Body.\n\nCaption text here."


def test_rotated_text_not_duplicated_and_missing_field_ok():
    p = page("Some **Caption text** here, already present.", rotated_text=["Caption text here, already present."])
    assert append_rotated_text(p)["cleaning_log"] == []
    assert append_rotated_text(page("x"))["markdown"] == "x"


# ---- pipeline
def test_rule_order_is_the_plan_order_plus_rotated_text_last():
    assert [r.__name__ for r in RULES] == [
        "normalize_unicode", "strip_running_headers_footers", "extract_footnotes", "strip_footnote_markers",
        "unwrap_markup_tags", "fix_line_hyphenation", "strip_table_leftovers", "collapse_whitespace",
        "append_rotated_text"]


def test_clean_page_logs_only_rules_that_changed_something():
    out = clean_page(page("AI RMF 1.0\n\nNIST AI 100-1\n\ntext\n\nPage 3\n"))
    assert out["cleaning_log"] == ["strip_running_headers_footers", "collapse_whitespace"] and out["markdown"] == "text"


# ---- real cleaned output
@pytest.fixture(scope="module")
def cleaned(need_pdfs):
    from ingest.classify import classify_all
    from ingest.clean import clean_all
    from ingest.config import load_manifest
    from ingest.extract import load_extract
    return clean_all(load_manifest(), classify_all(load_extract()))


def _get(cleaned, doc, pg):
    return next(r for r in cleaned if r["doc_id"] == doc and r["pdf_page"] == pg)


def test_real_ai_rmf_no_footnote_rule_hits_and_no_header_footer(cleaned):
    for r in (x for x in cleaned if x["doc_id"] == "ai_rmf_1_0"):
        assert "extract_footnotes" not in r["cleaning_log"] and "strip_footnote_markers" not in r["cleaning_log"]
        assert r["footnotes"] == []
        lines = [l.strip() for l in r["markdown"].splitlines() if l.strip()]
        assert not [l for l in lines[:3] if l.strip("#* ") in {"NIST AI 100-1", "AI RMF 1.0"}], r["pdf_page"]
        assert not lines or not lines[-1].startswith("Page ")


def test_real_genai_no_leftover_page_number_line(cleaned):
    for r in (x for x in cleaned if x["doc_id"] == "genai_profile" and x["printed_page"]):
        lines = [l.strip() for l in r["markdown"].splitlines() if l.strip()]
        assert lines[-1] != r["printed_page"] and lines[0] != r["printed_page"], r["pdf_page"]


def test_real_genai_p10_footnotes(cleaned):
    p = _get(cleaned, "genai_profile", 10)
    assert [f["number"] for f in p["footnotes"]] == ["9", "10"]
    assert "<sup>" not in p["markdown"] and "Confabulations of falsehoods are most" not in p["markdown"]
    assert "contexts. Confabulations" in p["markdown"]


def test_real_false_superscript_cell_survives(cleaned):
    p = _get(cleaned, "genai_profile", 21)
    assert "<sup>" not in p["markdown"] and "Protocols are put in place" in p["markdown"]


def test_real_ai_rmf_p28_table_leftovers_gone(cleaned):
    md = _get(cleaned, "ai_rmf_1_0", 28)["markdown"]
    assert "Continued on next page" not in md and "(Continued)" not in md and "|**Categories**" in md


def test_real_no_unexpected_tags_in_index_eligible_pages(cleaned):
    import re
    for r in (x for x in cleaned if x["index_eligible"]):
        tags = set(re.findall(r"</?(\w+)", r["markdown"])) - {"br"}
        assert not tags, (r["doc_id"], r["pdf_page"], tags)
