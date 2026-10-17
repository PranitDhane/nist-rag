"""Stage 10 validator tests: each hard check passes on a clean synthetic corpus and fails when broken."""
from __future__ import annotations

import copy

import pytest

from ingest.models import DocumentSpec, Manifest, PageTypeRange, Validation
from ingest.validate import CHECKS, validate


def spec() -> DocumentSpec:
    v = Validation(page_types=[PageTypeRange(1, 1, "cover"), PageTypeRange(2, 2, "body"), PageTypeRange(3, 3, "references")],
                   section_ids=["1", "2"], printed_page_spot_checks={2: "7"}, expected_figures=1, expected_footnotes=0,
                   subcategory_tables=1, unique_action_ids=2, forbidden_lines=["RUNNING HEADER"],
                   forbidden_line_patterns=["^Page [0-9]+$"])
    return DocumentSpec("d", "D", "https://x.org/d.pdf", "d.pdf", "", 3, "T", "page_label", v)


def page(n, ptype, **kw) -> dict:
    base = {"doc_id": "d", "pdf_page": n, "printed_page": None, "page_type": ptype, "index_eligible": ptype == "body",
            "headings": [], "markdown": "some text", "footnotes": [], "figures": [], "table_markers": [],
            "entities": {"action_ids": []}, "links": [], "cleaning_log": []}
    return {**base, **kw}


@pytest.fixture
def good():
    pages = [page(1, "cover"),
             page(2, "body", printed_page="7", headings=[{"section_id": "1", "title": "A", "level": 1},
                                                          {"section_id": "2", "title": "B", "level": 1}],
                  table_markers=["GOVERN 1.1"], entities={"action_ids": ["GV-1.1-001", "GV-1.1-002"]},
                  figures=[{"figure_number": 1, "caption": "Fig. 1.", "text": None, "method": "caption_only",
                            "confidence": None, "ocr_status": "skipped_unavailable"}],
                  links=[{"text": "a", "url": "https://x.org", "location": "body", "section_path": []}]),
             page(3, "references")]
    return pages, Manifest([spec()])


def failing(pages, manifest) -> set[str]:
    return {c.name.split()[0] for c in validate(pages, manifest) if not c.passed}


def test_clean_corpus_passes_all_eleven_checks(good):
    pages, manifest = good
    results = validate(pages, manifest)
    assert len(results) == len(CHECKS) == 11 and all(c.passed for c in results), [c for c in results if not c.passed]


def mutate(good, fn):
    pages, manifest = copy.deepcopy(good[0]), good[1]
    fn(pages)
    return failing(pages, manifest)


def test_1_missing_page_fails(good):
    assert "1" in mutate(good, lambda p: p.pop())


def test_2_wrong_printed_page_fails(good):
    assert mutate(good, lambda p: p[1].update(printed_page="8")) == {"2"}


def test_3_wrong_page_type_fails(good):
    assert "3" in mutate(good, lambda p: p[1].update(page_type="appendix"))


@pytest.mark.parametrize("markdown", ["intro\n\nRUNNING HEADER\n\nbody", "body\n\nPage 12", "body\n\n7"])
def test_4_leftover_header_footer_or_page_number_fails(good, markdown):
    assert mutate(good, lambda p: p[1].update(markdown=markdown)) == {"4"}


def test_4_header_text_inside_a_sentence_is_fine(good):
    assert mutate(good, lambda p: p[1].update(markdown="see RUNNING HEADER in the text, and Page 3 too")) == set()


def test_5_missing_extra_and_duplicate_sections_fail(good):
    assert mutate(good, lambda p: p[1].update(headings=[p[1]["headings"][0]])) == {"5"}
    assert mutate(good, lambda p: p[1]["headings"].append({"section_id": "9", "title": "X", "level": 1})) == {"5"}
    assert mutate(good, lambda p: p[1]["headings"].append(p[1]["headings"][0])) == {"5"}


def test_6_table_and_action_id_counts_fail(good):
    assert mutate(good, lambda p: p[1].update(table_markers=[])) == {"6"}
    assert mutate(good, lambda p: p[1].update(entities={"action_ids": ["GV-1.1-001"]})) == {"6"}
    assert mutate(good, lambda p: p[1].update(entities={"action_ids": ["GV-1.1-001", "GV-1.1-001", "GV-1.1-002"]})) == {"6"}


def test_7_figure_count_and_caption_fail(good):
    assert mutate(good, lambda p: p[1].update(figures=[])) == {"7"}
    assert mutate(good, lambda p: p[1]["figures"][0].update(caption=None)) == {"7"}


def test_8_empty_body_page_fails_but_empty_cover_does_not(good):
    assert mutate(good, lambda p: p[1].update(markdown="  \n")) == {"8"}
    assert mutate(good, lambda p: p[0].update(markdown="")) == set()


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///etc/passwd", "ftp://x.org", "#page=3"])
def test_9_bad_link_scheme_fails(good, url):
    assert mutate(good, lambda p: p[1]["links"].append({"text": "x", "url": url, "location": "body", "section_path": []})) == {"9"}


def test_9_bad_footnote_link_fails(good):
    assert mutate(good, lambda p: p[1].update(footnotes=[{"number": "1", "text": "t", "links": [{"text": "x", "url": "data:x"}]}])) == {"9", "11"}


def test_10_references_page_eligible_fails(good):
    assert mutate(good, lambda p: p[2].update(index_eligible=True)) == {"10"}


def test_11_footnotes_or_marker_removals_where_none_expected_fail(good):
    assert mutate(good, lambda p: p[1].update(footnotes=[{"number": "1", "text": "t", "links": []}])) == {"11"}
    assert mutate(good, lambda p: p[1].update(cleaning_log=["strip_footnote_markers"])) == {"11"}
