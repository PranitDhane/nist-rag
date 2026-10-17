import pytest

from chunking.join import join_document


def rec(page, md, printed=None, eligible=True, headings=(), markers=()):
    return {"doc_id": "d", "display_name": "D", "source_url": "u", "pdf_page": page,
            "printed_page": printed or str(page), "index_eligible": eligible, "markdown": md,
            "headings": list(headings), "table_markers": list(markers), "section_path_start": []}


def worked_example():
    """CHUNKING_PLAN Section 5: pages 22/23/24 of 2,400 / 2,100 / 1,800 characters."""
    # With the real "\\n\\n" separator the numbers shift by 2 per boundary; the plan's table
    # ignores separators, so check the mechanism with explicit separator-aware offsets.
    return join_document([rec(22, "a" * 2400), rec(23, "b" * 2100), rec(24, "c" * 1800)])


def test_span_inside_one_page():
    d = join_document([rec(1, "x" * 100), rec(2, "y" * 100)])
    a, b = d.pages_for_span(10, 50)
    assert a.pdf_page == b.pdf_page == 1


def test_span_crossing_one_boundary():
    d = join_document([rec(1, "x" * 100), rec(2, "y" * 100)])
    a, b = d.pages_for_span(90, 120)   # page 2 starts at 102
    assert (a.pdf_page, b.pdf_page) == (1, 2)


def test_span_crossing_two_boundaries():
    d = join_document([rec(1, "x" * 100), rec(2, "y" * 100), rec(3, "z" * 100)])
    a, b = d.pages_for_span(50, 250)
    assert (a.pdf_page, b.pdf_page) == (1, 3)


def test_span_starting_inside_separator_belongs_to_previous_page():
    d = join_document([rec(1, "x" * 100), rec(2, "y" * 100)])
    assert d.pages[0].end == 100 and d.pages[1].start == 102
    a, b = d.pages_for_span(101, 150)
    assert (a.pdf_page, b.pdf_page) == (1, 2)


def test_end_is_exclusive():
    d = join_document([rec(1, "x" * 100), rec(2, "y" * 100)])
    a, b = d.pages_for_span(0, 100)    # last char is offset 99, on page 1
    assert (a.pdf_page, b.pdf_page) == (1, 1)
    a, b = d.pages_for_span(0, 103)    # last char is offset 102, first char of page 2
    assert b.pdf_page == 2


def test_plan_worked_example_without_separators():
    """Same arithmetic as plan Section 5, using the page starts the join really produces."""
    d = worked_example()
    p22, p23, p24 = d.pages
    assert (p22.start, p22.end) == (0, 2400)
    assert p23.start == 2402 and p24.start == 4504
    a, b = d.pages_for_span(p23.start - 250, p23.start + 550)   # starts 2,152; ends 2,952
    assert (a.pdf_page, b.pdf_page) == (22, 23)
    assert d.pages_for_span(p24.start, p24.end)[1].pdf_page == 24


def test_only_index_eligible_pages_are_joined():
    d = join_document([rec(1, "cover", eligible=False), rec(2, "body")])
    assert d.text == "body" and [p.pdf_page for p in d.pages] == [2]


def test_heading_and_marker_offsets_become_absolute():
    d = join_document([
        rec(1, "intro\n\n# 1 Intro"),
        rec(2, "text\n\n## 1.1 Sub\n\nTable 1: t",
            headings=[{"section_id": "1.1", "title": "Sub", "level": 2, "char_offset": 6, "section_path": ["1 Intro", "1.1 Sub"]}],
            markers=[{"id": "Table 1", "kind": "rmf_table", "char_offset": 18}])])
    h = d.headings[0]
    assert d.text[h.start:].startswith("## 1.1 Sub")
    assert d.text[d.markers[0].start:].startswith("Table 1:")
    assert d.path_at(h.start + 3) == ["1 Intro", "1.1 Sub"]


def test_empty_document_raises():
    with pytest.raises(ValueError):
        join_document([rec(1, "x", eligible=False)])
