import pytest

from chunking.join import join_document
from chunking.segment import segment


def doc(pages):
    """pages: list of (markdown, headings, markers) with page-local offsets computed by the caller."""
    recs = []
    for i, (md, heads, marks) in enumerate(pages, start=1):
        recs.append({"doc_id": "d", "display_name": "D", "source_url": "u", "pdf_page": i, "printed_page": str(i),
                     "index_eligible": True, "markdown": md, "headings": heads, "table_markers": marks,
                     "section_path_start": []})
    return join_document(recs)


def head(md, text, sid, path):
    return {"section_id": sid, "title": text, "level": 1, "char_offset": md.index(text), "section_path": path}


def marker(md, text, id_, kind):
    return {"id": id_, "kind": kind, "char_offset": md.index(text)}


GEN_HDR = "|**Action ID**|**Suggested Action**|**GAI Risks**|\n|---|---|---|\n"


def coverage_ok(d, units):
    cov = [0] * len(d.text)
    for u in units:
        for i in range(u.start, u.end):
            cov[i] += 1
    return all(c == 1 or d.text[i].isspace() for i, c in enumerate(cov)) and all(
        c > 0 or d.text[i].isspace() for i, c in enumerate(cov))


def test_sections_and_prose_units():
    md = "# 1 A\n\ntext a\n\n# 2 B\n\ntext b"
    d = doc([(md, [head(md, "# 1 A", "1", ["1 A"]), head(md, "# 2 B", "2", ["2 B"])], [])])
    s = segment(d)
    assert [u.kind for u in s.units] == ["prose", "prose"]
    assert [u.section_path for u in s.units] == [["1 A"], ["2 B"]]
    assert coverage_ok(d, s.units)


def test_text_before_first_heading_is_a_document_level_section():
    md = "preface\n\n# 1 A\n\ntext"
    d = doc([(md, [head(md, "# 1 A", "1", ["1 A"])], [])])
    s = segment(d)
    assert s.units[0].section_path == [] and d.text[s.units[0].start:s.units[0].end] == "preface"


def test_genai_table_continues_across_page_separator_and_ends_before_next_marker():
    p1 = ("# 3 Actions\n\n**GOVERN 1.1:** Legal.\n\n" + GEN_HDR + "|GV-1.1-001|a|b|\n")
    p2 = ("|GV-1.1-002|c|d|\n|**AI Actor Tasks**|**: X**||\n\n**GOVERN 1.2:** Next.\n\n" + GEN_HDR + "|GV-1.2-001|e|f|")
    d = doc([(p1, [head(p1, "# 3", "3", ["3 Actions"])], [marker(p1, "**GOVERN 1.1:**", "GOVERN 1.1", "genai_subcategory")]),
             (p2, [], [marker(p2, "**GOVERN 1.2:**", "GOVERN 1.2", "genai_subcategory")])])
    s = segment(d)
    tables = [u for u in s.units if u.kind == "genai_table"]
    assert [t.marker_id for t in tables] == ["GOVERN 1.1", "GOVERN 1.2"]
    t1 = d.text[tables[0].start:tables[0].end]
    assert "GV-1.1-002" in t1 and "AI Actor Tasks" in t1 and "GOVERN 1.2" not in t1
    assert coverage_ok(d, s.units)


def test_genai_ai_actor_paragraph_belongs_to_the_table():
    md = "# 3 A\n\n**GOVERN 1.1:** L.\n\n" + GEN_HDR + "|GV-1.1-001|a|b|\n\n**AI Actor Tasks: Governance**\n\nTrailing prose."
    d = doc([(md, [head(md, "# 3", "3", ["3 A"])], [marker(md, "**GOVERN 1.1:**", "GOVERN 1.1", "genai_subcategory")])])
    s = segment(d)
    t = next(u for u in s.units if u.kind == "genai_table")
    assert d.text[t.start:t.end].endswith("**AI Actor Tasks: Governance**")
    assert s.units[-1].kind == "prose" and d.text[s.units[-1].start:s.units[-1].end] == "Trailing prose."


def test_table_never_crosses_a_heading():
    md = "# 3 A\n\n**GOVERN 1.1:** L.\n\n" + GEN_HDR + "|GV-1.1-001|a|b|\n\n# 4 B\n\n|x|y|"
    d = doc([(md, [head(md, "# 3", "3", ["3 A"]), head(md, "# 4", "4", ["4 B"])],
              [marker(md, "**GOVERN 1.1:**", "GOVERN 1.1", "genai_subcategory")])])
    s = segment(d)
    t = next(u for u in s.units if u.kind == "genai_table")
    assert "# 4 B" not in d.text[t.start:t.end]
    assert coverage_ok(d, s.units)


def test_rmf_table_runs_to_section_end_and_has_categories():
    md = ("# 5 Core\n\nintro\n\nTable 1: Cats.\n\n|**Categories**|**Subcategories**|\n|---|---|\n"
          "|**GOVERN 1:**|**GOVERN 1.1:** a|\n|text|b|\n|**GOVERN 2:**<br>x|**GOVERN 2.1:** c|\n\n# 6 Next\n\nend")
    d = doc([(md, [head(md, "# 5 Core", "5", ["5 Core"]), head(md, "# 6 Next", "6", ["6 Next"])],
              [marker(md, "Table 1:", "Table 1", "rmf_table")])])
    s = segment(d)
    t = next(u for u in s.units if u.kind == "rmf_table")
    cats = s.categories[t.sequence]
    assert [c.label for c in cats] == ["GOVERN 1", "GOVERN 2"]
    assert cats[0].start == t.start and cats[0].end == cats[1].start and cats[1].end == t.end
    assert "# 6 Next" not in d.text[t.start:t.end]
    assert coverage_ok(d, s.units)


def test_sequence_is_reading_order():
    md = "# 1 A\n\ntext\n\n# 2 B\n\nmore"
    d = doc([(md, [head(md, "# 1 A", "1", ["1 A"]), head(md, "# 2 B", "2", ["2 B"])], [])])
    assert [u.sequence for u in segment(d).units] == [0, 1]


def test_real_documents_segment_cleanly():
    from chunking import config
    if not config.PAGES_PATH.is_file():
        pytest.skip("pages.jsonl absent")
    from chunking.join import join_all, load_pages
    for d in join_all(load_pages()):
        s = segment(d)
        assert coverage_ok(d, s.units)
        for u in s.units:
            sec = s.sections[u.section_index]
            assert sec.start <= u.start and u.end <= sec.end
        if d.doc_id == "genai_profile":
            assert sum(u.kind == "genai_table" for u in s.units) == 49
        else:
            assert sum(len(c) for c in s.categories.values()) == 19
