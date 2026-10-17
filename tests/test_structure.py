"""Stage 8 tests: heading regexes, levels, section paths, table markers, entities (inline fixtures),
plus the 'Done when' counts on the real cleaned output (skipped without PDFs)."""
from __future__ import annotations

import collections

import pytest

from ingest.structure import (annotate_document, continues_table, detect_headings, extract_entities,
                              parse_heading, path_at, table_markers)


# ---- heading regexes and levels
@pytest.mark.parametrize("text,expected", [
    ("Executive Summary", ("Executive Summary", "Executive Summary", 1, "exec")),
    ("Part 1: Foundational Information", ("Part 1", "Foundational Information", 0, "part")),
    ("1. Framing Risk", ("1", "Framing Risk", 1, "number")),
    ("2.2. Confabulation", ("2.2", "Confabulation", 2, "number")),
    ("1.2.3 Risk Prioritization", ("1.2.3", "Risk Prioritization", 3, "number")),
    ("Appendix A: Descriptions of AI Actor Tasks", ("Appendix A", "Descriptions of AI Actor Tasks", 1, "appendix")),
    ("Appendix B. References", ("Appendix B", "References", 1, "appendix")),
    ("A.1. Governance", ("A.1", "Governance", 2, "sub")),
    ("A.1.1. Overview", ("A.1.1", "Overview", 3, "sub")),
])
def test_parse_heading(text, expected):
    assert parse_heading(text) == expected


@pytest.mark.parametrize("text", ["Overview", "AI Actor Tasks: Governance", "2023 report on AI", "Fig. 1. Harms", "1) Technical risks"])
def test_parse_heading_rejects(text):
    assert parse_heading(text) is None


def test_two_line_appendix_title():
    heads = detect_headings("# **Appendix A:**\n\nDescriptions of Tasks\n\ntext")
    assert [(h["section_id"], h["title"]) for h in heads] == [("Appendix A", "Descriptions of Tasks")]


def test_numbered_list_items_and_table_cells_are_not_headings():
    md = "1. Be risk-based and use-case driven.\n\n4. Presenting AI system outputs\n\n|2.1 Cell text|x|\n\n- 3. item"
    assert detect_headings(md) == []


def test_heading_from_markdown_heading_or_bold_only_line():
    md = "## **5.1 Govern**\n\n**2.4 Bold only heading**\n\nplain 2.5 Not heading\n\n**bold with 3.1 inside** text"
    assert [h["section_id"] for h in detect_headings(md)] == ["5.1", "2.4"]


# ---- section path across pages
def pages(*mds):
    return [{"doc_id": "d", "pdf_page": i + 1, "markdown": m} for i, m in enumerate(mds)]


def test_section_path_start_carried_across_pages():
    out = annotate_document(pages("# **Part 2: Core**\n\n## **5. AI RMF Core**\n\n# **5.1 Govern**\n\ntext",
                                  "more text", "# **5.2 Map**\n\ntext", "# **6. Profiles**\n\nx"))
    assert out[0]["section_path_start"] == []
    assert out[1]["section_path_start"] == ["Part 2", "5 AI RMF Core", "5.1 Govern"]
    assert out[2]["section_path_start"] == ["Part 2", "5 AI RMF Core", "5.1 Govern"]
    assert out[3]["section_path_start"] == ["Part 2", "5 AI RMF Core", "5.2 Map"]
    assert out[3]["section_path_end"] == ["Part 2", "6 Profiles"]


def test_appendix_resets_path_and_subsections_nest():
    out = annotate_document(pages("# **Part 2: Core**\n\n# **5. Core**", "# **Appendix A. Primary**\n\n# **A.1. Gov**\n\n# **A.1.1. Overview**"))
    assert out[1]["section_path_end"] == ["Appendix A: Primary", "A.1 Gov", "A.1.1 Overview"]


def test_path_at_offset():
    md = "intro\n\n# **2. Audience**\n\ntext here"
    page = annotate_document(pages(md))[0]
    assert path_at(page, 0) == []                                   # before any heading: path at page start
    assert path_at(page, md.index("text here")) == ["2 Audience"]
    nested = annotate_document(pages("# **1. Framing**\n\n" + md))[0]
    assert path_at(nested, 0) == ["1 Framing"]


# ---- tables
def test_table_markers_genai_require_action_id_table_next():
    md = ("**GOVERN 1.7:** Processes and procedures.\n\n|**Action ID**|**Suggested Action**|**GAI Risks**|\n|---|---|---|\n"
          "|GV-1.7-001|x|y|\n\n**MAP 1.3:** a row rendered as a paragraph.\n\n|a|b|")
    assert table_markers(md) == ["GOVERN 1.7"]


def test_table_marker_when_header_rendered_as_one_cell_table():
    md = "|**MEASURE 3.2:**Risk tracking approaches.\n|---|\n\n|**Action ID**|**Suggested Action**|**GAI Risks**|\n|---|---|---|"
    assert table_markers(md) == ["MEASURE 3.2"]


def test_table_markers_ai_rmf_captions():
    md = "Table 1: Categories and subcategories for the **GOVERN** function.\n\nTable 1: Categories. (Continued)\n"
    assert table_markers(md) == ["Table 1"]


@pytest.mark.parametrize("md,expected", [
    ("|a|b|\n|---|---|\n|x|y|", True),
    ("\n\n|GV-1.2-001|text|risk|", True),
    ("Table 1: Caption\n\n|a|b|", False),
    ("**GOVERN 1.7:** text\n\n|a|b|", False),
    ("plain paragraph\n\n|a|b|", False),
])
def test_continues_table(md, expected):
    assert continues_table(md) is expected


# ---- entities
def test_action_ids_only_from_first_cell_of_rows():
    md = ("e.g., GV-1.1-002 corresponds to the second action\n\n|GV-1.1-001|Align; see MS-2.1-003 here|Data Privacy|\n"
          "|MS-2.1-003|x|y|\n|**MG-3.2-007**|x|y|\n|text|GV-9.9-999|y|")
    assert extract_entities(md)["action_ids"] == ["GV-1.1-001", "MS-2.1-003", "MG-3.2-007"]


def test_gai_risks_split_and_cleaned():
    md = "|GV-1.1-001|Action text|Data Privacy; Harmful Bias and<br>Homogenization; Intellectual<br>Property|"
    assert extract_entities(md)["gai_risks"] == ["Data Privacy", "Harmful Bias and Homogenization", "Intellectual Property"]


def test_other_entities():
    md = ("**Trustworthy AI Characteristics:** Fair with Harmful Bias Managed, Safe, Valid and Reliable\n\n"
          "**GOVERN 1.7:** text\n\nFig. 2. Lifecycle.\n\nTable 3: Something.")
    e = extract_entities(md)
    assert e["trustworthy_characteristics"] == ["Fair with Harmful Bias Managed", "Safe", "Valid and Reliable"]
    assert e["subcategories"] == ["GOVERN 1.7"] and e["captions"] == ["Fig. 2. Lifecycle.", "Table 3: Something."]


# ---- real data ("Done when")
@pytest.fixture(scope="module")
def structured(need_pdfs):
    from ingest.clean import clean_all
    from ingest.classify import classify_all
    from ingest.config import load_manifest
    from ingest.extract import load_extract
    from ingest.structure import structure_all
    manifest = load_manifest()
    return manifest, structure_all(clean_all(manifest, classify_all(load_extract())))


def test_real_expected_section_ids_exactly_once_and_no_extras(structured):
    manifest, recs = structured
    for spec in manifest.documents:
        found = collections.Counter(h["section_id"] for r in recs if r["doc_id"] == spec.doc_id for h in r["headings"])
        assert set(found) == set(spec.validation.section_ids), (spec.doc_id, set(found) ^ set(spec.validation.section_ids))
        assert all(v == 1 for v in found.values()), [k for k, v in found.items() if v > 1]


def test_real_genai_tables_and_action_ids(structured):
    manifest, recs = structured
    spec = manifest.get("genai_profile")
    g = [r for r in recs if r["doc_id"] == "genai_profile"]
    assert sum(len(r["table_markers"]) for r in g) == spec.validation.subcategory_tables
    ids = [a for r in g for a in r["entities"]["action_ids"]]
    assert len(set(ids)) == spec.validation.unique_action_ids and len(ids) == len(set(ids))
    assert 15 <= sum(r["continues_table"] for r in g) <= 25    # plan: "about 19"


def test_real_ai_rmf_table_markers_are_the_four_tables(structured):
    _, recs = structured
    markers = [m for r in recs if r["doc_id"] == "ai_rmf_1_0" for m in r["table_markers"]]
    assert markers == ["Table 1", "Table 2", "Table 3", "Table 4"]
