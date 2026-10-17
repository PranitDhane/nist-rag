"""Stage S1: the shared citation formatter (ATTRIBUTION_PLAN Section 3). Labels are computed from structured metadata."""
import json

import pytest

from attribution import format as fmt
from ingest.config import load_manifest

RMF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
GAI_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"


def meta(chunk_type="prose", doc_id="ai_rmf_1_0", section=("Part 2", "5.1 Govern"), printed=("22", "22"), pdf=(27, 27), **extra):
    m = {"doc_id": doc_id, "chunk_type": chunk_type, "section_path": list(section), "printed_page_start": printed[0],
         "printed_page_end": printed[1], "pdf_page_start": pdf[0], "pdf_page_end": pdf[1],
         "source_url": RMF_URL if doc_id == "ai_rmf_1_0" else GAI_URL, "subcategory": None, "category": None,
         "figure_number": None, "footnote_number": None}
    m.update(extra)
    return m


# --- every row of the Section 3 table (formats as approved at Q1: current stored formats kept) -----------------------

def test_numbered_section_uses_section_sign_and_leaf_only():
    assert fmt.citation_label(meta(printed=("22", "23"), pdf=(27, 28))) == "NIST AI 100-1, §5.1 Govern, pp. 22–23"


def test_unnumbered_section_has_no_section_sign():
    m = meta(section=("Executive Summary",), printed=("1", "1"), pdf=(6, 6))
    assert fmt.citation_label(m) == "NIST AI 100-1, Executive Summary, p. 1"


def test_appendix_keeps_its_title():
    m = meta(doc_id="genai_profile", section=("Appendix A: Primary GAI Considerations",), printed=("47", "47"))
    assert fmt.citation_label(m) == "NIST AI 600-1, Appendix A: Primary GAI Considerations, p. 47"


def test_genai_table_chunk_adds_the_subcategory():
    m = meta("genai_action_table", doc_id="genai_profile", section=("3 Suggested Actions",), subcategory="GOVERN 1.7")
    assert fmt.citation_label(m) == "NIST AI 600-1, §3 Suggested Actions, GOVERN 1.7, p. 22"


def test_rmf_core_table_chunk_adds_the_category():
    m = meta("rmf_core_table", category="GOVERN 1")
    assert fmt.citation_label(m) == "NIST AI 100-1, §5.1 Govern, GOVERN 1, p. 22"


def test_figure_and_footnote_labels_name_the_item():
    assert fmt.citation_label(meta("figure", figure_number=3)) == "NIST AI 100-1, §5.1 Govern, Fig. 3, p. 22"
    assert fmt.citation_label(meta("footnote", doc_id="genai_profile", footnote_number=2)) == \
        "NIST AI 600-1, §5.1 Govern, footnote 2, p. 22"


def test_figure_label_never_says_ocr():
    """SA-D6: the OCR marker lives in `kind` only."""
    assert "OCR" not in fmt.citation_label(meta("figure", figure_number=1)).upper()


def test_one_page_is_p_and_a_range_is_pp_with_an_en_dash():
    assert fmt.pages_text(meta(printed=("21", "21"))) == "p. 21"
    rng = fmt.pages_text(meta(printed=("22", "23")))
    assert rng == "pp. 22–23" and "–" in rng and "-" not in rng


def test_null_printed_page_falls_back_to_pdf_page():
    one = meta(printed=(None, None), pdf=(26, 26))
    rng = meta(printed=(None, None), pdf=(26, 27))
    assert fmt.pages_text(one) == "PDF p. 26"
    assert fmt.pages_text(rng) == "PDF pp. 26–27"


def test_section_label_is_empty_without_a_section_path():
    assert fmt.section_label(meta(section=())) == ""
    assert fmt.citation_label(meta(section=(), printed=("3", "3"))) == "NIST AI 100-1, p. 3"


# --- URL ---------------------------------------------------------------------------------------------------------------

def test_url_uses_the_pdf_page_index_not_the_printed_number():
    m = meta(printed=("21", "21"), pdf=(26, 26))
    assert fmt.citation_url(m) == f"{RMF_URL}#page=26" and "p. 21" in fmt.citation_label(m)


def test_url_uses_pdf_page_start_for_a_range():
    assert fmt.citation_url(meta(printed=("22", "23"), pdf=(27, 28))).endswith("#page=27")


# --- kind --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_type,expected", [("prose", "prose"), ("genai_action_table", "table"),
                                                 ("rmf_core_table", "table"), ("figure", "figure-ocr"),
                                                 ("footnote", "footnote")])
def test_kind_mapping_for_every_chunk_type(chunk_type, expected):
    assert fmt.kind(meta(chunk_type)) == expected


# --- manifest ----------------------------------------------------------------------------------------------------------

def test_citation_names_come_from_the_manifest():
    names = {d.doc_id: d.citation_name for d in load_manifest().documents}
    assert names == {"ai_rmf_1_0": "NIST AI 100-1", "genai_profile": "NIST AI 600-1"}
    assert fmt.doc_name(meta()) == "NIST AI 100-1" and fmt.doc_name(meta(doc_id="genai_profile")) == "NIST AI 600-1"


def test_unknown_doc_id_fails_loudly():
    with pytest.raises(KeyError):
        fmt.doc_name(meta(doc_id="nope"))


def test_document_order_is_manifest_order():
    assert fmt.doc_order("ai_rmf_1_0") < fmt.doc_order("genai_profile")


# --- real data: the formatter equals what chunking stored (Q2: runtime formatter is the source of truth) -----------------

def real_chunks():
    from ingest.config import PROCESSED_DIR
    path = PROCESSED_DIR / "chunks.jsonl"
    if not path.exists():
        pytest.skip("chunks.jsonl not built")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_formatter_matches_every_stored_label_and_url():
    for c in real_chunks():
        assert fmt.citation_label(c) == c["citation_label"], c["chunk_id"]
        assert fmt.citation_url(c) == c["citation_url"], c["chunk_id"]


def test_no_real_label_mentions_ocr():
    assert not [c["chunk_id"] for c in real_chunks() if "ocr" in fmt.citation_label(c).lower()]
