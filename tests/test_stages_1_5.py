"""Tests for Stages 1-5: manifest, acquire, profile, extract, classify."""
from __future__ import annotations

import pymupdf
import pytest

from ingest import config
from ingest.classify import classify_document, heading_texts
from ingest.download import AcquireError, acquire_document, sha256_of
from ingest.extract import extract_page, marker_candidates, body_font_size
from ingest.profile import is_rotated, profile_document


# ---- Stage 1 ----
def test_manifest_has_both_documents(manifest):
    assert [d.doc_id for d in manifest.documents] == ["ai_rmf_1_0", "genai_profile"]
    assert {d.expected_pages for d in manifest.documents} == {48, 64}
    assert manifest.get("genai_profile").offset == 4


def test_manifest_unknown_doc(manifest):
    with pytest.raises(KeyError):
        manifest.get("nope")


def test_requirements_pinned():
    lines = [l for l in (config.PROJECT_ROOT / "requirements.txt").read_text().splitlines() if l.strip()]
    assert lines and all("==" in l for l in lines)


# ---- Stage 2 ----
def test_fake_pdf_rejected(manifest, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    spec = manifest.documents[0]
    (src / spec.filename).write_text("<html>not a pdf</html>")
    with pytest.raises(AcquireError, match="PDF|%PDF"):
        acquire_document(spec, tmp_path / "raw", from_dir=src)


def test_acquire_idempotent_and_hash(manifest, need_pdfs):
    for spec in manifest.documents:
        r = acquire_document(spec, config.RAW_DIR)
        assert r.status == "already_present"
        assert r.pages == spec.expected_pages
        assert sha256_of(config.RAW_DIR / spec.filename) == spec.sha256


def test_acquire_from_dir_alt_name(manifest, need_pdfs, tmp_path):
    """Local mode finds a file by the 100-1 / 600-1 part of the name."""
    spec = manifest.documents[0]
    src = tmp_path / "src"
    src.mkdir()
    (src / "nist_ai_100-1.pdf").write_bytes((config.RAW_DIR / spec.filename).read_bytes())
    r = acquire_document(spec, tmp_path / "raw", from_dir=src)
    assert r.status == "copied" and r.pages == 48
    assert (tmp_path / "raw" / spec.filename).is_file()


# ---- Stage 3 ----
def test_is_rotated():
    assert is_rotated((0, -632.41, 488.87, 0, 0, 0))
    assert not is_rotated((400, 0, 0, 300, 0, 0))


def test_profile_ai_rmf_figures(manifest, need_pdfs):
    spec = manifest.get("ai_rmf_1_0")
    p = profile_document(spec, config.RAW_DIR / spec.filename)
    figs = {i["figure_number"]: i for i in p["images"] if i["figure_number"] is not None}
    assert sorted(figs) == [1, 2, 3, 4, 5]
    assert [i["rotated"] for n, i in sorted(figs.items())] == [False, False, True, False, False]
    assert {n: figs[n]["pdf_page"] for n in figs} == {1: 10, 2: 15, 3: 16, 4: 17, 5: 25}
    assert p["page_labels_present"] and p["text_layer_present"]


def test_profile_genai_no_labels(manifest, need_pdfs):
    spec = manifest.get("genai_profile")
    p = profile_document(spec, config.RAW_DIR / spec.filename)
    assert p["pages"] == 64 and not p["page_labels_present"]
    assert not [i for i in p["images"] if i["figure_number"] is not None]


# ---- Stage 4 ----
@pytest.fixture(scope="module")
def extracted(manifest, need_pdfs):
    out = {}
    for spec in manifest.documents:
        with pymupdf.open(config.RAW_DIR / spec.filename) as doc:
            out[spec.doc_id] = [extract_page(spec, doc, i) for i in range(doc.page_count)]
    return out


def test_extract_counts_and_printed_pages(manifest, extracted):
    assert sum(len(v) for v in extracted.values()) == 112
    for spec in manifest.documents:
        by_page = {r["pdf_page"]: r for r in extracted[spec.doc_id]}
        for pg, expected in spec.validation.printed_page_spot_checks.items():
            assert by_page[pg]["printed_page"] == expected, (spec.doc_id, pg)


def test_extract_markers(extracted):
    assert sum(len(r["marker_candidates"]) for r in extracted["ai_rmf_1_0"]) == 0
    p10 = extracted["genai_profile"][9]
    # the footnote numbers at the page bottom are also small digit spans, so they appear too
    assert "9,10" in [m["text"] for m in p10["marker_candidates"]]
    starts = [b["first_span_text"] for b in p10["bottom_blocks"]]
    assert "9" in starts and "10" in starts


def test_marker_candidates_ignore_body_size():
    page_dict = {"blocks": [{"type": 0, "lines": [{"spans": [
        {"text": "body text here", "size": 11.0, "bbox": (0, 0, 1, 1)},
        {"text": "9,10", "size": 7.0, "bbox": (0, 0, 1, 1)}]}]}]}
    assert body_font_size(page_dict) == 11.0
    assert [m["text"] for m in marker_candidates(page_dict, 11.0)] == ["9,10"]


# ---- Stage 5 ----
def test_heading_texts():
    assert heading_texts("#### **2.2. Confabulation**\nbody\n## Appendix A:") == ["2.2. Confabulation", "Appendix A:"]


def test_classify_matches_validation_ranges(manifest, extracted):
    for spec in manifest.documents:
        result = classify_document(extracted[spec.doc_id])
        by_page = {r["pdf_page"]: r for r in result}
        for rng in spec.validation.page_types:
            for pg in range(rng.first, rng.last + 1):
                assert by_page[pg]["page_type"] == rng.page_type, (spec.doc_id, pg)
        assert all(r["index_eligible"] == (r["page_type"] in {"body", "appendix"}) for r in result)
        if spec.doc_id == "genai_profile":
            assert not any(by_page[p]["index_eligible"] for p in range(58, 65))


def test_classify_fails_clearly_without_body_heading():
    pages = [{"doc_id": "x", "pdf_page": i, "markdown": "plain text"} for i in (1, 2, 3)]
    with pytest.raises(ValueError, match="no body heading"):
        classify_document(pages)


def test_bottom_blocks_keep_footnotes_above_75_percent(extracted):
    """GenAI p.8 footnotes 6-8 start at 72% of the page height; they must not be lost."""
    p8 = extracted["genai_profile"][7]
    starts = [b["first_span_text"] for b in p8["bottom_blocks"]]
    assert {"6", "7", "8"} <= set(starts)


def test_rotated_caption_recovered_on_fig3_page(extracted):
    """AI RMF p.16 holds only a vertical caption, which pymupdf4llm drops from the Markdown."""
    p16 = extracted["ai_rmf_1_0"][15]
    assert len(p16["rotated_text"]) == 1                      # one caption, not one paragraph per line
    assert " ".join(p16["rotated_text"][0].split()).startswith("Fig. 3. AI actors across AI lifecycle stages")
    assert p16["rotated_text"][0].rstrip().endswith("verifying and validating the models.")
    assert "Fig. 3" not in p16["markdown"]
    assert all(not r["rotated_text"] for r in extracted["genai_profile"])
