"""Stage S4: the pure parts of eval/link_check.py, and the check's ability to FAIL on a wrong link."""
import json

import pytest

from eval import link_check as lc
from tests.test_build_sources import RMF, meta

PAGE_A = "Govern is a cross-cutting function that informs and is infused throughout the AI RMF. " * 3 + "Culture of risk management."
PAGE_B = "Map establishes the context to frame risks related to an AI system. " * 4
PRINTED = {("ai_rmf_1_0", 27): "22", ("ai_rmf_1_0", 30): "25"}


def chunk(text=PAGE_A, **kw):
    c = meta("a", printed=("22", "22"), pdf=(27, 27))
    c.update(display_text=text, citation_url=f"{RMF}#page=27", **kw)
    return c


def run(c, pages=None, printed=PRINTED, captions=None):
    return lc.check_chunk(c, pages or {27: PAGE_A, 28: PAGE_B, 30: PAGE_B}, printed, captions or {}, RMF)


def test_normalize_ignores_case_markup_ligatures_hyphenation_and_whitespace():
    assert lc.normalize("**Or-\nganizational** ﬁne\n| a |  B") == "organizationalfineab"


def test_matching_chunk_passes():
    assert run(chunk()) == []


def test_markdown_markup_in_the_chunk_does_not_matter():
    assert run(chunk("# **Govern** is a cross-cutting function that informs and is infused throughout the AI RMF. " * 3 + "Culture of risk management.")) == []


def test_a_link_to_the_wrong_page_is_a_real_failure():
    pages = {27: PAGE_B, 28: PAGE_A, 30: PAGE_B}                # the text lives on page 28, the link says 27
    [finding] = [f for f in run(chunk(), pages) if f["check"].startswith("text")]
    assert finding["check"] == "text"


def test_table_layout_difference_is_explained_when_the_linked_page_is_still_the_best_match():
    scrambled = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 2) + PAGE_A[:120]
    pages = {27: scrambled, 28: PAGE_B, 30: PAGE_B}
    [finding] = run(chunk(PAGE_A + " tail words that never appear on the page at all"), pages)
    assert finding["check"] in lc.EXPLAINED


def test_label_page_must_match_the_printed_page_ingestion_recorded():
    [finding] = run(chunk(), printed={("ai_rmf_1_0", 27): "99", ("ai_rmf_1_0", 30): "25"})
    assert finding["check"] == "printed_page" and "p. 22" in finding["detail"] and "p. 99" in finding["detail"]


def test_url_must_point_at_pdf_page_start():
    c = chunk(source_url=RMF + "x")
    assert any(f["check"] == "url" for f in run(c))


def test_figure_chunk_is_checked_on_caption_not_ocr_text():
    c = chunk("OCR garbage that is not in the text layer", chunk_type="figure", figure_number=5)
    assert run(c, captions={("ai_rmf_1_0", 27, "5"): "Govern is a cross-cutting function that informs"}) == []
    [f] = run(c, pages={27: PAGE_B, 28: PAGE_B, 30: PAGE_B}, captions={("ai_rmf_1_0", 27, "5"): "Fig. 5. Something else entirely here"})
    assert f["check"] == "figure_page"
    assert run(c, pages={27: "Figure 5. whatever", 28: PAGE_B, 30: PAGE_B}) == []       # falls back to 'Figure N'


def test_real_run_finds_no_failures_and_writes_a_report(tmp_path):
    from ingest.config import PROCESSED_DIR, RAW_DIR
    if not (PROCESSED_DIR / "chunks.jsonl").exists() or not list(RAW_DIR.glob("*.pdf")):
        pytest.skip("chunks or PDFs missing")
    out = tmp_path / "r.json"
    assert lc.main(["--out", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["chunks_checked"] == 208 and r["failed"] == 0 and r["pass_rate_pct"] == 100.0
