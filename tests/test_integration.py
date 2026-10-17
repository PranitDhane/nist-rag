"""Full pipeline on the real PDFs, into a temp directory (skipped if the PDFs are absent;
the OCR assertions are skipped if Tesseract is absent)."""
from __future__ import annotations

import json

import pytest

from ingest import config
from ingest.__main__ import main
from ingest.assemble import load_pages
from ingest.profile import detect_tesseract
from ingest.validate import validate


@pytest.fixture(scope="module")
def run(need_pdfs, tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "PROCESSED_DIR", out)
    mp.setattr(config, "STAGES_DIR", out / "stages")
    mp.setattr(config, "DEBUG_FIGURES_DIR", out / "debug" / "figures")
    code = main(["--rebuild"])
    yield code, out
    mp.undo()


def test_pipeline_exits_zero_and_writes_outputs(run):
    code, out = run
    assert code == 0
    assert (out / "pages.jsonl").is_file() and (out / "ingest_report.json").is_file()
    assert len(list((out / "stages").glob("*"))) >= 7


def test_hard_checks_pass_on_written_output(run, manifest):
    _, out = run
    pages = load_pages(out / "pages.jsonl")
    assert len(pages) == 112
    failed = [c for c in validate(pages, manifest) if not c.passed]
    assert not failed, failed


def test_pages_schema(run):
    _, out = run
    keys = {"doc_id", "display_name", "source_url", "pdf_page", "printed_page", "page_type", "index_eligible",
            "section_path_start", "headings", "markdown", "footnotes", "figures", "continues_table", "table_markers",
            "entities", "links", "cleaning_log"}
    for p in load_pages(out / "pages.jsonl"):
        assert set(p) == keys
        assert set(p["entities"]) == {"action_ids", "subcategories", "gai_risks", "trustworthy_characteristics", "captions"}
        assert all(set(l) == {"text", "url", "location", "section_path"} for l in p["links"])
        assert all(set(f) == {"number", "text", "links"} for f in p["footnotes"])


def test_report_layout(run):
    _, out = run
    report = json.loads((out / "ingest_report.json").read_text())
    assert list(report)[:2] == ["ocr", "documents"]            # OCR status first
    assert list(report) == ["ocr", "documents", "checks", "warnings", "rule_hits", "run_seconds"]
    assert all(c["passed"] for c in report["checks"]) and len(report["checks"]) == 11


def test_ocr_status_recorded(run):
    _, out = run
    report = json.loads((out / "ingest_report.json").read_text())
    assert len(report["ocr"]["figures"]) == 5
    if not detect_tesseract()["available"]:
        pytest.skip("Tesseract not installed")
    assert report["ocr"]["tesseract"]["available"]
    statuses = {f["figure_number"]: f["ocr_status"] for f in report["ocr"]["figures"]}
    assert statuses[1] == statuses[3] == statuses[4] == statuses[5] == "ok"


def test_chunking_inputs_only_in_pages_jsonl(run):
    """Fig. 3's caption, recovered from rotated text, is on the Fig. 3 page; OCR text is NOT in Markdown."""
    _, out = run
    p16 = next(p for p in load_pages(out / "pages.jsonl") if p["doc_id"] == "ai_rmf_1_0" and p["pdf_page"] == 16)
    assert p16["markdown"].startswith("Fig. 3. AI actors across AI lifecycle stages")
    assert p16["figures"][0]["figure_number"] == 3 and "TEVV includes" not in p16["markdown"]


def test_from_stage_with_missing_cache_fails_clearly(need_pdfs, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(config, "STAGES_DIR", tmp_path / "stages")
    assert main(["--from-stage", "7"]) == 1
    assert any("Run Stage" in r.message for r in caplog.records)


def test_rebuild_cannot_combine_with_from_stage():
    with pytest.raises(SystemExit):
        main(["--rebuild", "--from-stage", "5"])
