import json

import pytest

pytest.importorskip("transformers")
from chunking import config  # noqa: E402

pytestmark = pytest.mark.skipif(not config.PAGES_PATH.is_file(), reason="pages.jsonl absent")


def test_full_run_passes_all_hard_checks_and_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(config, "REPORT_PATH", tmp_path / "chunk_report.json")
    from chunking.__main__ import main
    assert main(["--rebuild"]) == 0
    first = (tmp_path / "chunks.jsonl").read_bytes()
    assert main([]) == 0
    assert (tmp_path / "chunks.jsonl").read_bytes() == first         # byte-for-byte
    report = json.loads((tmp_path / "chunk_report.json").read_text())
    assert report["total_chunks"] == len(first.splitlines()) and all(c["passed"] for c in report["checks"])
    assert "profile" in report


def test_schema_fields_and_types():
    from chunking.pipeline import run_all
    recs = [r for _, rs, _, _ in run_all() for r in rs]
    must = {"chunk_id", "doc_id", "display_name", "chunk_type", "section_path", "section_path_is_page_level",
            "merged_sections", "sequence", "part", "parts_total", "char_start", "char_end", "pdf_page_start",
            "pdf_page_end", "printed_page_start", "printed_page_end", "source_url", "citation_url", "citation_label",
            "display_text", "embed_text", "token_count_embed", "function", "category", "subcategory", "subcategories",
            "action_ids", "gai_risks", "ai_actor_tasks", "trustworthy_characteristics", "figure_number",
            "ocr_confidence", "footnote_number", "links"}
    for r in recs:
        assert must <= set(r), must - set(r)
        assert isinstance(r["links"], list) and all(set(l) == {"text", "url"} for l in r["links"])
        assert r["footnote_number"] is None or isinstance(r["footnote_number"], int)
        assert r["citation_url"] == f"{r['source_url']}#page={r['pdf_page_start']}"
        assert r["embed_text"].endswith(r["display_text"])
