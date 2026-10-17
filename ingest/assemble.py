"""Stage 10 (part 1): write the final page records and the run report.

Why a separate module from validate.py: the validator must stay an independent check of
what this module produces (plan decision log: "expectations used only by the validator").
Chunking will read ONLY pages.jsonl, so every field it needs is assembled here.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from ingest import config
from ingest.clean import rule_hit_counts
from ingest.figures import confidence_threshold
from ingest.models import Manifest

log = logging.getLogger(__name__)

SHORT_BODY_WORDS = 80
LEFTOVER_TAG_RE = re.compile(r"</?(?!br\b)[A-Za-z][^>]*>")   # <br> is legitimate Markdown inside table cells


def _page_figures(figures: list[dict], doc_id: str, pdf_page: int) -> list[dict]:
    keys = ("figure_number", "caption", "text", "method", "confidence", "ocr_status")
    return [{k: f[k] for k in keys} for f in figures if f["doc_id"] == doc_id and f["pdf_page"] == pdf_page]


def _footnotes_with_links(page: dict) -> list[dict]:
    """Footnotes plus the links whose rectangle sits inside that footnote's block."""
    out = []
    for fn in page["footnotes"]:
        links = [{"text": l["text"], "url": l["url"]} for l in page["links"] if l.get("footnote_number") == fn["number"]]
        out.append({"number": fn["number"], "text": fn["text"], "links": links})
    return out


def final_record(spec, page: dict, figures: list[dict]) -> dict:
    """One page in the pages.jsonl schema (plan Section 6, Stage 10)."""
    return {
        "doc_id": spec.doc_id, "display_name": spec.display_name, "source_url": spec.url,
        "pdf_page": page["pdf_page"], "printed_page": page["printed_page"],
        "page_type": page["page_type"], "index_eligible": page["index_eligible"],
        "section_path_start": page["section_path_start"], "headings": page["headings"],
        "markdown": page["markdown"], "footnotes": _footnotes_with_links(page),
        "figures": _page_figures(figures, spec.doc_id, page["pdf_page"]),
        "continues_table": page["continues_table"], "table_markers": page["table_markers"],
        "entities": page["entities"],
        "links": [{k: l[k] for k in ("text", "url", "location", "section_path")} for l in page["links"]],
        "cleaning_log": page["cleaning_log"],
    }


def assemble(manifest: Manifest, linked: list[dict], figures: list[dict]) -> list[dict]:
    """Final records in manifest order, written to pages.jsonl."""
    out = []
    for spec in manifest.documents:
        for page in sorted((p for p in linked if p["doc_id"] == spec.doc_id), key=lambda p: p["pdf_page"]):
            out.append(final_record(spec, page, figures))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_DIR / "pages.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("assemble: %d page records written to pages.jsonl", len(out))
    return out


def soft_warnings(linked: list[dict], figures: list[dict], pages: list[dict]) -> list[str]:
    """Problems that do not stop the run but that a reviewer should see."""
    out: list[str] = []
    for f in figures:
        if f["ocr_status"] != "ok":
            out.append(f"OCR {f['ocr_status']}: {f['doc_id']} figure {f['figure_number']} (p.{f['pdf_page']}, confidence {f['confidence']})")
    for p in linked:
        out += [f"{p['doc_id']} p.{p['pdf_page']}: {w}" for w in p["warnings"]]
        out += [f"{p['doc_id']} p.{p['pdf_page']}: anchor text looks cut off: {c['text'][:60]!r}" for c in p["cut_off_links"]]
        out += [f"{p['doc_id']} p.{p['pdf_page']}: body link without anchor text: {l['url'][:70]}"
                for l in p["links"] if l["location"] == "body" and not l["text"]]
    for r in pages:
        words = len(r["markdown"].split())
        if r["page_type"] == "body" and words < SHORT_BODY_WORDS:
            out.append(f"{r['doc_id']} p.{r['pdf_page']}: body page has only {words} words")
        for tag in sorted(set(LEFTOVER_TAG_RE.findall(r["markdown"]))):
            out.append(f"{r['doc_id']} p.{r['pdf_page']}: leftover tag {tag}")
    return out


def document_summary(spec, pages: list[dict]) -> dict:
    docs = [p for p in pages if p["doc_id"] == spec.doc_id]
    ids = [a for p in docs for a in p["entities"]["action_ids"]]
    links = Counter(l["location"] for p in docs for l in p["links"])
    return {
        "pages": len(docs),
        "page_types": dict(Counter(p["page_type"] for p in docs)),
        "index_eligible_pages": sum(p["index_eligible"] for p in docs),
        "headings": sum(len(p["headings"]) for p in docs),
        "table_markers": sum(len(p["table_markers"]) for p in docs),
        "continues_table_pages": sum(p["continues_table"] for p in docs),
        "table_row_action_ids": len(ids), "unique_action_ids": len(set(ids)),
        "footnotes": sum(len(p["footnotes"]) for p in docs),
        "figures": sum(len(p["figures"]) for p in docs),
        "links_by_location": dict(links),
    }


def build_report(manifest: Manifest, pages: list[dict], linked: list[dict], figures: list[dict],
                 profile: dict, checks: list, elapsed_seconds: float) -> dict:
    """Report layout: OCR status first, then per-document counts, checks, warnings, rule hits, run time."""
    report = {
        "ocr": {"tesseract": profile["tesseract"], "confidence_threshold": confidence_threshold(),
                "figures": [{"doc_id": f["doc_id"], "figure_number": f["figure_number"], "pdf_page": f["pdf_page"],
                             "ocr_status": f["ocr_status"], "confidence": f["confidence"]} for f in figures]},
        "documents": {spec.doc_id: document_summary(spec, pages) for spec in manifest.documents},
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "warnings": soft_warnings(linked, figures, pages),
        "rule_hits": {k: dict(v) for k, v in rule_hit_counts(linked).items()},
        "run_seconds": round(elapsed_seconds, 1),
    }
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / "ingest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_pages(path: Path | None = None) -> list[dict]:
    path = path or config.PROCESSED_DIR / "pages.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"assemble: {path} is missing. Run the pipeline first (python -m ingest).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
