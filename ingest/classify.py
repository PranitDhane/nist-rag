"""Stage 5: label every page so chunking knows what to index. Nothing is deleted.

Why content rules (not page ranges): the rules must work from what is on the
page. Expected ranges live only in the manifest's validation block, which this
module never reads, so validate.py stays an independent check.
"""
from __future__ import annotations

import json
import logging
import re

from ingest import config

log = logging.getLogger(__name__)

TOC_HEADINGS = {"table of contents", "list of tables", "list of figures"}
BODY_START_RE = re.compile(r"^(executive summary|1\.?\s+\S.*)$", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^Appendix [A-Z]\b")
REFERENCES_RE = re.compile(r"^Appendix [A-Z]\s*[.:]?\s*References\b", re.IGNORECASE)
BACK_COVER_MAX_WORDS = 30
INDEX_ELIGIBLE = {"body", "appendix"}


def heading_texts(markdown: str) -> list[str]:
    """Text of Markdown heading lines with '#', '*' and extra spaces stripped."""
    out = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("#"):
            text = re.sub(r"\s+", " ", line.lstrip("# ").replace("*", "")).strip()
            if text:
                out.append(text)
    return out


def classify_document(pages: list[dict]) -> list[dict]:
    """Add page_type, index_eligible and page_type_reason to one document's pages (in page order)."""
    if not pages:
        return []
    doc_id = pages[0]["doc_id"]
    heads = {p["pdf_page"]: heading_texts(p["markdown"]) for p in pages}
    first, last = pages[0]["pdf_page"], pages[-1]["pdf_page"]

    toc_pages = {n for n, hs in heads.items() if any(h.lower() in TOC_HEADINGS for h in hs)}

    def first_page_where(match, start: int = first) -> int | None:
        for n in sorted(heads):
            if n >= start and n != first and n not in toc_pages and any(match(h) for h in heads[n]):
                return n
        return None

    body_start = first_page_where(lambda h: bool(BODY_START_RE.match(h)))
    if body_start is None:
        raise ValueError(f"classify: {doc_id}: no body heading (Executive Summary / section 1) found outside "
                         "the table of contents. The document layout differs from what the rules expect.")
    appendix_start = first_page_where(lambda h: bool(APPENDIX_RE.match(h)), body_start)
    references_start = first_page_where(lambda h: bool(REFERENCES_RE.match(h)), body_start)
    log.info("classify: %s: body starts p.%s, appendix p.%s, references p.%s",
             doc_id, body_start, appendix_start, references_start)

    out = []
    for p in pages:
        n = p["pdf_page"]
        words = len(p["markdown"].split())
        if n == first:
            ptype, reason = "cover", "first page"
        elif n in toc_pages:
            ptype, reason = "toc", "toc heading"
        elif n < body_start:
            ptype, reason = "front_matter", f"before first body heading (p.{body_start})"
        elif n == last and words < BACK_COVER_MAX_WORDS and not heads[n]:
            ptype, reason = "back_cover", f"last page, {words} words, no heading"
        elif references_start is not None and n >= references_start:
            ptype, reason = "references", f"from references heading (p.{references_start})"
        elif appendix_start is not None and n >= appendix_start:
            ptype, reason = "appendix", f"from appendix heading (p.{appendix_start})"
        else:
            ptype, reason = "body", f"from first body heading (p.{body_start})"
        out.append({**p, "page_type": ptype, "index_eligible": ptype in INDEX_ELIGIBLE, "page_type_reason": reason})
    return out


def classify_all(records: list[dict]) -> list[dict]:
    by_doc: dict[str, list[dict]] = {}
    for r in records:
        by_doc.setdefault(r["doc_id"], []).append(r)
    result: list[dict] = []
    for pages in by_doc.values():
        result.extend(classify_document(sorted(pages, key=lambda p: p["pdf_page"])))
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STAGES_DIR / "05_classify.jsonl", "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return result


def load_classify() -> list[dict]:
    path = config.STAGES_DIR / "05_classify.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"classify: {path} is missing. Run Stage 5 first (python -m ingest --only-stage 5).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
