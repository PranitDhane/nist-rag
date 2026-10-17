"""Stage 4: per-page Markdown plus the page facts the Markdown cannot be trusted for.

Why both: pymupdf4llm gives good headings/tables but drops or mangles page
numbers, font sizes and link rectangles. Those come straight from PyMuPDF so
every citation traces back to the official PDF (plan principle 4).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf
import pymupdf4llm

from ingest import config
from ingest.models import DocumentSpec, Manifest

log = logging.getLogger(__name__)

MARKER_RE = re.compile(r"^\d+(,\d+)*$")
MARKER_MAX_RATIO = 0.75     # footnote markers are ~7pt vs 11pt body text
BOTTOM_FRACTION = 0.75      # blocks starting below 75% of the page height


def page_markdown(doc: pymupdf.Document, page_index: int) -> str:
    """Markdown for ONE page, with pymupdf4llm's automatic OCR disabled.

    `pages=[i]` (0-based) + `page_chunks=True` returns one dict per page; `use_ocr=False`
    stops it running Tesseract on image pages (figure OCR is done separately,
    labelled, in Stage 6). Converting page by page keeps pages independent.
    """
    chunks = pymupdf4llm.to_markdown(doc, pages=[page_index], page_chunks=True, use_ocr=False)
    return chunks[0]["text"]


def first_nonempty_line(page: pymupdf.Page) -> str | None:
    for line in page.get_text("text").splitlines():
        if line.strip():
            return line.strip()
    return None


def printed_page(spec: DocumentSpec, page: pymupdf.Page) -> str | None:
    """The page number printed on the page, per the manifest's strategy.

    first_line_offset: accept the first line only if it is digits AND equals
    pdf_page - offset, so a stray number (e.g. a table value) is never mistaken
    for the page number.
    """
    pdf_page = page.number + 1
    if spec.page_number_strategy == "page_label":
        return page.get_label() or None
    if spec.page_number_strategy == "first_line_offset":
        first = first_nonempty_line(page)
        if first and first.isdigit() and int(first) == pdf_page - (spec.offset or 0):
            return first
        return None
    raise ValueError(f"{spec.doc_id}: unknown page_number_strategy {spec.page_number_strategy!r}")


def _text_spans(page_dict: dict):
    for block in page_dict["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            yield from line["spans"]


def body_font_size(page_dict: dict) -> float | None:
    """Most common span size weighted by character count (what 'normal text' is on this page)."""
    weights: Counter[float] = Counter()
    for span in _text_spans(page_dict):
        weights[round(span["size"], 1)] += len(span["text"].strip())
    return weights.most_common(1)[0][0] if weights else None


def marker_candidates(page_dict: dict, body_size: float | None) -> list[dict]:
    """Small digit-only spans: probable footnote markers.

    Font size (not the superscript flag, which also fires on a normal table
    cell on GenAI p.17) is the signal.
    """
    if body_size is None:
        return []
    out = []
    for span in _text_spans(page_dict):
        text = span["text"].strip()
        if MARKER_RE.match(text) and span["size"] <= MARKER_MAX_RATIO * body_size:
            out.append({"text": text, "bbox": [round(v, 1) for v in span["bbox"]], "size": round(span["size"], 1)})
    return out


def _line_text(line: dict) -> str:
    return " ".join(s["text"] for s in line["spans"]).strip()


def _first_span(line: dict) -> dict | None:
    return next((s for s in line["spans"] if s["text"].strip()), None)


def _starts_footnote(line: dict, body_size: float | None) -> bool:
    """A line whose first span is a small digit-only span begins a footnote."""
    first = _first_span(line)
    return (first is not None and body_size is not None and bool(MARKER_RE.match(first["text"].strip()))
            and first["size"] <= MARKER_MAX_RATIO * body_size)


def bottom_blocks(page_dict: dict, page_height: float, body_size: float | None) -> list[dict]:
    """Footnote-candidate entries from the bottom quarter of the page.

    PyMuPDF may merge several footnotes into ONE text block (GenAI p.10 holds
    footnotes 9 and 10 together), so each block is split at every line that
    starts a footnote. Each entry therefore holds exactly one footnote (or one
    non-footnote block such as the page number).

    A block is also kept when it contains a footnote-start line even if it begins
    above the 75% line: on GenAI p.8 the footnotes start at 72% of the page height,
    so a pure position cut-off silently lost footnotes 6-8.
    """
    out = []
    for block in page_dict["blocks"]:
        if block["type"] != 0:
            continue
        has_footnote = any(_starts_footnote(l, body_size) for l in block["lines"])
        if block["bbox"][1] <= BOTTOM_FRACTION * page_height and not has_footnote:
            continue
        groups: list[list[dict]] = []
        for line in block["lines"]:
            if _first_span(line) is None:
                continue
            if not groups or _starts_footnote(line, body_size):
                groups.append([])
            groups[-1].append(line)
        for lines in groups:
            first = _first_span(lines[0])
            rects = [pymupdf.Rect(l["bbox"]) for l in lines]
            bbox = rects[0]
            for r in rects[1:]:
                bbox |= r
            out.append({
                "bbox": [round(v, 1) for v in bbox],
                "first_span_text": first["text"].strip(),
                "first_span_size": round(first["size"], 1),
                "text": " ".join(t for t in (_line_text(l) for l in lines) if t),
            })
    return out


def rotated_text(page_dict: dict) -> list[str]:
    """Paragraphs of non-horizontal text, in reading order.

    pymupdf4llm silently drops rotated text, so a page whose only text is a vertical caption
    (AI RMF p.16, Fig. 3) would come out empty. PyMuPDF splits such a caption into one block
    per line, so lines are re-grouped: ordered by x (ascending when the text reads upward,
    dir 0,-1; descending when it reads downward) and joined while the next line is adjacent
    (gap under one line height) and overlaps vertically.
    """
    lines = [l for block in page_dict["blocks"] if block["type"] == 0
             for l in block["lines"] if abs(l["dir"][1]) > 0.5 and _line_text(l)]
    out: list[str] = []
    for upward in (True, False):
        group = sorted((l for l in lines if (l["dir"][1] < 0) == upward), key=lambda l: l["bbox"][0], reverse=not upward)
        current: list[dict] = []
        for line in group:
            if current:
                prev = current[-1]
                x_gap = (line["bbox"][0] - prev["bbox"][2]) if upward else (prev["bbox"][0] - line["bbox"][2])
                overlaps = line["bbox"][1] < prev["bbox"][3] and prev["bbox"][1] < line["bbox"][3]
                if x_gap > line["bbox"][2] - line["bbox"][0] or not overlaps:
                    out.append(" ".join(_line_text(l) for l in current))
                    current = []
            current.append(line)
        if current:
            out.append(" ".join(_line_text(l) for l in current))
    return out


def raw_links(page: pymupdf.Page) -> list[dict]:
    """Every link annotation, unfiltered (Stage 9 decides what to keep)."""
    out = []
    for link in page.get_links():
        out.append({
            "kind": link.get("kind"),
            "uri": link.get("uri"),
            "page": link.get("page"),
            "nameddest": link.get("nameddest"),
            "rect": [round(v, 1) for v in link["from"]],
        })
    return out


def extract_page(spec: DocumentSpec, doc: pymupdf.Document, page_index: int) -> dict:
    page = doc[page_index]
    page_dict = page.get_text("dict")
    height = page.rect.height
    body = body_font_size(page_dict)
    return {
        "doc_id": spec.doc_id,
        "pdf_page": page_index + 1,
        "printed_page": printed_page(spec, page),
        "page_height": round(height, 1),
        "markdown": page_markdown(doc, page_index),
        "body_font_size": body,
        "marker_candidates": marker_candidates(page_dict, body),
        "bottom_blocks": bottom_blocks(page_dict, height, body),
        "rotated_text": rotated_text(page_dict),
        "links": raw_links(page),
    }


def extract_all(manifest: Manifest, raw_dir: Path | None = None) -> list[dict]:
    raw_dir = raw_dir or config.RAW_DIR
    records: list[dict] = []
    for spec in manifest.documents:
        path = raw_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"extract: {spec.doc_id}: {path} is missing. Run Stage 2 first.")
        with pymupdf.open(path) as doc:
            n_pages = doc.page_count
            for i in range(n_pages):
                records.append(extract_page(spec, doc, i))
        log.info("extract: %s: %d pages", spec.doc_id, n_pages)
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STAGES_DIR / "04_extract.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_extract() -> list[dict]:
    path = config.STAGES_DIR / "04_extract.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"extract: {path} is missing. Run Stage 4 first (python -m ingest --only-stage 4).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
