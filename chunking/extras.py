"""Stage C7: figure-OCR chunks and footnote chunks. Never merged with other chunks."""
from __future__ import annotations

from dataclasses import dataclass, field

from chunking import config
from chunking.join import JoinedDoc
from chunking.prose import split_segment
from chunking.tokens import count_tokens

OCR_LABEL = "Text extracted by OCR from Figure {n} (may contain recognition errors):"


@dataclass
class ExtraChunk:
    doc_id: str
    chunk_type: str                  # figure | footnote
    section_path: list[str]
    section_path_is_page_level: bool
    pdf_page: int
    page_end_offset: int             # absolute offset of the end of the page; orders the chunk after the page's text
    text: str
    tokens: int
    part: int = 1
    parts_total: int = 1
    figure_number: int | None = None
    ocr_confidence: float | None = None
    footnote_number: int | None = None
    links: list[dict] = field(default_factory=list)


def _figure_section(doc: JoinedDoc, page, caption: str | None) -> tuple[list[str], str]:
    """Section at the figure's position: located through its caption text on the page, else end of page."""
    if caption:
        key = " ".join(caption.split())[:40]
        md = doc.text[page.start:page.end]
        flat_pos = md.find(key) if key else -1
        if flat_pos >= 0:
            return doc.path_at(page.start + flat_pos), "caption"
    return doc.path_at(page.end), "end_of_page"


def build_figure_chunks(doc: JoinedDoc) -> tuple[list[ExtraChunk], list[str]]:
    out: list[ExtraChunk] = []
    log: list[str] = []
    for span in doc.pages:
        rec = doc.page_records[span.pdf_page]
        for fig in rec["figures"]:
            if not fig.get("text"):
                log.append(f"Figure {fig['figure_number']} (p.{span.pdf_page}): no OCR text ({fig.get('ocr_status')}); no chunk")
                continue
            path, how = _figure_section(doc, span, fig.get("caption"))
            if how != "caption":
                log.append(f"Figure {fig['figure_number']}: caption not found on page; section taken at end of page")
            label = OCR_LABEL.format(n=fig["figure_number"])
            head = "\n".join(x for x in (fig.get("caption"), label) if x)
            ocr = fig["text"].strip()
            room = config.TABLE_PIECE_MAX_TOKENS - count_tokens(head + "\n")
            pieces = split_segment(ocr, 0, max_tokens=room, min_tokens=0, overlap_fraction=0.0)
            if len(pieces) > 1:
                log.append(f"Figure {fig['figure_number']}: OCR text split into {len(pieces)} pieces")
            for n, (a, b, _) in enumerate(pieces, start=1):
                text = head + "\n" + ocr[a:b]
                out.append(ExtraChunk(doc.doc_id, "figure", path, False, span.pdf_page, span.end, text, count_tokens(text),
                                      n, len(pieces), figure_number=fig["figure_number"],
                                      ocr_confidence=fig.get("confidence")))
    return out, log


def build_footnote_chunks(doc: JoinedDoc) -> list[ExtraChunk]:
    """One chunk per footnote. Q6 (user): section path = section in effect at the END of the page."""
    out = []
    for span in doc.pages:
        rec = doc.page_records[span.pdf_page]
        for fn in rec["footnotes"]:
            number = int(fn["number"])
            text = f"Footnote {number}: {fn['text']}".strip()
            out.append(ExtraChunk(doc.doc_id, "footnote", doc.path_at(span.end), True, span.pdf_page, span.end, text,
                                  count_tokens(text), footnote_number=number,
                                  links=[{"text": l["text"], "url": l["url"]} for l in fn["links"]]))
    return out
