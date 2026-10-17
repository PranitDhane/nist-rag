"""Stage C8: breadcrumb, IDs, citations, links and final chunk records (CHUNKING_PLAN Section 7)."""
from __future__ import annotations

import re

from chunking import config
from chunking.extras import ExtraChunk, build_figure_chunks, build_footnote_chunks
from chunking.join import JoinedDoc
from chunking.prose import ProseChunk, build_prose_chunks
from chunking.segment import Segmentation
from chunking.tables import TableChunk, build_genai_chunks, build_rmf_chunks
from chunking.tokens import count_tokens

TYPE_ORDER = {"prose": 0, "genai_action_table": 0, "rmf_core_table": 0, "figure": 1, "footnote": 2}
TRUSTWORTHY_RE = re.compile(r"^\**\s*Trustworthy AI Characteristics?:\s*(.*?)\**\s*$")


class EmbedTooLong(Exception):
    """An embed_text exceeds the model's 512-token limit (never truncated silently)."""


# ----------------------------------------------------------------------------- formats (Q7)
def short_name(display_name: str) -> str:
    return display_name.split(" (")[0]


def breadcrumb(display_name: str, section_path: list[str], chunk_type: str, *, subcategory: str | None = None,
               category: str | None = None, part: int = 1, parts_total: int = 1, figure_number: int | None = None,
               footnote_number: int | None = None) -> str:
    parts = [display_name] + list(section_path)
    if chunk_type == "genai_action_table":
        parts.append(subcategory or "")
    elif chunk_type == "rmf_core_table":
        parts.append(category or "")
    elif chunk_type == "figure":
        parts.append(f"Figure {figure_number} (OCR)")
    elif chunk_type == "footnote":
        parts.append(f"Footnote {footnote_number}")
    line = " > ".join(parts)
    if parts_total > 1:
        line += f" [part {part} of {parts_total}]"
    return line


def _section_label(path: list[str]) -> str:
    if not path:
        return ""
    leaf = path[-1]
    return f"§{leaf}" if leaf[:1].isdigit() else leaf


def _pages(first: str | None, last: str | None, pdf_first: int, pdf_last: int) -> str:
    a, b = first or f"PDF {pdf_first}", last or f"PDF {pdf_last}"
    return f"p. {a}" if a == b else f"pp. {a}–{b}"


def citation_label(display_name: str, section_path: list[str], chunk_type: str, pages: tuple, *,
                   subcategory: str | None = None, category: str | None = None, figure_number: int | None = None,
                   footnote_number: int | None = None) -> str:
    bits = [short_name(display_name)]
    if sec := _section_label(section_path):
        bits.append(sec)
    if chunk_type == "genai_action_table":
        bits.append(subcategory or "")
    elif chunk_type == "rmf_core_table":
        bits.append(category or "")
    elif chunk_type == "figure":
        bits.append(f"Fig. {figure_number}")
    elif chunk_type == "footnote":
        bits.append(f"footnote {footnote_number}")
    bits.append(_pages(*pages))
    return ", ".join(bits)


# ----------------------------------------------------------------------------- links
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("<br>", " ").replace("**", "")).strip().lower()


def attach_links(doc: JoinedDoc, spans: list[tuple[int, int]], pdf_first: int, pdf_last: int) -> list[dict]:
    """Body links on the chunk's pages whose anchor text occurs in the chunk's raw text."""
    raw = _norm(" ".join(doc.text[a:b] for a, b in spans))
    out, seen = [], set()
    for page in range(pdf_first, pdf_last + 1):
        rec = doc.page_records.get(page)
        for l in (rec or {}).get("links", []):
            text = _norm(l["text"])
            if l["location"] != "body" or not text or (text, l["url"]) in seen:
                continue
            if text in raw:
                seen.add((text, l["url"]))
                out.append({"text": l["text"], "url": l["url"]})
    return out


def trustworthy_by_section(doc: JoinedDoc) -> dict[tuple, list[str]]:
    """'Trustworthy AI Characteristics: A, B' lines (GenAI risk sections) -> characteristics per section path."""
    from ingest.clean import plain
    out: dict[tuple, list[str]] = {}
    pos = 0
    for line in doc.text.split("\n"):
        m = TRUSTWORTHY_RE.match(plain(line)) if "Trustworthy AI Characteristic" in line else None
        if m:
            traits = [t.strip().rstrip(".") for t in m.group(1).split(",") if t.strip()]
            out.setdefault(tuple(doc.path_at(pos)), []).extend(traits)
        pos += len(line) + 1
    return out


# ----------------------------------------------------------------------------- assembly
def _record(doc: JoinedDoc, *, chunk_type: str, section_path: list[str], page_flag: bool, sequence: int, part: int,
            parts_total: int, display_text: str, pages: tuple, char_start=None, char_end=None, char_spans=None,
            merged_sections=None, function=None, category=None, subcategory=None, subcategories=None, action_ids=None,
            gai_risks=None, ai_actor_tasks=None, trustworthy=None, figure_number=None, ocr_confidence=None,
            footnote_number=None, links=None) -> dict:
    pdf_first, pdf_last, printed_first, printed_last = pages
    bc = breadcrumb(doc.display_name, section_path, chunk_type, subcategory=subcategory, category=category, part=part,
                    parts_total=parts_total, figure_number=figure_number, footnote_number=footnote_number)
    embed_text = bc + "\n\n" + display_text
    n = count_tokens(embed_text)
    if n > config.HARD_MAX_EMBED_TOKENS:
        raise EmbedTooLong(f"{doc.doc_id} {chunk_type} seq {sequence} part {part}: embed_text is {n} tokens "
                           f"(limit {config.HARD_MAX_EMBED_TOKENS})")
    return {
        "chunk_id": f"{doc.doc_id}:{chunk_type}:{sequence:04d}:{part}",
        "doc_id": doc.doc_id, "display_name": doc.display_name, "chunk_type": chunk_type,
        "section_path": section_path, "section_path_is_page_level": page_flag,
        "merged_sections": merged_sections or [], "sequence": sequence, "part": part, "parts_total": parts_total,
        "char_start": char_start, "char_end": char_end, "char_spans": char_spans,
        "pdf_page_start": pdf_first, "pdf_page_end": pdf_last,
        "printed_page_start": printed_first, "printed_page_end": printed_last,
        "source_url": doc.source_url, "citation_url": f"{doc.source_url}#page={pdf_first}",
        "citation_label": citation_label(doc.display_name, section_path, chunk_type,
                                         (printed_first, printed_last, pdf_first, pdf_last), subcategory=subcategory,
                                         category=category, figure_number=figure_number, footnote_number=footnote_number),
        "display_text": display_text, "embed_text": embed_text, "token_count_embed": n,
        "function": function, "category": category, "subcategory": subcategory,
        "subcategories": subcategories or [], "action_ids": action_ids or [], "gai_risks": gai_risks or [],
        "ai_actor_tasks": ai_actor_tasks or [], "trustworthy_characteristics": trustworthy or [],
        "figure_number": figure_number, "ocr_confidence": ocr_confidence, "footnote_number": footnote_number,
        "links": links or [],
    }


def _span_pages(doc: JoinedDoc, start: int, end: int) -> tuple:
    a, b = doc.pages_for_span(start, end)
    return a.pdf_page, b.pdf_page, a.printed_page, b.printed_page


def assemble_document(seg: Segmentation, prose: list[ProseChunk], genai: list[TableChunk], rmf: list[TableChunk],
                      figures: list[ExtraChunk], footnotes: list[ExtraChunk]) -> list[dict]:
    doc = seg.doc
    traits = trustworthy_by_section(doc)
    entries = []        # (sort_key, group_key, kind, obj)
    for c in prose:
        entries.append(((c.start, 0, 0, c.part), ("prose", c.unit_sequence), "prose", c))
    for c in genai + rmf:
        entries.append(((c.start, 0, 0, c.part), (c.chunk_type, c.unit_sequence, c.category), "table", c))
    for c in figures:
        entries.append(((c.page_end_offset, 1, c.figure_number, c.part), ("figure", c.figure_number), "extra", c))
    for c in footnotes:
        entries.append(((c.page_end_offset, 2, c.footnote_number, 1), ("footnote", c.pdf_page, c.footnote_number), "extra", c))
    entries.sort(key=lambda e: e[0])
    seq_of: dict[tuple, int] = {}
    records = []
    for _, group, kind, c in entries:
        sequence = seq_of.setdefault(group, len(seq_of))
        if kind == "prose":
            spans = c.spans
            pages = _span_pages(doc, c.start, c.end)
            records.append(_record(
                doc, chunk_type="prose", section_path=c.section_path, page_flag=False, sequence=sequence, part=c.part,
                parts_total=c.parts_total, display_text=c.text, pages=pages, char_start=c.start, char_end=c.end,
                char_spans=[list(s) for s in spans], merged_sections=c.merged_sections,
                trustworthy=traits.get(tuple(c.section_path)), links=attach_links(doc, spans, pages[0], pages[1])))
        elif kind == "table":
            pages = _span_pages(doc, c.start, c.end)
            records.append(_record(
                doc, chunk_type=c.chunk_type, section_path=c.section_path, page_flag=False, sequence=sequence,
                part=c.part, parts_total=c.parts_total, display_text=c.text, pages=pages, char_start=c.start,
                char_end=c.end, char_spans=[[c.start, c.end]], function=c.function, category=c.category,
                subcategory=c.subcategory, subcategories=c.subcategories, action_ids=c.action_ids,
                gai_risks=c.gai_risks, ai_actor_tasks=c.ai_actor_tasks,
                links=attach_links(doc, [(c.start, c.end)], pages[0], pages[1])))
        else:
            span = doc.page_by_number(c.pdf_page)
            records.append(_record(
                doc, chunk_type=c.chunk_type, section_path=c.section_path, page_flag=c.section_path_is_page_level,
                sequence=sequence, part=c.part, parts_total=c.parts_total, display_text=c.text,
                pages=(span.pdf_page, span.pdf_page, span.printed_page, span.printed_page),
                figure_number=c.figure_number, ocr_confidence=c.ocr_confidence, footnote_number=c.footnote_number,
                links=c.links))
    return records
