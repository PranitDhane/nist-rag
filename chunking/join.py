"""Stage C3: join index-eligible pages into one text per document, with an exact page map.

Page tracking is by character offsets only: never by searching for chunk text, never by
page markers in the text (CHUNKING_PLAN D9).
"""
from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from chunking import config

SEPARATOR = "\n\n"


@dataclass
class PageSpan:
    pdf_page: int
    printed_page: str | None
    start: int
    end: int          # exclusive; the separator after the page belongs to no page


@dataclass
class Heading:
    section_id: str
    title: str
    level: int
    start: int        # absolute offset of the heading line
    section_path: list[str]


@dataclass
class Marker:
    id: str
    kind: str         # genai_subcategory | rmf_table
    start: int


@dataclass
class JoinedDoc:
    doc_id: str
    display_name: str
    source_url: str
    text: str
    pages: list[PageSpan]
    headings: list[Heading]
    markers: list[Marker]
    page_records: dict[int, dict] = field(default_factory=dict)   # pdf_page -> original record
    initial_path: list[str] = field(default_factory=list)         # path in effect before the first heading

    def __post_init__(self):
        self._starts = [p.start for p in self.pages]

    def page_at(self, offset: int) -> PageSpan:
        """Page owning `offset`; an offset inside a separator belongs to the page before it."""
        i = bisect_right(self._starts, offset) - 1
        return self.pages[max(i, 0)]

    def pages_for_span(self, start: int, end: int) -> tuple[PageSpan, PageSpan]:
        """First and last page touched by [start, end). The last character is end-1."""
        last = max(end - 1, start)
        return self.page_at(start), self.page_at(last)

    def page_by_number(self, pdf_page: int) -> PageSpan:
        return next(p for p in self.pages if p.pdf_page == pdf_page)

    def path_at(self, offset: int) -> list[str]:
        """Section path in effect at an absolute offset."""
        path = self.initial_path
        for h in self.headings:
            if h.start <= offset:
                path = h.section_path
            else:
                break
        return path


def load_pages(path: Path | None = None) -> list[dict]:
    path = path or config.PAGES_PATH
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run `python -m ingest` first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def join_document(pages: list[dict]) -> JoinedDoc:
    """Join one document's index-eligible pages (any order in; pdf_page order out)."""
    eligible = sorted((p for p in pages if p["index_eligible"]), key=lambda p: p["pdf_page"])
    if not eligible:
        raise ValueError("no index-eligible pages")
    parts: list[str] = []
    spans: list[PageSpan] = []
    headings: list[Heading] = []
    markers: list[Marker] = []
    pos = 0
    for p in eligible:
        if parts:
            pos += len(SEPARATOR)
        md = p["markdown"]
        spans.append(PageSpan(p["pdf_page"], p["printed_page"], pos, pos + len(md)))
        for h in p["headings"]:
            headings.append(Heading(h["section_id"], h["title"], h["level"], pos + h["char_offset"], h["section_path"]))
        for m in p["table_markers"]:
            markers.append(Marker(m["id"], m["kind"], pos + m["char_offset"]))
        parts.append(md)
        pos += len(md)
    first = eligible[0]
    return JoinedDoc(first["doc_id"], first["display_name"], first["source_url"], SEPARATOR.join(parts),
                     spans, headings, markers, {p["pdf_page"]: p for p in eligible},
                     initial_path=list(eligible[0]["section_path_start"]))


def join_all(pages: list[dict]) -> list[JoinedDoc]:
    by_doc: dict[str, list[dict]] = {}
    for p in pages:
        by_doc.setdefault(p["doc_id"], []).append(p)
    return [join_document(v) for v in by_doc.values()]
