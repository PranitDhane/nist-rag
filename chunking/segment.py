"""Stage C4: cut each joined document into structural units before any size splitting.

Units never cross a section boundary. Every non-whitespace character of the joined text
belongs to exactly one unit (prose, genai_table or rmf_table).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from chunking.join import JoinedDoc

# A category label ("GOVERN 1:", "MAP 2:") - the digit is followed directly by a colon, unlike
# subcategory labels ("GOVERN 1.1:"). In the PDF text it may sit anywhere on a line.
CATEGORY_LABEL_RE = re.compile(r"\*\*(?:[A-Za-z ]+ )?(GOVERN|MAP|MEASURE|MANAGE) (\d+):")
AI_ACTOR_LINE_RE = re.compile(r"^\**\s*AI Actors?\b")


@dataclass
class Section:
    index: int
    start: int
    end: int
    path: list[str]
    has_heading: bool


@dataclass
class Unit:
    kind: str                      # prose | genai_table | rmf_table
    doc_id: str
    start: int
    end: int                       # exclusive, trimmed of surrounding whitespace
    section_index: int
    section_path: list[str]
    sequence: int = -1
    marker_id: str | None = None   # "GOVERN 1.7" / "Table 1" for tables


@dataclass
class Category:
    """One AI RMF Core category inside an rmf_table unit."""
    label: str                     # e.g. "GOVERN 1"
    start: int
    end: int


@dataclass
class Segmentation:
    doc: JoinedDoc
    sections: list[Section]
    units: list[Unit]
    categories: dict[int, list[Category]] = field(default_factory=dict)   # unit.sequence -> categories


def build_sections(doc: JoinedDoc) -> list[Section]:
    heads = sorted(doc.headings, key=lambda h: h.start)
    sections: list[Section] = []
    if not heads or heads[0].start > 0:
        end = heads[0].start if heads else len(doc.text)
        if doc.text[:end].strip():
            sections.append(Section(0, 0, end, list(doc.initial_path), False))
    for i, h in enumerate(heads):
        end = heads[i + 1].start if i + 1 < len(heads) else len(doc.text)
        sections.append(Section(len(sections), h.start, end, h.section_path, True))
    return sections


def _line_bounds(text: str, pos: int, limit: int) -> tuple[int, int]:
    """(start, end_exclusive_without_newline) of the line beginning at pos."""
    nl = text.find("\n", pos, limit)
    return pos, (limit if nl == -1 else nl)


def genai_table_end(text: str, start: int, limit: int) -> int:
    """End of the table unit that starts at `start`: the subcategory line, then Markdown table
    lines (continuing across blank lines and page separators) up to `limit`, which the caller
    sets to the next marker / section end. The first non-blank, non-table line ends the unit."""
    pos, line_end = _line_bounds(text, start, limit)   # the subcategory statement line (accepted as is)
    end = line_end
    pos = line_end + 1
    while pos < limit:
        _, line_end = _line_bounds(text, pos, limit)
        line = text[pos:line_end]
        if line.strip():
            if AI_ACTOR_LINE_RE.match(line.strip()):    # pymupdf4llm sometimes renders the last row as a paragraph
                return line_end
            if not line.lstrip().startswith("|"):
                break
            end = line_end
        pos = line_end + 1
    return end


def segment(doc: JoinedDoc) -> Segmentation:
    sections = build_sections(doc)
    markers = sorted(doc.markers, key=lambda m: m.start)
    units: list[Unit] = []
    for sec in sections:
        in_sec = [m for m in markers if sec.start <= m.start < sec.end]
        cursor = sec.start
        tables: list[Unit] = []
        for i, m in enumerate(in_sec):
            limit = in_sec[i + 1].start if i + 1 < len(in_sec) else sec.end
            if m.kind == "genai_subcategory":
                end = genai_table_end(doc.text, m.start, limit)
                kind = "genai_table"
            else:                                     # rmf_table: runs to the next marker / section end
                end = len(doc.text[:limit].rstrip())
                kind = "rmf_table"
            tables.append(Unit(kind, doc.doc_id, m.start, end, sec.index, sec.path, marker_id=m.id))
        for t in tables:
            _add_prose(units, doc, cursor, t.start, sec)
            units.append(t)
            cursor = t.end
        _add_prose(units, doc, cursor, sec.end, sec)
    units.sort(key=lambda u: u.start)
    for i, u in enumerate(units):
        u.sequence = i
    seg = Segmentation(doc, sections, units)
    for u in units:
        if u.kind == "rmf_table":
            seg.categories[u.sequence] = rmf_categories(doc, u)
    return seg


def _add_prose(units: list[Unit], doc: JoinedDoc, a: int, b: int, sec: Section) -> None:
    chunk = doc.text[a:b]
    if not chunk.strip():
        return
    lead = len(chunk) - len(chunk.lstrip())
    trail = len(chunk) - len(chunk.rstrip())
    units.append(Unit("prose", doc.doc_id, a + lead, b - trail, sec.index, sec.path))


def rmf_categories(doc: JoinedDoc, unit: Unit) -> list[Category]:
    """Categories (GOVERN 1, ...) of an rmf_table unit. A category starts at the table row whose
    first cell is a bold category label; rows after it up to the next label belong to it. The first
    category also owns the caption and header rows before it. Spans tile the unit."""
    text = doc.text
    starts: list[tuple[str, int]] = []
    for m in CATEGORY_LABEL_RE.finditer(text, unit.start, unit.end):
        label = f"{m.group(1)} {m.group(2)}"
        if starts and starts[-1][0] == label:
            continue
        starts.append((label, text.rfind("\n", 0, m.start()) + 1))
    cats = []
    for i, (label, s) in enumerate(starts):
        e = starts[i + 1][1] if i + 1 < len(starts) else unit.end
        cats.append(Category(label, unit.start if i == 0 else s, e))
    return cats
