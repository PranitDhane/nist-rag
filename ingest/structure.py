"""Stage 8: label sections, tables and key values on each page. Labels only; no chunking.

Why numbering, not bookmarks or heading levels: pymupdf4llm's '#' levels are inconsistent
and both PDFs' bookmarks contain errors, so the hierarchy comes from the section numbers
themselves (plan Section 5.3). Only real Markdown heading lines or bold-only lines can be
headings, so numbered list items and table cells are never mistaken for sections.
"""
from __future__ import annotations

import json
import logging
import re

from ingest import config
from ingest.clean import plain

log = logging.getLogger(__name__)

PART_RE = re.compile(r"^Part (\d+)\s*:\s*(.*)$")
APPENDIX_RE = re.compile(r"^Appendix ([A-Z])\s*[.:]\s*(.*)$")
SUBAPPENDIX_RE = re.compile(r"^([A-Z]\.\d+(?:\.\d+)*)\.\s+(.+)$")
NUMBERED_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+(.+)$")
BOLD_ONLY_RE = re.compile(r"^\*\*[^*]+\*\*$")
SUBCATEGORY_LINE_RE = re.compile(r"^\|?\s*((?:GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+):")
SEPARATOR_ROW_RE = re.compile(r"^\|[-:| ]+\|$")
SUBCATEGORY_BOLD_RE = re.compile(r"\*\*((?:GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+):\*\*")
TABLE_CAPTION_RE = re.compile(r"^(Table \d+):")
CAPTION_RE = re.compile(r"^(?:Fig\.|Figure|Table)\s*\d+\s*[.:]")
ACTION_ROW_RE = re.compile(r"^\|\s*(?:\*\*)?((?:GV|MP|MS|MG)-\d+\.\d+-\d{3})(?:\*\*)?\s*\|")
TRUSTWORTHY_RE = re.compile(r"^Trustworthy AI Characteristics?:\s*(.*)$")
ROOT_KINDS = {"part", "appendix", "exec"}  # these start a new top-level branch of the path


def parse_heading(text: str) -> tuple[str, str, int, str] | None:
    """(section_id, title, level, kind) for heading text already stripped of '#', '*', spaces."""
    if text == "Executive Summary":
        return "Executive Summary", text, 1, "exec"
    if m := PART_RE.match(text):
        return f"Part {m.group(1)}", m.group(2), 0, "part"
    if m := APPENDIX_RE.match(text):
        return f"Appendix {m.group(1)}", m.group(2), 1, "appendix"
    if m := SUBAPPENDIX_RE.match(text):
        return m.group(1), m.group(2), m.group(1).count(".") + 1, "sub"
    if m := NUMBERED_RE.match(text):
        return m.group(1), m.group(2), m.group(1).count(".") + 1, "number"
    return None


def detect_headings(markdown: str) -> list[dict]:
    """Headings in order, each {section_id, title, level, kind, offset}.

    `offset` is the character position of the heading line (used to place links in a section).
    """
    lines = markdown.split("\n")
    out: list[dict] = []
    pos = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        offset, pos = pos, pos + len(line) + 1
        if not (stripped.startswith("#") or BOLD_ONLY_RE.match(stripped)):
            continue
        parsed = parse_heading(plain(stripped))
        if parsed is None:
            continue
        section_id, title, level, kind = parsed
        if kind == "appendix" and not title:   # AI RMF-style split: title on the next line
            title = next((plain(l) for l in lines[i + 1:] if l.strip()), "")
        out.append({"section_id": section_id, "title": title, "level": level, "kind": kind, "offset": offset})
    return out


def path_label(h: dict) -> str:
    """One path element: 'Part 2', '5 AI RMF Core', 'Appendix A: Title'."""
    if h["kind"] == "part" or h["kind"] == "exec":
        return h["section_id"]
    if h["kind"] == "appendix":
        return f"{h['section_id']}: {h['title']}".rstrip(": ")
    return f"{h['section_id']} {h['title']}"


def update_path(stack: list[dict], h: dict) -> list[dict]:
    """Section stack after heading h: roots reset it; others replace same-or-deeper levels."""
    if h["kind"] in ROOT_KINDS:
        return [h]
    return [s for s in stack if s["level"] < h["level"]] + [h]


def first_content_line(markdown: str) -> str | None:
    return next((l.strip() for l in markdown.split("\n") if l.strip()), None)


def _action_table_follows(lines: list[str], i: int) -> bool:
    """True if the next real line after lines[i] is a table header row containing 'Action ID'.

    Separator rows ('|---|') are skipped: pymupdf4llm sometimes renders a subcategory header
    as a one-cell table, leaving a stray separator before the real table.
    """
    for line in lines[i + 1:]:
        s = line.strip()
        if not s or SEPARATOR_ROW_RE.match(s):
            continue
        return s.startswith("|") and "Action ID" in s
    return False


def table_markers(markdown: str) -> list[dict]:
    """Table starts, each {id, kind, char_offset}. A subcategory header ('GOVERN 1.7') starts a
    table only when an 'Action ID' table follows it (kind 'genai_subcategory'); in AI RMF the
    same bold text is just a table row that pymupdf4llm rendered as a paragraph. AI RMF tables
    start at a 'Table N' caption (kind 'rmf_table'; captions ending '(Continued)' are repeats,
    not starts). `char_offset` is the line's first character in `markdown`."""
    lines = markdown.split("\n")
    out = []
    pos = 0
    for i, line in enumerate(lines):
        offset, pos = pos, pos + len(line) + 1
        p = plain(line)
        if (m := SUBCATEGORY_LINE_RE.match(p)) and _action_table_follows(lines, i):
            out.append({"id": m.group(1), "kind": "genai_subcategory", "char_offset": offset})
        elif (m := TABLE_CAPTION_RE.match(p)) and not p.endswith("(Continued)"):
            out.append({"id": m.group(1), "kind": "rmf_table", "char_offset": offset})
    return out


def continues_table(markdown: str) -> bool:
    """True when the page opens mid-table: first content line is a table row and no marker precedes it."""
    first = first_content_line(markdown)
    return bool(first) and (first.startswith("|") or bool(ACTION_ROW_RE.match(first)))


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _clean_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.replace("<br>", " ").replace("**", "")).strip()


def extract_entities(markdown: str) -> dict:
    """Action IDs (first cell of table rows only), subcategories, GAI risks, trustworthy
    characteristics and captions. Prose mentions of an action ID are deliberately ignored."""
    action_ids: list[str] = []
    risks: list[str] = []
    subs: list[str] = []
    traits: list[str] = []
    captions: list[str] = []
    for line in markdown.split("\n"):
        if m := ACTION_ROW_RE.match(line.strip()):
            action_ids.append(m.group(1))
            cells = _cells(line)
            if len(cells) >= 3:
                risks += [r for r in (_clean_cell(x) for x in cells[2].split(";")) if r]
        p = plain(line)
        if m := TRUSTWORTHY_RE.match(p):
            traits += [t.strip().rstrip(".") for t in m.group(1).split(",") if t.strip()]
        if CAPTION_RE.match(p):
            captions.append(p)
    for m in SUBCATEGORY_BOLD_RE.finditer(markdown):
        subs.append(m.group(1))
    subs += [m.group(1) for line in markdown.split("\n") if (m := SUBCATEGORY_LINE_RE.match(plain(line)))]
    dedupe = lambda xs: list(dict.fromkeys(xs))
    return {"action_ids": action_ids, "subcategories": dedupe(subs), "gai_risks": dedupe(risks),
            "trustworthy_characteristics": dedupe(traits), "captions": dedupe(captions)}


def annotate_document(pages: list[dict]) -> list[dict]:
    """Annotate one document's pages (in page order), carrying the section path across pages."""
    stack: list[dict] = []
    out = []
    for p in pages:
        start = [path_label(s) for s in stack]
        heads = detect_headings(p["markdown"])
        heading_paths = []
        for h in heads:
            stack = update_path(stack, h)
            heading_paths.append({"offset": h["offset"], "path": [path_label(s) for s in stack]})
        md = p["markdown"]
        out.append({**p, "section_path_start": start,
                    "headings": [{**{k: h[k] for k in ("section_id", "title", "level")}, "char_offset": h["offset"],
                                 "section_path": hp["path"]} for h, hp in zip(heads, heading_paths)],
                    "heading_paths": heading_paths,
                    "section_path_end": [path_label(s) for s in stack],
                    "table_markers": table_markers(md), "continues_table": continues_table(md),
                    "entities": extract_entities(md)})
    return out


def path_at(page: dict, offset: int) -> list[str]:
    """Section path in effect at a character offset of the page's Markdown."""
    path = page["section_path_start"]
    for hp in page["heading_paths"]:
        if hp["offset"] <= offset:
            path = hp["path"]
    return path


def structure_all(cleaned: list[dict]) -> list[dict]:
    by_doc: dict[str, list[dict]] = {}
    for r in cleaned:
        by_doc.setdefault(r["doc_id"], []).append(r)
    result: list[dict] = []
    for doc_id, pages in by_doc.items():
        annotated = annotate_document(sorted(pages, key=lambda p: p["pdf_page"]))
        log.info("structure: %s: %d headings, %d table markers, %d continues_table pages, %d table-row action IDs",
                 doc_id, sum(len(p["headings"]) for p in annotated), sum(len(p["table_markers"]) for p in annotated),
                 sum(p["continues_table"] for p in annotated), sum(len(p["entities"]["action_ids"]) for p in annotated))
        result.extend(annotated)
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STAGES_DIR / "08_structure.jsonl", "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return result


def load_structure() -> list[dict]:
    path = config.STAGES_DIR / "08_structure.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"structure: {path} is missing. Run Stage 8 first (python -m ingest --only-stage 8).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
