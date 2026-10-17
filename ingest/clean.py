"""Stage 7: remove noise from page Markdown with named, ordered, tested rules.

Why one function per rule: every correction must be explainable and testable on
its own (plan principle 3). Each rule takes a page dict and returns a new page
dict; it appends its own name to `cleaning_log` only when it changed something,
so the log doubles as the per-rule hit count in the report.

A page dict needs: markdown, printed_page, page_number_strategy, body_font_size,
marker_candidates, bottom_blocks. Rules add `cleaning_log`, `footnotes`, `warnings`.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable

from ingest import config
from ingest.models import Manifest

log = logging.getLogger(__name__)

Page = dict
HEADER_LINES = {"NIST AI 100-1", "AI RMF 1.0"}
FOOTER_RE = re.compile(r"^Page [0-9ivx]+$")
FOOTNOTE_MAX_RATIO = 0.75   # footnote numbers are ~6.6pt vs 10-11pt body text
DIGIT_ONLY_SUP_RE = re.compile(r"<sup>\s*\d+(?:\s*,\s*\d+)*\s*</sup>")
TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
SPACES_RE = re.compile("[   -   　]")
ZERO_WIDTH_RE = re.compile("[​﻿]")
GARBLED_TASKS_RE = re.compile(r"AI Actor Tasks(?:\s+[A-Za-z]){1,3}(?=\*\*)")
GARBLED_ID_CELL_RE = re.compile(r"(\|(?:\*\*)?(?:GV|MP|MS|MG)-\d+\.\d+-\d{3})(?:\s+[A-Za-z]{1,2})+(?=(?:\*\*)?\|)")
CONTINUED_CAPTION_RE = re.compile(r"^Table \d+:.*\(Continued\)$")


# ---------------------------------------------------------------- helpers

def _changed(page: Page, name: str, markdown: str) -> Page:
    """New page with `markdown`; logs `name` only if the text really changed."""
    if markdown == page["markdown"]:
        return page
    return {**page, "markdown": markdown, "cleaning_log": [*page["cleaning_log"], name]}


def _warn(page: Page, message: str) -> Page:
    log.warning("clean: %s p.%s: %s", page["doc_id"], page["pdf_page"], message)
    return {**page, "warnings": [*page["warnings"], message]}


def plain(line: str) -> str:
    """A line without Markdown heading/emphasis marks, for exact comparisons."""
    return re.sub(r"\s+", " ", line.replace("#", " ").replace("*", " ")).strip()


def _nonempty(lines: list[str]) -> list[int]:
    return [i for i, l in enumerate(lines) if l.strip()]


def init_page(record: dict, page_number_strategy: str) -> Page:
    return {**record, "page_number_strategy": page_number_strategy,
            "cleaning_log": [], "footnotes": [], "warnings": []}


# ------------------------------------------------------------------ rules

def normalize_unicode(page: Page) -> Page:
    """NFKC (ligature 'fi' -> 'fi'), odd spaces -> ' '; zero-width characters are deleted
    (turning them into spaces would split words)."""
    md = unicodedata.normalize("NFKC", page["markdown"])
    md = ZERO_WIDTH_RE.sub("", SPACES_RE.sub(" ", md))
    return _changed(page, "normalize_unicode", md)


def strip_running_headers_footers(page: Page) -> Page:
    """Drop the running header/footer so it never pollutes retrieval.

    page_label documents (AI RMF): header lines are the exact strings, only within the first 3
    non-empty lines (order differs between pages); the footer is 'Page N', or a bare line equal
    to the printed label (p.4 shows a bare 'i').
    first_line_offset documents (GenAI): a lone line equal to the printed page. pymupdf4llm puts
    it LAST, not first as the plan assumed, so first and last non-empty lines are both checked
    and at most one is removed. Matching the computed printed page (never 'any digits') keeps
    real numbers such as a table value safe.
    """
    lines = page["markdown"].split("\n")
    idx = _nonempty(lines)
    if not idx:
        return page
    drop: set[int] = set()
    printed = page.get("printed_page")
    if page["page_number_strategy"] == "page_label":
        drop |= {i for i in idx[:3] if plain(lines[i]) in HEADER_LINES}
        last = plain(lines[idx[-1]])
        if FOOTER_RE.match(last) or (printed and last == printed):
            drop.add(idx[-1])
    elif printed:
        for i in (idx[-1], idx[0]):
            if plain(lines[i]) == printed:
                drop.add(i)
                break
    return _changed(page, "strip_running_headers_footers",
                    "\n".join(l for i, l in enumerate(lines) if i not in drop))


def _squash(text: str) -> tuple[str, list[int]]:
    """Text without whitespace, tags and Markdown emphasis/quote marks, plus each kept
    character's index in the original. Lets a footnote be found in the Markdown whatever
    its line breaks, <u>/<br> tags or '> ' quoting look like."""
    keep: list[str] = []
    index: list[int] = []
    i = 0
    while i < len(text):
        m = TAG_RE.match(text, i)
        if m:
            i = m.end()
            continue
        ch = text[i]
        if not (ch.isspace() or ch in "*_>"):
            keep.append(ch)
            index.append(i)
        i += 1
    return "".join(keep), index


def _is_footnote_block(block: dict, body: float | None) -> bool:
    first = block["first_span_text"].strip()
    return bool(body) and first.isdigit() and block["first_span_size"] <= FOOTNOTE_MAX_RATIO * body


def extract_footnotes(page: Page) -> Page:
    """Move footnote text out of the Markdown into `footnotes` (GenAI).

    Footnote blocks come from font size (small digit first span), not position. The text is
    located in the Markdown by a whitespace/tag-insensitive substring match, because
    pymupdf4llm renders footnotes as '> N text', plain paragraphs, or two footnotes in one
    paragraph. If a footnote is not found, it stays in place and a WARNING is logged.
    """
    body = page.get("body_font_size")
    md = page["markdown"]
    notes: list[dict] = []
    out = page
    for block in page["bottom_blocks"]:
        if not _is_footnote_block(block, body):
            continue
        number = block["first_span_text"].strip()
        raw = unicodedata.normalize("NFKC", block["text"]).strip()
        text = re.sub(r"\s+", " ", raw[len(number):]).strip()
        target = _squash(number + text)[0]
        squashed, index = _squash(md)
        pos = squashed.find(target)
        if not target or pos < 0:
            out = _warn(out, f"footnote {number} text not found in Markdown; left in place")
            continue
        md = md[:index[pos]] + md[index[pos + len(target) - 1] + 1:]
        notes.append({"number": number, "text": text})
    if not notes:
        return out
    md = re.sub(r"^[ \t]*>[ \t]*\n?", "", md, flags=re.M) if re.search(r"^[ \t]*>[ \t]*$", md, re.M) else md
    out = _changed(out, "extract_footnotes", md)
    return {**out, "footnotes": notes}


def _marker_candidates_outside_footnotes(page: Page) -> int:
    """Small digit spans in running text: footnote numbers at the page bottom are excluded
    (they sit inside a footnote block's bbox and are not <sup> markers in the Markdown)."""
    body = page.get("body_font_size")
    boxes = [b["bbox"] for b in page["bottom_blocks"] if _is_footnote_block(b, body)]
    def inside(c: list[float], b: list[float]) -> bool:
        return c[0] >= b[0] - 1 and c[1] >= b[1] - 1 and c[2] <= b[2] + 1 and c[3] <= b[3] + 1
    return sum(1 for c in page["marker_candidates"] if not any(inside(c["bbox"], b) for b in boxes))


def strip_footnote_markers(page: Page) -> Page:
    """Remove <sup> tags that hold ONLY digits (1 or 1,2), i.e. footnote markers.

    Digit-only content is the test, not the tag: pymupdf4llm also wraps ordinary 10pt table
    cells in <sup> (superscript flag false positive), and those must keep their text (rule 5).
    The count removed is cross-checked against the font-size marker candidates from Stage 4.
    """
    md, n = DIGIT_ONLY_SUP_RE.subn("", page["markdown"])
    out = _changed(page, "strip_footnote_markers", md)
    expected = _marker_candidates_outside_footnotes(page)
    if n != expected:
        out = _warn(out, f"footnote markers removed ({n}) != font-size marker candidates ({expected})")
    return out


def unwrap_markup_tags(page: Page) -> Page:
    """Remove <u>, </u>, <sup>, </sup> but keep the text inside (fixes false-superscript cells)."""
    return _changed(page, "unwrap_markup_tags", re.sub(r"</?(?:u|sup)>", "", page["markdown"]))


@lru_cache(maxsize=1)
def load_protect_list(path: Path | None = None) -> frozenset[str]:
    """Approved real hyphenated words (ingest/hyphen_protect.txt), lowercase, '#' comments allowed."""
    path = path or config.HYPHEN_PROTECT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"clean: hyphen protect-list {path} is missing; restore it from the repo.")
    words = {l.strip().lower() for l in path.read_text(encoding="utf-8").splitlines()}
    return frozenset(w for w in words if w and not w.startswith("#"))


HYPHEN_JOIN_RE = re.compile(r"(?P<pre>[A-Za-z]+)-(?:<br>|\n)(?P<post>[A-Za-z0-9][A-Za-z]*)")


def join_hyphenated(pre: str, post: str, protect: frozenset[str]) -> str:
    """Join the word fragments around a line-end hyphen ('pre-' + 'post').

    Letter + hyphen + lowercase: a soft hyphen, so drop it unless the word is on the
    protect-list. Letter + hyphen + digit or uppercase is a real hyphen ('GV-' + '1.1-001').
    """
    if not post[0].islower():
        return f"{pre}-{post}"
    return f"{pre}-{post}" if f"{pre}-{post}".lower() in protect else f"{pre}{post}"


def fix_line_hyphenation(page: Page, protect: frozenset[str] | None = None) -> Page:
    """Join words broken across a line ('-<br>' or '-\\n').

    'or-<br>ganizational' -> 'organizational' (soft hyphen dropped unless protected),
    'GV-' + '1.1-001' -> 'GV-1.1-001', 'human-' + 'AI' -> 'human-AI' (see join_hyphenated).
    """
    protect = load_protect_list() if protect is None else protect
    return _changed(page, "fix_line_hyphenation", HYPHEN_JOIN_RE.sub(
        lambda m: join_hyphenated(m.group("pre"), m.group("post"), protect), page["markdown"]))


def strip_table_leftovers(page: Page) -> Page:
    """Remove page-break residue inside tables, keep real table header rows.

    Removes 'Continued on next page', repeated captions ending '(Continued)' and a plain-text
    'Categories Subcategories' line. Also repairs the garbled 'AI Actor Tasks   i' cell (stray
    letters from an icon font): the letters go; a ':' is added only when the row does not
    already carry one at the start of the next cell ('|**: ...'). The same stray letters
    trail some action-ID cells ('|MP-3.4-004   i      i|'); they are removed so the ID is
    the whole first cell.
    """
    out_lines: list[str] = []
    for line in page["markdown"].split("\n"):
        p = plain(line)
        if p == "Continued on next page" or CONTINUED_CAPTION_RE.match(p) or p == "Categories Subcategories":
            continue
        if GARBLED_TASKS_RE.search(line):
            colon = "" if "|**:" in line else ":"
            line = GARBLED_TASKS_RE.sub(f"AI Actor Tasks{colon}", line)
        line = GARBLED_ID_CELL_RE.sub(r"\1", line)
        out_lines.append(line)
    return _changed(page, "strip_table_leftovers", "\n".join(out_lines))


def collapse_whitespace(page: Page) -> Page:
    """3+ blank lines -> 2, trailing spaces trimmed, no blank lines at the page edges."""
    md = "\n".join(l.rstrip() for l in page["markdown"].split("\n"))
    md = re.sub(r"\n{3,}", "\n\n", md).strip("\n")
    return _changed(page, "collapse_whitespace", md)


def append_rotated_text(page: Page) -> Page:
    """Add text that pymupdf4llm dropped because it is rotated (e.g. a vertical figure caption).

    Added AFTER the eight plan rules so the header/footer rule still sees the page edges as
    pymupdf4llm produced them. A paragraph already present in the Markdown is not added twice.
    """
    present = _squash(page["markdown"])[0]
    extra = []
    for para in page.get("rotated_text") or []:
        text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", para)).strip()
        if text and _squash(text)[0][:40] not in present:
            extra.append(text)
    if not extra:
        return page
    md = (page["markdown"].rstrip("\n") + "\n\n" if page["markdown"].strip() else "") + "\n\n".join(extra)
    return _changed(page, "append_rotated_text", md)


RULES: list[Callable[[Page], Page]] = [
    normalize_unicode, strip_running_headers_footers, extract_footnotes, strip_footnote_markers,
    unwrap_markup_tags, fix_line_hyphenation, strip_table_leftovers, collapse_whitespace, append_rotated_text,
]


def clean_page(page: Page) -> Page:
    for rule in RULES:
        page = rule(page)
    return page


def clean_all(manifest: Manifest, classified: list[dict]) -> list[dict]:
    strategy = {d.doc_id: d.page_number_strategy for d in manifest.documents}
    result = [clean_page(init_page(r, strategy[r["doc_id"]])) for r in classified]
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STAGES_DIR / "07_clean.jsonl", "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for doc_id, counts in rule_hit_counts(result).items():
        log.info("clean: %s rule hits (pages changed): %s", doc_id, dict(counts))
    return result


def rule_hit_counts(records: list[dict]) -> dict[str, Counter]:
    """Pages changed per rule, per document (from each page's cleaning_log)."""
    out: dict[str, Counter] = {}
    for r in records:
        out.setdefault(r["doc_id"], Counter()).update(r["cleaning_log"])
    return out


def load_clean() -> list[dict]:
    path = config.STAGES_DIR / "07_clean.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"clean: {path} is missing. Run Stage 7 first (python -m ingest --only-stage 7).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
