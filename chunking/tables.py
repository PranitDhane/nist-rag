"""Stage C6: table chunks. Chunk unit = natural table unit; split only between rows/subcategories;
the header (subcategory line / category text) and the "AI Actor Tasks" line repeat in every piece.

Both documents' tables are noisy in the Markdown (continued rows, icon-font garble, interleaved
cells), so parsing is deliberately forgiving and every repair is logged by the caller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from chunking import config
from chunking.segment import Category, Segmentation, Unit
from chunking.tokens import count_tokens

ACTION_ID_RE = re.compile(r"^(GV|MP|MS|MG)-\d+\.\d+-\d{3}$")
LOOSE_ACTION_ID_RE = re.compile(r"^(GV|MP|MS|MG)-?\d+\.\d+-+\d{3}$")
SUBCAT_LABEL_RE = re.compile(r"(?:GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+:")
FUNCTIONS = {"GOVERN": "GOVERN", "MAP": "MAP", "MEASURE": "MEASURE", "MANAGE": "MANAGE"}
ARTIFACT_RE = re.compile(r"\s{2,}(?:[fi]{1,2}\s{2,})*[fi]{1,2}(?=\s{2,}|\s*\*\*|\s*$|\s*<br>|\s*\|)")


class TableTooBig(Exception):
    """One row (or subcategory) with its header already exceeds the piece limit."""


def split_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return s.split("|")


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|[-:| ]+\|?", line.strip()))


def clean_text(s: str) -> str:
    """Cell text for display: no <br>, no bold markers, icon-font artifacts removed, single spaces."""
    s = s.replace("<br>", " ")
    for _ in range(3):
        s = ARTIFACT_RE.sub("", s)
    s = s.replace("**", "")
    return re.sub(r"\s+", " ", s).strip()


_NOISE = r"(?:[fi]|<br>|\*\*|\s)*"


def _fuzzy(term: str) -> re.Pattern:
    """Regex matching `term` with icon-font noise (stray f/i letters, <br>, spaces) between its letters."""
    parts = []
    for c in term:
        parts.append(r"[-\s]*" if c in " -" else re.escape(c))
    return re.compile(_NOISE.join(parts), re.I)


_ACTOR_PATTERNS = [(name, _fuzzy(name)) for name in config.AI_ACTOR_TASKS]


def canonical_actors(raw_line: str) -> tuple[list[str], bool]:
    """(actor names in order of appearance, repaired?) for a cleaned 'AI Actor Tasks: ...' line."""
    body = raw_line.split(":", 1)[1] if ":" in raw_line else raw_line
    found = []
    for name, pat in _ACTOR_PATTERNS:
        m = pat.search(body)
        if m:
            found.append((m.start(), name))
    names = [n for _, n in sorted(found)]
    listed = [a.strip() for a in body.split(",") if a.strip()]
    return names, listed != names


def clean_risk(s: str) -> str:
    s = re.sub(r"^(?:[fi]{1,2}\s+)+", "", s.strip())
    return re.sub(r"(?:\s+[fi]{1,2})+$", "", s).strip()


def normalize_action_id(raw: str) -> tuple[str, bool]:
    """(canonical ID, was_repaired). The PDF prints GV-4.3-001 as 'GV4.3--001'."""
    raw = raw.strip()
    if ACTION_ID_RE.match(raw):
        return raw, False
    if LOOSE_ACTION_ID_RE.match(raw):
        m = re.match(r"^([A-Z]{2})-?(\d+\.\d+)-+(\d{3})$", raw)
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", True
    return raw, False


# ----------------------------------------------------------------------------- GenAI tables
@dataclass
class ActionRow:
    action_id: str
    raw_id: str
    raw_lines: list[str]
    action: str
    risks: str
    start: int = 0                  # absolute offset of the row's first line


@dataclass
class GenaiTable:
    marker_id: str                  # "GOVERN 1.7"
    statement: str                  # cleaned subcategory line
    header_line: str                # original column header row (or a default)
    rows: list[ActionRow]
    actors_raw: str                 # cleaned "AI Actor Tasks: ..." line, "" if absent
    repairs: list[str] = field(default_factory=list)

    @property
    def function(self) -> str:
        return self.marker_id.split()[0]

    @property
    def actors(self) -> list[str]:
        if ":" not in self.actors_raw:
            return []
        return [a.strip() for a in self.actors_raw.split(":", 1)[1].split(",") if a.strip()]

    @property
    def risks(self) -> list[str]:
        out: list[str] = []
        for r in self.rows:
            out += [clean_risk(x) for x in r.risks.split(";") if x.strip()]
        return list(dict.fromkeys(out))


def parse_genai_table(text: str, marker_id: str, base: int = 0) -> GenaiTable:
    """Parse one GenAI subcategory table unit; `base` is the unit's absolute start offset."""
    offset, lines, starts = 0, [], []
    for l in text.split("\n"):
        if l.strip():
            lines.append(l)
            starts.append(base + offset)
        offset += len(l) + 1
    first = lines[0].strip()
    if first.startswith("|"):                       # header rendered as a one-cell table
        first = first.strip("|")
    statement = clean_text(first)
    header_line = "|**Action ID**|**Suggested Action**|**GAI Risks**|"
    rows: list[ActionRow] = []
    actor_parts: list[str] = []
    repairs: list[str] = []
    in_actor = False
    for line, line_start in zip(lines[1:], starts[1:]):
        s = line.strip()
        if is_separator(s):
            continue
        if re.match(r"^\**\s*AI Actors?\b", s) and not s.startswith("|"):        # paragraph form of the last row
            actor_parts.append(s)
            in_actor = True
            continue
        cells = split_cells(s)
        c0 = clean_text(cells[0])
        if c0 == "Action ID":
            if rows:
                repairs.append(f"repeated header row removed: {s}")
            else:
                header_line = s
            continue
        if re.match(r"^AI Actors?\b", c0) or in_actor or (c0 and not re.match(r"^[A-Z]{2}-?\d", c0) and rows and not cells[0].strip() == ""):
            in_actor = True
            actor_parts.append("".join(cells))       # cells were split mid-word by the table renderer
            continue
        if not c0 and rows:                          # wrapped row: continuation of the previous row
            prev = rows[-1]
            prev.raw_lines.append(s)
            if len(cells) > 1 and cells[1].strip():
                prev.action = (prev.action + " " + clean_text(cells[1])).strip()
            if len(cells) > 2 and cells[2].strip():
                prev.risks = (prev.risks + " " + clean_risk(clean_text(cells[2]))).strip()
            repairs.append(f"wrapped row joined into {prev.action_id}")
            continue
        aid, fixed = normalize_action_id(c0)
        if fixed:
            repairs.append(f"action ID repaired: {c0!r} -> {aid!r}")
        rows.append(ActionRow(aid, c0, [s], clean_text(cells[1]) if len(cells) > 1 else "",
                              clean_risk(clean_text(cells[2])) if len(cells) > 2 else "", line_start))
    actors = re.sub(r"\s+", " ", ARTIFACT_RE.sub("", "".join(actor_parts)).replace("**", "")).strip()
    actors = re.sub(r"^(AI Actors?(?: Tasks?)?)\s*:?\s*", "", actors) if actors else ""
    actors_line = f"AI Actor Tasks: {actors}" if actors else ""
    if actors_line:
        names, repaired = canonical_actors(actors_line)
        if names and repaired:
            repairs.append(f"AI actor line repaired: {actors_line!r} -> {names}")
            actors_line = "AI Actor Tasks: " + ", ".join(names)
    return GenaiTable(marker_id, statement, header_line, rows, actors_line, repairs)


def render_genai_piece(t: GenaiTable, rows: list[ActionRow], style: str) -> str:
    """style 'md': Markdown rows as in the PDF extraction; 'lines': one labelled line per row."""
    if style == "md":
        body = [f"**{t.marker_id}:** {t.statement.split(':', 1)[1].strip()}" if ":" in t.statement else t.statement,
                "", t.header_line, "|---|---|---|"]
        for r in rows:
            body.extend(r.raw_lines)
        if t.actors_raw:
            body.append(f"|**{t.actors_raw}**||")
        return "\n".join(body)
    body = [t.statement]
    for r in rows:
        line = f"{r.action_id} — Suggested action: {r.action}"
        if r.risks:
            line += f" — GAI risks: {r.risks}"
        body.append(line)
    if t.actors_raw:
        body.append(t.actors_raw)
    return "\n".join(body)


# ----------------------------------------------------------------------------- AI RMF categories
@dataclass
class Subcategory:
    label: str                      # "GOVERN 1.1"
    text: str                       # cleaned text without the label
    raw_start: int                  # absolute offset of the label in the joined text


@dataclass
class RmfCategory:
    label: str                      # "GOVERN 1"
    text: str                       # cleaned category description ("" if the PDF layout interleaved it)
    subcategories: list[Subcategory]
    span: tuple[int, int]
    table_id: str
    layout: str                     # 'rows' | 'mashed'


def parse_rmf_category(doc_text: str, cat: Category, table_id: str) -> RmfCategory:
    raw = doc_text[cat.start:cat.end]
    col1: list[str] = []
    body: list[str] = []
    saw_table_row = False
    for line in raw.split("\n"):
        s = line.strip()
        if not s or is_separator(s) or re.match(r"^\|\s*\*\*Categories\*\*\s*\|", s):
            continue
        if s.startswith("|"):
            saw_table_row = True
            cells = split_cells(s)
            col1.append(cells[0])
            seg = " ".join(c for c in cells[1:] if c.strip())
        else:
            seg = s
        body.append(seg)
    joined = "\n".join(body)
    plain_joined = joined.replace("**", "").replace("<br>", " ")
    subs: list[Subcategory] = []
    labels = list(SUBCAT_LABEL_RE.finditer(plain_joined))
    for i, m in enumerate(labels):
        end = labels[i + 1].start() if i + 1 < len(labels) else len(plain_joined)
        subs.append(Subcategory(m.group(0)[:-1], clean_text(plain_joined[m.end():end]), 0))
    # absolute offsets of subcategory labels in the raw text (first occurrence order)
    cursor = cat.start
    for sub in subs:
        m = re.compile(r"\*\*\s*" + re.escape(sub.label) + r"|" + re.escape(sub.label) + r":").search(doc_text, cursor, cat.end)
        sub.raw_start = m.start() if m else cursor
        cursor = sub.raw_start + 1
    cat_text = clean_text(" ".join(col1))
    cat_text = re.sub(r"^" + re.escape(cat.label) + r":?\s*", "", cat_text) if saw_table_row else ""
    layout = "rows" if saw_table_row and cat_text else "mashed"
    return RmfCategory(cat.label, cat_text, subs, (cat.start, cat.end), table_id, layout)


def render_rmf_piece(table_caption: str, cat: RmfCategory, subs: list[Subcategory], style: str, raw: str | None = None) -> str:
    head = f"{cat.label}: {cat.text}".strip()
    if style == "md" and raw is not None:
        return raw
    lines = [head] + [f"{s.label}: {s.text}" for s in subs]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- piece fitting
def partition_min_max(costs: list[int], n: int) -> list[int]:
    """Split `costs` into n consecutive groups minimising the largest group sum; returns group sizes."""
    lo, hi = max(costs), sum(costs)
    while lo < hi:
        mid = (lo + hi) // 2
        groups, cur = 1, 0
        for c in costs:
            if cur + c > mid:
                groups, cur = groups + 1, c
            else:
                cur += c
        lo, hi = (lo, mid) if groups <= n else (mid + 1, hi)
    sizes, cur, size = [], 0, 0
    remaining_groups = n
    for idx, c in enumerate(costs):
        items_left = len(costs) - idx
        if size and (cur + c > lo or items_left < remaining_groups):
            sizes.append(size)
            remaining_groups -= 1
            cur, size = 0, 0
        cur += c
        size += 1
    sizes.append(size)
    return sizes


def fit_groups(n_rows: int, render, limit: int) -> list[tuple[int, int]]:
    """Fewest consecutive row groups whose rendered text is <= limit tokens (exact count).
    `render(i, j)` returns the text for rows i..j-1. Raises TableTooBig if one row alone is too large."""
    single = [count_tokens(render(i, i + 1)) for i in range(n_rows)]
    if max(single) > limit:
        raise TableTooBig(f"row {single.index(max(single))} alone is {max(single)} tokens (limit {limit})")
    base = count_tokens(render(0, 0)) if n_rows else 0
    costs = [max(1, s - base) for s in single]
    for n in range(1, n_rows + 1):
        sizes = partition_min_max(costs, n)
        spans, i = [], 0
        for sz in sizes:
            spans.append((i, i + sz))
            i += sz
        if all(count_tokens(render(a, b)) <= limit for a, b in spans):
            return spans
    raise TableTooBig("no partition fits")


# ----------------------------------------------------------------------------- chunk builders
@dataclass
class TableChunk:
    doc_id: str
    chunk_type: str                 # genai_action_table | rmf_core_table
    section_index: int
    section_path: list[str]
    unit_sequence: int
    start: int
    end: int
    text: str
    tokens: int
    part: int
    parts_total: int
    function: str
    category: str | None
    subcategory: str | None
    subcategories: list[str]
    action_ids: list[str]
    gai_risks: list[str]
    ai_actor_tasks: list[str]
    caption: str | None = None      # "Table 1" for AI RMF


def build_genai_chunks(seg: Segmentation) -> tuple[list[TableChunk], list[str]]:
    doc, out, log = seg.doc, [], []
    limit = config.TABLE_PIECE_MAX_TOKENS
    for u in seg.units:
        if u.kind != "genai_table":
            continue
        t = parse_genai_table(doc.text[u.start:u.end], u.marker_id, u.start)
        log += [f"{u.marker_id}: {r}" for r in t.repairs]
        if not t.rows:
            raise TableTooBig(f"{u.marker_id}: no action rows parsed")
        try:
            groups = fit_groups(len(t.rows), lambda i, j: render_genai_piece(t, t.rows[i:j], "lines"), limit)
        except TableTooBig as e:
            raise TableTooBig(f"{doc.doc_id} {u.marker_id}: {e}") from e
        for n, (a, b) in enumerate(groups, start=1):
            rows = t.rows[a:b]
            text = render_genai_piece(t, rows, "lines")
            start = u.start if n == 1 else t.rows[a].start
            end = u.end if n == len(groups) else t.rows[b].start
            out.append(TableChunk(doc.doc_id, "genai_action_table", u.section_index, u.section_path, u.sequence, start, end,
                                  text, count_tokens(text), n, len(groups), t.function, None, u.marker_id, [],
                                  [r.action_id for r in rows], list(dict.fromkeys(
                                      clean_risk(x) for r in rows for x in r.risks.split(";") if x.strip())), t.actors))
    return out, log


def build_rmf_chunks(seg: Segmentation) -> tuple[list[TableChunk], list[str]]:
    doc, out, log = seg.doc, [], []
    limit = config.TABLE_PIECE_MAX_TOKENS
    for u in seg.units:
        if u.kind != "rmf_table":
            continue
        for cat in seg.categories[u.sequence]:
            rc = parse_rmf_category(doc.text, cat, u.marker_id)
            if rc.layout == "mashed":
                log.append(f"{rc.label}: category description not recoverable (interleaved PDF layout); label only")
            subs = rc.subcategories
            render = lambda i, j: render_rmf_piece(u.marker_id, rc, subs[i:j], "lines")
            try:
                groups = fit_groups(len(subs), render, limit) if subs else [(0, 0)]
            except TableTooBig as e:
                raise TableTooBig(f"{doc.doc_id} {rc.label}: {e}") from e
            for n, (a, b) in enumerate(groups, start=1):
                text = render(a, b)
                start = cat.start if n == 1 else subs[a].raw_start
                end = cat.end if n == len(groups) else subs[b].raw_start
                out.append(TableChunk(doc.doc_id, "rmf_core_table", u.section_index, u.section_path, u.sequence, start, end,
                                      text, count_tokens(text), n, len(groups), rc.label.split()[0], rc.label, None,
                                      [s.label for s in subs[a:b]], [], [], [], caption=u.marker_id))
    return out, log
