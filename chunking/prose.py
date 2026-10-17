"""Stage C5: prose chunks - recursive splitting by tokens, ~10% overlap within a section, tiny-section merge.

Custom splitter (user choice Q2): separators in priority order are blank line, line break,
sentence end, space. Every piece is a contiguous slice of the joined text, so its absolute
offsets are exact and no text search is ever needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from chunking import config
from chunking.segment import Segmentation, Unit
from chunking.tokens import count_tokens

SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"”’)\]*]*\s+(?=[A-Z0-9“\"(\[*#])")


@dataclass
class Atom:
    start: int
    end: int
    strength: int          # boundary after this atom: 2 blank line, 1 line break, 0 sentence end, -1 inside a word run
    heading: bool = False


@dataclass
class ProseChunk:
    doc_id: str
    section_index: int
    section_path: list[str]
    spans: list[tuple[int, int]]     # absolute spans making up the text (2+ only for merged chunks)
    text: str
    tokens: int
    unit_sequence: int
    part: int = 1
    parts_total: int = 1
    merged_sections: list[str] = field(default_factory=list)
    overlap_with_previous: bool = False

    @property
    def start(self) -> int:
        return self.spans[0][0]

    @property
    def end(self) -> int:
        return self.spans[-1][1]


def make_atoms(text: str, base: int, max_tokens: int) -> list[Atom]:
    """Split `text` into atoms (sentences / lines), each <= max_tokens, with boundary strengths."""
    atoms: list[Atom] = []
    lines = text.split("\n")
    pos = 0
    for li, line in enumerate(lines):
        line_start = pos
        pos += len(line) + 1
        if not line.strip():
            if atoms and atoms[-1].strength < 2:
                atoms[-1].strength = 2
            continue
        lead = len(line) - len(line.lstrip())
        body = line.strip()
        is_heading = body.startswith("#")
        cuts = [0]
        for m in SENTENCE_END_RE.finditer(body):
            cuts.append(m.end())
        cuts.append(len(body))
        for a, b in zip(cuts, cuts[1:]):
            piece = body[a:b].rstrip()
            if not piece:
                continue
            s = base + line_start + lead + a
            atoms.append(Atom(s, s + len(piece), 0, is_heading and a == 0 and len(cuts) == 2))
        atoms[-1].strength = 1
    return _split_oversized(atoms, base, text, max_tokens)


def _split_oversized(atoms: list[Atom], base: int, text: str, max_tokens: int) -> list[Atom]:
    out: list[Atom] = []
    for a in atoms:
        if count_tokens(text[a.start - base:a.end - base]) <= max_tokens:
            out.append(a)
            continue
        # no sentence/line boundary helps: cut at spaces into runs that fit
        words = [(m.start() + a.start, m.end() + a.start) for m in re.finditer(r"\S+", text[a.start - base:a.end - base])]
        i = 0
        while i < len(words):
            j = i + 1
            while j < len(words) and count_tokens(text[words[i][0] - base:words[j][1] - base]) <= max_tokens:
                j += 1
            out.append(Atom(words[i][0], words[j - 1][1], -1 if j < len(words) else a.strength))
            i = j
    return out


def split_segment(text: str, base: int, max_tokens: int | None = None, min_tokens: int | None = None,
                  overlap_fraction: float | None = None) -> list[tuple[int, int, bool]]:
    """Pieces (abs_start, abs_end, overlaps_previous) of one prose segment starting at absolute `base`.
    Defaults are the prose limits; figure OCR text passes its own."""
    mx = config.PROSE_TARGET_MAX if max_tokens is None else max_tokens
    mn = config.PROSE_TARGET_MIN if min_tokens is None else min_tokens
    overlap = config.PROSE_OVERLAP_FRACTION if overlap_fraction is None else overlap_fraction
    if count_tokens(text) <= mx:
        return [(base, base + len(text), False)]
    atoms = make_atoms(text, base, mx)
    n = len(atoms)

    def tok(i: int, j: int) -> int:       # tokens of atoms i..j-1 as one contiguous slice
        return count_tokens(text[atoms[i].start - base:atoms[j - 1].end - base])

    pieces: list[tuple[int, int, bool]] = []
    i, overlapped = 0, False
    while True:
        j = i + 1
        while j < n and tok(i, j + 1) <= mx:
            j += 1
        if j < n:
            # cut at the strongest boundary that still gives a piece >= minimum
            best = None
            for k in range(j, i, -1):
                if tok(i, k) < mn:
                    break
                if best is None or atoms[k - 1].strength > atoms[best - 1].strength:
                    best = k
            k = best or j
            if atoms[k - 1].heading and k - i > 1:        # never end a piece on a bare heading
                k -= 1
        else:
            k = n
        pieces.append((atoms[i].start, atoms[k - 1].end, overlapped))
        if k >= n:
            return pieces
        # next piece starts ov atoms back so that ~10% of this piece repeats
        target = overlap * tok(i, k)
        ov = 0
        while ov + 1 < k - i and tok(k - ov - 1, k) <= target:
            ov += 1
        if ov and tok(k - ov, k) > target * 1.5 and ov > 1:
            ov -= 1
        i, overlapped = k - ov, ov > 0


def _section_labels(path: list[str]) -> str:
    return path[-1] if path else "(document start)"


def build_prose_chunks(seg: Segmentation) -> tuple[list[ProseChunk], list[dict]]:
    """All prose chunks of a document in reading order, plus the merge log."""
    doc = seg.doc
    text = doc.text
    # own prose size per section, for the tiny test
    own: dict[int, int] = {}
    for u in seg.units:
        if u.kind == "prose":
            own[u.section_index] = own.get(u.section_index, 0) + count_tokens(text[u.start:u.end])

    chunks: list[ProseChunk] = []
    tiny_flags: list[bool] = []
    for u in seg.units:
        if u.kind != "prose":
            continue
        pieces = split_segment(text[u.start:u.end], u.start)
        for idx, (a, b, ov) in enumerate(pieces, start=1):
            t = text[a:b]
            chunks.append(ProseChunk(doc.doc_id, u.section_index, u.section_path, [(a, b)], t, count_tokens(t),
                                     u.sequence, idx, len(pieces), overlap_with_previous=ov))
            tiny_flags.append(own[u.section_index] <= config.TINY_SECTION_MAX_TOKENS)
    return merge_tiny(chunks, tiny_flags, doc.text)


def merge_tiny(chunks: list[ProseChunk], tiny: list[bool], text: str) -> tuple[list[ProseChunk], list[dict]]:
    """D5: merge tiny-section chunks forward into the next prose chunk; fall back to backward."""
    log: list[dict] = []
    out: list[ProseChunk] = []
    pending: list[ProseChunk] = []

    def joined(parts: list[ProseChunk], target: ProseChunk, backward: bool) -> tuple[str, int]:
        pieces = [p.text for p in parts]
        body = "\n\n".join(pieces + [target.text] if not backward else [target.text] + pieces)
        return body, count_tokens(body)

    for c, is_tiny in zip(chunks, tiny):
        if is_tiny:
            pending.append(c)
            continue
        if pending:
            body, n = joined(pending, c, backward=False)
            if n <= config.MERGED_MAX_TOKENS:
                merged_names = [_section_labels(p.section_path) for p in pending]
                spans = [s for p in pending for s in p.spans] + c.spans
                log.append({"direction": "forward", "merged": merged_names, "into": _section_labels(c.section_path),
                            "tokens": n, "first_page_span_start": spans[0][0]})
                c = ProseChunk(c.doc_id, c.section_index, c.section_path, spans, body, n, c.unit_sequence, c.part,
                               c.parts_total, c.merged_sections + merged_names, c.overlap_with_previous)
            else:
                log.append({"direction": "not_merged_too_large", "kept_small": [_section_labels(p.section_path) for p in pending],
                            "would_be_tokens": n})
                out.extend(pending)
            pending = []
        out.append(c)
    if pending:                                       # no next prose chunk: merge backward into the previous one
        prev = out[-1] if out else None
        if prev is not None:
            body, n = joined(pending, prev, backward=True)
            if n <= config.MERGED_MAX_TOKENS:
                names = [_section_labels(p.section_path) for p in pending]
                log.append({"direction": "backward", "merged": names, "into": _section_labels(prev.section_path), "tokens": n})
                idx = out.index(prev)
                out[idx] = ProseChunk(prev.doc_id, prev.section_index, prev.section_path,
                                      prev.spans + [s for p in pending for s in p.spans], body, n, prev.unit_sequence,
                                      prev.part, prev.parts_total, prev.merged_sections + names, prev.overlap_with_previous)
                pending = []
            else:
                log.append({"direction": "not_merged_too_large", "kept_small": [_section_labels(p.section_path) for p in pending],
                            "would_be_tokens": n})
        out.extend(pending)
    return out, log
