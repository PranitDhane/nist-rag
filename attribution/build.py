"""Stage S2: build_sources() and the consistency checks (ATTRIBUTION_PLAN Section 4). Deterministic: no LLM, no network."""
from __future__ import annotations

import logging
import re

from attribution import format as fmt
from attribution.models import SourceEntry

log = logging.getLogger(__name__)

MARKER = re.compile(r"\[(C\d+)\]")


_LINE_START_MARKERS = re.compile(r"(?m)^(?:\[C\d+\])+[ \t]*")
_INLINE_MARKERS = re.compile(r"[ \t]*(?:\[C\d+\])+")


def strip_markers(text: str) -> str:
    """Text for conversation history (Q3): keys are local to one answer, so `[C#]` markers are removed together with the
    space before them ('x [C1][C2].' -> 'x.'). The user-facing answer keeps its markers; this is never applied to it."""
    return _INLINE_MARKERS.sub("", _LINE_START_MARKERS.sub("", text))


class AttributionError(Exception):
    """Sources could not be built consistently (a bug upstream, never a user error)."""


def _key_number(key: str) -> int:
    return int(key[1:])


def build_sources(final_text: str, cited_keys: list[str], key_map: dict[str, str], chunk_lookup: dict[str, dict],
                  *, strict: bool = False) -> list[SourceEntry]:
    """One entry per distinct citation label, ordered document -> page -> sequence (SA-D3).

    `cited_keys` must come from the final parse, so they match `final_text`. Only cited chunks are listed.
    strict=True (tests) raises AttributionError on a problem; otherwise the problem is logged at ERROR and the
    list is still returned (the answer text is never touched)."""
    problems: list[str] = []
    merged: dict[str, dict] = {}
    for key in cited_keys:
        chunk_id = key_map.get(key)
        meta = chunk_lookup.get(chunk_id) if chunk_id is not None else None
        if meta is None:
            problems.append(f"{key} has no chunk ({'not in key_map' if chunk_id is None else chunk_id + ' not in chunk_lookup'})")
            continue
        label = fmt.citation_label(meta)
        order = (fmt.doc_order(meta["doc_id"]), meta["pdf_page_start"], meta["pdf_page_end"], meta.get("sequence", 0))
        slot = merged.get(label)
        if slot is None:
            merged[label] = {"keys": {key}, "url": fmt.citation_url(meta), "kind": fmt.kind(meta), "order": order}
            continue
        slot["keys"].add(key)
        if fmt.kind(meta) != slot["kind"]:
            problems.append(f"{label!r} merges chunks of different kinds ({slot['kind']}, {fmt.kind(meta)}); kept {slot['kind']}")
        slot["order"] = min(slot["order"], order)

    entries = [SourceEntry(keys=sorted(s["keys"], key=_key_number), label=label, url=s["url"], kind=s["kind"])
               for label, s in sorted(merged.items(), key=lambda item: (item[1]["order"], item[0]))]
    problems += check_consistency(final_text, entries)
    for p in problems:
        log.error("attribution: %s", p)
    if problems and strict:
        raise AttributionError("; ".join(problems))
    return entries


def check_consistency(final_text: str, entries: list[SourceEntry]) -> list[str]:
    """Problems between the answer text and the Sources list; empty when consistent (plan Section 4)."""
    problems: list[str] = []
    in_text = set(MARKER.findall(final_text))
    seen: dict[str, int] = {}
    for e in entries:
        if not e.keys:
            problems.append(f"entry {e.label!r} has no keys")
        for k in e.keys:
            seen[k] = seen.get(k, 0) + 1
    problems += [f"{k} is in the text but in no entry" for k in sorted(in_text - set(seen), key=_key_number)]
    problems += [f"{k} is in the list but not in the text" for k in sorted(set(seen) - in_text, key=_key_number)]
    problems += [f"{k} appears in {n} entries" for k, n in sorted(seen.items(), key=lambda kv: _key_number(kv[0])) if n > 1]
    labels = [e.label for e in entries]
    problems += [f"label {l!r} is listed twice" for l in sorted({l for l in labels if labels.count(l) > 1})]
    return problems
