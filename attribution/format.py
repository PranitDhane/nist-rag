"""The ONE citation label / URL formatter (INSUFFICIENT_CONTEXT_PLAN Section 4; reused by the attribution phase).

Computed at runtime from structured chunk metadata. Format approved at chunking Q7 and confirmed here (Q6): it equals
the `citation_label` stored in every chunk. Input is a chunk's metadata dict (RetrievedChunk.metadata).
"""
from __future__ import annotations

from functools import lru_cache

from augmentation.context_builder import TYPE_MAP, section_attr
from ingest.config import load_manifest

EN_DASH = "–"


@lru_cache(maxsize=1)
def _manifest():
    return load_manifest()


def doc_name(meta: dict) -> str:
    """Short document name from the manifest's `citation_name` (never derived from display_name by string slicing)."""
    try:
        name = _manifest().get(meta["doc_id"]).citation_name
    except KeyError:
        name = ""
    if not name:
        raise KeyError(f"no citation name for doc_id {meta.get('doc_id')!r}; add `citation_name` to it in ingest/sources.yaml")
    return name


def doc_order(doc_id: str) -> int:
    """Position of the document in the manifest (AI RMF first, then GenAI Profile): the document-order key for Sources."""
    return [d.doc_id for d in _manifest().documents].index(doc_id)


def section_label(meta: dict) -> str:
    """Leaf of the section path: '§5.1 Govern' for numbered sections, the plain title otherwise ('Executive Summary')."""
    path = meta.get("section_path") or []
    if not path:
        return ""
    leaf = path[-1]
    return f"§{leaf}" if leaf[:1].isdigit() else leaf


def pages_text(meta: dict) -> str:
    """Printed page range: 'p. 21' or 'pp. 22–23' (en dash). Null printed page -> 'PDF p. N' (never expected for indexed pages)."""
    first, last = meta.get("printed_page_start"), meta.get("printed_page_end")
    if first is None or last is None:
        pdf_first, pdf_last = meta["pdf_page_start"], meta["pdf_page_end"]
        return f"PDF p. {pdf_first}" if pdf_first == pdf_last else f"PDF pp. {pdf_first}{EN_DASH}{pdf_last}"
    return f"p. {first}" if first == last else f"pp. {first}{EN_DASH}{last}"


def kind(meta: dict) -> str:
    """prose | table | figure-ocr | footnote (same mapping as the prompt's `type` attribute)."""
    return TYPE_MAP[meta["chunk_type"]]


def citation_label(meta: dict) -> str:
    bits = [doc_name(meta)]
    if section := section_label(meta):
        bits.append(section)
    chunk_type = meta["chunk_type"]
    if chunk_type == "genai_action_table":
        bits.append(meta.get("subcategory") or "")
    elif chunk_type == "rmf_core_table":
        bits.append(meta.get("category") or "")
    elif chunk_type == "figure":
        bits.append(f"Fig. {meta.get('figure_number')}")
    elif chunk_type == "footnote":
        bits.append(f"footnote {meta.get('footnote_number')}")
    bits.append(pages_text(meta))
    return ", ".join(bits)


def citation_url(meta: dict) -> str:
    """Always the PDF page index (not the printed number): AI RMF printed 'p. 21' links to '#page=26'."""
    return f"{meta['source_url']}#page={meta['pdf_page_start']}"


def section_text(meta: dict) -> str:
    """Readable section for display: path joined with ' > ' (plus subcategory/category for table chunks)."""
    return section_attr(meta)
