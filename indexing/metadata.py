"""Stage E3: chunk <-> Chroma metadata, without losing information (EMBEDDING_INDEXING_PLAN Stage E3).

Chroma metadata values must be str/int/float/bool and never None. So:
  * scalars are kept as-is (null scalars are left out; a missing key means null on the way back);
  * lists are stored as JSON strings (user choice Q4) for an exact round trip;
  * `display_text` goes in metadata, `embed_text` is the Chroma document (user choice Q6).
"""
from __future__ import annotations

import json

SCALAR_FIELDS = (
    "chunk_id", "doc_id", "display_name", "chunk_type", "sequence", "part", "parts_total",
    "char_start", "char_end", "pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end",
    "function", "category", "subcategory", "figure_number", "ocr_confidence", "footnote_number",
    "section_path_is_page_level", "token_count_embed", "source_url", "citation_url", "citation_label",
    "display_text",
)
# chunk list field -> metadata key holding its JSON
LIST_FIELDS = {
    "section_path": "section_path_json", "merged_sections": "merged_sections_json",
    "subcategories": "subcategories_json", "action_ids": "action_ids_json", "gai_risks": "gai_risks_json",
    "ai_actor_tasks": "ai_actor_tasks_json", "trustworthy_characteristics": "trustworthy_characteristics_json",
    "links": "links_json", "char_spans": "char_spans_json",
}
# derived, for filtering/readability only; ignored when converting back
DERIVED_FIELDS = ("section_path_str", "section_top", "section_leaf")
PATH_SEPARATOR = " > "
VALID_TYPES = (str, int, float, bool)


def to_chroma_metadata(chunk: dict) -> dict:
    """Chroma-compatible metadata for one chunk record."""
    meta: dict = {}
    for key in SCALAR_FIELDS:
        value = chunk.get(key)
        if value is not None:
            meta[key] = value
    path = chunk.get("section_path") or []
    meta["section_path_str"] = PATH_SEPARATOR.join(path)
    if path:
        meta["section_top"], meta["section_leaf"] = path[0], path[-1]
    for field, key in LIST_FIELDS.items():
        value = chunk.get(field)
        if value is not None:                 # char_spans is null for figure/footnote chunks
            meta[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    bad = {k: type(v).__name__ for k, v in meta.items() if not isinstance(v, VALID_TYPES)}
    if bad:
        raise TypeError(f"{chunk.get('chunk_id')}: metadata values Chroma cannot store: {bad}")
    return meta


def from_chroma_metadata(meta: dict, document: str | None = None) -> dict:
    """Reverse of to_chroma_metadata. A missing scalar key is null; a missing list key is [] (char_spans: null).
    `document` is the Chroma document, i.e. the chunk's embed_text."""
    chunk: dict = {key: meta.get(key) for key in SCALAR_FIELDS}
    for field, key in LIST_FIELDS.items():
        if key in meta:
            chunk[field] = json.loads(meta[key])
        else:
            chunk[field] = None if field == "char_spans" else []
    chunk["embed_text"] = document
    return chunk
