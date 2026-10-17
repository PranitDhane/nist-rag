"""Stage A2: RetrievalResult -> (sources_block, key_map). Keys C1..Cn follow the GIVEN chunk order (never re-sorted)."""
from __future__ import annotations

import re

from augmentation.models import KeyMap
from retrieval.models import RetrievalResult, RetrievedChunk

TYPE_MAP = {"prose": "prose", "genai_action_table": "table", "rmf_core_table": "table",
            "figure": "figure-ocr", "footnote": "footnote"}
SECTION_SEPARATOR = " > "
_TAG_OPEN = re.compile(r"<(?=/?sources?)", re.IGNORECASE)     # '<source', '</source', '<sources', '</sources'


def neutralize_tags(text: str) -> str:
    """Make literal <source>/<sources> tags in chunk text harmless (< -> ‹). Nothing else in the text changes."""
    return _TAG_OPEN.sub("‹", text)


def escape_attr(value: str) -> str:
    """Attribute values: escape & and \" only."""
    return value.replace("&", "&amp;").replace('"', "&quot;")


def section_attr(meta: dict) -> str:
    parts = list(meta.get("section_path") or [])
    chunk_type = meta.get("chunk_type")
    if chunk_type == "genai_action_table" and meta.get("subcategory"):
        parts.append(meta["subcategory"])
    elif chunk_type == "rmf_core_table" and meta.get("category"):
        parts.append(meta["category"])
    return SECTION_SEPARATOR.join(parts)


def source_block(key: str, chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    attrs = {
        "id": key,
        "doc": meta.get("display_name", ""),
        "section": section_attr(meta),
        "type": TYPE_MAP[meta["chunk_type"]],
        "part": f"{meta.get('part', 1)} of {meta.get('parts_total', 1)}",
    }
    open_tag = "<source " + " ".join(f'{k}="{escape_attr(str(v))}"' for k, v in attrs.items()) + ">"
    return f"{open_tag}\n{neutralize_tags(chunk.display_text)}\n</source>"


def build_context(result: RetrievalResult) -> tuple[str, KeyMap]:
    """Sources block (one <source> per chunk, in the given order) and the key -> chunk_id map.
    Page numbers and URLs are deliberately not attributes (Q2): citations with pages are built in code."""
    key_map: KeyMap = {}
    blocks = []
    for i, chunk in enumerate(result.chunks, start=1):
        key = f"C{i}"
        key_map[key] = chunk.chunk_id
        blocks.append(source_block(key, chunk))
    return "\n\n".join(blocks), key_map
