"""Stage T5: add the missing parts of selected split tables (T-D8). Parts are found by metadata, never by position."""
from __future__ import annotations

from indexing.metadata import from_chroma_metadata
from retrieval import config
from retrieval.models import RetrievedChunk


def group_key(meta: dict) -> tuple | None:
    """Identity of a split table: same doc, chunk type and subcategory (GenAI) or category (AI RMF).
    None when the chunk is not a split table (never expanded)."""
    if meta.get("chunk_type") not in config.SIBLING_CHUNK_TYPES or (meta.get("parts_total") or 1) <= 1:
        return None
    scope = meta.get("subcategory") or meta.get("category")
    if scope is None:
        return None
    return (meta["doc_id"], meta["chunk_type"], scope)


def make_chunk(chunk_id: str, meta: dict, document: str | None = None, *, score=None, rank=None,
               added_as_sibling=False, sibling_of=None) -> RetrievedChunk:
    restored = from_chroma_metadata(meta, document)
    return RetrievedChunk(chunk_id=chunk_id, display_text=restored["display_text"], metadata=restored,
                          rerank_score=score, rerank_rank=rank, added_as_sibling=added_as_sibling,
                          sibling_of=sibling_of)


def fetch_parts(store, key: tuple) -> dict[int, dict]:
    """part number -> {"id", "meta", "document"} for every stored part of the split table `key`."""
    doc_id, chunk_type, scope = key
    scope_field = "subcategory" if chunk_type == "genai_action_table" else "category"
    where = {"$and": [{"doc_id": doc_id}, {"chunk_type": chunk_type}, {scope_field: scope}]}
    got = store.get(where=where, include=["metadatas", "documents"])
    return {int(m["part"]): {"id": i, "meta": m, "document": d}
            for i, m, d in zip(got["ids"], got["metadatas"], got["documents"])}


def missing_siblings(store, selected: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    """selected chunk_id -> sibling chunks that must be added because of it (one entry per group, attached to
    the best-ranked selected part). Never returns a chunk that is already selected, never a duplicate."""
    have = {c.chunk_id for c in selected}
    done_groups: set[tuple] = set()
    out: dict[str, list[RetrievedChunk]] = {}
    for chunk in selected:                                  # selected is in rerank order
        key = group_key(chunk.metadata)
        if key is None or key in done_groups:
            continue
        done_groups.add(key)
        parts = fetch_parts(store, key)
        total = int(chunk.metadata["parts_total"])
        added = []
        for part in range(1, total + 1):
            rec = parts.get(part)
            if rec is None:
                raise LookupError(f"{chunk.chunk_id}: part {part} of {total} of {key} not found in the vector store")
            if rec["id"] in have:
                continue
            have.add(rec["id"])
            added.append(make_chunk(rec["id"], rec["meta"], rec["document"], added_as_sibling=True,
                                    sibling_of=chunk.chunk_id))
        if added:
            out[chunk.chunk_id] = added
    return out


def expand(store, selected: list[RetrievedChunk], *, top_n: int | None = None,
           count_toward_top_n: bool | None = None) -> list[RetrievedChunk]:
    """Selected chunks (rerank order) plus their missing siblings, in no particular order (ordering.py orders).

    count_toward_top_n False (Q6a): `selected` is kept whole, siblings are extra.
    True (Q6b): the total is capped at top_n; selected chunks are taken best-first, each together with the
    siblings it needs, and a chunk whose group does not fit is skipped."""
    top_n = top_n or config.TOP_N
    count = config.SIBLINGS_COUNT_TOWARD_TOP_N if count_toward_top_n is None else count_toward_top_n
    extra = missing_siblings(store, selected)
    if not count:
        return list(selected) + [s for c in selected for s in extra.get(c.chunk_id, [])]
    kept: list[RetrievedChunk] = []
    total = 0
    for chunk in selected:
        need = 1 + len(extra.get(chunk.chunk_id, []))
        if total + need > top_n:
            continue
        kept.append(chunk)
        kept += extra.get(chunk.chunk_id, [])
        total += need
    return kept
