"""Stage T5: final order, best first by rerank score (T-D9), with added siblings placed per config (Q5)."""
from __future__ import annotations

from retrieval import config
from retrieval.models import RetrievedChunk
from retrieval.siblings import group_key


def _part(c: RetrievedChunk) -> int:
    return int(c.metadata.get("part") or 1)


def final_order(chunks: list[RetrievedChunk], placement: str | None = None) -> list[RetrievedChunk]:
    """`chunks` = selected (any order) + added siblings.

    selected chunks are sorted by (rerank_rank, chunk_id). Then:
      "group":  each split table is kept together in part order, at the position of its best selected part;
      "append": siblings go after all selected chunks (grouped by the part that caused them, in part order).
    """
    placement = placement or config.SIBLING_PLACEMENT
    selected = sorted((c for c in chunks if not c.added_as_sibling),
                      key=lambda c: (c.rerank_rank if c.rerank_rank is not None else 10**9, c.chunk_id))
    siblings = [c for c in chunks if c.added_as_sibling]
    if placement == "append":
        order = {c.chunk_id: i for i, c in enumerate(selected)}
        return selected + sorted(siblings, key=lambda c: (order.get(c.sibling_of, 10**9), _part(c)))
    if placement != "group":
        raise ValueError(f"unknown SIBLING_PLACEMENT {placement!r}")
    groups: dict[tuple, list[RetrievedChunk]] = {}
    for c in chunks:
        key = group_key(c.metadata)
        if key is not None:
            groups.setdefault(key, []).append(c)
    out: list[RetrievedChunk] = []
    emitted: set[tuple] = set()
    for c in selected:
        key = group_key(c.metadata)
        if key is None:
            out.append(c)
        elif key not in emitted:
            emitted.add(key)
            out += sorted(groups[key], key=_part)
    return out
