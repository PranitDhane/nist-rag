"""State boundary helpers (GRAPH_INTEGRATION_PLAN Section 4, Stage G1).

Graph state holds PLAIN DATA only. The retrieval and generation modules use dataclasses internally; they are converted
here at the node boundary. (The installed checkpointer can round-trip dataclasses, but warns that unregistered types
'will be blocked in a future version', and plain dicts also go straight into API responses and debug output.)

`retrieval` state value (built across hybrid_search -> rerank -> gate -> expand_context):
    query, candidate_count, config_version, timings_ms,
    candidates:   [{chunk_id, citation_label, fused_position, display_text, rerank_score, rerank_rank}]  (all fused candidates)
    top:          [RetrievedChunk as dict] the reranked top N, best first (before the gate and sibling expansion)
    best_score, threshold, relevant                                                                       (gate result)
    chunks:       [RetrievedChunk as dict]  final ordered context (after sibling expansion)
    siblings_added: [chunk_id]
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from retrieval.models import RetrievalResult, RetrievedChunk

_SCALARS = (type(None), bool, int, float, str)


def chunk_to_dict(chunk: RetrievedChunk) -> dict:
    return asdict(chunk)


def chunk_from_dict(d: dict) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=d["chunk_id"], display_text=d["display_text"], metadata=dict(d["metadata"]),
                          rerank_score=d.get("rerank_score"), rerank_rank=d.get("rerank_rank"),
                          added_as_sibling=bool(d.get("added_as_sibling")), sibling_of=d.get("sibling_of"))


def retrieval_result_from_state(retrieval: dict) -> RetrievalResult:
    """Rebuild the retrieval module's result object (final ordered chunks) from the state value."""
    return RetrievalResult(
        query=retrieval["query"], chunks=[chunk_from_dict(c) for c in retrieval.get("chunks") or []],
        relevant=bool(retrieval.get("relevant")), best_score=retrieval.get("best_score"),
        threshold=retrieval.get("threshold"), candidate_count=retrieval.get("candidate_count", 0),
        timings_ms=dict(retrieval.get("timings_ms") or {}), config_version=retrieval.get("config_version", ""))


def gate_result_from_state(retrieval: dict) -> RetrievalResult:
    """For the gate-failure reply: the reranked top chunks (best first, BEFORE sibling expansion), so insufficient.related
    can list them."""
    return RetrievalResult(query=retrieval["query"], chunks=[chunk_from_dict(c) for c in retrieval.get("top") or []],
                           relevant=bool(retrieval.get("relevant")), best_score=retrieval.get("best_score"),
                           threshold=retrieval.get("threshold"), candidate_count=retrieval.get("candidate_count", 0),
                           timings_ms={}, config_version=retrieval.get("config_version", ""))


def plain_data_violations(value: Any, path: str = "state") -> list[str]:
    """Paths of everything that is not plain data (None/bool/int/float/str, list, dict with str keys). Empty = OK.
    Use on the state without `messages` (those are LangChain message objects the checkpointer handles itself)."""
    if isinstance(value, _SCALARS):
        return []
    if isinstance(value, (list, tuple)):
        return [v for i, item in enumerate(value) for v in plain_data_violations(item, f"{path}[{i}]")]
    if isinstance(value, dict):
        bad = []
        for k, item in value.items():
            if not isinstance(k, str):
                bad.append(f"{path}: non-string key {k!r}")
            bad += plain_data_violations(item, f"{path}.{k}")
        return bad
    return [f"{path}: {type(value).__module__}.{type(value).__name__}"]
