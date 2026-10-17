"""`meta.debug` (GI-D6, contents approved in Stage G6 Q3): only when the request has debug=true.

Built from the FINAL graph state. It carries ids, labels, scores, counts, timings and versions: it NEVER contains chunk
text, prompt text or reasoning text (the state's `candidates[].display_text`, `chunks[]`, `generation.final_text` and
`streamed_text` are deliberately not copied)."""
from __future__ import annotations

import json
from functools import lru_cache

MAX_CANDIDATES = 20


@lru_cache(maxsize=1)
def _indexing_version() -> dict:
    try:
        from indexing import config as ic
        m = json.loads(ic.MANIFEST_PATH.read_text(encoding="utf-8"))
        return {"embedding_model": m.get("embedding_model"), "embedding_revision": m.get("model_revision"),
                "chunks_file_sha256": (m.get("chunks_file_sha256") or "")[:12], "chunk_count": m.get("chunk_count")}
    except Exception:                                          # noqa: BLE001 - debug output must never break a response
        return {}


def build_debug(state: dict) -> dict:
    retrieval = state.get("retrieval") or {}
    generation = state.get("generation") or {}
    block: dict = {
        "path": list(state.get("path") or []), "timings_ms": dict(state.get("timings_ms") or {}),
        "route": state.get("route"), "route_source": state.get("route_source"),
        "standalone_query": state.get("standalone_query"),
    }
    if retrieval:
        candidates = sorted(retrieval.get("candidates") or [], key=lambda c: (c.get("rerank_rank") is None, c.get("rerank_rank") or 0))
        block["retrieval"] = {
            "candidate_count": retrieval.get("candidate_count"),
            "candidates": [{"chunk_id": c["chunk_id"], "label": c.get("citation_label"), "rerank_score": c.get("rerank_score")}
                           for c in candidates[:MAX_CANDIDATES]],
            "best_score": retrieval.get("best_score"), "threshold": retrieval.get("threshold"), "relevant": retrieval.get("relevant"),
            "siblings_added": list(retrieval.get("siblings_added") or [])}
    if generation:
        block["generation"] = {k: generation.get(k) for k in ("model", "prompt_version", "reasoning_effort", "temperature", "usage",
                                                              "finish_reason", "fixes", "warnings", "corrected")}
    block["config_version"] = {"retrieval": retrieval.get("config_version"), "indexing": _indexing_version()}
    return block
