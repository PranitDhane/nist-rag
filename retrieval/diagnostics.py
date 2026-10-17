"""Stage T3: diagnostic mode. Evaluation/debugging only; the normal query path never calls this."""
from __future__ import annotations

from collections import defaultdict

from indexing.bm25_index import tokenize
from retrieval import config
from retrieval.fusion import fuse


def independent_rrf(rank_lists: list[list[str]], weights: list[float], c: int) -> dict[str, float]:
    """Weighted RRF recomputed from scratch: sum over retrievers of weight / (rank + c), rank 1-based."""
    score: dict[str, float] = defaultdict(float)
    for ids, w in zip(rank_lists, weights):
        for rank, cid in enumerate(ids, start=1):
            score[cid] += w / (rank + c)
    return dict(score)


def diagnose(comps, query: str) -> dict:
    """Per-candidate vector rank, BM25 rank (and BM25 raw score), fused position and independent RRF value.

    `order_matches` says whether sorting by the independent RRF value (ties: first seen) reproduces the
    ensemble's order; a mismatch is reported, never corrected."""
    vec = comps.vector.invoke(query)
    bm = comps.bm25.invoke(query)
    fused = fuse(comps.ensemble, vec, bm)
    raw = comps.bm25.vectorizer.get_scores(tokenize(query))
    chunk_ids = [d.metadata["chunk_id"] for d in comps.bm25.docs]
    bm_score = dict(zip(chunk_ids, (float(x) for x in raw)))

    v_ids = [d.metadata["chunk_id"] for d in vec]
    b_ids = [d.metadata["chunk_id"] for d in bm]
    f_ids = [d.metadata["chunk_id"] for d in fused]
    rrf = independent_rrf([v_ids, b_ids], config.WEIGHTS, config.RRF_C)
    by_rrf = sorted(f_ids, key=lambda cid: -rrf[cid])        # sorted() is stable: ties keep ensemble order
    label = {d.metadata["chunk_id"]: d.metadata.get("citation_label") for d in vec + bm}
    rows = []
    for pos, cid in enumerate(f_ids, start=1):
        rows.append({
            "fused_pos": pos, "chunk_id": cid, "citation_label": label[cid],
            "vector_rank": v_ids.index(cid) + 1 if cid in v_ids else None,
            "bm25_rank": b_ids.index(cid) + 1 if cid in b_ids else None,
            "bm25_score": bm_score[cid] if cid in b_ids else None,
            "bm25_padding": cid in b_ids and bm_score[cid] <= 0.0,   # BM25Retriever returned it with score 0
            "rrf": rrf[cid],
        })
    return {"query": query, "candidates": rows, "order_matches": by_rrf == f_ids,
            "n_vector": len(v_ids), "n_bm25": len(b_ids), "n_fused": len(f_ids)}


def format_table(diag: dict) -> str:
    lines = [f"query: {diag['query']}   (fused={diag['n_fused']}, independent RRF reproduces order: "
             f"{diag['order_matches']})",
             " pos  vec  bm25  bm25score  rrf       chunk_id                                  label"]
    for r in diag["candidates"]:
        pad = "*" if r["bm25_padding"] else " "
        bs = f"{r['bm25_score']:.2f}{pad}" if r["bm25_score"] is not None else "-"
        lines.append(f" {r['fused_pos']:>3}  {r['vector_rank'] or '-':>3}  {r['bm25_rank'] or '-':>4}  {bs:>9}  "
                     f"{r['rrf']:.5f}  {r['chunk_id']:41s} {r['citation_label']}")
    lines.append("* = BM25Retriever padding: score 0, returned only to fill k")
    return "\n".join(lines)
