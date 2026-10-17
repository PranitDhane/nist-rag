"""Stage T6/T8: retrieval evaluation. `python -m eval.retrieval_eval [--cases PATH] [--out PATH]`.

No LLM calls. Needs the real indexes and the reranker. Gold labels are structural (see retrieval_cases.yaml);
this script never edits them and a miss never fails the run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from retrieval import config

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "retrieval_cases.yaml"
DEFAULT_REPORT = EVAL_DIR / "retrieval_report.json"
CONFIGS = ("vector_only", "bm25_only", "ensemble", "ensemble_rerank")
TARGET_KEYS = {"doc_id", "section", "subcategory", "action_id", "function", "figure", "footnote", "chunk_type",
               "text_contains"}


# ------------------------------------------------------------------ gold matching (pure)
def matches(chunk: dict, target: dict) -> bool:
    """True if the chunk record satisfies every key of the target."""
    for key, want in target.items():
        if key == "doc_id" and chunk.get("doc_id") != want:
            return False
        if key == "chunk_type" and chunk.get("chunk_type") != want:
            return False
        if key == "section" and not any(str(p).startswith(want) for p in chunk.get("section_path") or []):
            return False
        if key == "subcategory" and want != chunk.get("subcategory") and want not in (chunk.get("subcategories") or []):
            return False
        if key == "action_id" and want not in (chunk.get("action_ids") or []):
            return False
        if key == "function" and chunk.get("function") != want:
            return False
        if key == "figure" and chunk.get("figure_number") != want:
            return False
        if key == "footnote" and chunk.get("footnote_number") != want:
            return False
        if key == "text_contains" and want.lower() not in (chunk.get("embed_text") or "").lower():
            return False
    return True


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(matches(chunk, t) for t in gold)


def first_hit_rank(chunk_ids: list[str], lookup: dict, gold: list[dict]) -> int | None:
    for rank, cid in enumerate(chunk_ids, start=1):
        if is_hit(lookup[cid], gold):
            return rank
    return None


def load_cases(path: Path = DEFAULT_CASES) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    for c in cases:
        c.setdefault("answerable", True)
        if c["answerable"] and not c.get("gold"):
            raise ValueError(f"{c['id']}: answerable case without gold")
        for t in c.get("gold") or []:
            unknown = set(t) - TARGET_KEYS
            if unknown or "doc_id" not in t:
                raise ValueError(f"{c['id']}: bad gold target {t}")
    return cases


def validate_gold(cases: list[dict], lookup: dict) -> list[str]:
    """Every gold target must match at least one chunk in the corpus (otherwise the label itself is wrong)."""
    problems = []
    for c in cases:
        for t in c.get("gold") or []:
            if not any(matches(rec, t) for rec in lookup.values()):
                problems.append(f"{c['id']}: gold target matches no chunk: {t}")
    return problems


# ------------------------------------------------------------------ metrics (pure)
def summarize(per_case: list[dict], config_name: str) -> dict:
    ans = [c for c in per_case if c["answerable"]]
    ranks = [c["configs"][config_name]["first_hit_rank"] for c in ans]
    hits = [r for r in ranks if r is not None]
    return {"hit_at_5": round(len(hits) / len(ans), 4) if ans else None,
            "mrr": round(sum(1 / r for r in hits) / len(ans), 4) if ans else None,
            "hits": len(hits), "answerable_cases": len(ans)}


def threshold_table(answerable_best: list[float], unanswerable_best: list[float], steps: int = 20) -> dict:
    """For `steps` evenly spaced thresholds over the observed score range: % of answerable kept (best >= t) and
    % of unanswerable rejected (best < t)."""
    scores = answerable_best + unanswerable_best
    if not scores:
        return {"rows": []}
    lo, hi = min(scores), max(scores)
    rows = []
    for i in range(steps):
        t = lo + (hi - lo) * i / (steps - 1) if steps > 1 else lo
        kept = sum(s >= t for s in answerable_best)
        rejected = sum(s < t for s in unanswerable_best)
        rows.append({"threshold": round(t, 4),
                     "answerable_kept_pct": round(100 * kept / len(answerable_best), 1) if answerable_best else None,
                     "unanswerable_rejected_pct": round(100 * rejected / len(unanswerable_best), 1)
                     if unanswerable_best else None})
    stat = lambda xs: {"best": round(max(xs), 4), "worst": round(min(xs), 4)} if xs else None  # noqa: E731
    return {"rows": rows, "answerable_scores": stat(answerable_best), "unanswerable_scores": stat(unanswerable_best)}


def latency_stats(values: list[float]) -> dict:
    xs = sorted(values)
    pick = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]  # noqa: E731
    return {"n": len(xs), "p50": round(pick(0.5), 1), "p95": round(pick(0.95), 1), "max": round(xs[-1], 1),
            "target_p95_ms": 1500, "meets_target": pick(0.95) <= 1500}


# ------------------------------------------------------------------ the run
def run(cases: list[dict]) -> dict:
    from indexing.bm25_index import tokenize
    from retrieval.components import get_components
    from retrieval.fusion import fuse
    from retrieval.rerank import count_truncated, load_reranker, rerank_text, score_candidates

    comps = get_components()
    lookup = comps.chunk_lookup
    problems = validate_gold(cases, lookup)
    if problems:
        raise SystemExit("gold labels do not match the corpus:\n  " + "\n  ".join(problems))
    reranker = load_reranker()
    per_case, truncated, pairs = [], 0, 0
    for case in cases:
        q = case["query"]
        vec = [d.metadata["chunk_id"] for d in comps.vector.invoke(q)]
        bm = [d.metadata["chunk_id"] for d in comps.bm25.invoke(q)]
        fused = [d.metadata["chunk_id"] for d in fuse(comps.ensemble, comps.vector.invoke(q), comps.bm25.invoke(q))]
        scored = score_candidates(reranker, q, fused, lookup)
        truncated += count_truncated(reranker, q, [rerank_text(lookup[c]) for c in fused])
        pairs += len(fused)
        tops = {"vector_only": vec[:5], "bm25_only": bm[:5], "ensemble": fused[:5],
                "ensemble_rerank": [s.chunk_id for s in scored[:5]]}
        nonzero = int((comps.bm25.vectorizer.get_scores(tokenize(q)) > 0).sum())
        row = {"id": case["id"], "kind": case["kind"], "query": q, "answerable": case["answerable"],
               "best_score": round(scored[0].score, 4) if scored else None,
               "candidate_count": len(fused), "bm25_padding": max(0, config.BM25_K - nonzero),
               "top5_rerank": [{"chunk_id": s.chunk_id, "score": round(s.score, 4), "fused_pos": s.fused_position,
                                "label": lookup[s.chunk_id]["citation_label"]} for s in scored[:5]],
               "configs": {}}
        if case["answerable"]:
            gold = case["gold"]
            row["recall_at_20_fused"] = first_hit_rank(fused[:20], lookup, gold) is not None
            for name, ids in tops.items():
                row["configs"][name] = {"first_hit_rank": first_hit_rank(ids, lookup, gold)}
            row["found_by"] = [n for n in CONFIGS if row["configs"][n]["first_hit_rank"] is not None]
        per_case.append(row)

    from retrieval.pipeline import retrieve, warmup
    warmup()
    totals = [retrieve(case["query"]).timings_ms["total"]
              for case in cases]
    ans = [c for c in per_case if c["answerable"]]
    unans = [c for c in per_case if not c["answerable"]]
    summary = {name: summarize(per_case, name) for name in CONFIGS}
    summary["recall_at_20_fused"] = {"found": sum(c["recall_at_20_fused"] for c in ans), "of": len(ans),
                                     "pct": round(100 * sum(c["recall_at_20_fused"] for c in ans) / len(ans), 1)}
    return {
        "config_version": config.config_version(), "config": config.as_dict(),
        "cases": len(per_case), "answerable": len(ans), "unanswerable": len(unans),
        "summary": summary,
        "thresholds": threshold_table([c["best_score"] for c in ans], [c["best_score"] for c in unans]),
        "latency_ms": latency_stats(totals),
        "bm25_padding_cases": sum(1 for c in per_case if c["bm25_padding"] > 0),
        "truncated_pairs": {"truncated": truncated, "pairs": pairs, "limit_tokens": config.RERANKER_MAX_LENGTH},
        "failures": {name: [c["id"] for c in ans if c["configs"][name]["first_hit_rank"] is None] for name in CONFIGS},
        "per_case": per_case,
    }


def format_report(rep: dict) -> str:
    s = rep["summary"]
    out = [f"config_version {rep['config_version']}   cases {rep['cases']} (answerable {rep['answerable']}, "
           f"unanswerable {rep['unanswerable']})", "",
           "configuration        Hit@5   MRR     found/answerable",
           *[f"{n:<20} {s[n]['hit_at_5']:<7} {s[n]['mrr']:<7} {s[n]['hits']}/{s[n]['answerable_cases']}" for n in CONFIGS],
           f"fused Recall@20: {s['recall_at_20_fused']['found']}/{s['recall_at_20_fused']['of']} "
           f"({s['recall_at_20_fused']['pct']}%)", "",
           "threshold   answerable kept %   unanswerable rejected %"]
    out += [f"{r['threshold']:<11} {r['answerable_kept_pct']:<19} {r['unanswerable_rejected_pct']}"
            for r in rep["thresholds"]["rows"]]
    t = rep["thresholds"]
    out += ["", f"best scores  answerable {t['answerable_scores']}   unanswerable {t['unanswerable_scores']}",
            f"latency retrieve(): {rep['latency_ms']}",
            f"BM25 zero-score padding present in {rep['bm25_padding_cases']} of {rep['cases']} queries",
            f"truncated pairs: {rep['truncated_pairs']['truncated']} of {rep['truncated_pairs']['pairs']} "
            f"(limit {rep['truncated_pairs']['limit_tokens']} tokens)", "", "failed cases (answerable, no gold in top 5):"]
    for name in CONFIGS:
        out.append(f"  {name}: {rep['failures'][name] or 'none'}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args(argv)
    rep = run(load_cases(args.cases))
    args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(rep))
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
