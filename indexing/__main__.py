"""python -m indexing [--rebuild] [--only bm25|chroma] [--validate]"""
from __future__ import annotations

import argparse
import logging
import sys

from indexing import config


def print_summary(result: dict) -> None:
    for c in result["checks"]:
        status = "SKIP" if c.skipped else ("PASS" if c.ok else "FAIL")
        print(f"[validate] {status}  {c.number} {c.name}: {c.detail}")
    print("\n[smoke] top-5 per index (soft check; misses do not fail the run)")
    print(f"{'query':58} {'vector':7} {'bm25':6}  expected")
    for s in result["smoke"]:
        rank = "" if s["bm25_rank_ok"] is None else f" (rank-1 {'ok' if s['bm25_rank_ok'] else 'MISS'})"
        print(f"{s['query'][:56]:58} {'HIT' if s['vector_hit'] else 'miss':7} {('HIT' if s['bm25_hit'] else 'miss') + rank:6}  {s['expected']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m indexing", description="Build and validate the Chroma + BM25 indexes.")
    parser.add_argument("--rebuild", action="store_true", help="delete both persisted indexes, then rebuild")
    parser.add_argument("--only", choices=("bm25", "chroma"), help="rebuild one index (development)")
    parser.add_argument("--validate", action="store_true", help="run Stage E6 only, against the existing indexes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.rebuild and args.only:
        parser.error("--rebuild rebuilds everything; do not combine it with --only")

    from indexing.manifest import IndexIncompatible
    from indexing.validate import validate
    from indexing.vector_store import TokenLimitExceeded
    try:
        second_run = ("skipped: --validate checks the existing indexes without rebuilding", None, None)
        if not args.validate:
            from indexing.build import build
            from indexing.vector_store import open_collection
            result = build(only=args.only, rebuild=args.rebuild)
            logging.info("build finished in %.1fs", result["build_seconds"])
            if args.only is None:
                # Q1 full-rebuild proof: build again and compare counts / ID sets (no duplicates)
                before = result["chroma"]
                again = build()["chroma"]
                second_run = (None, {"count": before["count"], "ids": before["ids"]},
                              {"count": again["count"], "ids": again["ids"]})
            else:
                second_run = (f"skipped: --only {args.only} does not rebuild both indexes", None, None)
        out = validate(second_run)
    except (IndexIncompatible, TokenLimitExceeded, FileNotFoundError) as e:
        logging.error("%s", e)
        return 1
    print_summary(out)
    failed = [c for c in out["checks"] if not c.ok]
    print(f"\n[index] report written to {config.REPORT_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
