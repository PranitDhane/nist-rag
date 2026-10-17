"""python -m chunking [--rebuild] [--only-stage C1..C9]"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from chunking import config
from chunking.assemble import EmbedTooLong
from chunking.pipeline import run_all
from chunking.tables import TableTooBig
from chunking.validate import (load_chunks, run_checks, serialize, write_chunks, write_report)

STAGES = {"C1": "tokenizer check", "C2": "profile only", "C3": "build chunks (no validation)",
          "C4": "build chunks (no validation)", "C5": "build chunks (no validation)", "C6": "build chunks (no validation)",
          "C7": "build chunks (no validation)", "C8": "build chunks (no validation)", "C9": "validate existing chunks.jsonl"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chunking", description="Build chunks.jsonl from pages.jsonl.")
    parser.add_argument("--rebuild", action="store_true", help="delete chunks.jsonl and chunk_report.json first")
    parser.add_argument("--only-stage", choices=sorted(STAGES), help="run one stage: C1 tokenizer, C2 profile, C3-C8 build only, C9 validate only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from ingest.config import load_manifest
    manifest = load_manifest()
    if args.rebuild:
        for p in (config.CHUNKS_PATH, config.REPORT_PATH):
            p.unlink(missing_ok=True)

    if args.only_stage == "C1":
        from chunking.tokens import cache_location, count_tokens
        print(f"[C1] count_tokens('hello world') = {count_tokens('hello world')}; tokenizer cache: {cache_location()}")
        return 0
    if args.only_stage == "C2":
        from chunking.profile import write_profile
        prof = write_profile()
        for doc_id, p in prof.items():
            print(f"[C2] {doc_id}: {p['units']}, tiny sections {len(p['tiny_sections_under_60'])}, "
                  f"genai tables {p['genai_tables'].get('count', 0)}, rmf categories {p['rmf_categories'].get('count', 0)}")
        return 0

    started = time.time()
    try:
        if args.only_stage == "C9":
            records = load_chunks()
            checks, warnings = run_checks(records, manifest, rerun=[r for _, rs, _, _ in run_all() for r in rs])
            logs = {}
        else:
            results = run_all()
            records = [r for _, rs, _, _ in results for r in rs]
            logs = {doc.doc_id: log for doc, _, log, _ in results}
            write_chunks(records)
            if args.only_stage:
                print(f"[build] {len(records)} chunks written to {config.CHUNKS_PATH}")
                return 0
            checks, warnings = run_checks(records, manifest, rerun=[r for _, rs, _, _ in run_all() for r in rs])
    except (EmbedTooLong, TableTooBig, FileNotFoundError) as e:
        logging.error("%s", e)
        return 1

    profile = None
    if not args.only_stage:
        from chunking.profile import profile as build_profile
        profile = build_profile()
    report = write_report(records, checks, warnings, logs, time.time() - started, profile)
    for doc_id, c in report["counts"].items():
        print(f"[chunks] {doc_id}: {sum(c.values())} chunks {c}")
    for c in checks:
        print(f"[validate] {'PASS' if c.ok else 'FAIL'}  {c.number} {c.name}: {c.detail}")
    print(f"[validate] {len(warnings)} soft warnings (see {config.REPORT_PATH.name})")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
