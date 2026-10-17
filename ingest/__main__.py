"""CLI entry point: python -m ingest.

Stages run in plan order (2-10). Each stage reads the previous stage's cache file, so
`--from-stage N` and `--only-stage N` work from stages/NN_*.jsonl and fail with a clear
message if a needed cache is missing. Exit code 0 = success, 1 = any error or failed hard check.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from ingest import config
from ingest.assemble import assemble, build_report
from ingest.classify import classify_all, load_classify
from ingest.clean import clean_all, rule_hit_counts
from ingest.config import load_manifest
from ingest.download import AcquireError, acquire_all
from ingest.extract import extract_all, load_extract
from ingest.figures import confidence_threshold, figures_all, load_figures
from ingest.links import link_counts, links_all, load_links
from ingest.models import Manifest
from ingest.profile import load_profile, profile_all
from ingest.structure import load_structure, structure_all
from ingest.validate import validate

FIRST_STAGE, LAST_STAGE = 2, 10


def _stage_acquire(manifest: Manifest, from_dir: Path | None) -> None:
    for r in acquire_all(manifest, from_dir=from_dir):
        pin = "pinned" if r.sha256_pinned else "NOT PINNED - add to sources.yaml"
        print(f"[acquire] {r.doc_id}: {r.status}, {r.pages} pages, sha256={r.sha256} ({pin})")


def _stage_profile(manifest: Manifest, from_dir: Path | None) -> None:
    profile = profile_all(manifest)
    t = profile["tesseract"]
    print(f"[profile] tesseract: available={t['available']} version={t['version']}")
    for d in profile["documents"]:
        figs = [i for i in d["images"] if i["figure_number"] is not None]
        print(f"[profile] {d['doc_id']}: {d['pages']} pages, text_layer={d['text_layer_present']}, "
              f"page_labels={d['page_labels_present']}, images={len(d['images'])}, "
              f"tiny_skipped={d['tiny_images_skipped']}, figures={len(figs)}")


def _stage_extract(manifest: Manifest, from_dir: Path | None) -> None:
    print(f"[extract] {len(extract_all(manifest))} page records written")


def _stage_classify(manifest: Manifest, from_dir: Path | None) -> None:
    print(f"[classify] {len(classify_all(load_extract()))} pages classified")


def _stage_figures(manifest: Manifest, from_dir: Path | None) -> None:
    for f in figures_all(manifest, load_profile(), load_classify(), threshold=confidence_threshold()):
        print(f"[figures] {f['doc_id']} fig{f['figure_number']} p.{f['pdf_page']}: {f['ocr_status']} "
              f"conf={f['confidence']} png={f['png']}")


def _stage_clean(manifest: Manifest, from_dir: Path | None) -> None:
    cleaned = clean_all(manifest, load_classify())
    for doc_id, counts in rule_hit_counts(cleaned).items():
        print(f"[clean] {doc_id}: {dict(counts)}")
    print(f"[clean] {len(cleaned)} pages cleaned, {sum(len(r['warnings']) for r in cleaned)} warnings")


def _stage_structure(manifest: Manifest, from_dir: Path | None) -> None:
    from ingest.clean import load_clean
    structured = structure_all(load_clean())
    for doc_id in dict.fromkeys(r["doc_id"] for r in structured):
        docs = [r for r in structured if r["doc_id"] == doc_id]
        ids = [a for r in docs for a in r["entities"]["action_ids"]]
        print(f"[structure] {doc_id}: headings={sum(len(r['headings']) for r in docs)} "
              f"table_markers={sum(len(r['table_markers']) for r in docs)} "
              f"continues_table={sum(r['continues_table'] for r in docs)} "
              f"table_row_action_ids={len(ids)} unique={len(set(ids))}")


def _stage_links(manifest: Manifest, from_dir: Path | None) -> None:
    linked = links_all(manifest, load_structure())
    for doc_id, counts in link_counts(linked).items():
        print(f"[links] {doc_id}: {counts}")
    empty = sum(1 for r in linked for l in r["links"] if l["location"] == "body" and not l["text"])
    print(f"[links] body links with empty anchor text: {empty}")


STAGES = {2: _stage_acquire, 3: _stage_profile, 4: _stage_extract, 5: _stage_classify, 6: _stage_figures,
          7: _stage_clean, 8: _stage_structure, 9: _stage_links}


def run_final_stage(manifest: Manifest, started: float) -> bool:
    """Stage 10: assemble, validate, write the report, print the summary. True if all hard checks passed."""
    linked, figures, profile = load_links(), load_figures(), load_profile()
    pages = assemble(manifest, linked, figures)
    checks = validate(pages, manifest)
    report = build_report(manifest, pages, linked, figures, profile, checks, time.monotonic() - started)
    for doc_id, s in report["documents"].items():
        print(f"[validate] {doc_id}: {s['pages']} pages, {s['headings']} headings, {s['table_markers']} table markers, "
              f"{s['unique_action_ids']} action IDs, {s['footnotes']} footnotes, {s['figures']} figures, links {s['links_by_location']}")
    for c in checks:
        print(f"[validate] {'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}")
    print(f"[validate] {len(report['warnings'])} soft warnings (see ingest_report.json); "
          f"{sum(c.passed for c in checks)}/{len(checks)} hard checks passed")
    return all(c.passed for c in checks)


def rebuild_clean(processed_dir: Path) -> None:
    """Delete every regenerable output (stage caches, debug figures, pages.jsonl, report); PDFs stay."""
    for target in (config.STAGES_DIR, config.DEBUG_FIGURES_DIR):
        if target.is_dir() and processed_dir in target.parents:
            shutil.rmtree(target)
    for name in ("pages.jsonl", "ingest_report.json"):
        (processed_dir / name).unlink(missing_ok=True)


def run_stages(first: int, last: int, manifest: Manifest, from_dir: Path | None = None) -> bool:
    """Run stages first..last. Returns False if the Stage 10 hard checks failed."""
    started = time.monotonic()
    for n in range(first, min(last, 9) + 1):
        STAGES[n](manifest, from_dir)
    return run_final_stage(manifest, started) if last >= LAST_STAGE else True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest", description="NIST AI RMF ingestion pipeline")
    parser.add_argument("--from-dir", type=Path, help="acquire PDFs from this local directory instead of downloading")
    parser.add_argument("--rebuild", action="store_true", help="delete cached stage outputs first (PDFs are kept)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-stage", type=int, choices=range(3, LAST_STAGE + 1), metavar="3-10",
                       help="rerun from this stage using cached earlier outputs")
    group.add_argument("--only-stage", type=int, choices=range(FIRST_STAGE, LAST_STAGE + 1), metavar="2-10",
                       help="run a single stage (development)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.only_stage:
        first = last = args.only_stage
    else:
        first, last = args.from_stage or FIRST_STAGE, LAST_STAGE
    if args.rebuild and (first, last) != (FIRST_STAGE, LAST_STAGE):
        parser.error("--rebuild reruns everything; do not combine it with --from-stage or --only-stage")

    manifest = load_manifest()
    try:
        if args.rebuild:
            rebuild_clean(config.PROCESSED_DIR)
        ok = run_stages(first, last, manifest, args.from_dir)
    except (AcquireError, FileNotFoundError) as e:
        logging.error("%s", e)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
