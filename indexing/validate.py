"""Stage E6: hard checks, soft smoke checks and index_report.json."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass

from indexing import bm25_index, config
from indexing.loaders import load_bm25, load_chunk_lookup, load_vector_store, read_chunks
from indexing.manifest import (IndexIncompatible, check_compatibility, config_with, read_manifest, sha256_file,
                               sha256_ids)
from indexing.metadata import from_chroma_metadata


@dataclass
class Check:
    number: int
    name: str
    ok: bool
    detail: str
    skipped: bool = False


def _chroma_ids(store) -> set[str]:
    return set(store._collection.get(include=[])["ids"])


def check_counts(store, bm25_ids: list[str], chunks: list[dict]) -> Check:
    n_chroma, n_bm25, n_chunks = store._collection.count(), len(bm25_ids), len(chunks)
    ok = n_chroma == n_bm25 == n_chunks
    return Check(1, "Chroma count = BM25 corpus size = chunk count", ok,
                 f"chroma={n_chroma}, bm25={n_bm25}, chunks.jsonl={n_chunks}")


def check_id_sets(store, bm25_ids: list[str], chunks: list[dict]) -> Check:
    a, b, c = _chroma_ids(store), set(bm25_ids), {x["chunk_id"] for x in chunks}
    problems = []
    if a != c:
        problems.append(f"chroma vs chunks: {len(a - c)} extra, {len(c - a)} missing")
    if b != c:
        problems.append(f"bm25 vs chunks: {len(b - c)} extra, {len(c - b)} missing")
    if len(bm25_ids) != len(b):
        problems.append("duplicate IDs in the BM25 corpus")
    return Check(2, "Chroma, BM25 and chunk ID sets are identical", not problems, "; ".join(problems) or f"{len(c)} IDs identical")


def check_fresh(chunks: list[dict]) -> Check:
    manifest = read_manifest()
    problems = []
    if manifest.get("chunks_file_sha256") != sha256_file(config.CHUNKS_PATH):
        problems.append("chunks.jsonl changed since the indexes were built (stale)")
    if manifest.get("chunk_ids_sha256") != sha256_ids([c["chunk_id"] for c in chunks]):
        problems.append("chunk ID set differs from the one indexed")
    if manifest.get("chunk_count") != len(chunks):
        problems.append(f"manifest chunk_count={manifest.get('chunk_count')} but chunks.jsonl has {len(chunks)}")
    return Check(3, "manifest hash matches the current chunks.jsonl (indexes not stale)", not problems,
                 "; ".join(problems) or f"sha256 {manifest['chunks_file_sha256'][:12]}…")


def check_vectors(store) -> Check:
    from indexing.vector_store import read_back_settings, sample_norms
    col = store._collection
    problems = []
    settings = read_back_settings(col)
    if settings["distance"] != "cosine":
        problems.append(f"distance reads back as {settings['distance']!r}")
    sample = col.get(limit=1, include=["embeddings"])["embeddings"]
    dim = len(sample[0])
    if dim != config.DIMENSION:
        problems.append(f"dimension {dim}, expected {config.DIMENSION}")
    norms = sample_norms(col, 10, seed=1)
    if any(abs(n - 1.0) > 1e-3 for n in norms):
        problems.append(f"norms off 1.0: {[round(n, 4) for n in norms]}")
    return Check(4, "dimension 384, sampled norms ~1.0, distance cosine", not problems,
                 "; ".join(problems) or f"dim={dim}, distance={settings['distance']}, 10 sampled norms within 1e-3 of 1.0")


def check_metadata_roundtrip(store, lookup: dict[str, dict]) -> Check:
    ids = random.Random(7).sample(sorted(lookup), min(10, len(lookup)))
    got = store._collection.get(ids=ids, include=["documents", "metadatas"])
    bad = [i for i, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
           if from_chroma_metadata(meta, doc) != lookup[i]]
    return Check(5, "metadata round trip for 10 random chunks matches chunks.jsonl exactly", not bad,
                 f"mismatch: {bad}" if bad else "10/10 identical")


def check_fresh_process() -> Check:
    code = ("from indexing.loaders import load_vector_store, load_bm25\n"
            "s = load_vector_store(); b, ids = load_bm25()\n"
            "print(s._collection.count(), len(ids))")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.PROJECT_ROOT, timeout=300)
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    ok = proc.returncode == 0 and len(last.split()) == 2
    return Check(6, "load_vector_store() and load_bm25() work in a fresh Python process", ok,
                 f"fresh process loaded chroma/bm25 counts: {last}" if ok else (proc.stderr.strip().splitlines() or ["failed"])[-1])


def check_mismatch_detected() -> Check:
    try:
        check_compatibility(config_with(MODEL_NAME="BAAI/bge-base-en-v1.5"))
    except IndexIncompatible as e:
        return Check(7, "check_compatibility raises when the configured model name is changed", True, str(e)[:120])
    return Check(7, "check_compatibility raises when the configured model name is changed", False, "no error raised")


def check_second_run(skipped_reason: str | None, before: dict | None, after: dict | None) -> Check:
    name = "a second full run creates no duplicates and keeps the same counts"
    if skipped_reason:
        return Check(8, name, True, skipped_reason, skipped=True)
    ok = before == after
    return Check(8, name, ok, f"count {before['count']} -> {after['count']}, same ID set: {before['ids'] == after['ids']}"
                 if ok else f"before={before and before['count']}, after={after and after['count']}")


# ----------------------------------------------------------------------------- smoke checks
def matches(chunk: dict, expect: dict) -> bool:
    for key, want in expect.items():
        if key == "section_leaf_startswith":
            path = chunk["section_path"]
            if not path or not path[-1].startswith(want):
                return False
        elif key == "text_contains":
            if want not in chunk["display_text"]:
                return False
        elif chunk.get(key) != want:
            return False
    return True


def smoke_checks(store, bm25, bm25_ids: list[str], lookup: dict[str, dict], k: int = 5) -> list[dict]:
    out = []
    for spec in config.SMOKE_CHECKS:
        vec = [(d.metadata["chunk_id"], 1.0 - s) for d, s in store.similarity_search_with_score(spec["query"], k=k)]
        kw = bm25_index.search(bm25, bm25_ids, spec["query"], k)

        def hit(results):
            return any(matches(lookup[cid], e) for cid, _ in results for e in spec["expect"])

        def rows(results):
            return [{"chunk_id": cid, "citation_label": lookup[cid]["citation_label"], "score": round(sc, 4)} for cid, sc in results]

        rank1_ok = None
        if "bm25_rank" in spec:
            rank1_ok = bool(kw) and any(matches(lookup[kw[spec["bm25_rank"] - 1][0]], e) for e in spec["expect"])
        out.append({"query": spec["query"], "expected": spec["note"], "vector_hit": hit(vec), "bm25_hit": hit(kw),
                    "bm25_rank_ok": rank1_ok, "vector_top5": rows(vec), "bm25_top5": rows(kw)})
    return out


# ----------------------------------------------------------------------------- run + report
def validate(second_run: tuple[str | None, dict | None, dict | None] = ("skipped: needs a build run", None, None)) -> dict:
    chunks = read_chunks()
    lookup = load_chunk_lookup()
    store = load_vector_store()
    bm25, bm25_ids = load_bm25()
    checks = [check_counts(store, bm25_ids, chunks), check_id_sets(store, bm25_ids, chunks), check_fresh(chunks),
              check_vectors(store), check_metadata_roundtrip(store, lookup), check_fresh_process(),
              check_mismatch_detected(), check_second_run(*second_run)]
    smoke = smoke_checks(store, bm25, bm25_ids, lookup)

    manifest = read_manifest()
    from indexing.vector_store import read_back_settings
    report = {
        "counts_by_doc": dict(Counter(c["doc_id"] for c in chunks)),
        "counts_by_chunk_type": dict(Counter(c["chunk_type"] for c in chunks)),
        "counts_by_doc_and_type": {d: dict(Counter(c["chunk_type"] for c in chunks if c["doc_id"] == d))
                                   for d in sorted({c["doc_id"] for c in chunks})},
        "build_seconds": manifest.get("build_seconds"), "device": manifest.get("device"),
        "model_settings_read_back": {k: manifest.get(k) for k in ("embedding_model", "model_revision", "dimension", "normalized",
                                                                  "max_seq_length", "document_prefix")},
        "chroma_settings_read_back": {**read_back_settings(store._collection), "collection_name": config.COLLECTION_NAME,
                                      "chroma_version": manifest.get("chroma_version")},
        "bm25": {k: manifest["bm25"].get(k) for k in ("library", "version", "tokenizer_version", "k1", "b", "stemming",
                                                      "stopwords", "indexed_text", "vocab_size", "avg_doc_len")},
        "checks": [{"number": c.number, "name": c.name, "passed": c.ok, "skipped": c.skipped, "detail": c.detail} for c in checks],
        "smoke_checks": smoke,
    }
    config.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"report": report, "checks": checks, "smoke": smoke}
