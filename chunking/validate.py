"""Stage C9: hard checks, soft warnings, chunks.jsonl and chunk_report.json."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass

from chunking import config
from chunking.join import JoinedDoc, join_all, load_pages
from chunking.tokens import count_tokens

REQUIRED = ("chunk_id", "doc_id", "section_path", "pdf_page_start", "pdf_page_end", "citation_url", "display_text")


@dataclass
class Check:
    number: int
    name: str
    ok: bool
    detail: str


def serialize(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def write_chunks(records: list[dict]) -> None:
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(serialize(records), encoding="utf-8")


def load_chunks() -> list[dict]:
    if not config.CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"{config.CHUNKS_PATH} is missing. Run `python -m chunking` first.")
    return [json.loads(l) for l in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]


def _result(n: int, name: str, problems: list[str], ok_detail: str) -> Check:
    return Check(n, name, not problems, ok_detail if not problems else "; ".join(problems[:6]) +
                 (f" (+{len(problems) - 6} more)" if len(problems) > 6 else ""))


def check_embed_limit(records) -> Check:
    bad = [f"{r['chunk_id']}={r['token_count_embed']}" for r in records
           if count_tokens(r["embed_text"]) > config.HARD_MAX_EMBED_TOKENS or r["token_count_embed"] != count_tokens(r["embed_text"])]
    mx = max(r["token_count_embed"] for r in records)
    return _result(1, f"every embed_text <= {config.HARD_MAX_EMBED_TOKENS} tokens", bad, f"all {len(records)} chunks fit (max {mx})")


def check_coverage(records, docs: dict[str, JoinedDoc]) -> Check:
    problems = []
    for doc_id, doc in docs.items():
        covered = bytearray(len(doc.text))
        for r in records:
            if r["doc_id"] == doc_id and r["chunk_type"] in ("prose", "genai_action_table", "rmf_core_table"):
                for a, b in r["char_spans"]:
                    covered[a:b] = b"\x01" * (b - a)
        missing = [i for i, ch in enumerate(doc.text) if not ch.isspace() and not covered[i]]
        if missing:
            problems.append(f"{doc_id}: {len(missing)} uncovered characters, first at {missing[0]} "
                            f"(p.{doc.page_at(missing[0]).pdf_page}: {doc.text[missing[0]:missing[0] + 40]!r})")
    return _result(2, "coverage: every non-whitespace character is in a prose/table chunk", problems,
                   "all index-eligible text covered")


def check_sections(records, docs: dict[str, JoinedDoc]) -> Check:
    problems = []
    by_doc: dict[str, list[dict]] = {}
    for r in records:
        if r["chunk_type"] == "prose":
            by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_id, chunks in by_doc.items():
        doc = docs[doc_id]
        for r in chunks:
            for a, b in r["char_spans"]:
                if any(a < h.start < b for h in doc.headings):
                    problems.append(f"{r['chunk_id']} spans a heading")
            if doc.path_at(r["char_spans"][-1][0]) != r["section_path"]:
                problems.append(f"{r['chunk_id']} section_path differs from the section at its start")
        for prev, cur in zip(chunks, chunks[1:]):
            if cur["char_spans"][0][0] < prev["char_spans"][-1][1] and cur["section_path"] != prev["section_path"]:
                problems.append(f"{cur['chunk_id']} overlaps a different section")
    return _result(3, "no prose chunk spans two sections; overlap only within a section", problems, "sections respected")


def check_tables(records, manifest) -> Check:
    problems, parts = [], []
    for spec in manifest.documents:
        v = spec.validation
        tabs = [r for r in records if r["doc_id"] == spec.doc_id and r["chunk_type"] == "genai_action_table"]
        if v.chunk_action_ids is not None:
            ids = [a for r in tabs for a in r["action_ids"]]
            counts = Counter(ids)
            parts.append(f"{len(counts)} action IDs")
            if len(ids) != len(counts) or len(counts) != v.chunk_action_ids:
                problems.append(f"{spec.doc_id}: {len(counts)} distinct / {len(ids)} listed action IDs, expected {v.chunk_action_ids} each once")
        if v.subcategory_tables is not None:
            subs = {r["subcategory"] for r in tabs}
            parts.append(f"{len(subs)} subcategories")
            if len(subs) != v.subcategory_tables:
                problems.append(f"{spec.doc_id}: {len(subs)} GenAI subcategories with a chunk, expected {v.subcategory_tables}")
    return _result(4, "action IDs each in exactly one table chunk; every subcategory has a chunk", problems, ", ".join(parts))


def check_table_integrity(records) -> Check:
    problems = []
    for r in records:
        if r["chunk_type"] == "genai_action_table":
            t = r["display_text"]
            if not t.startswith(f"{r['subcategory']}:"):
                problems.append(f"{r['chunk_id']}: header line missing")
            if "AI Actor Tasks:" not in t.splitlines()[-1]:
                problems.append(f"{r['chunk_id']}: AI Actor Tasks line missing")
            for aid in r["action_ids"]:
                lines = [l for l in t.splitlines() if l.startswith(aid + " — ")]
                if len(lines) != 1:
                    problems.append(f"{r['chunk_id']}: row {aid} not present exactly once as a whole line")
        elif r["chunk_type"] == "rmf_core_table":
            if not r["display_text"].startswith(f"{r['category']}:"):
                problems.append(f"{r['chunk_id']}: category line missing")
            for sub in r["subcategories"]:
                if f"\n{sub}: " not in r["display_text"]:
                    problems.append(f"{r['chunk_id']}: subcategory {sub} missing")
    return _result(5, "every table piece has its repeated header; no row split", problems, "headers repeated, rows whole")


def check_fields(records) -> Check:
    problems = []
    for r in records:
        for k in REQUIRED:
            if r.get(k) in (None, "") and not (k == "section_path"):
                problems.append(f"{r['chunk_id']}: {k} empty")
        if not isinstance(r["section_path"], list):
            problems.append(f"{r['chunk_id']}: section_path not a list")
        if not r["display_text"].strip():
            problems.append(f"{r['chunk_id']}: empty display_text")
    ids = [r["chunk_id"] for r in records]
    if len(ids) != len(set(ids)):
        problems.append("duplicate chunk_ids")
    return _result(6, "every chunk has doc_id, section_path, pages, citation_url, display_text", problems, "all fields present, IDs unique")


def check_no_breadcrumb(records) -> Check:
    bad = [r["chunk_id"] for r in records if r["embed_text"] != r["embed_text"].split("\n\n", 1)[0] + "\n\n" + r["display_text"]
           or r["embed_text"].split("\n\n", 1)[0] in r["display_text"]]
    return _result(7, "no display_text contains the breadcrumb line", bad, "display_text is clean")


def check_extras(records, pages: list[dict]) -> Check:
    problems = []
    elig = [p for p in pages if p["index_eligible"]]
    n_fn = sum(len(p["footnotes"]) for p in elig)
    got_fn = sum(r["chunk_type"] == "footnote" for r in records)
    if n_fn != got_fn:
        problems.append(f"{got_fn} footnote chunks, {n_fn} footnotes in pages.jsonl")
    with_text = {(p["doc_id"], f["figure_number"]) for p in elig for f in p["figures"] if f["text"]}
    got = {(r["doc_id"], r["figure_number"]) for r in records if r["chunk_type"] == "figure"}
    if with_text != got:
        problems.append(f"figure chunks {sorted(got)} != figures with OCR text {sorted(with_text)}")
    return _result(8, "footnote chunks = footnotes; figure chunks only for figures with OCR text", problems,
                   f"{got_fn} footnotes, {len(got)} figures with OCR")


def check_determinism(records, rerun) -> Check:
    same = serialize(records) == serialize(rerun)
    return Check(9, "determinism: two runs give identical chunks", same, "identical output" if same else "second run differs")


def soft_warnings(records, merge_logs: list[dict]) -> list[str]:
    w = []
    merged_ids = {r["chunk_id"] for r in records if r["merged_sections"]}
    for r in records:
        t = r["token_count_embed"]
        if r["chunk_type"] == "prose" and r["chunk_id"] not in merged_ids:
            from chunking.tokens import count_tokens as ct
            if ct(r["display_text"]) < 100 and r["part"] == r["parts_total"] and r["parts_total"] == 1:
                w.append(f"short prose chunk (<100 tokens): {r['chunk_id']} {r['section_path'][-1:]}")
        if r["merged_sections"] and count_tokens(r["display_text"]) > config.PROSE_TARGET_MAX:
            w.append(f"merged chunk above {config.PROSE_TARGET_MAX} tokens: {r['chunk_id']}")
        if r["chunk_type"] == "figure" and r["ocr_confidence"] is not None and r["ocr_confidence"] < 70:
            w.append(f"figure OCR confidence near threshold: {r['chunk_id']} ({r['ocr_confidence']})")
        if r["chunk_type"] == "footnote" and len(r["display_text"].split()) < 6:     # "Footnote N:" is 2 words
            w.append(f"short footnote (<5 words): {r['chunk_id']}")
    return w


def stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    hist = Counter((v // 50) * 50 for v in values)
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values),
            "histogram_50": {f"{k}-{k + 49}": hist[k] for k in sorted(hist)}}


def run_checks(records, manifest, rerun=None) -> tuple[list[Check], list[str]]:
    pages = load_pages()
    docs = {d.doc_id: d for d in join_all(pages)}
    checks = [check_embed_limit(records), check_coverage(records, docs), check_sections(records, docs),
              check_tables(records, manifest), check_table_integrity(records), check_fields(records),
              check_no_breadcrumb(records), check_extras(records, pages)]
    if rerun is not None:
        checks.append(check_determinism(records, rerun))
    return checks, soft_warnings(records, [])


def write_report(records, checks, warnings, logs: dict, build_seconds: float, profile: dict | None = None) -> dict:
    report = {}
    if config.REPORT_PATH.is_file():
        report = json.loads(config.REPORT_PATH.read_text(encoding="utf-8"))
    if profile is not None:
        report["profile"] = profile
    by_doc_type = Counter((r["doc_id"], r["chunk_type"]) for r in records)
    report.update({
        "build_seconds": round(build_seconds, 1),
        "counts": {d: {t: n for (dd, t), n in sorted(by_doc_type.items()) if dd == d} for d in sorted({r["doc_id"] for r in records})},
        "total_chunks": len(records),
        "token_histograms_embed": {t: stats([r["token_count_embed"] for r in records if r["chunk_type"] == t])
                                   for t in sorted({r["chunk_type"] for r in records})},
        "merges": {d: l["merges"] for d, l in logs.items()},
        "splits": {"table_pieces": Counter(f"{r['doc_id']}:{r['chunk_type']}" for r in records
                                           if r["chunk_type"] in ("genai_action_table", "rmf_core_table") and r["part"] == 1 and r["parts_total"] > 1),
                   "prose_pieces": Counter(r["doc_id"] for r in records if r["chunk_type"] == "prose" and r["part"] == 1 and r["parts_total"] > 1),
                   "figures": Counter(r["doc_id"] for r in records if r["chunk_type"] == "figure" and r["part"] == 1 and r["parts_total"] > 1)},
        "repairs": {d: {"table_repairs": l["table_repairs"], "rmf_notes": l["rmf_notes"], "figure_notes": l["figure_notes"]}
                    for d, l in logs.items()},
        "links_attached": sum(len(r["links"]) for r in records),
        "checks": [{"number": c.number, "name": c.name, "passed": c.ok, "detail": c.detail} for c in checks],
        "warnings": warnings,
    })
    config.REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
    return report
