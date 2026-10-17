"""Stage S4: does every citation link open a page that contains the cited text? `python -m eval.link_check [--out PATH]`.

No LLM, no network: only chunks.jsonl, pages.jsonl and the local PDFs in data/raw. Checks EVERY chunk (not just cited ones):
  text     (see COVERAGE_OK) the first and last SAMPLE normalized characters of the chunk's display_text appear in the PDF text of pages
           pdf_page_start..pdf_page_end (the pages the URL points at). Figure-OCR chunks are skipped for this (OCR text is
           not in the text layer); their caption (or 'Figure N') must be on the page instead.
  page     the label's printed page(s) equal the printed page(s) ingestion recorded for those PDF pages.
  url      '#page=' equals pdf_page_start and the base URL is the manifest URL.
Failures are reported, never fixed: a failure usually points at an ingestion or chunking bug."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from attribution import format as fmt
from ingest import config as ingest_config

SAMPLE = 60
EXPLAINED = ("text_sample_only", "text_low_coverage")
COVERAGE_OK = 0.9      # below this a missing sample is a real failure; at or above it is an ordering/synthetic-line artifact
DEFAULT_REPORT = Path(__file__).resolve().parent / "link_check_report.json"
_NON_ALNUM = re.compile(r"[\W_]+")


def normalize(text: str) -> str:
    """NFKC (ligatures), lowercase, then drop everything that is not a letter or digit. This subsumes 'collapse whitespace'
    and 'remove hyphenation', and makes Markdown markup (**, |, #, bullets) irrelevant to the comparison."""
    return _NON_ALNUM.sub("", unicodedata.normalize("NFKC", text).lower())


def samples(display_text: str, size: int = SAMPLE) -> list[str]:
    norm = normalize(display_text)
    return [norm[:size], norm[-size:]] if len(norm) > size else [norm]


def text_failures(display_text: str, page_text: str) -> list[str]:
    """Which samples (start / end of the chunk) are missing from the normalized page text."""
    page = normalize(page_text)
    found = [s in page for s in samples(display_text)]
    return [] if all(found) else [name for name, ok in zip(("start", "end"), found) if not ok][:len(found)]


def _windows(text: str, n: int = 4) -> set[str]:
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", re.sub(r"-\s*\n\s*", "", text)).lower())
    return {" ".join(words[i:i + n]) for i in range(max(len(words) - n + 1, 1))}


def coverage(display_text: str, page_text: str) -> float:
    """Share of the chunk's 4-word windows that appear on the linked pages. Insensitive to the order of table cells and to
    the synthetic header/footer lines a chunk adds ('Footnote 1:', 'AI Actor Tasks: ...'); a link to the WRONG page scores ~0."""
    mine = _windows(display_text)
    return len(mine & _windows(page_text)) / len(mine)


def figure_failure(page_text: str, figure_number, caption: str | None) -> str | None:
    """OCR chunk: the caption (first 40 normalized chars) or at least 'Figure N' must be on the page."""
    page = normalize(page_text)
    if caption and normalize(caption)[:40] in page:
        return None
    return None if f"figure{figure_number}" in page or f"fig{figure_number}" in page else "caption"


def expected_pages_text(first, last) -> str:
    return f"p. {first}" if first == last else f"pp. {first}–{last}"


def best_page(display_text: str, pdf_pages: dict[int, str]) -> int:
    """The PDF page whose text covers the chunk best (ties: the earliest)."""
    return max(pdf_pages, key=lambda p: (coverage(display_text, pdf_pages[p]), -p))


def check_chunk(chunk: dict, pdf_pages: dict[int, str], printed: dict[tuple[str, int], str | None], captions: dict, url_base: str) -> list[dict]:
    """Findings for one chunk: [{'check': ..., 'detail': ...}]; empty = passed. A missing start/end sample is graded:
      text_sample_only   word coverage >= COVERAGE_OK (synthetic 'Footnote N:' / 'AI Actor Tasks' lines, cell order)
      text_low_coverage  coverage lower, but the linked pages are still the best match in the whole PDF (table layout)
      text               neither: the link may point at the wrong page (a real failure)"""
    out: list[dict] = []
    first, last = chunk["pdf_page_start"], chunk["pdf_page_end"]
    page_text = "\n".join(pdf_pages[p] for p in range(first, last + 1))
    if chunk["chunk_type"] == "figure":
        if (miss := figure_failure(page_text, chunk["figure_number"], captions.get((chunk["doc_id"], first, str(chunk["figure_number"]))))):
            out.append({"check": "figure_page", "detail": f"{miss} of Figure {chunk['figure_number']} not found on PDF page {first}"})
    elif (missing := text_failures(chunk["display_text"], page_text)):
        cov = coverage(chunk["display_text"], page_text)
        grade = "text_sample_only" if cov >= COVERAGE_OK else \
            "text_low_coverage" if best_page(chunk["display_text"], pdf_pages) in range(first, last + 1) else "text"
        out.append({"check": grade, "coverage": round(cov, 3),
                    "detail": f"{'+'.join(missing)} sample of the chunk not found on PDF pages {first}-{last}; word coverage {cov:.0%}"})
    want_first, want_last = printed.get((chunk["doc_id"], first)), printed.get((chunk["doc_id"], last))
    if want_first is None or want_last is None:
        out.append({"check": "printed_page", "detail": f"ingestion has no printed page for PDF page {first} or {last}"})
    elif fmt.pages_text(chunk) != expected_pages_text(want_first, want_last):
        out.append({"check": "printed_page", "detail": f"label says {fmt.pages_text(chunk)!r}, pages say {expected_pages_text(want_first, want_last)!r}"})
    url = fmt.citation_url(chunk)
    if url != f"{url_base}#page={first}":
        out.append({"check": "url", "detail": f"{url!r} != {url_base}#page={first}"})
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(processed_dir: Path | None = None, raw_dir: Path | None = None) -> dict:
    import pymupdf
    processed_dir, raw_dir = processed_dir or ingest_config.PROCESSED_DIR, raw_dir or ingest_config.RAW_DIR
    manifest = ingest_config.load_manifest()
    chunks = read_jsonl(processed_dir / "chunks.jsonl")
    pages = read_jsonl(processed_dir / "pages.jsonl")
    printed = {(p["doc_id"], p["pdf_page"]): p["printed_page"] for p in pages}
    captions = {(p["doc_id"], p["pdf_page"], str(f["figure_number"])): f.get("caption") for p in pages for f in p["figures"]}
    pdf_text: dict[str, dict[int, str]] = {}
    for spec in manifest.documents:
        with pymupdf.open(raw_dir / spec.filename) as doc:
            pdf_text[spec.doc_id] = {i + 1: page.get_text() for i, page in enumerate(doc)}
    failures, explained = [], []
    for c in chunks:
        problems = check_chunk(c, pdf_text[c["doc_id"]], printed, captions, manifest.get(c["doc_id"]).url)
        if problems:
            row = {"chunk_id": c["chunk_id"], "chunk_type": c["chunk_type"], "label": fmt.citation_label(c),
                   "url": fmt.citation_url(c), "failures": problems}
            (explained if all(p["check"] in EXPLAINED for p in problems) else failures).append(row)
    by_grade: dict[str, int] = {}
    for f in explained:
        for p in f["failures"]:
            by_grade[f"{p['check']}:{f['chunk_type']}"] = by_grade.get(f"{p['check']}:{f['chunk_type']}", 0) + 1
    return {"chunks_checked": len(chunks), "passed_strict": len(chunks) - len(failures) - len(explained),
            "explained_mismatch": len(explained), "failed": len(failures),
            "pass_rate_pct": round(100 * (len(chunks) - len(failures)) / len(chunks), 1) if chunks else None,
            "explained_by_grade_and_type": by_grade, "sample_chars": SAMPLE, "coverage_ok": COVERAGE_OK,
            "failures": failures, "explained": explained}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args(argv)
    report = run()
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"checked {report['chunks_checked']}  passed strict {report['passed_strict']}  "
          f"explained mismatch {report['explained_mismatch']} {report['explained_by_grade_and_type']}  FAILED {report['failed']}  "
          f"(pass rate counting explained as pass: {report['pass_rate_pct']}%)")
    for f in report["failures"]:
        print(f"FAIL {f['chunk_id']}  {f['label']}  {f['url']}")
        for p in f["failures"]:
            print(f"     {p['check']}: {p['detail']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
