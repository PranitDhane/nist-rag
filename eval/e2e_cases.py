"""Stage V1: load, validate and verify the end-to-end case file (`eval/e2e_cases.yaml`).

`python -m eval.e2e_cases` checks the GOLD DATA itself, never the system:
  * schema: required fields, known routes/outcomes/text keys, unique ids;
  * gold locations: every verbatim quote is found on the stated PDF page of the stated document (read from the PDF
    with PyMuPDF, not from our pipeline output, so a pipeline bug cannot hide a wrong gold fact); every 'absence'
    claim has zero hits in the stated documents;
  * must_contain: each fact has at least one variant that occurs in the documents its expected sections name;
  * expected sections: each one exists in the chunk corpus.
Expected values live only in the case file; this module and the later harness only read them (CLAUDE.md section 5).
Nothing here is ever edited to improve a result."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "e2e_cases.yaml"
ROOT = EVAL_DIR.parent

ROUTES = {"greeting", "nist_question", "out_of_scope"}
OUTCOMES = {"greeting", "answered", "insufficient", "out_of_scope"}
TEXT_KEYS = {"INSUFFICIENT_TEXT", "OUT_OF_SCOPE_TEXT", "GREETING_TEXT"}
DOC_IDS = {"ai_rmf_1_0", "genai_profile"}


# ------------------------------------------------------------------ text matching (shared with the V3 checks)
def squash(text: str) -> str:
    """Lower-case letters and digits only: used to find gold QUOTES in PDF text, where words are broken across lines
    ('trustworthi- ness') and punctuation differs."""
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", text).lower())


def tokens(text: str) -> str:
    """Lower-case words separated by single spaces; hyphens, dashes, quotes and punctuation count as separators."""
    return re.sub(r"[^0-9a-z]+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def contains(text: str, needle: str) -> bool:
    """Plan checks B1/B2: case-insensitive, hyphens/dashes/extra spaces ignored ('cross-cutting' = 'cross cutting').
    The needle must START at a word boundary and may continue (prefix match: 'deactivat' finds 'deactivated',
    'rain' does NOT find 'train'). A needle with no letters or digits (e.g. '$', '€') is a plain substring test."""
    n = tokens(needle)
    if not n:
        return needle.lower() in text.lower() if needle.strip() else False
    return f" {n}" in f" {tokens(text)}"


def fact_found(text: str, variants: list[str]) -> bool:
    """A fact passes if ANY of its variants is found."""
    return any(contains(text, v) for v in variants)


# ------------------------------------------------------------------ loading
def text_constants() -> dict[str, str]:
    """The fixed texts, resolved from the app's own config at run time (never copied into the case file)."""
    from augmentation.config import INSUFFICIENT_CONTEXT
    from graph.replies import GREETING_REPLY, OUT_OF_SCOPE_REPLY
    return {"INSUFFICIENT_TEXT": INSUFFICIENT_CONTEXT, "OUT_OF_SCOPE_TEXT": OUT_OF_SCOPE_REPLY, "GREETING_TEXT": GREETING_REPLY}


def load_cases(path: Path = DEFAULT_CASES) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def all_turns(data: dict) -> list[dict]:
    """Flat list of every request: single-turn cases first, then conversation turns in order (conversation id added)."""
    turns = [dict(c) for c in data.get("single_turn", [])]
    for conv in data.get("conversations", []):
        for t in conv["turns"]:
            turns.append({**t, "conversation": conv["id"]})
    return turns


# ------------------------------------------------------------------ schema
@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    def count(self, key: str, n: int = 1) -> None:
        self.checked[key] = self.checked.get(key, 0) + n


def validate_schema(data: dict) -> Report:
    rep = Report()
    seen: set[str] = set()
    for t in all_turns(data):
        tid = t.get("id", "?")
        rep.count("turns")
        if tid in seen:
            rep.errors.append(f"{tid}: duplicate id")
        seen.add(tid)
        for key in ("id", "message", "expected_route", "expected_outcome"):
            if not t.get(key):
                rep.errors.append(f"{tid}: missing '{key}'")
        if t.get("expected_route") not in ROUTES:
            rep.errors.append(f"{tid}: unknown expected_route {t.get('expected_route')!r}")
        if t.get("expected_outcome") not in OUTCOMES:
            rep.errors.append(f"{tid}: unknown expected_outcome {t.get('expected_outcome')!r}")
        if t.get("expected_text_key") and t["expected_text_key"] not in TEXT_KEYS:
            rep.errors.append(f"{tid}: unknown expected_text_key {t['expected_text_key']!r}")
        if t.get("expected_insufficient_reason_any"):
            from insufficient.reasons import Reason  # noqa: PLC0415
            unknown = set(t["expected_insufficient_reason_any"]) - {r.value for r in Reason}
            if unknown:
                rep.errors.append(f"{tid}: unknown insufficient reason(s) {sorted(unknown)}")
        if t.get("expected_outcome") in {"greeting", "out_of_scope"} and not t.get("expected_text_key"):
            rep.errors.append(f"{tid}: a fixed-text outcome needs expected_text_key")
        for fact in (t.get("must_contain") or []) + (t.get("must_not_contain") or []):
            if not fact.get("fact") or not fact.get("variants"):
                rep.errors.append(f"{tid}: every must_contain/must_not_contain entry needs 'fact' and non-empty 'variants'")
        if t.get("expected_outcome") == "answered":
            if not t.get("must_contain"):
                rep.errors.append(f"{tid}: an answered turn needs must_contain facts")
            if not t.get("expected_sections"):
                rep.errors.append(f"{tid}: an answered turn needs expected_sections")
            if not t.get("gold_locations"):
                rep.errors.append(f"{tid}: an answered turn needs gold_locations (plan V1 step 2)")
            if t.get("min_facts") is None or not 1 <= t["min_facts"] <= len(t.get("must_contain") or []):
                rep.errors.append(f"{tid}: min_facts must be between 1 and the number of must_contain facts")
        for sec in (t.get("expected_sections") or []) + (t.get("must_not_cite") or []):
            if sec.get("doc") not in DOC_IDS or not sec.get("section"):
                rep.errors.append(f"{tid}: section entries need doc in {sorted(DOC_IDS)} and a section")
        for loc in t.get("gold_locations") or []:
            if "quote" not in loc and "absent_terms" not in loc:
                rep.errors.append(f"{tid}: gold location {loc.get('fact')!r} needs 'quote' or 'absent_terms'")
    for conv in data.get("conversations", []):
        if len(conv["turns"]) != 3:
            rep.warnings.append(f"{conv['id']}: plan E-D2 says 3 turns, found {len(conv['turns'])}")
    core = [c for c in data.get("single_turn", []) if not c.get("optional")]
    if len(core) != 10:
        rep.warnings.append(f"plan E-D2 says 10 single-turn cases, found {len(core)} core")
    return rep


# ------------------------------------------------------------------ gold verification against the PDFs
def pdf_paths() -> dict[str, Path]:
    """doc_id -> PDF in data/raw, via the manifest (filenames live only there). Case-insensitive file lookup."""
    from ingest.config import load_manifest  # noqa: PLC0415
    raw = ROOT / "data" / "raw"
    have = {p.name.lower(): p for p in raw.iterdir()}
    return {d.doc_id: have[d.filename.lower()] for d in load_manifest().documents}


def read_pdf_pages(path: Path) -> list[str]:
    import pymupdf as fitz  # PyMuPDF, the same library the pipeline uses for page numbers
    with fitz.open(path) as doc:
        return [page.get_text() for page in doc]


def quote_segments(quote: str) -> list[str]:
    """A gold quote may elide text with '...' or an ellipsis; each piece must occur in order on the page."""
    return [s.strip() for s in re.split(r"\.\.\.|…", quote) if len(squash(s)) >= 8]


def segments_in_order(page_text: str, segments: list[str]) -> bool:
    haystack, pos = squash(page_text), 0
    for seg in segments:
        i = haystack.find(squash(seg), pos)
        if i < 0:
            return False
        pos = i + len(squash(seg))
    return True


def verify_gold(data: dict, pages: dict[str, list[str]]) -> Report:
    rep = Report()
    for t in all_turns(data):
        for loc in t.get("gold_locations") or []:
            label = f"{t['id']} / {loc.get('fact')}"
            if "absent_terms" in loc:
                rep.count("absence_claims")
                docs = list(pages) if loc.get("doc") == "both" else [loc["doc"]]
                for term in loc["absent_terms"]:
                    for doc in docs:
                        hits = [i + 1 for i, p in enumerate(pages[doc]) if contains(p, term)]
                        if hits:
                            rep.errors.append(f"{label}: '{term}' is NOT absent from {doc} (pdf pages {hits})")
                continue
            rep.count("quotes")
            doc, pg = loc["doc"], loc["pdf_page"]
            if doc not in pages or not isinstance(pg, int) or not 1 <= pg <= len(pages[doc]):
                rep.errors.append(f"{label}: bad doc/pdf_page ({doc}, {pg})")
                continue
            segs = quote_segments(loc["quote"])
            if not segs:
                rep.errors.append(f"{label}: quote has no checkable text")
            elif not segments_in_order(pages[doc][pg - 1], segs):
                where = [i + 1 for i, p in enumerate(pages[doc]) if segments_in_order(p, segs)]
                rep.errors.append(f"{label}: quote not found on {doc} pdf page {pg}" +
                                  (f" (found on pdf page(s) {where})" if where else " (found on no page)"))
    return rep


def verify_printed_pages(data: dict, page_records: list[dict]) -> Report:
    """The case file gives both numbers for each quote; they must agree with the page records (printed label = what
    the citation label shows, pdf page = what the #page= link opens)."""
    rep = Report()
    printed = {(r["doc_id"], r["pdf_page"]): str(r["printed_page"]) for r in page_records}
    for t in all_turns(data):
        for loc in t.get("gold_locations") or []:
            if "quote" not in loc:
                continue
            rep.count("printed_pages")
            want = printed.get((loc["doc"], loc["pdf_page"]))
            if want != str(loc.get("printed")):
                rep.errors.append(f"{t['id']} / {loc.get('fact')}: pdf page {loc['pdf_page']} of {loc['doc']} is printed page "
                                  f"{want!r}, the case file says {loc.get('printed')!r}")
    return rep


def verify_traps_not_in_gold(data: dict) -> Report:
    """A must_not_contain variant that appears in the turn's own gold quote would make a correct answer fail."""
    rep = Report()
    for t in all_turns(data):
        gold = " ".join(loc.get("quote", "") for loc in t.get("gold_locations") or [])
        for trap in t.get("must_not_contain") or []:
            rep.count("traps")
            hit = [v for v in trap["variants"] if contains(gold, v)]
            if hit:
                rep.errors.append(f"{t['id']}: trap {trap['fact']!r} variant(s) {hit} occur in the turn's own gold quote")
    return rep


def verify_facts_reachable(data: dict, pages: dict[str, list[str]]) -> Report:
    """Every must_contain fact must be possible: one variant occurs somewhere in the expected document(s)."""
    rep = Report()
    for t in all_turns(data):
        docs = sorted({s["doc"] for s in t.get("expected_sections") or []})
        text = " ".join(" ".join(pages[d]) for d in docs)
        for fact in t.get("must_contain") or []:
            rep.count("facts")
            if not fact_found(text, fact["variants"]):
                rep.errors.append(f"{t['id']}: fact {fact['fact']!r} has no variant in {docs}")
    return rep


# ------------------------------------------------------------------ expected sections exist in the corpus
def section_matches(chunk: dict, section: str) -> bool:
    if section in (chunk.get("category"), chunk.get("subcategory")) or section in (chunk.get("subcategories") or []):
        return True
    return any(p == section or p.startswith(section + " ") or p.startswith(section + ":") for p in chunk.get("section_path") or [])


def verify_sections(data: dict, chunks: list[dict]) -> Report:
    rep = Report()
    for t in all_turns(data):
        for key in ("expected_sections", "must_not_cite"):
            for s in t.get(key) or []:
                rep.count(key)
                if any(c["doc_id"] == s["doc"] and section_matches(c, s["section"]) for c in chunks):
                    continue
                if key == "expected_sections":
                    rep.errors.append(f"{t['id']}: expected_sections {s['doc']} / {s['section']!r} matches no chunk in the corpus")
                else:      # a trap on a section that does not exist can never fire: allowed, but say so
                    rep.warnings.append(f"{t['id']}: must_not_cite {s['doc']} / {s['section']!r} matches no chunk, so it can never fire")
    return rep


# ------------------------------------------------------------------ command line
def run(path: Path = DEFAULT_CASES) -> dict:
    data = load_cases(path)
    chunks = [json.loads(line) for line in (ROOT / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    pages = {doc: read_pdf_pages(p) for doc, p in pdf_paths().items()}
    page_records = [json.loads(line) for line in (ROOT / "data" / "processed" / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    reports = {"schema": validate_schema(data), "gold_locations": verify_gold(data, pages),
               "printed_pages": verify_printed_pages(data, page_records), "traps_vs_gold": verify_traps_not_in_gold(data),
               "facts_reachable": verify_facts_reachable(data, pages), "sections": verify_sections(data, chunks)}
    return {"turns": len(all_turns(data)), "reports": reports}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = ap.parse_args()
    result = run(args.cases)
    failed = False
    print(f"{result['turns']} requests in {args.cases.name}")
    for name, rep in result["reports"].items():
        print(f"[{'FAIL' if rep.errors else 'ok'}] {name}: {rep.checked}")
        for msg in rep.errors:
            print("   ERROR  ", msg)
        for msg in rep.warnings:
            print("   warning", msg)
        failed |= bool(rep.errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
