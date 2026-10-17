"""Stage 10 (part 2): hard checks on the final records.

Why independent: this module reads only pages.jsonl and the manifest's `validation:` block
(the single home of expected values), never any pipeline code, so a wrong rule cannot hide
by also being wrong in its own test. Each check returns a result instead of raising, so one
run reports every failure.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ingest.models import Manifest

ALLOWED_PREFIXES = ("http://", "https://", "mailto:")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _plain(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("#", " ").replace("*", " ")).strip()


def _lines(markdown: str) -> list[str]:
    return [l for l in markdown.split("\n") if l.strip()]


def _doc_pages(pages: list[dict], doc_id: str) -> list[dict]:
    return sorted((p for p in pages if p["doc_id"] == doc_id), key=lambda p: p["pdf_page"])


def _result(name: str, problems: list[str], ok_detail: str) -> Check:
    return Check(name, not problems, ok_detail if not problems else "; ".join(problems[:8]) + (
        f" (+{len(problems) - 8} more)" if len(problems) > 8 else ""))


def check_page_counts(pages, manifest) -> Check:
    problems = [f"{d.doc_id}: {len(_doc_pages(pages, d.doc_id))} pages, expected {d.expected_pages}" for d in manifest.documents
                if len(_doc_pages(pages, d.doc_id)) != d.expected_pages]
    total = sum(d.expected_pages for d in manifest.documents)
    if len(pages) != total:
        problems.append(f"{len(pages)} records in total, expected {total}")
    return _result("1 page counts", problems, f"{len(pages)} records ({', '.join(f'{d.doc_id}={d.expected_pages}' for d in manifest.documents)})")


def check_printed_pages(pages, manifest) -> Check:
    problems = []
    for d in manifest.documents:
        by_page = {p["pdf_page"]: p for p in _doc_pages(pages, d.doc_id)}
        for pdf_page, expected in d.validation.printed_page_spot_checks.items():
            got = by_page.get(pdf_page, {}).get("printed_page", "<missing>")
            if got != expected:
                problems.append(f"{d.doc_id} p.{pdf_page}: printed {got!r}, expected {expected!r}")
    return _result("2 printed-page spot checks", problems, "all spot checks match")


def check_page_types(pages, manifest) -> Check:
    problems = []
    for d in manifest.documents:
        by_page = {p["pdf_page"]: p for p in _doc_pages(pages, d.doc_id)}
        for rng in d.validation.page_types:
            for n in range(rng.first, rng.last + 1):
                got = by_page.get(n, {}).get("page_type")
                if got != rng.page_type:
                    problems.append(f"{d.doc_id} p.{n}: {got!r}, expected {rng.page_type!r}")
    return _result("3 page types", problems, "all pages match the expected ranges")


def check_no_running_text(pages, manifest) -> Check:
    """No stand-alone header/footer/page-number line survives anywhere on a page."""
    problems = []
    for d in manifest.documents:
        patterns = [re.compile(x) for x in d.validation.forbidden_line_patterns]
        for p in _doc_pages(pages, d.doc_id):
            plain = [_plain(l) for l in _lines(p["markdown"])]
            for line in plain:
                if line in d.validation.forbidden_lines or any(rx.match(line) for rx in patterns):
                    problems.append(f"{d.doc_id} p.{p['pdf_page']}: leftover line {line!r}")
            printed = p["printed_page"]
            if printed and plain and printed in (plain[0], plain[-1]):
                problems.append(f"{d.doc_id} p.{p['pdf_page']}: leftover page-number line {printed!r}")
    return _result("4 no header/footer/page-number lines", problems, "none found")


def check_section_ids(pages, manifest) -> Check:
    problems = []
    for d in manifest.documents:
        found = Counter(h["section_id"] for p in _doc_pages(pages, d.doc_id) for h in p["headings"])
        expected = set(d.validation.section_ids)
        problems += [f"{d.doc_id}: missing section {s!r}" for s in sorted(expected - set(found))]
        problems += [f"{d.doc_id}: unexpected section {s!r}" for s in sorted(set(found) - expected)]
        problems += [f"{d.doc_id}: section {s!r} found {n} times" for s, n in found.items() if n > 1]
    return _result("5 section IDs exactly once", problems, "all expected sections found once, none extra")


def check_tables_and_actions(pages, manifest) -> Check:
    problems, parts = [], []
    for d in manifest.documents:
        v = d.validation
        docs = _doc_pages(pages, d.doc_id)
        if v.subcategory_tables is not None:
            n = sum(len(p["table_markers"]) for p in docs)
            parts.append(f"{n} tables")
            if n != v.subcategory_tables:
                problems.append(f"{d.doc_id}: {n} subcategory table markers, expected {v.subcategory_tables}")
        if v.unique_action_ids is not None:
            ids = [a for p in docs for a in p["entities"]["action_ids"]]
            parts.append(f"{len(set(ids))} unique action IDs")
            if len(set(ids)) != v.unique_action_ids:
                problems.append(f"{d.doc_id}: {len(set(ids))} unique table-row action IDs, expected {v.unique_action_ids}")
            problems += [f"{d.doc_id}: action ID {a} is a table-row ID {n} times" for a, n in Counter(ids).items() if n > 1]
    return _result("6 subcategory tables and action IDs", problems, ", ".join(parts))


def check_figures(pages, manifest) -> Check:
    problems = []
    for d in manifest.documents:
        if d.validation.expected_figures is None:
            continue
        figs = [f for p in _doc_pages(pages, d.doc_id) for f in p["figures"]]
        if len(figs) != d.validation.expected_figures:
            problems.append(f"{d.doc_id}: {len(figs)} figure records, expected {d.validation.expected_figures}")
        problems += [f"{d.doc_id}: figure {f['figure_number']} has no caption" for f in figs if not f["caption"]]
    return _result("7 figure records with captions", problems, "expected figure records, all captioned")


def check_body_not_empty(pages, manifest) -> Check:
    problems = [f"{p['doc_id']} p.{p['pdf_page']}: empty markdown" for p in pages
                if p["page_type"] == "body" and not p["markdown"].strip()]
    return _result("8 no empty body page", problems, "every body page has text")


def check_link_schemes(pages, manifest) -> Check:
    urls = [(p, l["url"]) for p in pages for l in p["links"]]
    urls += [(p, l["url"]) for p in pages for fn in p["footnotes"] for l in fn["links"]]
    problems = [f"{p['doc_id']} p.{p['pdf_page']}: {u[:60]}" for p, u in urls if not u.startswith(ALLOWED_PREFIXES)]
    return _result("9 link URL schemes", problems, f"{len(urls)} link URLs, all http/https/mailto")


def check_references_not_eligible(pages, manifest) -> Check:
    problems = []
    for d in manifest.documents:
        by_page = {p["pdf_page"]: p for p in _doc_pages(pages, d.doc_id)}
        for rng in d.validation.page_types:
            if rng.page_type == "references":
                problems += [f"{d.doc_id} p.{n}: index_eligible is true" for n in range(rng.first, rng.last + 1)
                             if by_page.get(n, {}).get("index_eligible")]
    return _result("10 reference pages not index-eligible", problems, "all reference pages are index_eligible=false")


def check_footnote_counts(pages, manifest) -> Check:
    problems = []
    for d in manifest.documents:
        if d.validation.expected_footnotes is None:
            continue
        docs = _doc_pages(pages, d.doc_id)
        n = sum(len(p["footnotes"]) for p in docs)
        removals = sum("strip_footnote_markers" in p["cleaning_log"] for p in docs)
        if n != d.validation.expected_footnotes:
            problems.append(f"{d.doc_id}: {n} footnotes, expected {d.validation.expected_footnotes}")
        if d.validation.expected_footnotes == 0 and removals:
            problems.append(f"{d.doc_id}: footnote markers removed on {removals} pages, expected none")
    return _result("11 footnote counts", problems, "footnote counts match (none where none are expected)")


CHECKS = [check_page_counts, check_printed_pages, check_page_types, check_no_running_text, check_section_ids,
          check_tables_and_actions, check_figures, check_body_not_empty, check_link_schemes,
          check_references_not_eligible, check_footnote_counts]


def validate(pages: list[dict], manifest: Manifest) -> list[Check]:
    """Run every hard check; the caller exits non-zero if any failed."""
    return [check(pages, manifest) for check in CHECKS]
