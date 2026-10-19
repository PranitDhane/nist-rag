"""Stage V3: deterministic end-to-end checks (E2E plan Section 4: A, B, C, E, F) as pure functions over SAVED responses.

`python -m eval.e2e_checks --from-run eval/e2e_runs/<timestamp>` applies them to a run saved by `eval.e2e`. Nothing here
calls the server or an LLM. Every check returns pass / fail / na with a reason; checks are REPORTED, never gated (E-D6),
and the case file is only read, never adjusted to improve a result (CLAUDE.md section 5).

Group D (groundedness) is stage V4. C3 (link check) needs the PDFs, so it is skipped (na) unless a LinkContext is given."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval import e2e_cases as cases_mod
from eval.e2e_cases import contains, fact_found, section_matches

PASS, FAIL, NA = "pass", "fail", "na"
MARKER = re.compile(r"\[C(\d+)\]")
GATE_REASONS = {"gate_low_score", "no_candidates"}          # insufficient answers that never reached the model
MAX_WORDS = 150                                              # prompt rule; reported, not failed (plan B3)
RELATED_COUNT = 5
EN_STOPWORDS = {"the", "and", "of", "is", "are", "to", "in", "a", "that", "for", "with", "by", "as", "it", "this", "be"}


@dataclass
class Check:
    id: str
    name: str
    status: str
    detail: str = ""
    data: dict = field(default_factory=dict)      # structured extras for the report (facts found, words, labels ...)


@dataclass
class LinkContext:
    """What the C3 link check needs: PDF text per page, printed pages, figure captions and the base URL per document."""
    pdf_text: dict[str, dict[int, str]]
    printed: dict[tuple[str, int], str | None]
    captions: dict
    url_base: dict[str, str]


@dataclass
class Context:
    texts: dict[str, str]                         # fixed texts by config key (INSUFFICIENT_TEXT ...)
    chunks: dict[str, dict]                       # chunk_id -> chunk record (from chunks.jsonl)
    llm_calls: dict[str, int]                     # expected LLM calls per outcome (case file `defaults`)
    link: LinkContext | None = None


# ------------------------------------------------------------------ reading one saved record
@dataclass
class View:
    ok: bool                                      # HTTP 200 with a JSON body
    status: int | None
    error: str | None
    body: dict
    meta: dict
    debug: dict

    @property
    def answer(self) -> str:
        return self.body.get("answer") or ""

    @property
    def route(self) -> str | None:
        return self.body.get("route")

    @property
    def outcome(self) -> str | None:
        return self.meta.get("outcome")

    @property
    def sources(self) -> list[dict]:
        return self.body.get("sources") or []

    @property
    def key_map(self) -> dict:
        return self.debug.get("key_map") or {}


def view(record: dict) -> View:
    if "skipped" in record:
        return View(False, None, record["skipped"], {}, {}, {})
    body = record.get("body") if isinstance(record.get("body"), dict) else {}
    status = record.get("status")
    err = None
    if status != 200:
        detail = body.get("detail", body) if isinstance(body, dict) else {}
        err = (detail.get("error", {}) if isinstance(detail, dict) else {}).get("code") or body.get("harness_error") or f"HTTP {status}"
    meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
    debug = meta.get("debug") if isinstance(meta.get("debug"), dict) else {}
    return View(status == 200 and bool(body), status, err, body, meta, debug)


def expectation(case: dict, actual_route: str | None) -> dict:
    """What this turn is expected to do. If the case names an acceptable alternative and the system took it, the
    alternative is the expectation (and the result says so); otherwise the primary one."""
    primary = {"route": case["expected_route"], "outcome": case["expected_outcome"], "text_key": case.get("expected_text_key"),
               "alternative": False}
    alt = case.get("acceptable_alternative")
    if alt and actual_route == alt.get("expected_route") != primary["route"]:
        return {"route": alt["expected_route"], "outcome": alt.get("expected_outcome", alt["expected_route"]),
                "text_key": alt.get("expected_text_key"), "alternative": True}
    return primary


def _na(cid: str, name: str, why: str) -> Check:
    return Check(cid, name, NA, why)


def _no_response(v: View) -> str:
    return f"no response ({v.error})"


# ------------------------------------------------------------------ A. routing and outcome
def check_a(case: dict, v: View, ctx: Context) -> list[Check]:
    names = {"A1": "route", "A2": "outcome", "A3": "fixed text", "A4": "standalone query", "A5": "llm calls"}
    if not v.ok:
        return [Check("A1", names["A1"], FAIL, _no_response(v)), Check("A2", names["A2"], FAIL, _no_response(v)),
                *(_na(k, names[k], _no_response(v)) for k in ("A3", "A4", "A5"))]
    exp = expectation(case, v.route)
    alt_note = " (acceptable alternative)" if exp["alternative"] else ""
    out: list[Check] = []

    want_source = case.get("expected_route_source")
    route_ok = v.route == exp["route"]
    source_ok = want_source is None or v.meta.get("route_source") == want_source
    detail = f"route={v.route}" + (f", route_source={v.meta.get('route_source')}" if want_source else "")
    if not (route_ok and source_ok):
        detail += f"; expected route={exp['route']}" + (f", route_source={want_source}" if want_source else "")
    out.append(Check("A1", names["A1"], PASS if route_ok and source_ok else FAIL, detail + alt_note,
                     {"alternative": exp["alternative"], "preferred_route": case["expected_route"]}))

    out.append(Check("A2", names["A2"], PASS if v.outcome == exp["outcome"] else FAIL,
                     f"outcome={v.outcome}" + ("" if v.outcome == exp["outcome"] else f"; expected {exp['outcome']}") + alt_note))

    if exp["text_key"]:
        want = ctx.texts[exp["text_key"]]
        out.append(Check("A3", names["A3"], PASS if v.answer == want else FAIL,
                         f"{exp['text_key']} " + ("matches exactly" if v.answer == want else f"differs: got {v.answer[:80]!r}")))
    else:
        out.append(_na("A3", names["A3"], "no fixed text expected"))

    terms = case.get("expected_standalone_terms")
    if terms:
        sq = v.debug.get("standalone_query") or v.body.get("standalone_query") or ""
        missing_all = [t for t in terms.get("all_of", []) if not contains(sq, t)]
        any_of = terms.get("any_of", [])
        any_ok = not any_of or any(contains(sq, t) for t in any_of)
        ok = not missing_all and any_ok
        why = f"standalone_query={sq!r}" + (f"; missing all_of {missing_all}" if missing_all else "") + \
            ("" if any_ok else f"; none of any_of {any_of}")
        out.append(Check("A4", names["A4"], PASS if ok else FAIL, why))
    else:
        out.append(_na("A4", names["A4"], "not a follow-up case"))

    out.append(_a5(case, v, ctx, exp, names["A5"]))
    return out


def expected_llm_calls(outcome: str, v: View, table: dict[str, int]) -> int | None:
    """Plan A5: 0 for a rule greeting, 1 for out of scope / gate failure / LLM greeting, 2 when generation ran."""
    if outcome == "greeting":
        return table["greeting_rule"] if v.meta.get("route_source") == "rule" else table["out_of_scope"]
    if outcome == "insufficient":
        return table["insufficient_gate"] if v.meta.get("insufficient_reason") in GATE_REASONS else table["insufficient_model"]
    return table.get(outcome)


def _a5(case: dict, v: View, ctx: Context, exp: dict, name: str) -> Check:
    want = expected_llm_calls(exp["outcome"], v, ctx.llm_calls)
    got = v.meta.get("llm_calls")
    if want is None or got is None:
        return _na("A5", name, "no expected value or no llm_calls in the response")
    return Check("A5", name, PASS if got == want else FAIL, f"llm_calls={got}, expected {want}")


# ------------------------------------------------------------------ B. content
def strip_markers(text: str) -> str:
    return re.sub(r"\s+", " ", MARKER.sub("", text)).strip()


def is_english(text: str) -> bool:
    """Heuristic for plan B3: almost all letters are Latin and common English words are present. A Hindi answer fails."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin = sum(("a" <= c.lower() <= "z") for c in letters)
    words = set(re.findall(r"[a-z']+", text.lower()))
    return latin / len(letters) >= 0.95 and len(words & EN_STOPWORDS) >= 3


def check_b(case: dict, v: View, ctx: Context) -> list[Check]:
    names = {"B1": "must-contain facts", "B2": "traps", "B3": "language and length"}
    if not v.ok:
        return [_na(k, n, _no_response(v)) for k, n in names.items()]
    text = strip_markers(v.answer)
    out: list[Check] = []

    facts = case.get("must_contain") or []
    if facts:
        found = [f["fact"] for f in facts if fact_found(text, f["variants"])]
        missing = [f["fact"] for f in facts if f["fact"] not in found]
        need = case["min_facts"]
        out.append(Check("B1", names["B1"], PASS if len(found) >= need else FAIL,
                         f"{len(found)}/{len(facts)} facts found (need {need})" + (f"; missing: {missing}" if missing else ""),
                         {"found": found, "missing": missing, "min_facts": need, "total": len(facts)}))
    else:
        out.append(_na("B1", names["B1"], "no must-contain facts"))

    traps = case.get("must_not_contain") or []
    if traps:
        hit = [t["fact"] for t in traps if fact_found(text, t["variants"])]
        out.append(Check("B2", names["B2"], FAIL if hit else PASS, f"trap(s) hit: {hit}" if hit else "no trap hit", {"hit": hit}))
    else:
        out.append(_na("B2", names["B2"], "no traps"))

    if v.outcome == "answered":
        words = len(text.split())
        eng = is_english(text)
        out.append(Check("B3", names["B3"], PASS if eng else FAIL,
                         f"{'English' if eng else 'NOT English'}, {words} words" + (f" (over {MAX_WORDS})" if words > MAX_WORDS else ""),
                         {"words": words, "over_length": words > MAX_WORDS, "english": eng}))
    else:
        out.append(_na("B3", names["B3"], "not an answered turn"))
    return out


# ------------------------------------------------------------------ C. citations
def cited_keys(v: View) -> list[str]:
    return sorted({k for s in v.sources for k in s.get("keys", [])}, key=lambda k: int(k[1:]) if k[1:].isdigit() else 0)


def cited_chunks(v: View, ctx: Context) -> tuple[dict[str, dict], list[str]]:
    """{key: chunk record} for the cited keys, via debug.key_map and chunks.jsonl; plus the keys that could not be resolved."""
    found, missing = {}, []
    for k in cited_keys(v):
        chunk = ctx.chunks.get(v.key_map.get(k, ""))
        (found.__setitem__(k, chunk) if chunk else missing.append(k))
    return found, missing


def check_c(case: dict, v: View, ctx: Context) -> list[Check]:
    names = {"C1": "markers match sources", "C2": "expected sections", "C3": "link check", "C4": "fixes and warnings"}
    if case["expected_outcome"] != "answered":
        return [_na(k, n, "no citations expected for this outcome") for k, n in names.items() if k != "C4"] + \
               [_c4(v, names["C4"])]
    if not v.ok:
        return [_na(k, n, _no_response(v)) for k, n in names.items()]
    return [_c1(v, names["C1"]), _c2(case, v, ctx, names["C2"]), _c3(v, ctx, names["C3"]), _c4(v, names["C4"])]


def _c1(v: View, name: str) -> Check:
    in_text = {f"C{n}" for n in MARKER.findall(v.answer)}
    in_sources = set(cited_keys(v))
    if not in_text and not in_sources:
        return Check("C1", name, FAIL, "no [C#] marker and no source: an answered turn must cite")
    only_text, only_src = sorted(in_text - in_sources), sorted(in_sources - in_text)
    if only_text or only_src:
        return Check("C1", name, FAIL, f"in text only: {only_text}; in sources only: {only_src}")
    return Check("C1", name, PASS, f"{len(in_text)} marker(s) = {len(in_sources)} source key(s)")


def _c2(case: dict, v: View, ctx: Context, name: str) -> Check:
    if not v.sources:
        return Check("C2", name, FAIL, "no sources, so no expected section can be cited")
    if not v.key_map:
        return _na("C2", name, "debug.key_map missing: cannot map citations to chunks")
    found, unresolved = cited_chunks(v, ctx)
    if unresolved:
        return Check("C2", name, FAIL, f"cited key(s) {unresolved} not resolvable through key_map/chunks.jsonl")
    expected = case.get("expected_sections") or []
    forbidden = case.get("must_not_cite") or []
    matched = sorted({f"{s['doc']} {s['section']}" for c in found.values() for s in expected
                      if c["doc_id"] == s["doc"] and section_matches(c, s["section"])})
    bad = sorted({f"{s['doc']} {s['section']}" for c in found.values() for s in forbidden
                  if c["doc_id"] == s["doc"] and section_matches(c, s["section"])})
    wanted_docs = {s["doc"] for s in expected}
    other_docs = sorted({c["doc_id"] for c in found.values()} - wanted_docs)
    labels = sorted({c["citation_label"] for c in found.values()})
    ok = bool(matched) and not bad
    detail = (f"matched {matched}" if matched else f"none of {[s['section'] for s in expected]} cited") + \
             (f"; FORBIDDEN section cited: {bad}" if bad else "") + (f"; also cited from other document(s): {other_docs}" if other_docs else "")
    return Check("C2", name, PASS if ok else FAIL, detail, {"matched": matched, "forbidden_cited": bad, "other_docs": other_docs, "labels": labels})


def _c3(v: View, ctx: Context, name: str) -> Check:
    """Each source's link must open a page that contains the cited chunk's text (reusing eval/link_check.py), and the
    URL in the response must be the one the chunk's page gives."""
    from eval import link_check as lc
    if ctx.link is None:
        return _na("C3", name, "link context not loaded (needs the PDFs; run without --no-links)")
    if not v.key_map or not v.sources:
        return _na("C3", name, "no sources or no key_map")
    found, unresolved = cited_chunks(v, ctx)
    problems, explained = [], []
    for src in v.sources:
        for key in src.get("keys", []):
            chunk = found.get(key)
            if chunk is None:
                continue
            doc, first = chunk["doc_id"], chunk["pdf_page_start"]
            want_url = f"{ctx.link.url_base[doc]}#page={first}"
            if src.get("url") != want_url:
                problems.append(f"{key}: response url {src.get('url')!r} != {want_url!r}")
            for f in lc.check_chunk(chunk, ctx.link.pdf_text[doc], ctx.link.printed, ctx.link.captions, ctx.link.url_base[doc]):
                (explained if f["check"] in lc.EXPLAINED else problems).append(f"{key}: {f['check']}: {f['detail']}")
    if unresolved:
        problems.append(f"unresolvable key(s) {unresolved}")
    if problems:
        return Check("C3", name, FAIL, "; ".join(problems), {"explained": explained})
    return Check("C3", name, PASS, f"{sum(len(s.get('keys', [])) for s in v.sources)} cited chunk(s) open the right page" +
                 (f" ({len(explained)} sample mismatch(es) explained by layout)" if explained else ""), {"explained": explained})


def _c4(v: View, name: str) -> Check:
    if not v.ok:
        return _na("C4", name, _no_response(v))
    fixes, warnings = v.meta.get("fixes"), v.meta.get("warnings")
    if fixes is None and warnings is None:
        return _na("C4", name, "generation did not run")
    fixes, warnings = fixes or [], warnings or []
    uncited = sum(w.startswith("UNCITED_SENTENCE") for w in warnings)
    if not fixes and not warnings:
        return Check("C4", name, PASS, "no fixes, no warnings", {"fixes": [], "warnings": []})
    return Check("C4", name, FAIL, f"fixes={fixes} warnings={warnings}", {"fixes": fixes, "warnings": warnings, "uncited_sentence_warnings": uncited})


# ------------------------------------------------------------------ E. insufficient path
def check_e(case: dict, v: View, ctx: Context) -> list[Check]:
    names = {"E1": "exact sentence", "E2": "sources empty", "E3": "related sections", "E4": "reason recorded"}
    exp = expectation(case, v.route) if v.ok else {"outcome": case["expected_outcome"], "alternative": False}
    if exp["outcome"] != "insufficient":
        return [_na(k, n, "not an insufficient turn") for k, n in names.items()]
    if not v.ok:
        return [_na(k, n, _no_response(v)) for k, n in names.items()]
    related = v.body.get("related_sections") or []
    reason = v.meta.get("insufficient_reason")
    allowed = case.get("expected_insufficient_reason_any")
    out = [Check("E1", names["E1"], PASS if v.answer == ctx.texts["INSUFFICIENT_TEXT"] else FAIL, f"answer={v.answer[:80]!r}"),
           Check("E2", names["E2"], PASS if not v.sources else FAIL, f"{len(v.sources)} source(s)"),
           Check("E3", names["E3"], PASS if len(related) == RELATED_COUNT else FAIL, f"{len(related)} related section(s), expected {RELATED_COUNT}",
                 {"labels": [r.get("citation_label") for r in related]})]
    if not reason:
        out.append(Check("E4", names["E4"], FAIL, "insufficient_reason missing"))
    elif allowed and reason not in allowed:
        out.append(Check("E4", names["E4"], FAIL, f"insufficient_reason={reason}, expected one of {allowed}", {"reason": reason}))
    else:
        out.append(Check("E4", names["E4"], PASS, f"insufficient_reason={reason}", {"reason": reason}))
    return out


# ------------------------------------------------------------------ F. operational
def check_f(case: dict, record: dict, v: View) -> list[Check]:
    status = record.get("status")
    f1 = Check("F1", "HTTP status", PASS if status == 200 else FAIL,
               "HTTP 200" if status == 200 else f"status={status}" + (f" ({v.error})" if v.error else ""),
               {"status": status, "retried": bool(record.get("retried"))})
    if not v.ok:
        return [f1, _na("F2", "latency and tokens", _no_response(v))]
    usage = v.meta.get("usage") or {}
    data = {"latency_ms": v.meta.get("latency_ms"), "client_ms": record.get("elapsed_ms"), "tokens": usage.get("total"),
            "router_tokens": usage.get("router"), "generation_tokens": usage.get("generation"), "llm_calls": v.meta.get("llm_calls"),
            "waited_s": record.get("waited_s")}
    have = data["latency_ms"] is not None and data["llm_calls"] is not None
    return [f1, Check("F2", "latency and tokens", PASS if have else NA,
                      f"{data['latency_ms']} ms, {data['tokens']} tokens, {data['llm_calls']} LLM call(s)" if have else "meta incomplete", data)]


# ------------------------------------------------------------------ one turn, a whole run
def run_turn_checks(case: dict, record: dict, ctx: Context) -> list[Check]:
    v = view(record)
    return [*check_a(case, v, ctx), *check_b(case, v, ctx), *check_c(case, v, ctx), *check_e(case, v, ctx), *check_f(case, record, v)]


def run_checks(data: dict, records: list[dict], ctx: Context) -> list[dict]:
    """[{id, conversation, message, checks: [Check as dict]}] in run order. A saved turn the case file does not know is
    reported as an error row (the case file changed since the run), never silently dropped."""
    by_id = {t["id"]: t for t in cases_mod.all_turns(data)}
    rows = []
    for rec in records:
        case = by_id.get(rec["id"])
        if case is None:
            rows.append({"id": rec["id"], "conversation": rec.get("conversation"), "error": "turn id not in the case file", "checks": []})
            continue
        rows.append({"id": rec["id"], "conversation": case.get("conversation"), "message": case["message"],
                     "checks": [asdict(c) for c in run_turn_checks(case, rec, ctx)]})
    return rows


# ------------------------------------------------------------------ context and command line
def load_link_context() -> LinkContext:
    import pymupdf
    from eval import link_check as lc
    from ingest import config as ingest_config
    manifest = ingest_config.load_manifest()
    pages = lc.read_jsonl(ingest_config.PROCESSED_DIR / "pages.jsonl")
    pdf_text = {}
    for spec in manifest.documents:
        with pymupdf.open(ingest_config.RAW_DIR / spec.filename) as doc:
            pdf_text[spec.doc_id] = {i + 1: p.get_text() for i, p in enumerate(doc)}
    return LinkContext(pdf_text=pdf_text, printed={(p["doc_id"], p["pdf_page"]): p["printed_page"] for p in pages},
                       captions={(p["doc_id"], p["pdf_page"], str(f["figure_number"])): f.get("caption") for p in pages for f in p["figures"]},
                       url_base={s.doc_id: s.url for s in manifest.documents})


def build_context(data: dict, with_links: bool = True) -> Context:
    chunks = {c["chunk_id"]: c for c in (json.loads(line) for line in
              (cases_mod.ROOT / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    return Context(texts=cases_mod.text_constants(), chunks=chunks, llm_calls=dict(data["defaults"]["llm_calls"]),
                   link=load_link_context() if with_links else None)


def results_table(rows: list[dict]) -> str:
    """One line per turn: the failed check ids, so a glance shows where to look; details are in the JSON."""
    sym = {PASS: ".", FAIL: "X", NA: "-"}
    out = [f"{'id':8} {'checks':44} failed"]
    for r in rows:
        if r.get("error"):
            out.append(f"{r['id']:8} {r['error']}")
            continue
        cells = "".join(f"{c['id']}{sym[c['status']]} " for c in r["checks"])
        failed = [c["id"] for c in r["checks"] if c["status"] == FAIL]
        out.append(f"{r['id']:8} {cells:44} {','.join(failed) or '-'}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from eval import e2e
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-run", type=Path, required=True)
    ap.add_argument("--cases", type=Path, default=cases_mod.DEFAULT_CASES)
    ap.add_argument("--no-links", action="store_true", help="skip C3 (does not read the PDFs)")
    ap.add_argument("--out", type=Path, help="write the per-turn results as JSON")
    args = ap.parse_args(argv)
    try:
        run = e2e.load_run(args.from_run)
    except e2e.HarnessError as exc:
        print(f"e2e_checks: {exc}", file=sys.stderr)
        return 2
    data = cases_mod.load_cases(args.cases)
    rows = run_checks(data, run["records"], build_context(data, with_links=not args.no_links))
    print(results_table(rows))
    for r in rows:
        for c in r["checks"]:
            if c["status"] == FAIL:
                print(f"  FAIL {r['id']} {c['id']} {c['name']}: {c['detail']}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
