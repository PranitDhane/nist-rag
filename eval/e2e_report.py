"""Stage V6: apply every check to a saved run and write the report.
`python -m eval.e2e_report --from-run eval/e2e_runs/<timestamp>` writes <run>/report.json and docs/E2E_RESULTS.md.

Report only (E-D6): nothing here gates, every failed check is listed, and every number in the Markdown comes from the
report dict (CLAUDE.md: no hand-typed numbers). The case file and the saved responses are read, never changed."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval import e2e, e2e_checks as ck, e2e_config as cfg, e2e_ground as ground
from eval import e2e_cases as cases_mod

LIMITATIONS = [
    "The support score (group D) uses the same reranker model as retrieval, so it is not an independent judge; a sentence can "
    "score high on topic overlap with a subtle factual error. The gold facts (B) and the citation checks (C) complement it.",
    "Temperature 0.2 means wording can differ between two runs of the same question; each case ran once, so single results carry variance.",
    "Fact matching is lexical (case-insensitive, hyphen-insensitive, word-start prefix): a correct answer in other words can miss a "
    "fact, and a fact word in a wrong context can count.",
    "Sixteen requests is a small sample: the percentages are descriptive, not statistically tight.",
    "The English check is a heuristic (script and common English words).",
]


# ------------------------------------------------------------------ aggregates
def percentile(values: list[float], p: float) -> float | None:
    """Nearest rank, so the value is one that occurred."""
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, math.ceil(p * len(v)) - 1))]


def rate(rows: list[dict], check_id: str) -> dict:
    """{passed, applicable, pct}: na results are not counted (they did not apply)."""
    got = [c for r in rows for c in r["checks"] if c["id"] == check_id and c["status"] != ck.NA]
    passed = sum(c["status"] == ck.PASS for c in got)
    return {"passed": passed, "applicable": len(got), "pct": round(100 * passed / len(got), 1) if got else None}


def checks_of(rows: list[dict], check_id: str) -> list[dict]:
    return [c for r in rows for c in r["checks"] if c["id"] == check_id and c["status"] != ck.NA]


def turn_type(record: dict) -> str:
    v = ck.view(record)
    if not v.ok:
        return "error"
    if v.outcome == "insufficient":
        return "insufficient (gate)" if v.meta.get("insufficient_reason") in ck.GATE_REASONS else "insufficient (model)"
    return v.outcome or "unknown"


def aggregates(rows: list[dict], records: list[dict], stream_rows: list[dict]) -> dict:
    facts = checks_of(rows, "B1")
    found = sum(len(c["data"].get("found", [])) for c in facts)
    total = sum(c["data"].get("total", 0) for c in facts)
    d2 = checks_of(rows, "D2")
    sup, sup_total = sum(c["data"].get("supported", 0) for c in d2), sum(c["data"].get("total", 0) for c in d2)
    d3 = checks_of(rows, "D3")
    uncited, cited = sum(c["data"].get("count", 0) for c in d3), sum(c["data"].get("cited_sentences", 0) for c in d3)
    e_turns = [r for r in rows if any(c["id"] == "E1" and c["status"] != ck.NA for c in r["checks"])]
    e_ok = [r for r in e_turns if all(c["status"] == ck.PASS for c in r["checks"] if c["id"] in ("E1", "E2", "E3", "E4"))]
    b3 = checks_of(rows, "B3")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_type[turn_type(rec)].append(rec)
    per_type = {}
    for name, recs in sorted(by_type.items()):
        views = [ck.view(r) for r in recs if ck.view(r).ok]
        lat = [v.meta["latency_ms"] for v in views if v.meta.get("latency_ms") is not None]
        tokens = [(v.meta.get("usage") or {}).get("total") for v in views]
        tokens = [t for t in tokens if t is not None]
        calls = [v.meta["llm_calls"] for v in views if v.meta.get("llm_calls") is not None]
        per_type[name] = {"turns": len(recs), "latency_ms_p50": percentile(lat, 0.5), "latency_ms_p95": percentile(lat, 0.95),
                          "tokens_mean": round(sum(tokens) / len(tokens)) if tokens else None, "tokens_max": max(tokens) if tokens else None,
                          "llm_calls": sorted(set(calls))}
    all_lat = [ck.view(r).meta["latency_ms"] for r in records if ck.view(r).ok and ck.view(r).meta.get("latency_ms") is not None]
    return {
        "route_accuracy": rate(rows, "A1"), "outcome_accuracy": rate(rows, "A2"), "exact_text_accuracy": rate(rows, "A3"),
        "follow_up_resolution": rate(rows, "A4"), "llm_calls_as_expected": rate(rows, "A5"),
        "key_fact_coverage": {"found": found, "total": total, "pct": round(100 * found / total, 1) if total else None},
        "turns_meeting_min_facts": rate(rows, "B1"),
        "trap_hits": {"turns_with_a_hit": sum(bool(c["data"].get("hit")) for c in checks_of(rows, "B2")),
                      "traps_checked_turns": len(checks_of(rows, "B2")),
                      "hits": [h for c in checks_of(rows, "B2") for h in c["data"].get("hit", [])]},
        "english": rate(rows, "B3"),
        "over_length_answers": {"count": sum(bool(c["data"].get("over_length")) for c in b3), "of": len(b3),
                                "words_max": max((c["data"].get("words", 0) for c in b3), default=None)},
        "citation_consistency": rate(rows, "C1"), "expected_section_hit": rate(rows, "C2"), "link_check": rate(rows, "C3"),
        "no_fixes_or_warnings": rate(rows, "C4"),
        "support_rate": {"supported": sup, "total": sup_total, "pct": round(100 * sup / sup_total, 1) if sup_total else None,
                         "threshold": next((c["data"].get("threshold") for c in d2), None)},
        "uncited_sentence_rate": {"uncited": uncited, "cited": cited, "pct": round(100 * uncited / (uncited + cited), 1) if uncited + cited else None},
        "insufficient_path": {"turns": len(e_turns), "all_checks_passed": len(e_ok)},
        "http_ok": rate(rows, "F1"),
        "latency_ms": {"p50": percentile(all_lat, 0.5), "p95": percentile(all_lat, 0.95), "n": len(all_lat)},
        "by_turn_type": per_type,
        "streaming": {cid: rate(stream_rows, cid) for cid in ("ST1", "ST2", "ST3")},
        "alternatives_taken": [r["id"] for r in rows for c in r["checks"] if c["id"] == "A1" and c["data"].get("alternative")],
    }


def failures(rows: list[dict], stream_rows: list[dict]) -> list[dict]:
    """Every failed check, unsoftened, in run order (plan Section 7)."""
    out = []
    for kind, group in (("chat", rows), ("stream", stream_rows)):
        for r in group:
            out += [{"turn": r["id"], "via": kind, "check": c["id"], "name": c["name"], "detail": c["detail"]}
                    for c in r["checks"] if c["status"] == ck.FAIL]
    return out


def build_report(data: dict, run: dict, rows: list[dict], stream_rows: list[dict], calibration: dict | None, threshold: float | None,
                 run_dir: Path) -> dict:
    man = run["manifest"]
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": run_dir.name,
            "run_started_at": man.get("started_at"), "base_url": man.get("base_url"), "pace_seconds": man.get("pace_seconds"),
            "cases_sha256": man.get("cases_sha256"), "requests": len(run["records"]), "streamed": len(stream_rows),
            "server": {k: man.get("server_health", {}).get(k) for k in ("model",)},
            "support_threshold": threshold, "calibration": calibration and {k: calibration[k] for k in
                ("positives", "negatives", "negatives_best_of_3", "sanity", "overlap", "seed")},
            "aggregates": aggregates(rows, run["records"], stream_rows), "failures": failures(rows, stream_rows),
            "per_turn": rows, "stream": stream_rows, "limitations": LIMITATIONS}


# ------------------------------------------------------------------ Markdown (every number comes from the report dict)
def pct(r: dict) -> str:
    return "n/a" if r["pct"] is None else f"{r['pct']}% ({r['passed']}/{r['applicable']})"


def fnum(x, unit: str = "") -> str:
    return "n/a" if x is None else f"{x:,}{unit}"


def render_markdown(rep: dict) -> str:
    a = rep["aggregates"]
    L = ["# End-to-end evaluation results", "",
         "_Generated by `python -m eval.e2e_report`; do not edit by hand. Report only: nothing here is a pass/fail gate._", "",
         f"- Run: `{rep['run']}` against `{rep['base_url']}` (router model `{rep['server'].get('model')}`; the answering model is in each turn's `meta.model`)",
         f"- {rep['requests']} `/chat` requests + {rep['streamed']} `/chat/stream` requests, paced {fnum(rep['pace_seconds'], ' s')} after generation",
         f"- Case file sha256 `{(rep['cases_sha256'] or '')[:12]}`; support-score threshold: "
         f"{rep['support_threshold'] if rep['support_threshold'] is not None else 'not chosen'}", "",
         "## Summary", "", "| Metric | Result |", "|---|---|",
         f"| Route accuracy (A1) | {pct(a['route_accuracy'])} |", f"| Outcome accuracy (A2) | {pct(a['outcome_accuracy'])} |",
         f"| Fixed texts exact (A3) | {pct(a['exact_text_accuracy'])} |", f"| Follow-up resolved in standalone query (A4) | {pct(a['follow_up_resolution'])} |",
         f"| LLM calls as expected (A5) | {pct(a['llm_calls_as_expected'])} |",
         f"| Key-fact coverage (B1, all facts) | {a['key_fact_coverage']['pct']}% ({a['key_fact_coverage']['found']}/{a['key_fact_coverage']['total']}) |"
         if a["key_fact_coverage"]["total"] else "| Key-fact coverage (B1) | n/a |",
         f"| Answered turns meeting `min_facts` (B1) | {pct(a['turns_meeting_min_facts'])} |",
         f"| Turns with a trap hit (B2) | {a['trap_hits']['turns_with_a_hit']} of {a['trap_hits']['traps_checked_turns']} |",
         f"| English answers (B3) | {pct(a['english'])} |",
         f"| Answers over ~150 words (B3) | {a['over_length_answers']['count']} of {a['over_length_answers']['of']} (longest {fnum(a['over_length_answers']['words_max'], ' words')}) |",
         f"| `[C#]` markers match sources (C1) | {pct(a['citation_consistency'])} |", f"| Expected section cited (C2) | {pct(a['expected_section_hit'])} |",
         f"| Links open the cited text (C3) | {pct(a['link_check'])} |", f"| No fixes or warnings (C4) | {pct(a['no_fixes_or_warnings'])} |",
         f"| Sentences supported by their cited chunk (D2, threshold {a['support_rate']['threshold']}) | "
         f"{'n/a' if a['support_rate']['pct'] is None else str(a['support_rate']['pct']) + '% (' + str(a['support_rate']['supported']) + '/' + str(a['support_rate']['total']) + ')'} |",
         f"| Uncited fact-stating sentences (D3) | {'n/a' if a['uncited_sentence_rate']['pct'] is None else str(a['uncited_sentence_rate']['pct']) + '% (' + str(a['uncited_sentence_rate']['uncited']) + ' uncited, ' + str(a['uncited_sentence_rate']['cited']) + ' cited)'} |",
         f"| Insufficient path fully correct (E1-E4) | {a['insufficient_path']['all_checks_passed']} of {a['insufficient_path']['turns']} turns |",
         f"| HTTP 200 (F1) | {pct(a['http_ok'])} |", f"| Latency p50 / p95 | {fnum(a['latency_ms']['p50'], ' ms')} / {fnum(a['latency_ms']['p95'], ' ms')} (n={a['latency_ms']['n']}) |",
         *( [f"| Stream {cid} | {pct(r)} |" for cid, r in a["streaming"].items()] ),
         ""]
    if a["alternatives_taken"]:
        L += [f"Acceptable alternative route taken (reported, counted as a pass): {', '.join(a['alternatives_taken'])}.", ""]
    L += ["## Per turn type", "", "| Type | Turns | LLM calls | Tokens (mean / max) | Latency p50 / p95 |", "|---|---|---|---|---|"]
    for name, t in a["by_turn_type"].items():
        L.append(f"| {name} | {t['turns']} | {', '.join(map(str, t['llm_calls'])) or '-'} | {fnum(t['tokens_mean'])} / {fnum(t['tokens_max'])} | "
                 f"{fnum(t['latency_ms_p50'], ' ms')} / {fnum(t['latency_ms_p95'], ' ms')} |")
    L += ["", "## Per turn", "", "| Turn | Message | Route / outcome | Failed checks | Words | Latency | Tokens |", "|---|---|---|---|---|---|---|"]
    for r in rep["per_turn"]:
        cs = {c["id"]: c for c in r["checks"]}
        failed = ", ".join(c["id"] for c in r["checks"] if c["status"] == ck.FAIL) or "-"
        a1 = cs.get("A1", {"detail": ""})["detail"].split(";")[0]
        a2 = cs.get("A2", {"detail": ""})["detail"].split(";")[0]
        f2 = cs.get("F2", {"data": {}})["data"]
        words = cs.get("B3", {"data": {}})["data"].get("words")
        L.append(f"| {r['id']} | {(r.get('message') or '')[:60].replace('|', '/')} | {a1}, {a2} | {failed} | {words if words is not None else '-'} | "
                 f"{fnum(f2.get('latency_ms'), ' ms')} | {fnum(f2.get('tokens'))} |")
    cal = rep.get("calibration")
    if cal:
        L += ["", "## Groundedness calibration (support score)", "", "| Set | n | min | p10 | median | p90 | max |", "|---|---|---|---|---|---|---|"]
        for key, label in (("positives", "positives (sentence vs its cited chunk)"), ("negatives", "negative controls (3 random other-section chunks)"),
                           ("sanity", "sanity (verbatim chunk sentence)")):
            d = cal[key]
            L.append(f"| {label} | {d.get('n', 0)} | " + " | ".join(f"{d[k]:.3f}" if d.get(k) is not None else "-" for k in ("min", "p10", "median", "p90", "max")) + " |")
        if cal.get("overlap"):
            o = cal["overlap"]
            L += ["", f"Overlap: {o['positives_at_or_below_negative_p90_pct']}% of positives score at or below the negative p90; "
                      f"{o['negatives_at_or_above_positive_p10_pct']}% of negatives score at or above the positive p10; "
                      f"separable: {o['separable']}."]
    if rep["stream"]:
        L += ["", "## Streaming subset", "", "| Turn | ST1 order | ST2 nothing after error | ST3 same structure as /chat | First token |", "|---|---|---|---|---|"]
        for r in rep["stream"]:
            cs = {c["id"]: c for c in r["checks"]}
            L.append(f"| {r['id']} | {cs['ST1']['status']} | {cs['ST2']['status']} | {cs['ST3']['status']} | {fnum(cs['ST4']['data'].get('first_token_ms'), ' ms')} |")
    L += ["", f"## Failed checks ({len(rep['failures'])}, none filtered)", ""]
    if rep["failures"]:
        L += ["| Turn | Via | Check | What happened |", "|---|---|---|---|"]
        L += [f"| {f['turn']} | {f['via']} | {f['check']} {f['name']} | {f['detail'].replace('|', '/')} |" for f in rep["failures"]]
    else:
        L.append("No check failed.")
    L += ["", "## Limitations", ""] + [f"- {t}" for t in rep["limitations"]] + [""]
    return "\n".join(L)


# ------------------------------------------------------------------ command line
def evaluate_run(run_dir: Path, cases_path: Path, with_links: bool, scorer, threshold: float | None, calibration: dict | None) -> dict:
    data = cases_mod.load_cases(cases_path)
    run = e2e.load_run(run_dir)
    ctx = ck.build_context(data, with_links=with_links)
    ctx.scorer, ctx.support_threshold = scorer, threshold
    rows = ck.run_checks(data, run["records"], ctx)
    stream_rows = ck.run_stream_checks(run["stream_records"], run["records"], data)
    return build_report(data, run, rows, stream_rows, calibration, threshold, run_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-run", type=Path, required=True)
    ap.add_argument("--cases", type=Path, default=cases_mod.DEFAULT_CASES)
    ap.add_argument("--support-threshold", type=float, default=cfg.SUPPORT_THRESHOLD)
    ap.add_argument("--no-links", action="store_true", help="skip C3 (does not read the PDFs)")
    ap.add_argument("--no-scorer", action="store_true", help="skip D1/D2 (does not load the reranker)")
    ap.add_argument("--md", type=Path, default=cases_mod.ROOT / cfg.RESULTS_MD, help="where to write the Markdown summary")
    args = ap.parse_args(argv)
    try:
        calibration = None
        cal_path = args.from_run / "calibration.json"
        if cal_path.is_file():
            calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        scorer = None if args.no_scorer else ground.reranker_scorer()
        rep = evaluate_run(args.from_run, args.cases, not args.no_links, scorer, args.support_threshold, calibration)
    except e2e.HarnessError as exc:
        print(f"e2e_report: {exc}", file=sys.stderr)
        return cfg.EXIT_HARNESS_FAILURE
    (args.from_run / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md.write_text(render_markdown(rep), encoding="utf-8")
    print(f"wrote {args.from_run / 'report.json'} and {args.md}; {len(rep['failures'])} failed check(s) listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
