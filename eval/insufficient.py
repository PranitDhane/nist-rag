"""Stage I5: the insufficient-context path on REAL data.  python -m eval.insufficient [--no-live]

* every `answerable: false` case of the retrieval evaluation set goes through retrieval + the gate; a case that
  passes the gate goes on to LIVE generation (read-only use of the generation layer) so the model-refusal path is seen too;
* 3 answerable cases run the full path (retrieve -> gate -> live generation) and must NOT end up insufficient.
Writes eval/insufficient_report.json. Misses never fail the run; they are for the user to judge (IC-D4 consequence).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import yaml

from insufficient.reply import finalize_model_refusal, gate_failure_reply

CASES_PATH = Path(__file__).with_name("retrieval_cases.yaml")
REPORT_PATH = Path(__file__).with_name("insufficient_report.json")
ANSWERABLE_IDS = ("c02", "c06", "c09")          # direct, exact action ID (table), and the paraphrase the router once mis-routed
PACE_SECONDS = 35                               # between live generation calls (free tier: 8,000 tokens/minute)
STOPWORDS = {"what", "does", "with", "that", "this", "from", "have", "your", "about", "which", "should", "under", "their",
             "they", "them", "when", "where", "there", "write", "code", "nist", "say", "says", "company", "companies",
             "many", "much", "steps", "get", "how", "can", "used", "uses", "use", "for", "the", "and", "are", "its", "set"}


def load_cases() -> list[dict]:
    return yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def query_terms(query: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]{3,}", query.lower()) if w not in STOPWORDS]


def overlap_flag(query: str, related: list[dict]) -> bool:
    """True when NO content word of the query appears in any related section title: the list looks unrelated.
    A heuristic to point the user at cases worth reading; the labels themselves are in the report."""
    terms = query_terms(query)
    titles = " ".join(e["section"].lower() for e in related)
    return bool(terms) and not any(t in titles for t in terms)


def summarize(case: dict, path: str, outcome: str, reply=None, retrieval=None, generation=None, answer=None, error=None) -> dict:
    related = [e.to_dict() for e in reply.related_sections] if reply else []
    row = {"id": case["id"], "kind": case.get("kind"), "query": case["query"], "expected_answerable": case.get("answerable", True),
           "path": path, "outcome": outcome, "reason": reply.reason.value if reply else None,
           "best_score": None if retrieval is None or retrieval.best_score is None else round(retrieval.best_score, 4),
           "threshold": None if retrieval is None else retrieval.threshold,
           "relevant": None if retrieval is None else retrieval.relevant,
           "related_labels": [e["citation_label"] for e in related],
           "generation": None if generation is None else {"outcome": generation.outcome, "fixes": list(generation.fixes),
                                                           "usage": generation.usage, "finish_reason": generation.finish_reason},
           "answer": answer, "error": error, "flags": []}
    if reply and overlap_flag(case["query"], related):
        row["flags"].append("related sections share no query term (look unrelated)")
    if row["expected_answerable"] and outcome != "answered":
        row["flags"].append("answerable case did not end answered")
    if not row["expected_answerable"] and outcome == "answered":
        row["flags"].append("unanswerable case was ANSWERED (hallucination risk)")
    return row


async def run_case(case: dict, retrieve, generate, live: bool) -> dict:
    res = retrieve(case["query"])
    if not res.relevant:
        return summarize(case, "gate", "insufficient", reply=gate_failure_reply(res), retrieval=res)
    if not live:
        return summarize(case, "passed_gate", "skipped_live", retrieval=res)
    try:
        gen = await generate(case["query"], res)
    except Exception as e:                                   # noqa: BLE001 - recorded, the run continues
        return summarize(case, "generation", "error", retrieval=res, error=f"{type(e).__name__}: {e}")
    if gen.outcome == "INSUFFICIENT":
        return summarize(case, "model", "insufficient", reply=finalize_model_refusal(gen, res), retrieval=res, generation=gen)
    if gen.outcome == "ANSWERED":
        return summarize(case, "model", "answered", retrieval=res, generation=gen, answer=gen.final_text)
    return summarize(case, "generation", "error", retrieval=res, generation=gen, error=json.dumps(gen.error))


def build_report(rows: list[dict]) -> dict:
    un = [r for r in rows if not r["expected_answerable"]]
    an = [r for r in rows if r["expected_answerable"]]
    return {
        "unanswerable": {"cases": len(un), "insufficient": sum(r["outcome"] == "insufficient" for r in un),
                         "via_gate": sum(r["path"] == "gate" for r in un), "via_model": sum(r["path"] == "model" and r["outcome"] == "insufficient" for r in un),
                         "answered": sum(r["outcome"] == "answered" for r in un), "flagged_unrelated": sum(bool(r["flags"]) and r["outcome"] == "insufficient" for r in un)},
        "answerable": {"cases": len(an), "answered": sum(r["outcome"] == "answered" for r in an),
                       "insufficient": sum(r["outcome"] == "insufficient" for r in an)},
        "live_generation_calls": sum(1 for r in rows if r["generation"] is not None),
        "rows": rows}


async def run(live: bool = True) -> dict:
    from retrieval.pipeline import retrieve
    generate = None
    if live:
        from generation.generate import generate
    cases = load_cases()
    chosen = [c for c in cases if c.get("answerable") is False] + [c for c in cases if c["id"] in ANSWERABLE_IDS]
    rows = []
    for n, case in enumerate(chosen, 1):
        row = await run_case(case, retrieve, generate, live)
        rows.append(row)
        print(f"[{n:2d}/{len(chosen)}] {case['id']:<4} {row['path']:<11} {row['outcome']:<13} score={row['best_score']} "
              f"{row['reason'] or ''} {'FLAG' if row['flags'] else ''}", flush=True)
        if row["generation"] is not None and n < len(chosen) and live:
            await asyncio.sleep(PACE_SECONDS)
    return build_report(rows)


def print_table(rep: dict) -> None:
    print(f"\n{'id':<5}{'path':<11}{'outcome':<13}{'score':>7}  {'reason':<19} flags")
    for r in rep["rows"]:
        print(f"{r['id']:<5}{r['path']:<11}{r['outcome']:<13}{(r['best_score'] if r['best_score'] is not None else '-')!s:>7}  "
              f"{(r['reason'] or '-'):<19} {'; '.join(r['flags']) or '-'}")
    print("\nSummary:", json.dumps({k: v for k, v in rep.items() if k != "rows"}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.insufficient")
    parser.add_argument("--no-live", action="store_true", help="skip live generation (gate path only)")
    args = parser.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.WARNING)
    report = asyncio.run(run(live=not args.no_live))
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(report)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
