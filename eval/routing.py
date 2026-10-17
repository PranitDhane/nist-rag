"""Stage R8: routing evaluation with REAL Groq calls.  python -m eval.routing [--only id,id] [--no-pace]

Misses never fail the run (exit 0): they go in eval/routing_report.json for you to review. Exit 1 only if the
evaluation could not run at all (no key, LLM outage).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter, deque
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, HumanMessage

from app.config import ConfigError, get_settings
from graph.build import build_app_graph
from llm.groq_client import GroqClient

CASES_PATH = Path(__file__).with_name("routing_cases.yaml")
REPORT_PATH = Path(__file__).with_name("routing_report.json")
ROUTES = ("greeting", "nist_question", "out_of_scope")
TPM_BUDGET = 6500          # stay under Groq's 8,000 tokens/minute free-tier limit
EST_TOKENS = 1400          # worst-case cost of one routing call, used before its real usage is known


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_case(case: dict, out: dict) -> list[str]:
    """Reasons this case failed (empty list = pass)."""
    problems = []
    if out.get("error"):
        err = out["error"]
        return [f"error: {err.get('code') if isinstance(err, dict) else err}"]
    if out.get("route") != case["expect_route"]:
        problems.append(f"route {out.get('route')!r}, expected {case['expect_route']!r}")
    if case.get("expect_source") and out.get("route_source") != case["expect_source"]:
        problems.append(f"route_source {out.get('route_source')!r}, expected {case['expect_source']!r}")
    if case["expect_route"] == "nist_question" and out.get("route") == "nist_question":
        q = (out.get("standalone_query") or "").lower()
        missing = [t for t in case.get("query_terms", []) if t.lower() not in q]
        if missing:
            problems.append(f"standalone query {out.get('standalone_query')!r} is missing terms {missing}")
    return problems


class Pacer:
    """Sliding 60-second token window so a 35-case run never trips the tokens/minute limit."""

    def __init__(self, enabled: bool = True):
        self.enabled, self.window = enabled, deque()

    def _used(self, now: float) -> int:
        while self.window and now - self.window[0][0] > 60:
            self.window.popleft()
        return sum(t for _, t in self.window)

    async def wait(self) -> None:
        while self.enabled:
            now = time.monotonic()
            if self._used(now) + EST_TOKENS <= TPM_BUDGET:
                return
            await asyncio.sleep(max(0.5, 60 - (now - self.window[0][0]) + 0.2))

    def record(self, tokens: int) -> None:
        self.window.append((time.monotonic(), tokens))


async def _no_rag(state: dict) -> dict:
    """The routing evaluation measures routing only: the RAG subgraph (retrieval + generation) is not part of it."""
    return {"answer": "", "sources": []}


async def run(only: set[str] | None = None, pace: bool = True) -> dict:
    settings = get_settings().require_key()
    client = GroqClient(settings)
    graph = build_app_graph(settings, client, rag_node=_no_rag)
    pacer = Pacer(pace)
    cases = [c for c in load_cases() if not only or c["id"] in only]
    results = []
    for n, case in enumerate(cases, start=1):
        cfg = {"configurable": {"thread_id": f"eval-{case['id']}"}}
        seed = []
        for turn in case.get("history", []):
            seed += [HumanMessage(content=turn["user"]), AIMessage(content=turn["assistant"])]
        if seed:
            await graph.aupdate_state(cfg, {"messages": seed}, as_node="finalize")
        await pacer.wait()
        before, started = client.stats.tokens, time.perf_counter()
        out = await graph.ainvoke({"user_message": case["message"]}, cfg)
        latency_ms = round((time.perf_counter() - started) * 1000)
        used = client.stats.tokens - before
        if used:
            pacer.record(used)
        problems = check_case(case, out)
        results.append({"id": case["id"], "category": case["category"], "message": case["message"],
                        "history_turns": len(case.get("history", [])), "expected_route": case["expect_route"],
                        "route": out.get("route"), "route_source": out.get("route_source"),
                        "standalone_query": out.get("standalone_query"), "reason": out.get("route_reason"),
                        "llm_calls": out.get("llm_calls", 0), "latency_ms": latency_ms, "tokens": used,
                        "passed": not problems, "problems": problems})
        print(f"[{n:2d}/{len(cases)}] {'PASS' if not problems else 'FAIL'} {case['id']:<11} {out.get('route')!s:<14} "
              f"{out.get('route_source')!s:<8} {latency_ms:>5}ms", flush=True)
    return build_report(results, client)


def build_report(results: list[dict], client: GroqClient | None = None) -> dict:
    per_route = {}
    for r in ROUTES:
        rows = [x for x in results if x["expected_route"] == r]
        per_route[r] = {"cases": len(rows), "correct": sum(x["route"] == r for x in rows),
                        "accuracy": round(sum(x["route"] == r for x in rows) / len(rows), 3) if rows else None}
    confusion = {e: {a: sum(1 for x in results if x["expected_route"] == e and x["route"] == a) for a in (*ROUTES, None)}
                 for e in ROUTES}
    nist = [x for x in results if x["expected_route"] == "nist_question" and x["route"] == "nist_question"]
    term_failures = [x["id"] for x in nist if any("missing terms" in p for p in x["problems"])]
    by_cat = {}
    for x in results:
        c = by_cat.setdefault(x["category"], {"cases": 0, "passed": 0})
        c["cases"] += 1
        c["passed"] += x["passed"]
    return {
        "model": get_settings().router_model, "cases": len(results), "passed": sum(x["passed"] for x in results),
        "route_accuracy": round(sum(x["route"] == x["expected_route"] for x in results) / max(len(results), 1), 3),
        "accuracy_per_route": per_route, "confusion_expected_vs_actual": confusion,
        "standalone_query_term_check": {"nist_cases_routed_correctly": len(nist), "term_failures": term_failures},
        "by_category": by_cat, "llm_calls": sum(x["llm_calls"] for x in results),
        "rule_decided": sum(x["route_source"] == "rule" for x in results), "fallbacks": sum(x["route_source"] == "fallback" for x in results),
        "tokens": sum(x["tokens"] for x in results), "avg_latency_ms_llm": round(
            sum(x["latency_ms"] for x in results if x["llm_calls"]) / max(sum(1 for x in results if x["llm_calls"]), 1)),
        "failures": [x for x in results if not x["passed"]], "results": results,
    }


def print_report(rep: dict) -> None:
    print(f"\nModel {rep['model']}: {rep['passed']}/{rep['cases']} cases passed, route accuracy {rep['route_accuracy']:.1%}, "
          f"{rep['llm_calls']} LLM calls ({rep['rule_decided']} decided by the rule), {rep['tokens']} tokens, "
          f"avg LLM latency {rep['avg_latency_ms_llm']} ms")
    print("\nAccuracy per expected route:")
    for r, v in rep["accuracy_per_route"].items():
        print(f"  {r:<14} {v['correct']}/{v['cases']}  {v['accuracy']:.0%}")
    print("\nConfusion (rows = expected, columns = actual):")
    print(f"  {'':<14}" + "".join(f"{a:>15}" for a in (*ROUTES, "none")))
    for e, row in rep["confusion_expected_vs_actual"].items():
        print(f"  {e:<14}" + "".join(f"{row[a]:>15}" for a in (*ROUTES, None)))
    t = rep["standalone_query_term_check"]
    print(f"\nStandalone-query term check: {t['nist_cases_routed_correctly'] - len(t['term_failures'])}/{t['nist_cases_routed_correctly']} ok"
          + (f"; failures: {t['term_failures']}" if t["term_failures"] else ""))
    print("\nBy category: " + ", ".join(f"{k} {v['passed']}/{v['cases']}" for k, v in rep["by_category"].items()))
    if rep["failures"]:
        print(f"\n{len(rep['failures'])} FAILED CASE(S):")
        for f in rep["failures"]:
            print(f"  - {f['id']} [{f['category']}] {f['message']!r}")
            for p in f["problems"]:
                print(f"      {p}")
            print(f"      reason: {f['reason']!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.routing", description="Routing evaluation with real Groq calls.")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--no-pace", action="store_true", help="skip the tokens/minute pacing (will hit the free-tier limit)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        report = asyncio.run(run(set(args.only.split(",")) if args.only else None, not args.no_pace))
    except ConfigError as e:
        print(f"Cannot run: {e}", file=sys.stderr)
        return 1
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
