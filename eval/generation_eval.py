"""Stages N4/N5: live runs of the generation phase against the real Groq model.

    python -m eval.generation_eval measure   [--retrieval-queries 10]   -> eval/generation_usage_report.json
    python -m eval.generation_eval behaviour [--only id1,id2]           -> eval/generation_behaviour_report.json

Skipped without GROQ_API_KEY. Calls are paced under the per-minute token limit (generation.config.TPM_LIMIT).
Every case runs through generate_stream(), which gives TTFT, usage and the correction flag in one call.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

from app.config import Settings
from augmentation.parser import parse_answer, split_units
from eval.generation_checks import actual_case, run_checks
from generation import config
from generation.client import GenerationClient
from generation.generate import estimate_tokens, generate_stream
from retrieval.models import RetrievalResult, RetrievedChunk

EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "answer_behaviour_cases.yaml"
TYPE_TO_CHUNK_TYPE = {"prose": "prose", "table": "genai_action_table", "figure-ocr": "figure", "footnote": "footnote"}


# ------------------------------------------------------------------ contexts
def fixture_result(sources: list[dict], query: str) -> RetrievalResult:
    """A RetrievalResult from a case's fixture sources, in the given order (C1, C2, ...)."""
    chunks = []
    for i, s in enumerate(sources, start=1):
        part, total = (int(x) for x in s["part"].split(" of "))
        meta = {"chunk_id": f"fixture:{i}", "display_name": s["doc"], "chunk_type": TYPE_TO_CHUNK_TYPE[s["type"]],
                "section_path": s["section"].split(" > "), "part": part, "parts_total": total,
                "subcategory": None, "category": None}
        chunks.append(RetrievedChunk(chunk_id=f"fixture:{i}", display_text=s["text"], metadata=meta,
                                     rerank_score=1.0, rerank_rank=i))
    return RetrievalResult(query=query, chunks=chunks, relevant=True, best_score=1.0, threshold=0.15, candidate_count=len(chunks))


def context_for(case: dict) -> RetrievalResult:
    ctx = case["context"]
    if ctx["mode"] == "fixture":
        return fixture_result(ctx["sources"], case["query"])
    from retrieval.pipeline import retrieve
    return retrieve(case["query"])


# ------------------------------------------------------------------ pacing under the tokens-per-minute limit
class Pacer:
    """Waits until (tokens used in the last 60 s) + (this request's worst case) fits the limit."""

    def __init__(self, limit: int, margin: float = 0.9, clock=time.monotonic, sleep=asyncio.sleep):
        self.limit, self.margin, self.clock, self.sleep = limit, margin, clock, sleep
        self.window: list[tuple[float, int]] = []

    def used(self) -> int:
        now = self.clock()
        self.window = [(t, n) for t, n in self.window if now - t < 60]
        return sum(n for _, n in self.window)

    async def wait_for(self, tokens: int) -> float:
        waited = 0.0
        while self.used() + tokens > self.limit * self.margin and self.window:
            step = max(1.0, 60 - (self.clock() - self.window[0][0]) + 0.5)
            await self.sleep(step)
            waited += step
        return waited

    def record(self, tokens: int) -> None:
        self.window.append((self.clock(), tokens))


# ------------------------------------------------------------------ one case
async def run_case(case: dict, client: GenerationClient, pacer: Pacer, *, measurement_mode: bool, ctx: RetrievalResult | None = None) -> dict:
    from augmentation.messages import build_messages
    ctx = ctx or context_for(case)
    messages, key_map = build_messages(case["query"], ctx)
    limit = client.max_tokens
    await pacer.wait_for(estimate_tokens(messages) + limit)
    t0 = time.perf_counter()
    result, error, tokens_streamed = None, None, 0
    async for ev in generate_stream(case["query"], ctx, client=client, measurement_mode=measurement_mode, max_tokens=limit):
        if ev["type"] == "token":
            tokens_streamed += 1
        elif ev["type"] == "result":
            result = ev["result"]
        elif ev["type"] == "error":
            error = ev
    row = {"id": case["id"], "kind": case.get("kind"), "query": case["query"], "expect_case": case.get("expect_case"),
           "chunks": len(ctx.chunks), "wall_ms": round((time.perf_counter() - t0) * 1000)}
    if result is None:
        row.update(actual_case="ERROR", error={"code": error["code"], "message": error["message"]} if error else None)
        pacer.record(estimate_tokens(messages))
        return row
    pacer.record(result.usage.get("total", 0))
    raw = parse_answer(result.streamed_text or "", key_map)
    units = split_units(result.streamed_text or "")
    row.update(
        actual_case=actual_case(result.outcome, result.final_text, key_map), outcome=result.outcome,
        final_text=result.final_text, streamed_text=result.streamed_text, corrected=result.corrected,
        fixes=result.fixes, warnings=result.warnings, cited_keys=result.cited_keys,
        word_count=parse_answer(result.final_text or "", key_map).word_count, usage=result.usage,
        finish_reason=result.finish_reason, ttft_ms=result.timings_ms.get("ttft"), total_ms=result.timings_ms.get("total"),
        raw_violations=raw.violations, raw_units=len(units), raw_uncited_units=len(raw.uncited_sentences),
        checks=run_checks(case.get("checks") or [], result.final_text or "", key_map) if case.get("checks") else [])
    return row


def _stats(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {}
    pick = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]  # noqa: E731
    return {"min": xs[0], "median": statistics.median(xs), "p95": pick(0.95), "max": xs[-1], "n": len(xs)}


# ------------------------------------------------------------------ N4: usage report
def usage_report(rows: list[dict]) -> dict:
    ok = [r for r in rows if "usage" in r]
    visible = [r["usage"]["completion"] - r["usage"]["reasoning"] for r in ok]
    return {
        "model": config.GEN_MODEL, "reasoning_effort": config.REASONING_EFFORT, "tpm_limit": config.TPM_LIMIT,
        "requests": len(rows), "errors": [{"id": r["id"], "error": r.get("error")} for r in rows if r.get("actual_case") == "ERROR"],
        "prompt_tokens": _stats([r["usage"]["prompt"] for r in ok]),
        "reasoning_tokens": _stats([r["usage"]["reasoning"] for r in ok]),
        "visible_answer_tokens": _stats(visible),
        "completion_tokens_incl_reasoning": _stats([r["usage"]["completion"] for r in ok]),
        "total_tokens": _stats([r["usage"]["total"] for r in ok]),
        "ttft_ms": _stats([r["ttft_ms"] for r in ok]), "total_ms": _stats([r["total_ms"] for r in ok]),
        "finish_reasons": {k: sum(1 for r in ok if r["finish_reason"] == k) for k in {r["finish_reason"] for r in ok}},
        "largest_request_total_tokens": max((r["usage"]["total"] for r in ok), default=0),
        "largest_vs_tpm_limit": f"{max((r['usage']['total'] for r in ok), default=0)} of {config.TPM_LIMIT}",
        "estimate_vs_actual_prompt": [{"id": r["id"], "actual": r["usage"]["prompt"]} for r in ok][:5],
        "rows": rows,
    }


# ------------------------------------------------------------------ N5: behaviour report
def behaviour_report(rows: list[dict]) -> dict:
    ran = [r for r in rows if r.get("actual_case") != "ERROR"]
    ans = [r for r in ran if r["outcome"] == "ANSWERED"]
    pct = lambda a, b: round(100 * a / b, 1) if b else None  # noqa: E731
    units = sum(r["raw_units"] for r in ans)
    uncited = sum(r["raw_uncited_units"] for r in ans)
    mism = [r for r in rows if r.get("actual_case") != r["expect_case"]]
    check_fail = [{"id": r["id"], "failed": [c for c in r.get("checks", []) if not c["passed"]]} for r in ran
                  if any(not c["passed"] for c in r.get("checks", []))]
    return {
        "model": config.GEN_MODEL, "cases": len(rows), "errors": len(rows) - len(ran),
        "case_accuracy": {"correct": len(rows) - len(mism), "of": len(rows), "pct": pct(len(rows) - len(mism), len(rows))},
        "uncited_sentence_rate": {"answers_with_uncited_pct": pct(sum(1 for r in ans if r["raw_uncited_units"]), len(ans)),
                                  "sentences_uncited_pct": pct(uncited, units), "sentences": units, "uncited": uncited},
        "over_length_rate_pct": pct(sum(1 for r in ans if "OVER_LENGTH" in r["raw_violations"]), len(ans)),
        "invented_citation_rate_pct": pct(sum(1 for r in ran if "UNKNOWN_CITATION" in r["raw_violations"]), len(ran)),
        "correction_rate_pct": pct(sum(1 for r in ran if r["corrected"]), len(ran)),
        "insufficient_by_fix_rate_pct": pct(sum(1 for r in ran if r["outcome"] == "INSUFFICIENT" and r["fixes"]), len(ran)),
        "avg_total_tokens": round(statistics.mean(r["usage"]["total"] for r in ran), 1) if ran else None,
        "avg_total_ms": round(statistics.mean(r["total_ms"] for r in ran), 1) if ran else None,
        "case_mismatches": [{"id": r["id"], "expected": r["expect_case"], "actual": r.get("actual_case"),
                             "text": r.get("final_text"), "fixes": r.get("fixes")} for r in mism],
        "check_failures": check_fail,
        "checks_total": sum(len(r.get("checks", [])) for r in ran),
        "checks_passed_strict": sum(c["passed"] for r in ran for c in r.get("checks", [])),
        "checks_passed_normalized": sum(c.get("passed_normalized", c["passed"]) for r in ran for c in r.get("checks", [])),
        "rows": rows,
    }


def rescore(path: Path) -> dict:
    """Recompute every row's checks from the saved final text (no API calls) and rewrite the behaviour report."""
    cases = {c["id"]: c for c in yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]}
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    for r in rows:
        if r.get("actual_case") == "ERROR" or r["id"] not in cases:
            continue
        key_map = {f"C{i}": f"chunk{i}" for i in range(1, r["chunks"] + 1)}
        r["checks"] = run_checks(cases[r["id"]].get("checks") or [], r.get("final_text") or "", key_map)
    rep = behaviour_report(rows)
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


# ------------------------------------------------------------------ CLI
async def amain(args) -> int:
    settings = Settings.from_env()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set; live generation evaluation skipped")
        return 0
    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    measure = args.mode == "measure"
    limit = args.max_tokens or (config.MEASUREMENT_MAX_COMPLETION_TOKENS if measure else config.MAX_COMPLETION_TOKENS)
    if limit is None:
        print("MAX_COMPLETION_TOKENS is not set: run `measure` first (Stage N4)")
        return 1
    if measure and args.retrieval_queries:
        from eval.retrieval_eval import load_cases
        extra = [c for c in load_cases() if c["answerable"]][:: max(1, 33 // args.retrieval_queries)][: args.retrieval_queries]
        cases += [{"id": f"r-{c['id']}", "kind": "retrieval_eval", "query": c["query"], "context": {"mode": "retrieve", "note": ""},
                   "expect_case": "A", "checks": []} for c in extra]
    client = GenerationClient(settings, max_tokens=limit)
    pacer = Pacer(config.TPM_LIMIT)
    rows, out = [], Path(args.out)
    for i, case in enumerate(cases, start=1):
        row = await run_case(case, client, pacer, measurement_mode=measure)
        rows.append(row)
        print(f"[{i}/{len(cases)}] {case['id']:<34} expect={case.get('expect_case')} got={row.get('actual_case')} "
              f"tokens={row.get('usage', {}).get('total')} ttft={row.get('ttft_ms')}ms fixes={row.get('fixes')} finish={row.get('finish_reason')}", flush=True)
        rep = usage_report(rows) if measure else behaviour_report(rows)
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")      # incremental: a crash loses nothing
    print(f"report written to {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["measure", "behaviour", "rescore"])
    ap.add_argument("--only")
    ap.add_argument("--retrieval-queries", type=int, default=10)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    args.out = args.out or str(EVAL_DIR / ("generation_usage_report.json" if args.mode == "measure" else "generation_behaviour_report.json"))
    if args.mode == "rescore":
        rep = rescore(Path(args.out))
        print(f"rescored {rep['cases']} cases: checks strict {rep['checks_passed_strict']}/{rep['checks_total']}, "
              f"normalized {rep['checks_passed_normalized']}/{rep['checks_total']}")
        return 0
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
