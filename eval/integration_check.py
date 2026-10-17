"""Stage G8: live integration run and memory measurement.

    python -m eval.integration_check live      one message per path through the REAL app (real Groq, real indexes)
    python -m eval.integration_check memory    20-message conversation: checkpoint bytes per message + process RSS
    python -m eval.integration_check all       both, then write eval/integration_report.json

`live` needs GROQ_API_KEY (skipped without it) and paces itself under the per-minute token limit. `memory` uses the REAL
retrieval + reranker + indexes but a deterministic fake answer model, so it spends no tokens; the stored state has the
same shape and size as in production (the answer text is the only thing that differs).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPORT = Path(__file__).resolve().parent / "integration_report.json"

LIVE_PATHS = [
    ("greeting_by_rule", "Hello!"),
    ("out_of_scope", "What's the capital of France?"),
    ("answered", "Which of the four AI RMF functions is cross-cutting?"),
    ("gate_failure", "What penalties does the EU AI Act set for non-compliance?"),
    ("model_refusal_known_case_d", "What are the four core functions of the AI RMF and what penalties does NIST impose for not following them?"),
]


def rss_mb() -> float:
    return round(int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1024, 1)


# ------------------------------------------------------------------ live
def live(pause: float = 15.0) -> dict:
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import create_app
    from app.sse import parse_events

    settings = get_settings()
    if not settings.groq_api_key:
        return {"skipped": "GROQ_API_KEY is not set"}
    rows = []
    started = time.perf_counter()
    with TestClient(create_app(settings)) as client:                 # lifespan: loads the real components once
        startup_s = round(time.perf_counter() - started, 2)
        for name, message in LIVE_PATHS:
            t0 = time.perf_counter()
            r = client.post("/chat", json={"message": message, "debug": True})
            wall = round((time.perf_counter() - t0) * 1000)
            row = {"path_name": name, "message": message, "status": r.status_code, "wall_ms": wall}
            if r.status_code == 200:
                j = r.json()
                m, dbg = j["meta"], j["meta"]["debug"]
                row.update(route=j["route"], outcome=m["outcome"], insufficient_reason=m["insufficient_reason"], llm_calls=m["llm_calls"],
                           latency_ms=m["latency_ms"], usage=m["usage"], path=dbg["path"], answer=j["answer"],
                           sources=[s["label"] for s in j["sources"]], related=[s["citation_label"] for s in (j.get("related_sections") or [])],
                           fixes=m.get("fixes"), warnings=m.get("warnings"), timings_ms=dbg["timings_ms"],
                           best_score=(dbg.get("retrieval") or {}).get("best_score"))
            else:
                row["error"] = r.json()
            rows.append(row)
            print(f"{name:<28} {r.status_code} {row.get('outcome')} llm_calls={row.get('llm_calls')} {row.get('latency_ms')}ms "
                  f"tokens={(row.get('usage') or {}).get('total')}", flush=True)
            time.sleep(pause)
        # the same answered question through the stream: tokens must flow live from the tagged ChatGroq call
        t0 = time.perf_counter()
        with client.stream("POST", "/chat/stream", json={"message": "Which of the four AI RMF functions is cross-cutting?"}) as r:
            first_token_ms, frames = None, []
            buf = ""
            for chunk in r.iter_text():
                buf += chunk
                if first_token_ms is None and "event: token" in buf:
                    first_token_ms = round((time.perf_counter() - t0) * 1000)
            frames = parse_events(buf)
        stream = {"events": [e for e, _ in frames], "tokens": sum(1 for e, _ in frames if e == "token"), "first_token_ms": first_token_ms,
                  "total_ms": round((time.perf_counter() - t0) * 1000), "done": dict(frames).get("done")}
        print("stream:", stream["events"][:3], "...", stream["events"][-2:], "first token", first_token_ms, "ms", flush=True)
    return {"startup_seconds": startup_s, "paths": rows, "stream": stream}


# ------------------------------------------------------------------ memory
def checkpoint_bytes(saver) -> dict:
    cp = sum(len(entry[0][1]) for t in saver.storage.values() for ns in t.values() for entry in ns.values())
    blobs = sum(len(v[1]) for v in saver.blobs.values())
    writes = sum(len(w[2][1]) if isinstance(w[2], tuple) else 0 for per in saver.writes.values() for w in per.values())
    n_cp = sum(len(ns) for t in saver.storage.values() for ns in t.values())
    return {"checkpoints": n_cp, "checkpoint_bytes": cp, "blob_bytes": blobs, "writes_bytes": writes, "total_bytes": cp + blobs + writes}


def memory_run(turns: int = 20, durability: str | None = None) -> dict:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from app.config import get_settings
    from eval.retrieval_eval import load_cases
    from generation.client import GenerationClient
    from generation.generate import resolve_max_tokens
    from graph.build import build_app_graph
    from graph.components import load_components
    from graph.router import RouteDecision
    from llm.groq_client import StructuredResult

    settings = get_settings()
    answer = AIMessage(content="The AI RMF describes this in the sections cited here [C1]. Further detail is in the companion text [C2].")
    model = GenericFakeChatModel(messages=itertools.cycle([answer]), tags=["answer_generation"])
    comps = load_components(settings, generation_client=GenerationClient(settings, chat_model=model, max_tokens=resolve_max_tokens(False)))

    cases = load_cases()
    nist = [c["query"] for c in cases if c["answerable"]][:11] + [c["query"] for c in cases if not c["answerable"]][:3]
    script = []                                           # (message, decision-or-None): 14 NIST, 3 greetings by rule, 3 out of scope
    greetings, oos = ["Hello!", "thanks", "bye"], ["What's the capital of France?", "Write me a poem.", "Who won the 2018 World Cup?"]
    for i in range(turns):
        if i % 7 == 3 and greetings:
            script.append((greetings.pop(0), None))
        elif i % 7 == 5 and oos:
            script.append((oos.pop(0), RouteDecision(route="out_of_scope", standalone_query="", reason="r")))
        else:
            q = nist.pop(0)
            script.append((q, RouteDecision(route="nist_question", standalone_query=q, reason="r")))

    class Router:
        def __init__(self):
            self.queue = [d for _, d in script if d is not None]

        async def structured(self, schema, messages):
            return StructuredResult(self.queue.pop(0), "{}", None, 1, total_tokens=0)

    graph = build_app_graph(settings, Router(), components=comps)
    run_kwargs = {} if durability is None else {"durability": durability}
    cfg = {"configurable": {"thread_id": "memory-check"}}
    rss_before, rows, last_total = rss_mb(), [], 0

    async def run():
        nonlocal last_total
        for i, (message, _) in enumerate(script, start=1):
            out = await graph.ainvoke({"user_message": message}, cfg, **run_kwargs)
            sizes = checkpoint_bytes(graph.checkpointer)
            rows.append({"turn": i, "outcome": out.get("outcome"), "route": out.get("route"), "checkpoints_total": sizes["checkpoints"],
                         "bytes_added": sizes["total_bytes"] - last_total, "bytes_total": sizes["total_bytes"],
                         "state_json_bytes": len(json.dumps({k: v for k, v in out.items() if k != "messages"}, default=str)),
                         "rss_mb": rss_mb()})
            last_total = sizes["total_bytes"]
    asyncio.run(run())
    per = [r["bytes_added"] for r in rows]
    nist_added = [r["bytes_added"] for r in rows if r["route"] == "nist_question"]
    return {"durability": durability or "per-step (LangGraph default)", "turns": turns, "rss_before_mb": rss_before, "rss_after_mb": rows[-1]["rss_mb"], "total_checkpoint_bytes": last_total,
            "avg_bytes_per_turn": round(sum(per) / len(per)), "avg_bytes_per_nist_turn": round(sum(nist_added) / max(len(nist_added), 1)),
            "max_bytes_per_turn": max(per), "checkpoints_total": rows[-1]["checkpoints_total"], "per_turn": rows}


def memory(turns: int = 20) -> dict:
    """Production setting (app.api.DURABILITY = "exit") next to the per-step default, same 20-message script."""
    from app.api import DURABILITY
    return {"production": memory_run(turns, DURABILITY), "per_step_default": memory_run(turns, None)}


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["all"])[0]
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    if mode in ("live", "all"):
        report["live"] = live()
    if mode in ("memory", "all"):
        report["memory"] = memory()
        for label, m in report["memory"].items():
            print(f"memory [{label}]: {m['turns']} turns, {m['total_checkpoint_bytes'] / 1e6:.2f} MB of checkpoints "
                  f"({m['avg_bytes_per_nist_turn'] / 1e3:.0f} KB per NIST turn, {m['checkpoints_total']} checkpoints), "
                  f"RSS {m['rss_before_mb']} -> {m['rss_after_mb']} MB")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report written to {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
