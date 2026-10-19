"""Stage V2: the end-to-end harness. `python -m eval.e2e --base-url http://127.0.0.1:8000`.

Sends every case of `eval/e2e_cases.yaml` to a RUNNING server over real HTTP (`POST /chat`, always `debug=true`), the
way a reviewer would, and saves each request and response under `eval/e2e_runs/<timestamp>/raw/`. This stage only
observes: no checks, no scores (those are stages V3-V4), and the harness makes no LLM calls of its own (E-D3).
`--from-run <dir>` reloads a saved run without touching the network, so checks can be re-run for free.

Exit code: 0 whenever the harness itself worked (E-D6: report only); EXIT_HARNESS_FAILURE if the server is unreachable,
the case file is unreadable or a `--from-run` directory is missing. A failed turn is data, not a harness failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from eval import e2e_config as cfg
from eval.e2e_cases import DEFAULT_CASES, all_turns, load_cases

log = logging.getLogger("e2e")
EVAL_DIR = Path(__file__).resolve().parent


class HarnessError(Exception):
    """The harness (not the system under test) cannot continue. Message says what failed and what to do."""


# ------------------------------------------------------------------ one HTTP exchange
@dataclass
class Client:
    """Everything a run needs from the outside world; tests replace `sleep` and `clock` so nothing really waits."""
    base_url: str
    session: requests.Session = field(default_factory=requests.Session)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    pace_seconds: float = cfg.PACE_SECONDS
    _last_generation_end: float | None = None

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path


def check_server(client: Client) -> dict:
    """The one fatal condition: fail early with a clear message if the server cannot be reached or is not healthy."""
    try:
        r = client.session.get(client.url("/health"), timeout=cfg.HEALTH_TIMEOUT_S)
    except requests.RequestException as exc:
        raise HarnessError(f"server not reachable at {client.base_url} ({type(exc).__name__}). Start it with "
                           f"`make dev` (or `python -m app.main`) and pass --base-url if it is not on the default "
                           f"address.") from exc
    if r.status_code != 200:
        raise HarnessError(f"{client.url('/health')} answered HTTP {r.status_code}; the server is up but not healthy. "
                           f"Check its log.")
    body = r.json()
    if not body.get("key_configured"):
        raise HarnessError("the server reports no Groq API key configured; set GROQ_API_KEY in .env and restart it.")
    return body


def error_code(body: object) -> str | None:
    """The API's error code from either body shape ({"error": {...}} or FastAPI's {"detail": {"error": {...}}})."""
    if isinstance(body, dict):
        body = body.get("detail", body)
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            return body["error"].get("code")
    return None


def reached_generation(body: object) -> bool:
    """True if the turn spent generation tokens (so the next request must wait for the per-minute budget)."""
    meta = body.get("meta") if isinstance(body, dict) else None
    if not isinstance(meta, dict):
        return False
    usage = meta.get("usage") or {}
    return bool(usage.get("generation")) or (meta.get("llm_calls") or 0) >= 2


def pace(client: Client) -> float:
    """Sleep until PACE_SECONDS have passed since the last request that reached generation. Returns seconds waited."""
    if client._last_generation_end is None:
        return 0.0
    wait = client.pace_seconds - (client.clock() - client._last_generation_end)
    if wait > 0:
        client.sleep(wait)
        return wait
    return 0.0


def send(client: Client, message: str, thread_id: str | None) -> dict:
    """POST /chat once, with pacing and the single rate-limit retry. Never raises for HTTP errors: they are recorded.
    Returns {status, body, attempts, retried, waited_s, elapsed_ms}."""
    payload: dict = {"message": message, "debug": True}
    if thread_id:
        payload["thread_id"] = thread_id
    waited = pace(client)
    attempts: list[dict] = []
    for attempt in range(1 + cfg.RATE_LIMIT_RETRIES):
        started = client.clock()
        try:
            r = client.session.post(client.url("/chat"), json=payload, timeout=cfg.REQUEST_TIMEOUT_S)
            status, body, headers = r.status_code, _json_or_text(r), r.headers
        except requests.RequestException as exc:
            status, body, headers = None, {"harness_error": f"{type(exc).__name__}: {exc}"}, {}
        elapsed_ms = round((client.clock() - started) * 1000)
        attempts.append({"status": status, "elapsed_ms": elapsed_ms, "error_code": error_code(body)})
        if reached_generation(body):
            client._last_generation_end = client.clock()
        retryable = status == 503 and error_code(body) in cfg.RATE_LIMIT_CODES
        if retryable and attempt < cfg.RATE_LIMIT_RETRIES:
            retry_after = _retry_after(headers)
            attempts[-1]["retry_after_s"] = retry_after
            wait = max(retry_after, client.pace_seconds)      # a rate limit means the minute budget is spent: wait it out
            log.info("e2e rate limited (503); waiting %.0f s, then retrying once", wait)
            client.sleep(wait)
            waited += wait
            continue
        break
    return {"status": status, "body": body, "attempts": attempts, "retried": len(attempts) > 1,
            "waited_s": round(waited, 2), "elapsed_ms": sum(a["elapsed_ms"] for a in attempts)}


def _json_or_text(r: requests.Response) -> object:
    try:
        return r.json()
    except ValueError:
        return {"non_json_body": r.text[:500]}


def _retry_after(headers) -> float:
    try:
        return max(0.0, float(headers.get("Retry-After")))
    except (TypeError, ValueError):
        return cfg.DEFAULT_RETRY_AFTER_S


# ------------------------------------------------------------------ a whole run
def plan_requests(data: dict, only: set[str] | None = None) -> list[dict]:
    """The requests in run order. Conversation turns stay in order and share one thread; `only` filters by turn id
    OR conversation id."""
    turns = all_turns(data)
    if only:
        turns = [t for t in turns if t["id"] in only or t.get("conversation") in only]
        unknown = only - {t["id"] for t in all_turns(data)} - {t.get("conversation") for t in all_turns(data)}
        if unknown:
            raise HarnessError(f"--only names unknown case/conversation id(s): {sorted(unknown)}")
    return turns


def run_turns(client: Client, turns: list[dict], raw_dir: Path | None = None) -> list[dict]:
    """Send the turns in order. A conversation's first response supplies the thread_id for its later turns; if that
    first turn fails, the rest of the conversation is recorded as skipped (sending them without a thread would test
    something else)."""
    threads: dict[str, str | None] = {}
    records = []
    for t in turns:
        conv = t.get("conversation")
        base = {"id": t["id"], "conversation": conv, "message": t["message"]}
        if conv and conv in threads and threads[conv] is None:
            rec = {**base, "skipped": f"conversation {conv} has no thread_id (its first turn failed)"}
        else:
            result = send(client, t["message"], threads.get(conv) if conv else None)
            rec = {**base, "thread_id_sent": threads.get(conv) if conv else None, **result}
            if conv and conv not in threads:
                body = result["body"]
                threads[conv] = body.get("thread_id") if isinstance(body, dict) and result["status"] == 200 else None
        rec["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        records.append(rec)
        if raw_dir is not None:
            write_raw(raw_dir, rec)
        log.info("e2e %s status=%s", t["id"], rec.get("status", "skipped"))
    return records


def write_raw(raw_dir: Path, record: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{record['id']}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_run(run_dir: Path) -> dict:
    """Reload a saved run (no network). Returns {"manifest": ..., "records": [...in the saved run order...]}."""
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file() or not (run_dir / "raw").is_dir():
        raise HarnessError(f"{run_dir} is not a saved run (needs run.json and raw/). Pass a directory under "
                           f"eval/{cfg.RUNS_DIR_NAME}/.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for turn_id in manifest["turn_ids"]:
        path = run_dir / "raw" / f"{turn_id}.json"
        if not path.is_file():
            raise HarnessError(f"saved run {run_dir} is missing raw/{turn_id}.json listed in run.json.")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return {"manifest": manifest, "records": records}


def write_manifest(run_dir: Path, client: Client, cases_path: Path, records: list[dict], server: dict, started: str) -> dict:
    manifest = {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_url": client.base_url, "pace_seconds": client.pace_seconds, "server_health": server,
                "cases_file": str(cases_path.name), "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                "turn_ids": [r["id"] for r in records]}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def summary_table(records: list[dict]) -> str:
    rows = [f"{'id':8} {'status':6} {'route':14} {'outcome':13} {'calls':5} {'tokens':6} {'ms':>6}  retried/waited"]
    for r in records:
        if "skipped" in r:
            rows.append(f"{r['id']:8} SKIPPED ({r['skipped']})")
            continue
        body = r["body"] if isinstance(r["body"], dict) else {}
        meta = body.get("meta") or {}
        usage = meta.get("usage") or {}
        rows.append(f"{r['id']:8} {str(r['status']):6} {str(body.get('route')):14} {str(meta.get('outcome') or error_code(body)):13} "
                    f"{str(meta.get('llm_calls', '-')):5} {str(usage.get('total', '-')):6} {r['elapsed_ms']:>6}  "
                    f"{'retried ' if r['retried'] else ''}{r['waited_s']}s")
    return "\n".join(rows)


# ------------------------------------------------------------------ command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=cfg.DEFAULT_BASE_URL)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR / cfg.RUNS_DIR_NAME,
                    help="parent directory; the run is saved in a timestamped sub-directory")
    ap.add_argument("--only", nargs="+", help="turn ids or conversation ids to run (default: all)")
    ap.add_argument("--from-run", type=Path, help="reload a saved run instead of calling the server")
    ap.add_argument("--warmup", action="store_true", help="send one throwaway NIST question first (cold models make the "
                    "first request slow); costs ~4,000 tokens and is not part of the results")
    ap.add_argument("--pace", type=float, default=cfg.PACE_SECONDS, help="seconds to wait after a request that reached generation")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.from_run:
            run = load_run(args.from_run)
            print(f"reloaded {len(run['records'])} saved turns from {args.from_run} (no network)")
            print(summary_table(run["records"]))
            return 0
        data = load_cases(args.cases)
        turns = plan_requests(data, set(args.only) if args.only else None)
        client = Client(base_url=args.base_url, pace_seconds=args.pace)
        server = check_server(client)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_dir = args.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.warmup:
            log.info("e2e warm-up request (not recorded)")
            send(client, cfg.WARMUP_MESSAGE, None)
        records = run_turns(client, turns, run_dir / "raw")
        write_manifest(run_dir, client, args.cases, records, server, started)
        print(summary_table(records))
        print(f"\nsaved {len(records)} turns to {run_dir}")
        return 0
    except HarnessError as exc:
        print(f"e2e: {exc}", file=sys.stderr)
        return cfg.EXIT_HARNESS_FAILURE


if __name__ == "__main__":
    sys.exit(main())
