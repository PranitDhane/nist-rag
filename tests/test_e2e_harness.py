"""Stage V2: the E2E harness against a fake local server. No Groq, no sleeping (time is faked)."""
import json
import socket
import uuid

import pytest
import requests

from eval import e2e, e2e_config as cfg
from eval.e2e_cases import all_turns, load_cases
from tests.fake_server import FakeServer, answered, greeting


class FakeTime:
    """Clock + sleep that never really wait; `elapse` moves time like a slow request would."""
    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.now += s


def client_for(server, ft: FakeTime, pace: float = 35.0) -> e2e.Client:
    return e2e.Client(base_url=server.url, sleep=ft.sleep, clock=ft.clock, pace_seconds=pace)


def conv(*ids):
    return [{"id": i, "conversation": "C", "message": f"m-{i}"} for i in ids]


# ---- server check
def test_unreachable_server_is_a_clear_harness_error_and_exit_code(tmp_path, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                       # a port nobody listens on
    with pytest.raises(e2e.HarnessError, match="not reachable.*make dev"):
        e2e.check_server(e2e.Client(base_url=f"http://127.0.0.1:{port}"))
    code = e2e.main(["--base-url", f"http://127.0.0.1:{port}", "--out-dir", str(tmp_path)])
    assert code == cfg.EXIT_HARNESS_FAILURE and "not reachable" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []              # nothing saved


def test_unhealthy_server_and_missing_key_are_harness_errors():
    with FakeServer(health=(500, {}, {})) as bad, pytest.raises(e2e.HarnessError, match="not healthy"):
        e2e.check_server(e2e.Client(base_url=bad.url))
    with FakeServer(health=(200, {"status": "ok", "key_configured": False}, {})) as nokey, pytest.raises(e2e.HarnessError, match="GROQ_API_KEY"):
        e2e.check_server(e2e.Client(base_url=nokey.url))
    with FakeServer() as ok:
        assert e2e.check_server(e2e.Client(base_url=ok.url))["status"] == "ok"


# ---- single request
def test_every_request_has_debug_true_and_a_first_turn_has_no_thread_id():
    ft = FakeTime()
    with FakeServer() as srv:
        res = e2e.send(client_for(srv, ft), "What is GOVERN?", None)
    assert srv.posts == [{"message": "What is GOVERN?", "debug": True}]
    assert res["status"] == 200 and res["body"]["route"] == "nist_question" and not res["retried"]


# ---- conversations
def test_conversation_turns_share_the_thread_id_from_the_first_response():
    ft = FakeTime()
    with FakeServer() as srv:
        recs = e2e.run_turns(client_for(srv, ft, pace=0), conv("T1", "T2", "T3"))
    first, second, third = srv.posts
    assert "thread_id" not in first
    assert second["thread_id"] == third["thread_id"] == recs[0]["body"]["thread_id"]
    assert [r["thread_id_sent"] for r in recs] == [None, second["thread_id"], second["thread_id"]]


def test_two_conversations_and_single_turns_never_share_threads():
    ft = FakeTime()
    turns = [{"id": "S1", "message": "a"}, {"id": "A1", "conversation": "A", "message": "b"},
             {"id": "A2", "conversation": "A", "message": "c"}, {"id": "B1", "conversation": "B", "message": "d"},
             {"id": "S2", "message": "e"}]
    with FakeServer() as srv:
        e2e.run_turns(client_for(srv, ft, pace=0), turns)
    p = srv.posts
    assert "thread_id" not in p[0] and "thread_id" not in p[1] and "thread_id" in p[2]
    assert "thread_id" not in p[3] and "thread_id" not in p[4]


def test_failed_first_turn_skips_the_rest_of_its_conversation_but_not_other_cases():
    ft = FakeTime()

    def handler(path, payload, n):
        return (500, {"error": {"code": "internal_error", "message": "x"}}, {}) if n == 1 else (200, answered(str(uuid.uuid4()), "q"), {})
    turns = conv("T1", "T2") + [{"id": "S9", "message": "after"}]
    with FakeServer(handler) as srv:
        recs = e2e.run_turns(client_for(srv, ft, pace=0), turns)
    assert recs[0]["status"] == 500 and "skipped" in recs[1] and recs[2]["status"] == 200
    assert len(srv.posts) == 2                           # T2 was never sent


# ---- pacing
def test_waits_after_a_generation_request_but_not_after_a_greeting():
    ft = FakeTime()

    def handler(path, payload, n):
        tid = str(uuid.uuid4())
        return 200, (greeting(tid) if payload["message"] == "hi" else answered(tid, payload["message"])), {}
    with FakeServer(handler) as srv:
        c = client_for(srv, ft)
        e2e.send(c, "hi", None)                          # no generation
        e2e.send(c, "q1", None)                          # no wait before it; reaches generation at t0
        t0 = ft.now
        ft.now += 10                                     # 10 s pass
        e2e.send(c, "hi", None)                          # must wait 25 s more
        assert ft.sleeps == [25.0]
        e2e.send(c, "q2", None)                          # last generation was q1; 35 s already passed -> no wait
        assert ft.sleeps == [25.0]
        assert ft.now == t0 + 10 + 25


def test_pacing_is_skipped_when_enough_real_time_already_passed():
    ft = FakeTime()
    with FakeServer() as srv:
        c = client_for(srv, ft)
        e2e.send(c, "q1", None)
        ft.now += 40
        e2e.send(c, "q2", None)
    assert ft.sleeps == []


def test_reached_generation_detection():
    assert e2e.reached_generation({"meta": {"llm_calls": 2}})
    assert e2e.reached_generation({"meta": {"llm_calls": 1, "usage": {"generation": 5}}})
    assert not e2e.reached_generation({"meta": {"llm_calls": 1, "usage": {"generation": 0}}})
    assert not e2e.reached_generation({"detail": "x"}) and not e2e.reached_generation("text")


# ---- rate limit retry
RATE = (503, {"error": {"code": "llm_unavailable", "message": "later"}}, {"Retry-After": "50"})


def test_rate_limit_is_retried_once_after_waiting_and_recorded():
    ft = FakeTime()
    seen = []

    def handler(path, payload, n):
        seen.append(n)
        return RATE if n == 1 else (200, answered(str(uuid.uuid4()), "q"), {})
    with FakeServer(handler) as srv:
        res = e2e.send(client_for(srv, ft), "q", None)
    assert res["status"] == 200 and res["retried"] and len(res["attempts"]) == 2 and seen == [1, 2]
    assert res["attempts"][0]["status"] == 503 and res["attempts"][0]["retry_after_s"] == 50.0
    assert ft.sleeps == [50.0]                           # max(Retry-After 50, pace 35)


def test_rate_limit_wait_is_at_least_the_pace_when_retry_after_is_small():
    ft = FakeTime()
    low = (503, {"error": {"code": "llm_unavailable", "message": "x"}}, {"Retry-After": "2"})
    with FakeServer(lambda p, pl, n: low if n == 1 else (200, answered("t", "q"), {})) as srv:
        e2e.send(client_for(srv, ft), "q", None)
    assert ft.sleeps == [35.0]


def test_a_second_rate_limit_is_recorded_not_retried_again():
    ft = FakeTime()
    with FakeServer(lambda p, pl, n: RATE) as srv:
        res = e2e.send(client_for(srv, ft), "q", None)
        assert srv.chat_calls == 2                       # exactly one retry
    assert res["status"] == 503 and res["retried"] and len(res["attempts"]) == 2


def test_other_errors_are_recorded_and_never_retried():
    ft = FakeTime()
    err = (500, {"error": {"code": "retrieval_failed", "message": "x"}}, {})
    with FakeServer(lambda p, pl, n: err) as srv:
        res = e2e.send(client_for(srv, ft), "q", None)
        assert srv.chat_calls == 1
    assert res["status"] == 500 and not res["retried"] and ft.sleeps == []


def test_error_code_reads_both_body_shapes():
    assert e2e.error_code({"error": {"code": "a"}}) == "a"
    assert e2e.error_code({"detail": {"error": {"code": "b"}}}) == "b"
    assert e2e.error_code({"x": 1}) is None and e2e.error_code("s") is None


def test_a_connection_failure_mid_run_is_recorded_not_raised():
    ft = FakeTime()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    res = e2e.send(e2e.Client(base_url=f"http://127.0.0.1:{port}", sleep=ft.sleep, clock=ft.clock), "q", None)
    assert res["status"] is None and "harness_error" in res["body"]


# ---- planning, raw files, reload
def test_plan_requests_keeps_order_and_filters_by_turn_or_conversation_id():
    data = load_cases()
    ids = [t["id"] for t in all_turns(data)]
    assert [t["id"] for t in e2e.plan_requests(data)] == ids and len(ids) == 16
    assert [t["id"] for t in e2e.plan_requests(data, {"C1"})] == ["C1-T1", "C1-T2", "C1-T3"]
    assert [t["id"] for t in e2e.plan_requests(data, {"S9", "S10"})] == ["S9", "S10"]
    with pytest.raises(e2e.HarnessError, match="unknown"):
        e2e.plan_requests(data, {"NOPE"})


def test_dry_run_against_the_fake_server_saves_raw_files_and_reloads_identically(tmp_path, capsys):
    with FakeServer() as srv:
        code = e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path), "--pace", "0"])
        assert code == 0 and len(srv.posts) == 16 and all(p["debug"] is True for p in srv.posts)
    (run_dir,) = list(tmp_path.iterdir())
    raw = sorted(p.stem for p in (run_dir / "raw").iterdir())
    assert raw == sorted(t["id"] for t in all_turns(load_cases()))
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["base_url"] == srv.url and manifest["pace_seconds"] == 0.0 and len(manifest["cases_sha256"]) == 64
    one = json.loads((run_dir / "raw" / "S1.json").read_text())
    assert one["message"].startswith("What are the four core functions") and one["status"] == 200 and one["body"]["meta"]["debug"]["key_map"]
    assert len({r["body"]["thread_id"] for r in map(lambda i: json.loads((run_dir / "raw" / f"{i}.json").read_text()), ("C1-T1", "S1"))}) == 2
    # --from-run: no network (the server is already stopped), same records in the same order
    out = capsys.readouterr().out
    assert e2e.main(["--from-run", str(run_dir)]) == 0
    reloaded = e2e.load_run(run_dir)
    assert [r["id"] for r in reloaded["records"]] == manifest["turn_ids"]
    assert reloaded["records"][0]["body"]["answer"] == one["body"]["answer"]
    assert "reloaded 16 saved turns" in capsys.readouterr().out and "S1" in out


def test_from_run_errors_are_clear_and_exit_nonzero(tmp_path, capsys):
    assert e2e.main(["--from-run", str(tmp_path / "missing")]) == cfg.EXIT_HARNESS_FAILURE
    assert "not a saved run" in capsys.readouterr().err
    run = tmp_path / "r"
    (run / "raw").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"turn_ids": ["S1"]}))
    with pytest.raises(e2e.HarnessError, match="missing raw/S1.json"):
        e2e.load_run(run)


def test_a_server_that_returns_errors_is_still_exit_zero(tmp_path):
    err = (500, {"error": {"code": "internal_error", "message": "x"}}, {})
    with FakeServer(lambda p, pl, n: err) as srv:
        assert e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path), "--pace", "0", "--only", "S9", "S10"]) == 0


def test_warmup_request_is_sent_but_not_recorded(tmp_path):
    with FakeServer() as srv:
        e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path), "--pace", "0", "--only", "S10", "--warmup"])
    assert srv.posts[0]["message"] == cfg.WARMUP_MESSAGE and len(srv.posts) == 2
    (run_dir,) = list(tmp_path.iterdir())
    assert [p.stem for p in (run_dir / "raw").iterdir()] == ["S10"]


def test_summary_table_shows_skipped_errors_and_retries():
    recs = [{"id": "S1", "status": 200, "body": answered("t", "q"), "elapsed_ms": 1500, "retried": True, "waited_s": 50},
            {"id": "S2", "status": 500, "body": {"error": {"code": "internal_error"}}, "elapsed_ms": 10, "retried": False, "waited_s": 0},
            {"id": "C1-T2", "skipped": "no thread"}]
    table = e2e.summary_table(recs)
    assert "answered" in table and "internal_error" in table and "SKIPPED" in table and "retried" in table


# ---- streaming subset (V5 / Q4)
def test_sse_frame_parsing():
    assert e2e.parse_sse_frame(["event: token", 'data: {"text":"hi"}']) == ("token", {"text": "hi"})
    assert e2e.parse_sse_frame([": comment"]) is None
    assert e2e.parse_sse_frame(["event: x", "data: not json"])[1] == {"unparsed": "not json"}


def test_send_stream_records_events_in_order_with_timing_and_paces():
    ft = FakeTime()
    with FakeServer() as srv:
        c = client_for(srv, ft)
        res = e2e.send_stream(c, "What is GOVERN?")
        assert srv.stream_posts == [{"message": "What is GOVERN?", "debug": True}]
        assert [e["event"] for e in res["events"]] == ["meta", "token", "token", "sources", "done"]
        assert res["status"] == 200 and res["first_token_ms"] is not None and res["error"] is None
        res["events"][0]["data"]["route"] == "nist_question"
        e2e.send_stream(c, "again")                              # generation ran, so the next stream request must wait
    assert ft.sleeps and ft.sleeps[0] > 0


def test_send_stream_http_error_and_connection_failure_are_recorded():
    ft = FakeTime()
    with FakeServer(stream_handler=lambda p, n: (422, [("error", {"code": "bad"})])) as srv:
        res = e2e.send_stream(client_for(srv, ft), "q")
    assert res["status"] == 422 and res["events"][0]["event"] == "http_error"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    res = e2e.send_stream(e2e.Client(base_url=f"http://127.0.0.1:{port}", sleep=ft.sleep, clock=ft.clock), "q")
    assert res["status"] is None and "Error" in res["error"] and res["events"] == []


def test_stream_subset_runs_only_the_chosen_single_turns_and_is_saved_and_reloaded(tmp_path):
    with FakeServer() as srv:
        code = e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path), "--pace", "0", "--only", "S1", "S7", "C1"])
        assert code == 0
        assert [p["message"][:12] for p in srv.stream_posts] == ["What are the", "What penalti"]       # default STREAM_TURNS = S1, S7
    (run_dir,) = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["stream_ids"] == ["S1", "S7"] and sorted(p.stem for p in (run_dir / "stream").iterdir()) == ["S1", "S7"]
    run = e2e.load_run(run_dir)
    assert [r["id"] for r in run["stream_records"]] == ["S1", "S7"] and run["stream_records"][0]["events"][0]["event"] == "meta"


def test_stream_subset_can_be_skipped_or_chosen(tmp_path):
    with FakeServer() as srv:
        e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path / "a"), "--pace", "0", "--only", "S1", "--no-stream"])
        assert srv.stream_calls == 0
        e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path / "b"), "--pace", "0", "--only", "S1", "S2", "--stream", "S2"])
        assert [p["message"][:12] for p in srv.stream_posts] == ["What charact"]


def test_conversation_turns_are_never_streamed(tmp_path):
    with FakeServer() as srv:
        e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path), "--pace", "0", "--only", "C1", "--stream", "C1-T1"])
        assert srv.stream_calls == 0
