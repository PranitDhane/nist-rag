"""Stage V6: aggregates, failure listing and the Markdown report, from hand-made saved responses (no server, no model)."""
import copy
import json

import pytest

from eval import e2e, e2e_checks as ck, e2e_config as cfg, e2e_report as rp
from tests.fake_server import FakeServer
from tests.test_e2e_checks import (CTX, GOOD_STREAM, INSUFF_CASE, TEXTS, answered_body, case, ev, insufficient_body, rec, srec)

OOS = {"route": "out_of_scope", "answer": TEXTS["OUT_OF_SCOPE_TEXT"], "sources": [], "related_sections": [],
       "meta": {"llm_calls": 1, "route_source": "llm", "latency_ms": 700, "outcome": "out_of_scope", "usage": {"router": 1100, "generation": 0, "total": 1100}}}


def fixture():
    """Three turns: S1 answered (all good), S7 insufficient with the out-of-scope alternative, S9 wrong (answered instead of out_of_scope)."""
    data = {"single_turn": [case(id="S1"), case(id="S7", **INSUFF_CASE, acceptable_alternative={"expected_route": "out_of_scope", "expected_text_key": "OUT_OF_SCOPE_TEXT"}),
                            case(id="S9", expected_route="out_of_scope", expected_outcome="out_of_scope", expected_text_key="OUT_OF_SCOPE_TEXT",
                                 must_contain=[], min_facts=None, must_not_contain=[], expected_sections=[], must_not_cite=[])],
            "conversations": [], "defaults": {"llm_calls": {"greeting_rule": 0, "out_of_scope": 1, "insufficient_gate": 1, "insufficient_model": 2, "answered": 2}}}
    records = [rec(id="S1"), rec(OOS, id="S7"), rec(answered_body(), id="S9")]
    return data, records


def evaluate():
    data, records = fixture()
    rows = ck.run_checks(data, records, CTX)
    stream_rows = ck.run_stream_checks([dict(srec(GOOD_STREAM), id="S1", message="q")], records, data)
    run = {"manifest": {"started_at": "t", "base_url": "http://x", "pace_seconds": 35.0, "cases_sha256": "ab" * 32, "server_health": {"model": "m"}},
           "records": records, "stream_records": []}
    return data, records, rows, stream_rows, run


# ---------------------------------------------------------------- aggregates
def test_rate_counts_only_applicable_checks():
    _, _, rows, _, _ = evaluate()
    a1 = rp.rate(rows, "A1")
    assert a1 == {"passed": 2, "applicable": 3, "pct": 66.7}                      # S1 pass, S7 alternative pass, S9 fail
    assert rp.rate(rows, "A3") == {"passed": 1, "applicable": 2, "pct": 50.0}      # only S7(alt) and S9 expect a fixed text
    assert rp.rate(rows, "E1")["applicable"] == 0 and rp.rate(rows, "E1")["pct"] is None


def test_aggregates_cover_the_plan_list():
    data, records, rows, stream_rows, _ = evaluate()
    a = rp.aggregates(rows, records, stream_rows)
    assert a["route_accuracy"]["pct"] == 66.7 and a["outcome_accuracy"]["applicable"] == 3
    assert a["key_fact_coverage"] == {"found": 2, "total": 2, "pct": 100.0}       # only S1 has must-contain facts
    assert a["alternatives_taken"] == ["S7"] and a["streaming"]["ST1"]["pct"] == 100.0
    assert a["latency_ms"]["n"] == 3 and a["by_turn_type"]["answered"]["turns"] == 2 and a["by_turn_type"]["out_of_scope"]["turns"] == 1
    assert a["by_turn_type"]["answered"]["tokens_mean"] == 4000 and a["by_turn_type"]["out_of_scope"]["llm_calls"] == [1]
    assert a["insufficient_path"] == {"turns": 0, "all_checks_passed": 0}          # S7 took the alternative: E checks do not apply


def test_turn_type_distinguishes_the_two_insufficient_paths_and_errors():
    assert rp.turn_type(rec(insufficient_body("gate_low_score"))) == "insufficient (gate)"
    assert rp.turn_type(rec(insufficient_body("model_case_d", llm_calls=2))) == "insufficient (model)"
    assert rp.turn_type(rec({"error": {"code": "x"}}, status=500)) == "error"


def test_percentile_is_nearest_rank_and_handles_empty():
    assert rp.percentile([5, 1, 3, 2, 4], 0.5) == 3 and rp.percentile([5, 1, 3, 2, 4], 0.95) == 5 and rp.percentile([], 0.5) is None


def test_support_and_uncited_rates_come_from_the_d_checks():
    data = {"single_turn": [case(id="S1")], "conversations": [], "defaults": {"llm_calls": {"answered": 2}}}
    ctx = ck.Context(texts=TEXTS, chunks=CTX.chunks, llm_calls={"answered": 2}, scorer=lambda s, texts: [0.9 if "GOVERN" in s else 0.1] * len(texts),
                     support_threshold=0.5)
    body = answered_body(answer="GOVERN is cross-cutting [C1]. Unrelated claim [C2]. No citation here.")
    body["meta"]["debug"]["key_map"] = {"C1": "g:core", "C2": "g:govern"}
    body["sources"] = [{"keys": ["C1"]}, {"keys": ["C2"]}]
    for chunk in ctx.chunks.values():
        chunk.setdefault("display_text", "some text")
    rows = ck.run_checks(data, [rec(body, id="S1")], ctx)
    a = rp.aggregates(rows, [rec(body, id="S1")], [])
    assert a["support_rate"] == {"supported": 1, "total": 2, "pct": 50.0, "threshold": 0.5}
    assert a["uncited_sentence_rate"] == {"uncited": 1, "cited": 2, "pct": 33.3}


# ---------------------------------------------------------------- failures
def test_failures_list_every_failed_check_with_its_turn_and_detail():
    _, records, rows, stream_rows, _ = evaluate()
    fails = rp.failures(rows, stream_rows)
    failed_in_rows = [(r["id"], c["id"]) for r in rows for c in r["checks"] if c["status"] == "fail"]
    assert [(f["turn"], f["check"]) for f in fails if f["via"] == "chat"] == failed_in_rows and failed_in_rows
    s9 = [f for f in fails if f["turn"] == "S9" and f["check"] == "A1"][0]
    assert "expected route=out_of_scope" in s9["detail"]


def test_a_passing_alternative_is_not_a_failure():
    _, _, rows, stream_rows, _ = evaluate()
    assert not [f for f in rp.failures(rows, stream_rows) if f["turn"] == "S7"]


# ---------------------------------------------------------------- markdown
def test_markdown_numbers_come_from_the_report_not_from_the_template(tmp_path):
    data, records, rows, stream_rows, run = evaluate()
    report = rp.build_report(data, run, rows, stream_rows, None, 0.4, tmp_path / "20261019-000000")
    md = rp.render_markdown(report)
    assert "66.7% (2/3)" in md and "Acceptable alternative route taken" in md and "S7" in md
    assert f"## Failed checks ({len(report['failures'])}, none filtered)" in md and "## Limitations" in md and "support-score threshold: 0.4" in md
    changed = copy.deepcopy(report)
    changed["aggregates"]["route_accuracy"] = {"passed": 1, "applicable": 9, "pct": 11.1}
    assert "| Route accuracy (A1) | 11.1% (1/9) |" in rp.render_markdown(changed)
    assert "| Route accuracy (A1) | 66.7% (2/3) |" in md and "| Route accuracy (A1) | 66.7% (2/3) |" not in rp.render_markdown(changed)


def test_markdown_lists_every_failure_row_and_escapes_pipes(tmp_path):
    data, records, rows, stream_rows, run = evaluate()
    report = rp.build_report(data, run, rows, stream_rows, None, None, tmp_path)
    report["failures"].append({"turn": "X", "via": "chat", "check": "B1", "name": "n", "detail": "a | b"})
    md = rp.render_markdown(report)
    assert md.count("| X | chat | B1 n |") == 1 and "a / b" in md
    assert sum(1 for line in md.splitlines() if line.startswith("| ") and " | chat | " in line) == len(report["failures"])


def test_markdown_shows_calibration_when_present(tmp_path):
    data, records, rows, stream_rows, run = evaluate()
    d = {"n": 5, "min": 0.5, "p10": 0.5, "median": 0.8, "p90": 0.9, "max": 0.95}
    cal = {"positives": d, "negatives": dict(d, max=0.2), "negatives_best_of_3": d, "sanity": d, "seed": 1,
           "overlap": {"positives_at_or_below_negative_p90_pct": 0.0, "negatives_at_or_above_positive_p10_pct": 0.0, "separable": True}}
    md = rp.render_markdown(rp.build_report(data, run, rows, stream_rows, cal, 0.4, tmp_path))
    assert "Groundedness calibration" in md and "separable: True" in md and "0.950" in md


def test_report_is_json_serialisable(tmp_path):
    data, records, rows, stream_rows, run = evaluate()
    json.dumps(rp.build_report(data, run, rows, stream_rows, None, None, tmp_path))


# ---------------------------------------------------------------- end to end on a saved fake run
def test_main_writes_report_json_and_markdown_and_exits_zero_even_with_failures(tmp_path):
    with FakeServer() as srv:
        assert e2e.main(["--base-url", srv.url, "--out-dir", str(tmp_path), "--pace", "0"]) == 0      # every fake reply is 'answered'
    (run_dir,) = list(tmp_path.iterdir())
    md_path = tmp_path / "RESULTS.md"
    assert rp.main(["--from-run", str(run_dir), "--no-links", "--no-scorer", "--md", str(md_path)]) == 0
    report = json.loads((run_dir / "report.json").read_text())
    assert report["requests"] == 16 and report["streamed"] == 2 and report["failures"]
    assert {f["turn"] for f in report["failures"]} >= {"S9", "S10"}                # the fake answers everything, so these must fail
    md = md_path.read_text()
    assert md.startswith("# End-to-end evaluation results") and "S9" in md and "do not edit by hand" in md


def test_main_with_a_missing_run_is_a_harness_failure(tmp_path, capsys):
    assert rp.main(["--from-run", str(tmp_path / "nope"), "--no-links", "--no-scorer"]) == cfg.EXIT_HARNESS_FAILURE
    assert "not a saved run" in capsys.readouterr().err
