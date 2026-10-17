from eval.routing import Pacer, build_report, check_case, load_cases


def test_cases_file_meets_the_plan():
    cases = load_cases()
    assert len(cases) >= 25 and len({c["id"] for c in cases}) == len(cases)
    cats = {c["category"] for c in cases}
    assert {"pure_greeting", "greeting_llm", "nist_direct", "follow_up", "mixed", "out_of_scope", "assistant_itself", "near_scope"} <= cats
    for c in cases:
        assert c["expect_route"] in ("greeting", "nist_question", "out_of_scope") and len(c.get("history", [])) <= 2
    assert any("EU AI Act" in c["message"] for c in cases)


def test_check_case_logic():
    case = {"expect_route": "nist_question", "query_terms": ["MAP"], "expect_source": None}
    assert check_case(case, {"route": "nist_question", "standalone_query": "What about the map function?"}) == []
    assert "missing terms" in check_case(case, {"route": "nist_question", "standalone_query": "What about it?"})[0]
    assert "expected 'nist_question'" in check_case(case, {"route": "out_of_scope"})[0]
    assert check_case(case, {"error": "llm_unavailable"}) == ["error: llm_unavailable"]
    rule_case = {"expect_route": "greeting", "expect_source": "rule"}
    assert check_case(rule_case, {"route": "greeting", "route_source": "llm"})[0].startswith("route_source")


def test_report_shapes_and_confusion():
    rows = [{"id": "a", "category": "x", "expected_route": "greeting", "route": "greeting", "route_source": "rule", "llm_calls": 0,
             "latency_ms": 1, "tokens": 0, "passed": True, "problems": []},
            {"id": "b", "category": "x", "expected_route": "out_of_scope", "route": "nist_question", "route_source": "llm", "llm_calls": 1,
             "latency_ms": 500, "tokens": 1200, "passed": False, "problems": ["route"]}]
    rep = build_report(rows)
    assert rep["passed"] == 1 and rep["route_accuracy"] == 0.5 and rep["llm_calls"] == 1 and rep["rule_decided"] == 1
    assert rep["confusion_expected_vs_actual"]["out_of_scope"]["nist_question"] == 1 and len(rep["failures"]) == 1


def test_pacer_waits_when_the_window_is_full(monkeypatch):
    import asyncio
    p = Pacer(True)
    p.record(6000)
    sleeps = []
    async def fake_sleep(s):
        sleeps.append(s)
        p.window.clear()
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    asyncio.run(p.wait())
    assert sleeps and asyncio.run(Pacer(False).wait()) is None
