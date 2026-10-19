"""Stage V3: deterministic E2E checks over hand-made saved responses. One passing and one failing example per check."""
import copy

import pytest

from eval import e2e_checks as ck
from eval import e2e_cases as ec

TEXTS = {"INSUFFICIENT_TEXT": "The NIST documents provided do not contain this information.",
         "OUT_OF_SCOPE_TEXT": "Sorry, we don't have this answer.", "GREETING_TEXT": "Hello! I can answer..."}
LLM = {"greeting_rule": 0, "out_of_scope": 1, "insufficient_gate": 1, "insufficient_model": 2, "answered": 2}
CHUNKS = {
    "g:core": {"chunk_id": "g:core", "doc_id": "ai_rmf_1_0", "section_path": ["Part 2", "5 AI RMF Core"], "citation_label": "NIST AI 100-1, §5 AI RMF Core, pp. 20–21"},
    "g:govern": {"chunk_id": "g:govern", "doc_id": "ai_rmf_1_0", "section_path": ["Part 2", "5 AI RMF Core", "5.1 Govern"], "citation_label": "NIST AI 100-1, §5.1 Govern, pp. 21–22"},
    "g:conf": {"chunk_id": "g:conf", "doc_id": "genai_profile", "section_path": ["2 Overview", "2.2 Confabulation"], "citation_label": "NIST AI 600-1, §2.2 Confabulation, p. 6"},
}
CTX = ck.Context(texts=TEXTS, chunks=CHUNKS, llm_calls=LLM)


def case(**over):
    c = {"id": "T", "message": "q", "expected_route": "nist_question", "expected_outcome": "answered",
         "must_contain": [{"fact": "govern", "variants": ["govern"]}, {"fact": "cross-cutting", "variants": ["cross-cutting", "cross cutting"]}],
         "min_facts": 2, "must_not_contain": [{"fact": "map is cross-cutting", "variants": ["map is cross"]}],
         "expected_sections": [{"doc": "ai_rmf_1_0", "section": "5.1"}], "must_not_cite": [{"doc": "genai_profile", "section": "2.2"}]}
    c.update(over)
    return c


def answered_body(**over):
    b = {"thread_id": "t", "route": "nist_question", "answer": "GOVERN is a cross-cutting function of the AI RMF [C1][C2].",
         "sources": [{"keys": ["C1"], "label": "x", "url": "u#page=25", "kind": "prose"}, {"keys": ["C2"], "label": "y", "url": "u#page=26", "kind": "prose"}],
         "standalone_query": "Which of the four AI RMF functions is cross-cutting?", "related_sections": [],
         "meta": {"llm_calls": 2, "route_source": "llm", "latency_ms": 3000, "outcome": "answered", "insufficient_reason": None,
                  "usage": {"router": 1200, "generation": 2800, "total": 4000}, "fixes": [], "warnings": [],
                  "debug": {"standalone_query": "Which of the four AI RMF functions is cross-cutting?", "key_map": {"C1": "g:core", "C2": "g:govern"}}}}
    b.update(over)
    return b


def rec(body=None, status=200, **over):
    r = {"id": "T", "status": status, "body": body if body is not None else answered_body(), "elapsed_ms": 3100, "retried": False, "waited_s": 0}
    r.update(over)
    return r


def result(c, r, ctx=CTX):
    return {x.id: x for x in ck.run_turn_checks(c, r, ctx)}


def insufficient_body(reason="gate_low_score", llm_calls=1, **over):
    b = {"thread_id": "t", "route": "nist_question", "answer": TEXTS["INSUFFICIENT_TEXT"], "sources": [], "standalone_query": "q",
         "related_sections": [{"citation_label": f"L{i}"} for i in range(5)],
         "meta": {"llm_calls": llm_calls, "route_source": "llm", "latency_ms": 900, "outcome": "insufficient", "insufficient_reason": reason,
                  "usage": {"router": 1100, "generation": 0, "total": 1100}, "debug": {}}}
    b.update(over)
    return b


INSUFF_CASE = dict(expected_outcome="insufficient", expected_text_key="INSUFFICIENT_TEXT", must_contain=[], min_facts=None,
                   must_not_contain=[], expected_sections=[], must_not_cite=[], expected_insufficient_reason_any=["gate_low_score", "model_case_d"])


# ---------------------------------------------------------------- a fully good answered turn
def test_a_good_answered_turn_passes_everything_except_link_context():
    r = result(case(), rec())
    assert {k: v.status for k, v in r.items()} == {"A1": "pass", "A2": "pass", "A3": "na", "A4": "na", "A5": "pass", "B1": "pass", "B2": "pass",
                                                  "B3": "pass", "C1": "pass", "C2": "pass", "C3": "na", "C4": "pass", "D1": "na", "D2": "na", "D3": "pass", "E1": "na", "E2": "na",
                                                  "E3": "na", "E4": "na", "F1": "pass", "F2": "pass"}
    assert r["B3"].data["words"] == 10 and r["F2"].data["tokens"] == 4000 and "link context" in r["C3"].detail


# ---------------------------------------------------------------- A
def test_a1_a2_route_and_outcome_fail_when_wrong():
    body = answered_body(route="out_of_scope")
    body["meta"]["outcome"] = "out_of_scope"
    r = result(case(), rec(body))
    assert r["A1"].status == r["A2"].status == "fail" and "expected route=nist_question" in r["A1"].detail


def test_a1_route_source_is_checked_when_the_case_names_one():
    c = case(expected_route_source="rule")
    assert result(c, rec())["A1"].status == "fail"          # the fake answered turn came from the LLM router
    body = answered_body()
    body["meta"]["route_source"] = "rule"
    assert result(c, rec(body))["A1"].status == "pass"


def test_acceptable_alternative_is_accepted_and_flagged():
    c = case(**INSUFF_CASE, acceptable_alternative={"expected_route": "out_of_scope", "expected_text_key": "OUT_OF_SCOPE_TEXT"})
    oos = {"route": "out_of_scope", "answer": TEXTS["OUT_OF_SCOPE_TEXT"], "sources": [], "related_sections": [],
           "meta": {"llm_calls": 1, "route_source": "llm", "latency_ms": 700, "outcome": "out_of_scope", "usage": {"total": 1100}}}
    r = result(c, rec(oos))
    assert (r["A1"].status, r["A2"].status, r["A3"].status, r["A5"].status) == ("pass",) * 4
    assert r["A1"].data["alternative"] is True and "acceptable alternative" in r["A1"].detail
    assert r["E1"].status == "na"                          # the insufficient-path checks do not apply to the alternative
    # a route that is neither the primary nor the alternative still fails
    greet = dict(oos, route="greeting", meta=dict(oos["meta"], outcome="greeting"))
    assert result(c, rec(greet))["A1"].status == "fail"


def test_a3_fixed_text_must_match_exactly():
    c = case(expected_outcome="out_of_scope", expected_route="out_of_scope", expected_text_key="OUT_OF_SCOPE_TEXT", must_contain=[], min_facts=None)
    good = {"route": "out_of_scope", "answer": TEXTS["OUT_OF_SCOPE_TEXT"], "sources": [], "meta": {"llm_calls": 1, "outcome": "out_of_scope", "latency_ms": 1}}
    assert result(c, rec(good))["A3"].status == "pass"
    for bad_text in ("Sorry, we don't have this answer", "sorry, we don't have this answer.", TEXTS["OUT_OF_SCOPE_TEXT"] + " "):
        assert result(c, rec(dict(good, answer=bad_text)))["A3"].status == "fail"


def test_a4_standalone_terms_all_of_and_any_of():
    c = case(expected_standalone_terms={"all_of": ["cross-cutting"], "any_of": ["function", "AI RMF"]})
    assert result(c, rec())["A4"].status == "pass"
    body = answered_body()
    body["meta"]["debug"]["standalone_query"] = "Which one is cross-cutting?"          # follow-up not resolved
    assert result(c, rec(body))["A4"].status == "fail"
    body["meta"]["debug"]["standalone_query"] = "Which function is important?"          # all_of missing
    r = result(c, rec(body))["A4"]
    assert r.status == "fail" and "missing all_of" in r.detail


@pytest.mark.parametrize("outcome,route_source,reason,calls,ok", [
    ("greeting", "rule", None, 0, True), ("greeting", "rule", None, 1, False), ("greeting", "llm", None, 1, True),
    ("out_of_scope", "llm", None, 1, True), ("out_of_scope", "llm", None, 2, False),
    ("insufficient", "llm", "gate_low_score", 1, True), ("insufficient", "llm", "gate_low_score", 2, False),
    ("insufficient", "llm", "model_case_d", 2, True), ("insufficient", "llm", "model_case_d", 1, False),
    ("answered", "llm", None, 2, True), ("answered", "llm", None, 3, False)])
def test_a5_expected_llm_calls_per_outcome(outcome, route_source, reason, calls, ok):
    body = {"route": "greeting" if outcome == "greeting" else "nist_question", "answer": "x", "sources": [],
            "meta": {"llm_calls": calls, "route_source": route_source, "outcome": outcome, "insufficient_reason": reason, "latency_ms": 1}}
    c = case(expected_route=body["route"], expected_outcome=outcome)
    assert (result(c, rec(body))["A5"].status == "pass") is ok


# ---------------------------------------------------------------- B
def test_b1_fact_variants_threshold_and_missing_list():
    body = answered_body(answer="GOVERN is cross cutting [C1][C2].")               # 'cross cutting' = variant of cross-cutting
    assert result(case(), rec(body))["B1"].status == "pass"
    body = answered_body(answer="GOVERN is important [C1][C2].")
    r = result(case(), rec(body))["B1"]
    assert r.status == "fail" and r.data["missing"] == ["cross-cutting"] and "1/2" in r.detail
    assert result(case(min_facts=1), rec(body))["B1"].status == "pass"              # min_facts is the pass bar


def test_b1_ignores_citation_markers_so_they_cannot_create_facts():
    c = case(must_contain=[{"fact": "c1", "variants": ["c1"]}], min_facts=1)
    assert result(c, rec(answered_body(answer="Something else [C1].")))["B1"].status == "fail"


def test_b2_trap_hit_fails_and_clean_answer_passes():
    assert result(case(), rec())["B2"].status == "pass"
    body = answered_body(answer="GOVERN is cross-cutting, and MAP is cross-cutting too [C1].")
    r = result(case(), rec(body))["B2"]
    assert r.status == "fail" and r.data["hit"] == ["map is cross-cutting"]


def test_b2_runs_on_non_answered_turns_too():
    c = case(expected_outcome="out_of_scope", expected_route="out_of_scope", must_contain=[], min_facts=None,
             must_not_contain=[{"fact": "answered from general knowledge", "variants": ["paris"]}], expected_text_key="OUT_OF_SCOPE_TEXT")
    leak = {"route": "out_of_scope", "answer": "The capital is Paris.", "sources": [], "meta": {"outcome": "out_of_scope", "llm_calls": 1, "latency_ms": 1}}
    assert result(c, rec(leak))["B2"].status == "fail"


def test_b3_english_word_count_and_language():
    r = result(case(), rec(answered_body(answer="AI RMF के चार मुख्य कार्य GOVERN, MAP, MEASURE और MANAGE हैं [C1][C2]।")))["B3"]
    assert r.status == "fail" and not r.data["english"]
    long_text = ("The framework says that the function is cross-cutting and is infused in the AI RMF. " * 20) + "[C1][C2]"
    r = result(case(), rec(answered_body(answer=long_text)))["B3"]
    assert r.status == "pass" and r.data["over_length"] is True and r.data["words"] > ck.MAX_WORDS   # reported, not failed


# ---------------------------------------------------------------- C
def test_c1_markers_and_sources_must_match_both_ways():
    assert result(case(), rec())["C1"].status == "pass"
    body = answered_body(answer="GOVERN is cross-cutting [C1][C2][C3].")
    r = result(case(), rec(body))["C1"]
    assert r.status == "fail" and "in text only: ['C3']" in r.detail
    body = answered_body(answer="GOVERN is cross-cutting [C1].")
    assert "in sources only: ['C2']" in result(case(), rec(body))["C1"].detail
    body = answered_body(answer="GOVERN is cross-cutting.", sources=[])
    assert "must cite" in result(case(), rec(body))["C1"].detail


def test_c2_expected_section_cited_or_not():
    assert result(case(), rec())["C2"].status == "pass"
    body = answered_body()
    body["meta"]["debug"]["key_map"] = {"C1": "g:core", "C2": "g:core"}             # §5.1 no longer cited
    r = result(case(), rec(body))["C2"]
    assert r.status == "fail" and "none of ['5.1'] cited" in r.detail


def test_c2_forbidden_section_and_other_document():
    body = answered_body()
    body["meta"]["debug"]["key_map"] = {"C1": "g:govern", "C2": "g:conf"}
    r = result(case(), rec(body))["C2"]
    assert r.status == "fail" and r.data["forbidden_cited"] == ["genai_profile 2.2"] and r.data["other_docs"] == ["genai_profile"]
    body["meta"]["debug"]["key_map"] = {"C1": "g:govern", "C2": "g:conf"}
    ok = result(case(must_not_cite=[]), rec(body))["C2"]
    assert ok.status == "pass" and "other document" in ok.detail                    # an extra document is listed, not failed


def test_c2_missing_key_map_is_na_and_unknown_chunk_fails():
    body = answered_body()
    del body["meta"]["debug"]["key_map"]
    assert result(case(), rec(body))["C2"].status == "na"
    body["meta"]["debug"]["key_map"] = {"C1": "g:core", "C2": "no-such-chunk"}
    r = result(case(), rec(body))["C2"]
    assert r.status == "fail" and "not resolvable" in r.detail


def test_c3_link_check_with_a_fake_link_context(monkeypatch):
    from eval import link_check as lc
    calls = []
    monkeypatch.setattr(lc, "check_chunk", lambda chunk, pdf, printed, captions, base: calls.append(chunk["chunk_id"]) or [])
    chunks = {k: dict(v, pdf_page_start=25 if k == "g:core" else 26) for k, v in CHUNKS.items()}
    link = ck.LinkContext(pdf_text={"ai_rmf_1_0": {}, "genai_profile": {}}, printed={}, captions={}, url_base={"ai_rmf_1_0": "u", "genai_profile": "w"})
    ctx = ck.Context(texts=TEXTS, chunks=chunks, llm_calls=LLM, link=link)
    assert result(case(), rec(), ctx)["C3"].status == "pass" and calls == ["g:core", "g:govern"]
    body = answered_body()
    body["sources"][0]["url"] = "u#page=20"                                          # printed page instead of the PDF page
    r = result(case(), rec(body), ctx)["C3"]
    assert r.status == "fail" and "u#page=20" in r.detail
    monkeypatch.setattr(lc, "check_chunk", lambda *a: [{"check": "text", "detail": "start sample not found"}])
    assert result(case(), rec(), ctx)["C3"].status == "fail"
    monkeypatch.setattr(lc, "check_chunk", lambda *a: [{"check": "text_sample_only", "detail": "layout"}])
    r = result(case(), rec(), ctx)["C3"]
    assert r.status == "pass" and r.data["explained"]                               # layout artifacts are passed but listed


def test_c4_fixes_and_warnings_are_reported():
    body = answered_body()
    body["meta"]["warnings"] = ["UNCITED_SENTENCE:2"]
    body["meta"]["fixes"] = ["fix_urls"]
    r = result(case(), rec(body))["C4"]
    assert r.status == "fail" and r.data["uncited_sentence_warnings"] == 1 and "fix_urls" in r.detail


def test_c_checks_are_na_when_no_citations_are_expected():
    c = case(**INSUFF_CASE)
    r = result(c, rec(insufficient_body()))
    assert [r[k].status for k in ("C1", "C2", "C3")] == ["na"] * 3


# ---------------------------------------------------------------- E
def test_e_checks_pass_for_a_correct_insufficient_turn():
    r = result(case(**INSUFF_CASE), rec(insufficient_body()))
    assert [r[k].status for k in ("E1", "E2", "E3", "E4")] == ["pass"] * 4 and r["A5"].status == "pass" and r["A3"].status == "pass"


@pytest.mark.parametrize("mutate,failing", [
    (lambda b: b.update(answer=TEXTS["INSUFFICIENT_TEXT"] + " Also GOVERN."), "E1"),
    (lambda b: b.update(answer=TEXTS["INSUFFICIENT_TEXT"].rstrip(".")), "E1"),
    (lambda b: b.update(sources=[{"keys": ["C1"]}]), "E2"),
    (lambda b: b.update(related_sections=[{"citation_label": "a"}] * 4), "E3"),
    (lambda b: b["meta"].update(insufficient_reason=None), "E4"),
    (lambda b: b["meta"].update(insufficient_reason="fix_empty_output"), "E4"),
])
def test_e_checks_fail_on_each_violation(mutate, failing):
    body = insufficient_body()
    mutate(body)
    assert result(case(**INSUFF_CASE), rec(body))[failing].status == "fail"


def test_e_checks_fail_when_an_expected_insufficient_turn_got_an_answer():
    r = result(case(**INSUFF_CASE), rec())
    assert r["A2"].status == "fail" and r["E1"].status == "fail" and r["E3"].status == "fail"


# ---------------------------------------------------------------- F and broken turns
def test_f_http_errors_and_skipped_turns_are_reported_not_hidden():
    err = {"error": {"code": "llm_unavailable", "message": "later"}}
    r = result(case(), rec(err, status=503, retried=True))
    assert r["F1"].status == "fail" and "llm_unavailable" in r["F1"].detail and r["F1"].data["retried"] is True
    assert r["A1"].status == r["A2"].status == "fail" and r["B1"].status == r["C1"].status == r["F2"].status == "na"
    skipped = result(case(), {"id": "T", "skipped": "conversation C has no thread_id"})
    assert skipped["F1"].status == "fail" and "no thread_id" in skipped["A1"].detail


def test_f2_records_latency_tokens_and_calls():
    r = result(case(), rec(waited_s=12.5))["F2"]
    assert r.status == "pass" and r.data == {"latency_ms": 3000, "client_ms": 3100, "tokens": 4000, "router_tokens": 1200,
                                              "generation_tokens": 2800, "llm_calls": 2, "waited_s": 12.5}


# ---------------------------------------------------------------- whole run
def test_run_checks_keeps_order_reports_unknown_ids_and_never_mutates_inputs():
    data = {"single_turn": [case(id="S1")], "conversations": [], "defaults": {"llm_calls": LLM}}
    records = [rec(id="S1"), rec(id="OLD-CASE")]
    before = copy.deepcopy((data, records))
    rows = ck.run_checks(data, records, CTX)
    assert [r["id"] for r in rows] == ["S1", "OLD-CASE"] and "not in the case file" in rows[1]["error"]
    assert (data, records) == before
    table = ck.results_table(rows)
    assert "S1" in table and "OLD-CASE" in table and "A1." in table


def test_english_heuristic():
    assert ck.is_english("The framework is a voluntary guide to manage risk.")
    assert not ck.is_english("यह एक परीक्षण है") and not ck.is_english("")
    assert ck.is_english("The AI RMF Core comprises four functions: **GOVERN**, **MAP**, **MEASURE**, and **MANAGE**.")   # real live answer (2 stop words)
    assert ck.is_english("GOVERN is cross-cutting [C1].")                         # short but English
    assert not ck.is_english("Le cadre de gestion des risques est un guide volontaire pour les organisations")  # long, Latin script, French


# ---------------------------------------------------------------- the real case file and real data
def real_ctx(links: bool):
    return ck.build_context(ec.load_cases(), with_links=links)


needs_data = pytest.mark.skipif(not (ec.ROOT / "data" / "processed" / "chunks.jsonl").exists() or not (ec.ROOT / "data" / "raw").exists(),
                                reason="needs data/raw and data/processed")


@needs_data
def test_every_real_case_runs_through_every_check_without_crashing():
    data = ec.load_cases()
    ctx = real_ctx(False)
    for t in ec.all_turns(data):
        for fake in (rec(id=t["id"]), rec(None, status=500, id=t["id"]), {"id": t["id"], "skipped": "x"}):
            checks = ck.run_turn_checks(t, fake, ctx)
            assert {c.id for c in checks} >= {"A1", "A2", "B1", "C1", "E1", "F1"} and all(c.status in ("pass", "fail", "na") for c in checks)


@needs_data
def test_real_chunks_and_real_pdfs_for_the_cross_cutting_answer():
    """C1-T2 as the live system answered it (real chunk ids and urls): C2 and C3 pass against the real corpus and PDFs."""
    data = ec.load_cases()
    c = next(t for t in ec.all_turns(data) if t["id"] == "C1-T2")
    body = answered_body(sources=[
        {"keys": ["C1"], "label": "NIST AI 100-1, §5 AI RMF Core, pp. 20–21", "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf#page=25", "kind": "prose"},
        {"keys": ["C2"], "label": "NIST AI 100-1, §5.1 Govern, pp. 21–22", "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf#page=26", "kind": "prose"}])
    body["meta"]["debug"]["key_map"] = {"C1": "ai_rmf_1_0:prose:0020:1", "C2": "ai_rmf_1_0:prose:0022:1"}
    ctx = real_ctx(True)
    r = result(c, rec(body, id="C1-T2"), ctx)
    assert r["C2"].status == "pass" and r["C3"].status == "pass", (r["C2"].detail, r["C3"].detail)
    bad = copy.deepcopy(body)
    bad["sources"][0]["url"] = bad["sources"][0]["url"].replace("page=25", "page=20")
    assert result(c, rec(bad, id="C1-T2"), ctx)["C3"].status == "fail"


# ---------------------------------------------------------------- streaming subset
def ev(*pairs):
    return [{"event": e, "data": d, "t_ms": i * 10} for i, (e, d) in enumerate(pairs)]


def srec(events, status=200, first=20):
    return {"id": "T", "status": status, "events": events, "first_token_ms": first, "elapsed_ms": 900, "error": None}


GOOD_STREAM = ev(("meta", {"route": "nist_question"}), ("token", {"text": "GOVERN is cross-cutting "}), ("token", {"text": "[C1][C2]."}),
                 ("sources", {"sources": [{"keys": ["C1"]}, {"keys": ["C2"]}]}), ("done", {"outcome": "answered", "llm_calls": 2}))


def stream_result(events, chat=None, **kw):
    chat = rec() if chat is None else chat
    return {c.id: c for c in ck.check_stream({}, srec(events, **kw), chat)}


def test_stream_valid_answered_stream_passes_all():
    r = stream_result(GOOD_STREAM)
    assert [r[k].status for k in ("ST1", "ST2", "ST3", "ST4")] == ["pass"] * 4 and r["ST4"].data["first_token_ms"] == 20


@pytest.mark.parametrize("names,ok", [
    (["meta", "token", "sources", "done"], True), (["meta", "token", "token", "token", "sources", "done"], True),
    (["meta", "token", "correction", "sources", "done"], True), (["meta", "token", "sources", "related", "done"], False),
    (["token", "meta", "sources", "done"], False), (["meta", "sources", "token", "done"], False),
    (["meta", "token", "sources"], False), (["meta", "token", "done"], False),
    (["meta", "token", "error"], True), (["meta", "token", "error", "done"], False), (["meta", "error", "token"], False),
    (["meta", "correction", "token", "sources", "done"], False)])
def test_stream_event_order(names, ok):
    assert ck.valid_event_order(names, "answered")[0] is ok


def test_stream_related_event_only_for_insufficient():
    assert ck.valid_event_order(["meta", "token", "sources", "related", "done"], "insufficient")[0]
    assert not ck.valid_event_order(["meta", "token", "sources", "done"], "insufficient")[0]


def test_stream_token_after_error_fails_st1_and_st2():
    events = ev(("meta", {}), ("token", {"text": "a"}), ("error", {"code": "x"}), ("token", {"text": "b"}))
    r = stream_result(events)
    assert r["ST1"].status == "fail" and r["ST2"].status == "fail" and "token" in r["ST2"].detail


def test_stream_ending_with_error_is_valid_order_but_fails_the_comparison():
    r = stream_result(ev(("meta", {}), ("token", {"text": "a"}), ("error", {"code": "x"})))
    assert r["ST1"].status == "pass" and r["ST2"].status == "pass" and r["ST3"].status == "fail"


def test_stream_correction_replaces_the_streamed_draft():
    events = ev(("meta", {}), ("token", {"text": "Draft with no citation."}), ("correction", {"text": "GOVERN is cross-cutting [C1][C2]."}),
                ("sources", {"sources": [{"keys": ["C1"]}, {"keys": ["C2"]}]}), ("done", {"outcome": "answered"}))
    assert ck.streamed_text(srec(events)) == "GOVERN is cross-cutting [C1][C2]."
    assert stream_result(events)["ST3"].status == "pass"
    no_fix = ev(("meta", {}), ("token", {"text": "Draft with no citation."}), ("sources", {"sources": [{"keys": ["C1"]}]}), ("done", {"outcome": "answered"}))
    r = stream_result(no_fix)
    assert r["ST3"].status == "fail" and "no [C#] marker" in r["ST3"].detail


def test_stream_st3_detects_outcome_and_citation_mismatches():
    wrong_keys = ev(("meta", {}), ("token", {"text": "x [C1][C3]."}), ("sources", {"sources": [{"keys": ["C1"]}]}), ("done", {"outcome": "answered"}))
    assert "markers ['C1', 'C3'] != sources keys ['C1']" in stream_result(wrong_keys)["ST3"].detail
    other_outcome = ev(("meta", {}), ("token", {"text": TEXTS["INSUFFICIENT_TEXT"]}), ("sources", {"sources": []}), ("related", {}), ("done", {"outcome": "insufficient"}))
    assert "outcome stream=insufficient /chat=answered" in stream_result(other_outcome)["ST3"].detail


def test_stream_insufficient_pair_passes_and_sources_in_insufficient_stream_fail():
    chat = rec(insufficient_body())
    ok = ev(("meta", {}), ("token", {"text": TEXTS["INSUFFICIENT_TEXT"]}), ("sources", {"sources": []}), ("related", {}), ("done", {"outcome": "insufficient"}))
    assert stream_result(ok, chat)["ST3"].status == "pass" and stream_result(ok, chat)["ST1"].status == "pass"
    bad = ev(("meta", {}), ("token", {"text": "x"}), ("sources", {"sources": [{"keys": ["C1"]}]}), ("related", {}), ("done", {"outcome": "insufficient"}))
    assert stream_result(bad, chat)["ST3"].status == "fail"


def test_stream_http_failure_and_missing_chat_run():
    r = {c.id: c for c in ck.check_stream({}, {"id": "T", "status": 500, "events": [{"event": "http_error", "data": {}}], "error": None}, rec())}
    assert r["ST1"].status == "fail" and r["ST3"].status == "na"
    r = {c.id: c for c in ck.check_stream({}, {"id": "T", "status": None, "events": [], "error": "ConnectionError: x"}, rec())}
    assert "ConnectionError" in r["ST1"].detail
    assert stream_result(GOOD_STREAM, chat=rec(None, status=503))["ST3"].status == "na"


def test_run_stream_checks_pairs_each_stream_with_its_chat_record():
    data = {"single_turn": [case(id="S1")], "conversations": [], "defaults": {"llm_calls": LLM}}
    rows = ck.run_stream_checks([dict(srec(GOOD_STREAM), id="S1", message="q")], [rec(id="S1")], data)
    assert rows[0]["id"] == "S1" and [c["status"] for c in rows[0]["checks"]] == ["pass"] * 4
