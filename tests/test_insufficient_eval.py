import asyncio

from augmentation.config import INSUFFICIENT_CONTEXT as D
from eval.insufficient import ANSWERABLE_IDS, build_report, load_cases, overlap_flag, query_terms, run_case, summarize
from generation.models import ANSWERED, INSUFFICIENT, GenerationResult
from tests.test_insufficient_reply import gen, retrieval


def test_case_selection_follows_the_plan():
    cases = load_cases()
    assert sum(c.get("answerable") is False for c in cases) == 10 and all(any(c["id"] == i for c in cases) for i in ANSWERABLE_IDS)
    assert len(ANSWERABLE_IDS) == 3


def test_overlap_heuristic():
    related = [{"section": "5.1 Govern"}, {"section": "3 Suggested Actions"}]
    assert query_terms("What does NIST say about quantum computing risks?") == ["quantum", "computing", "risks"]
    assert overlap_flag("What about quantum computing?", related) is True
    assert overlap_flag("What does the Govern function do?", related) is False
    assert overlap_flag("what is it", related) is False                      # no content words: nothing to flag


def test_gate_failure_case_never_calls_generation():
    async def boom(*a, **k):
        raise AssertionError("generation must not run after a gate failure")
    row = asyncio.run(run_case({"id": "u0", "query": "What fines does the GDPR impose?", "answerable": False}, lambda q: retrieval(False), boom, True))
    assert row["path"] == "gate" and row["outcome"] == "insufficient" and row["reason"] == "gate_low_score"
    assert len(row["related_labels"]) == 5 and row["generation"] is None


def test_gate_pass_goes_to_generation_and_a_model_refusal_is_finalized():
    async def fake_generate(q, res):
        return gen(fixes=["NO_CITATIONS"])
    row = asyncio.run(run_case({"id": "u1", "query": "q", "answerable": False}, lambda q: retrieval(True), fake_generate, True))
    assert row["path"] == "model" and row["outcome"] == "insufficient" and row["reason"] == "fix_no_citations"


def test_answered_unanswerable_case_is_flagged_and_answerable_miss_is_flagged():
    async def answered(q, res):
        return GenerationResult(outcome=ANSWERED, final_text="Made up [C1].", cited_keys=["C1"])
    row = asyncio.run(run_case({"id": "u2", "query": "q", "answerable": False}, lambda q: retrieval(True), answered, True))
    assert row["outcome"] == "answered" and any("ANSWERED" in f for f in row["flags"])
    miss = asyncio.run(run_case({"id": "c9", "query": "q"}, lambda q: retrieval(False), None, True))
    assert miss["outcome"] == "insufficient" and "answerable case did not end answered" in miss["flags"]


def test_no_live_mode_skips_generation_and_errors_are_recorded():
    row = asyncio.run(run_case({"id": "u3", "query": "q", "answerable": False}, lambda q: retrieval(True), None, False))
    assert row["outcome"] == "skipped_live"
    async def broken(q, res):
        raise RuntimeError("boom")
    row = asyncio.run(run_case({"id": "u4", "query": "q", "answerable": False}, lambda q: retrieval(True), broken, True))
    assert row["outcome"] == "error" and "RuntimeError: boom" in row["error"]


def test_report_counts():
    rows = [summarize({"id": "a", "query": "q", "answerable": False}, "gate", "insufficient"),
            summarize({"id": "b", "query": "q", "answerable": False}, "model", "answered"),
            summarize({"id": "c", "query": "q"}, "model", "answered")]
    rep = build_report(rows)
    assert rep["unanswerable"]["cases"] == 2 and rep["unanswerable"]["answered"] == 1 and rep["answerable"]["answered"] == 1
