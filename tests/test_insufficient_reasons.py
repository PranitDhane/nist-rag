import pytest

from generation.models import ANSWERED, ERROR, INSUFFICIENT, GenerationResult
from insufficient.reasons import FIX_TO_REASON, Reason, reason_from_generation, reason_from_retrieval
from retrieval.models import RetrievalResult


def gen(outcome=INSUFFICIENT, fixes=(), **kw):
    return GenerationResult(outcome=outcome, final_text="x", fixes=list(fixes), **kw)


def ret(relevant, chunks=1, candidates=10):
    return RetrievalResult(query="q", chunks=[object()] * chunks, relevant=relevant, best_score=0.05, threshold=0.15,
                           candidate_count=candidates)


# ---- retrieval (gate) ---------------------------------------------------------------------------------
def test_gate_low_score():
    assert reason_from_retrieval(ret(False)) is Reason.GATE_LOW_SCORE


def test_no_candidates_when_retrieval_returned_nothing():
    assert reason_from_retrieval(ret(False, chunks=0, candidates=0)) is Reason.NO_CANDIDATES


def test_relevant_retrieval_has_no_reason():
    assert reason_from_retrieval(ret(True)) is None


def test_chunks_present_but_zero_candidate_count_is_still_a_gate_failure():
    assert reason_from_retrieval(ret(False, chunks=3, candidates=0)) is Reason.GATE_LOW_SCORE


# ---- generation: one test per row of the plan's table --------------------------------------------------
def test_row_model_case_d():
    assert reason_from_generation(gen(fixes=[])) is Reason.MODEL_CASE_D


def test_row_fix_partial_answer():
    assert reason_from_generation(gen(fixes=["D_WITH_EXTRA_TEXT"])) is Reason.FIX_PARTIAL_ANSWER


def test_row_fix_no_citations():
    assert reason_from_generation(gen(fixes=["NO_CITATIONS"])) is Reason.FIX_NO_CITATIONS


def test_row_fix_empty_output():
    assert reason_from_generation(gen(fixes=["EMPTY"])) is Reason.FIX_EMPTY_OUTPUT


def test_row_fix_truncated_only_when_generation_marks_length():
    assert reason_from_generation(gen(fixes=["LENGTH"], finish_reason="length")) is Reason.FIX_TRUNCATED


# ---- priority, inputs, non-insufficient outcomes --------------------------------------------------------
def test_priority_order_when_several_fixes_apply():
    codes = [c for c, _ in FIX_TO_REASON]
    assert codes == ["D_WITH_EXTRA_TEXT", "NO_CITATIONS", "EMPTY", "LENGTH"]
    assert reason_from_generation(gen(fixes=["LENGTH", "EMPTY", "NO_CITATIONS", "D_WITH_EXTRA_TEXT"])) is Reason.FIX_PARTIAL_ANSWER
    assert reason_from_generation(gen(fixes=["LENGTH", "EMPTY", "NO_CITATIONS"])) is Reason.FIX_NO_CITATIONS
    assert reason_from_generation(gen(fixes=["LENGTH", "EMPTY"])) is Reason.FIX_EMPTY_OUTPUT


def test_harmless_extra_fixes_do_not_hide_model_case_d():
    assert reason_from_generation(gen(fixes=["MALFORMED_CITATION", "HAS_URL"])) is Reason.MODEL_CASE_D


@pytest.mark.parametrize("outcome", [ANSWERED, ERROR])
def test_answered_and_error_results_have_no_reason(outcome):
    assert reason_from_generation(gen(outcome, fixes=["NO_CITATIONS"])) is None


def test_dict_form_of_the_result_is_accepted():
    d = gen(fixes=["EMPTY"]).to_dict()
    assert reason_from_generation(d) is Reason.FIX_EMPTY_OUTPUT
    assert reason_from_retrieval({"relevant": False, "candidate_count": 4, "chunks": [1]}) is Reason.GATE_LOW_SCORE


def test_reason_values_are_the_documented_strings():
    assert {r.value for r in Reason} == {"gate_low_score", "no_candidates", "model_case_d", "fix_partial_answer",
                                         "fix_no_citations", "fix_empty_output", "fix_truncated"}
    assert Reason.MODEL_CASE_D == "model_case_d"                                  # str enum: JSON-friendly
