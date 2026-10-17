import json
import logging
import socket

import pytest

from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation.models import ANSWERED, ERROR, INSUFFICIENT, GenerationResult
from insufficient.reasons import Reason
from insufficient.reply import InsufficientReply, finalize_model_refusal, gate_failure_reply, log_insufficient
from retrieval.models import RetrievalResult, RetrievedChunk


def chunk(i, sibling=False):
    meta = {"chunk_id": f"c{i}", "doc_id": "ai_rmf_1_0", "display_name": "NIST AI 100-1 (AI RMF 1.0)", "chunk_type": "prose",
            "section_path": ["Part 2", f"5.{i} Section"], "printed_page_start": str(10 + i), "printed_page_end": str(10 + i),
            "pdf_page_start": 15 + i, "pdf_page_end": 15 + i, "source_url": "https://x.org/a.pdf", "part": 1, "parts_total": 1,
            "subcategory": None, "category": None}
    return RetrievedChunk(chunk_id=f"c{i}", display_text="SECRET ANSWER CONTENT", metadata=meta, rerank_score=None if sibling else 0.05,
                          rerank_rank=None if sibling else i, added_as_sibling=sibling)


def retrieval(relevant=False, n=5, candidates=12, siblings=0):
    chunks = [chunk(i) for i in range(1, n + 1)] + [chunk(100 + i, sibling=True) for i in range(siblings)]
    return RetrievalResult(query="What penalties does the EU AI Act set?", chunks=chunks, relevant=relevant, best_score=0.04,
                           threshold=0.15, candidate_count=candidates)


def gen(fixes=(), final_text=D, outcome=INSUFFICIENT, **kw):
    return GenerationResult(outcome=outcome, final_text=final_text, fixes=list(fixes), **kw)


@pytest.fixture
def no_llm(monkeypatch):
    """Any LLM call or network access in the code under test fails the test."""
    def boom(*a, **k):
        raise AssertionError("an LLM / network call was made in the insufficient path")
    import langchain_groq
    from generation.client import GenerationClient
    from llm.groq_client import GroqClient
    monkeypatch.setattr(GroqClient, "structured", boom)
    monkeypatch.setattr(GroqClient, "ping", boom)
    monkeypatch.setattr(GenerationClient, "complete", boom)
    for name in ("invoke", "ainvoke", "stream", "astream", "_generate", "_agenerate"):
        monkeypatch.setattr(langchain_groq.ChatGroq, name, boom, raising=False)
    monkeypatch.setattr(socket.socket, "connect", boom)


# ---- gate path ------------------------------------------------------------------------------------------
def test_gate_failure_reply_is_the_exact_sentence_with_related_sections_and_no_sources(no_llm):
    r = gate_failure_reply(retrieval())
    assert isinstance(r, InsufficientReply) and r.answer == D and r.sources == [] and r.failure_source == "gate"
    assert r.reason is Reason.GATE_LOW_SCORE and r.llm_calls_added == 0 and r.outcome == "insufficient"
    assert [e.section for e in r.related_sections] == [f"Part 2 > 5.{i} Section" for i in range(1, 6)]
    assert (r.best_score, r.threshold) == (0.04, 0.15)


def test_gate_failure_with_no_candidates(no_llm):
    empty = RetrievalResult(query="q", chunks=[], relevant=False, best_score=None, threshold=0.15, candidate_count=0)
    r = gate_failure_reply(empty)
    assert r.reason is Reason.NO_CANDIDATES and r.related_sections == [] and r.answer == D and r.sources == []


def test_gate_path_ignores_siblings(no_llm):
    r = gate_failure_reply(retrieval(n=3, siblings=2))
    assert len(r.related_sections) == 3 and all("100" not in e.citation_url for e in r.related_sections)


def test_gate_reply_refuses_to_run_for_a_relevant_retrieval(no_llm):
    with pytest.raises(ValueError, match="not a gate failure"):
        gate_failure_reply(retrieval(relevant=True))


# ---- model-refusal path -----------------------------------------------------------------------------------
def test_model_refusal_reply_adds_no_llm_call_and_lists_prompt_chunks(no_llm):
    r = finalize_model_refusal(gen(), retrieval(relevant=True))
    assert r.answer == D and r.sources == [] and r.failure_source == "model" and r.llm_calls_added == 0
    assert r.reason is Reason.MODEL_CASE_D and len(r.related_sections) == 5


@pytest.mark.parametrize("fixes,reason", [([], Reason.MODEL_CASE_D), (["D_WITH_EXTRA_TEXT"], Reason.FIX_PARTIAL_ANSWER),
                                          (["NO_CITATIONS"], Reason.FIX_NO_CITATIONS), (["EMPTY"], Reason.FIX_EMPTY_OUTPUT),
                                          (["LENGTH"], Reason.FIX_TRUNCATED)])
def test_reason_codes_on_the_reply(fixes, reason, no_llm):
    r = finalize_model_refusal(gen(fixes=fixes), retrieval(relevant=True))
    assert r.reason is reason and r.fixes == fixes


def test_both_paths_produce_the_identical_sentence(no_llm):
    assert gate_failure_reply(retrieval()).answer == finalize_model_refusal(gen(), retrieval(True)).answer == D


def test_model_sentence_without_final_period_is_normalised_to_the_exact_sentence(no_llm):
    r = finalize_model_refusal(gen(final_text=D.rstrip(".")), retrieval(True))
    assert r.answer == D and r.answer.endswith(".")


def test_a_result_carrying_extra_content_is_refused_not_passed_through(no_llm):
    with pytest.raises(ValueError, match="exactly the Case D sentence"):
        finalize_model_refusal(gen(final_text=D + " GOVERN is cross-cutting [C1]."), retrieval(True))
    with pytest.raises(ValueError):
        finalize_model_refusal(gen(final_text=None), retrieval(True))


@pytest.mark.parametrize("outcome", [ANSWERED, ERROR])
def test_non_insufficient_generation_is_not_a_refusal(outcome, no_llm):
    with pytest.raises(ValueError, match="not a model refusal"):
        finalize_model_refusal(gen(outcome=outcome, final_text="x"), retrieval(True))


def test_dict_form_of_the_generation_result_works(no_llm):
    r = finalize_model_refusal(gen(fixes=["EMPTY"]).to_dict(), retrieval(True))
    assert r.reason is Reason.FIX_EMPTY_OUTPUT and r.answer == D


def test_siblings_in_the_prompt_are_not_listed_for_model_refusals(no_llm):
    r = finalize_model_refusal(gen(), retrieval(True, n=3, siblings=2))
    assert len(r.related_sections) == 3


# ---- invariants for every insufficient reply --------------------------------------------------------------
@pytest.mark.parametrize("reply_factory", [lambda: gate_failure_reply(retrieval()), lambda: finalize_model_refusal(gen(), retrieval(True))])
def test_no_chunk_text_and_sources_always_empty_and_serialisable(reply_factory, no_llm):
    d = reply_factory().to_dict()
    assert d["sources"] == [] and d["answer"] == D and d["outcome"] == "insufficient"
    blob = json.dumps(d)
    assert "SECRET" not in blob and d["reason"] in {r.value for r in Reason}
    assert set(d["related_sections"][0]) == {"doc", "section", "pages", "citation_label", "citation_url", "kind"}


def test_log_line_has_the_required_fields_and_never_the_message(caplog):
    r = finalize_model_refusal(gen(fixes=["NO_CITATIONS"]), retrieval(True))
    with caplog.at_level(logging.INFO, logger="insufficient"):
        log_insufficient(r, "thread-1", llm_calls=2)
    line = caplog.records[0].getMessage()
    for part in ("thread_id=thread-1", "source=model", "insufficient_reason=fix_no_citations", "best_score=0.04", "threshold=0.15",
                 "related_sections=5", "llm_calls=2", "fixes=NO_CITATIONS"):
        assert part in line, part
    assert "penalties" not in line and "SECRET" not in line
