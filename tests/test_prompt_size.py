"""Stage A6: the pure parts of eval/prompt_size.py (no models, no network)."""
from augmentation.messages import build_messages
from eval.prompt_size import MAX_CHUNKS, chunk_from_record, est_tokens, measure, stats
from retrieval.models import RetrievalResult


def _result(n: int) -> RetrievalResult:
    chunks = [chunk_from_record({"chunk_id": f"c{i}", "display_text": "x" * 400, "doc_id": "ai_rmf_1_0",
                                 "display_name": "NIST AI 100-1 (AI RMF 1.0)", "chunk_type": "prose",
                                 "section_path": ["5 AI RMF Core"]}) for i in range(n)]
    return RetrievalResult(query="q", chunks=chunks, relevant=True, best_score=0.9, threshold=0.15, candidate_count=n)


def test_stats_are_values_that_occurred():
    s = stats([5, 1, 3, 2, 4])
    assert (s["min"], s["median"], s["max"]) == (1, 3, 5) and s["p95"] == 5


def test_estimate_rounds_up_and_margin_increases():
    assert est_tokens(9) == 3 and est_tokens(400, 1.2) > est_tokens(400)


def test_measure_matches_the_real_prompt():
    r = _result(3)
    m = measure("What is X?", r)
    messages, _ = build_messages("What is X?", r)
    assert m["total_chars"] == sum(len(x.content) for x in messages)
    assert m["chunks"] == 3 and m["siblings"] == 0 and m["sources_chars"] > 3 * 400


def test_more_chunks_means_bigger_prompt_and_bound_is_ten():
    assert measure("q", _result(5))["total_chars"] > measure("q", _result(2))["total_chars"]
    assert MAX_CHUNKS == 10
