"""Stage I1: why an answer was insufficient (IC-D6). Pure functions; no I/O, no LLM."""
from __future__ import annotations

from enum import Enum
from typing import Any

from generation.models import INSUFFICIENT          # read-only import: the outcome string owned by the generation layer


class Reason(str, Enum):
    GATE_LOW_SCORE = "gate_low_score"               # retrieval ran, best rerank score below the threshold
    NO_CANDIDATES = "no_candidates"                 # retrieval returned nothing
    MODEL_CASE_D = "model_case_d"                   # the model wrote exactly the Case D sentence
    FIX_PARTIAL_ANSWER = "fix_partial_answer"       # fixer: Case D sentence mixed with other text (D_WITH_EXTRA_TEXT)
    FIX_NO_CITATIONS = "fix_no_citations"           # fixer: no citation left after removing invented ones (NO_CITATIONS)
    FIX_EMPTY_OUTPUT = "fix_empty_output"           # fixer: empty output (EMPTY)
    FIX_TRUNCATED = "fix_truncated"                 # finish_reason == "length" under generation LENGTH_POLICY == "insufficient"


# Fix markers written by generation (generation.fixer / generation.generate) -> reason, in the plan's priority order.
FIX_TO_REASON: tuple[tuple[str, Reason], ...] = (
    ("D_WITH_EXTRA_TEXT", Reason.FIX_PARTIAL_ANSWER),
    ("NO_CITATIONS", Reason.FIX_NO_CITATIONS),
    ("EMPTY", Reason.FIX_EMPTY_OUTPUT),
    ("LENGTH", Reason.FIX_TRUNCATED),
)
FORCING_FIXES = frozenset(code for code, _ in FIX_TO_REASON)


def _get(obj: Any, name: str, default=None):
    """GenerationResult objects and their dict form (graph state stores the dict) are both accepted."""
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def reason_from_retrieval(result: Any) -> Reason | None:
    """Gate-failure reason, or None when retrieval found something relevant."""
    if _get(result, "relevant"):
        return None
    if not _get(result, "candidate_count") and not _get(result, "chunks"):
        return Reason.NO_CANDIDATES
    return Reason.GATE_LOW_SCORE


def reason_from_generation(gen: Any) -> Reason | None:
    """Model-refusal reason, or None unless the outcome is INSUFFICIENT. First matching row of the plan's table wins:
    model_case_d, fix_partial_answer, fix_no_citations, fix_empty_output, fix_truncated."""
    if _get(gen, "outcome") != INSUFFICIENT:
        return None
    fixes = set(_get(gen, "fixes") or [])
    if not fixes & FORCING_FIXES:
        return Reason.MODEL_CASE_D
    for code, reason in FIX_TO_REASON:
        if code in fixes:
            return reason
    raise AssertionError("unreachable")
