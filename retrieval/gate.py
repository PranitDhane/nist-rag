"""Stage T7: the relevance gate (T-D10). relevant = best rerank score >= threshold."""
from __future__ import annotations

from dataclasses import dataclass

from retrieval import config
from retrieval.rerank import Scored


class ThresholdNotSet(Exception):
    """The gate must not run without a user-chosen threshold (evaluation mode excepted)."""


@dataclass
class GateResult:
    relevant: bool
    best_score: float | None
    threshold: float | None
    kept: list[Scored]            # top-N after the gate (unchanged unless the gate removes chunks)


def apply_gate(top: list[Scored], *, threshold: float | None = None, removes_chunks: bool | None = None,
               evaluation_mode: bool = False) -> GateResult:
    """`top` is the reranked top-N, best first.

    The gate sets `relevant`; it never hides the chunks (the next phase may want to log and explain the decision)
    unless removes_chunks is on (Q7b), in which case chunks below the threshold are dropped here, i.e. BEFORE
    sibling expansion. Empty `top` -> relevant False, best_score None."""
    threshold = config.THRESHOLD if threshold is None else threshold
    removes = config.GATE_REMOVES_CHUNKS if removes_chunks is None else removes_chunks
    best = top[0].score if top else None
    if threshold is None:
        if not evaluation_mode:
            raise ThresholdNotSet("retrieval.config.THRESHOLD is None; choose it from the Stage T6 table")
        return GateResult(relevant=best is not None, best_score=best, threshold=None, kept=list(top))
    relevant = best is not None and best >= threshold
    kept = [s for s in top if s.score >= threshold] if removes else list(top)
    return GateResult(relevant=relevant, best_score=best, threshold=threshold, kept=kept)
