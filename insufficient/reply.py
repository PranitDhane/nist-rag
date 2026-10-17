"""Stage I3: the insufficient-context reply, for both failure sources. NEVER calls an LLM and never retries (Section 1).

gate_failure_reply():       retrieval said `relevant = False` -> reply straight away (IC-D1), no generation call.
finalize_model_refusal():   generation ended INSUFFICIENT -> final (IC-D2); nothing is generated again.
Both return the same InsufficientReply, so the API and the graph treat them identically.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from augmentation.config import INSUFFICIENT_CONTEXT
from insufficient.reasons import Reason, reason_from_generation, reason_from_retrieval
from insufficient.related import RelatedEntry, build_related_sections
from retrieval.models import RetrievalResult

log = logging.getLogger("insufficient")

OUTCOME = "insufficient"                      # meta.outcome for these replies (answered replies: "answered")
# Q4 (approved): shown with related_sections; makes clear they are the closest matches searched, not evidence.
RELATED_NOTE = "These were the closest sections found. They do not answer the question, but you may want to check them."


@dataclass
class InsufficientReply:
    answer: str                               # always the exact Case D sentence (IC-D7)
    reason: Reason                            # meta.insufficient_reason (IC-D6)
    related_sections: list[RelatedEntry]      # closest sections searched; NO answer content (IC-D3)
    failure_source: str                       # "gate" | "model"
    sources: list = field(default_factory=list)    # always empty: nothing supports an answer
    llm_calls_added: int = 0                  # this phase never adds an LLM call
    best_score: float | None = None
    threshold: float | None = None
    fixes: list[str] = field(default_factory=list)  # model refusals: the fixes generation applied (for logging)

    @property
    def outcome(self) -> str:
        return OUTCOME

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reason"] = self.reason.value
        d["outcome"] = OUTCOME
        d["related_sections"] = [e.to_dict() for e in self.related_sections]
        return d


def gate_failure_reply(retrieval_result: RetrievalResult) -> InsufficientReply:
    """Gate failure: the Case D sentence + the reranked candidates as related sections. No LLM call."""
    reason = reason_from_retrieval(retrieval_result)
    if reason is None:
        raise ValueError("retrieval was relevant: this is not a gate failure")
    return InsufficientReply(
        answer=INSUFFICIENT_CONTEXT, reason=reason, failure_source="gate", sources=[], llm_calls_added=0,
        related_sections=build_related_sections(retrieval_result.chunks, "gate"),
        best_score=retrieval_result.best_score, threshold=retrieval_result.threshold)


def _final_text(gen: Any) -> str | None:
    return gen.get("final_text") if isinstance(gen, dict) else getattr(gen, "final_text", None)


def _fixes(gen: Any) -> list[str]:
    return list((gen.get("fixes") if isinstance(gen, dict) else getattr(gen, "fixes", None)) or [])


def finalize_model_refusal(gen_result: Any, retrieval_result: RetrievalResult) -> InsufficientReply:
    """Model refusal (outcome INSUFFICIENT): the sentence + the chunks that were in the prompt as related sections.
    `gen_result` may be a GenerationResult or its dict form (graph state)."""
    reason = reason_from_generation(gen_result)
    if reason is None:
        raise ValueError("generation outcome is not INSUFFICIENT: this is not a model refusal")
    text = (_final_text(gen_result) or "").strip()
    if text.rstrip(".") != INSUFFICIENT_CONTEXT.rstrip("."):
        # Defence against leaking content: an insufficient reply is ONLY the sentence (IC-D3, G-D7).
        raise ValueError("an INSUFFICIENT generation result must carry exactly the Case D sentence as its final text")
    return InsufficientReply(
        answer=INSUFFICIENT_CONTEXT,          # the exact sentence (the model may have omitted the final period)
        reason=reason, failure_source="model", sources=[], llm_calls_added=0,
        related_sections=build_related_sections(retrieval_result.chunks, "prompt"),
        best_score=retrieval_result.best_score, threshold=retrieval_result.threshold, fixes=_fixes(gen_result))


def log_insufficient(reply: InsufficientReply, thread_id: str, llm_calls: int) -> None:
    """One structured INFO line per insufficient outcome (Section 6). Never the message text."""
    log.info("insufficient thread_id=%s source=%s insufficient_reason=%s best_score=%s threshold=%s related_sections=%d "
             "llm_calls=%s fixes=%s", thread_id, reply.failure_source, reply.reason.value,
             None if reply.best_score is None else round(reply.best_score, 4), reply.threshold,
             len(reply.related_sections), llm_calls, ",".join(reply.fixes) or "-")
