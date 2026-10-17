"""Stage T4: cross-encoder scoring of every fused candidate, then top-N selection.

Uses sentence-transformers' CrossEncoder directly (Q3): LangChain's CrossEncoderReranker drops the scores,
and the scores are needed for the relevance threshold (T-D10) and for best-first order (T-D9)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from retrieval import config

log = logging.getLogger(__name__)


@dataclass
class Scored:
    chunk_id: str
    score: float
    fused_position: int         # 1-based position in the ensemble result
    rank: int = 0               # 1-based position after reranking (set by score_candidates)


@lru_cache(maxsize=2)
def load_reranker(device: str | None = None):
    from sentence_transformers import CrossEncoder
    device = device or config.resolve_device()
    for local_only in (True, False):
        try:
            return CrossEncoder(config.RERANKER_MODEL, revision=config.RERANKER_REVISION, device=device,
                                max_length=config.RERANKER_MAX_LENGTH, local_files_only=local_only)
        except OSError:
            if not local_only:
                raise
            log.info("reranker not cached yet; downloading %s@%s", config.RERANKER_MODEL, config.RERANKER_REVISION[:8])
    raise AssertionError("unreachable")


def _activation():
    import torch
    if config.RERANK_SCORE_SCALE == "sigmoid":
        return torch.nn.Sigmoid()
    if config.RERANK_SCORE_SCALE == "logit":
        return torch.nn.Identity()
    raise ValueError(f"unknown RERANK_SCORE_SCALE {config.RERANK_SCORE_SCALE!r}")


def rerank_text(chunk: dict) -> str:
    return chunk[config.RERANK_TEXT]


def predict(reranker, query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    return [float(x) for x in reranker.predict([(query, t) for t in texts], activation_fn=_activation(),
                                               show_progress_bar=False)]


def score_candidates(reranker, query: str, candidate_ids: list[str], lookup: dict[str, dict]) -> list[Scored]:
    """Score every candidate. Sorted by score (desc); ties by fused position, then chunk_id (deterministic).
    `candidate_ids` is in fused order."""
    scores = predict(reranker, query, [rerank_text(lookup[cid]) for cid in candidate_ids])
    scored = [Scored(cid, s, pos) for pos, (cid, s) in enumerate(zip(candidate_ids, scores), start=1)]
    scored.sort(key=lambda x: (-x.score, x.fused_position, x.chunk_id))
    for rank, item in enumerate(scored, start=1):
        item.rank = rank
    return scored


def select_top(scored: list[Scored], n: int | None = None) -> list[Scored]:
    return scored[: n or config.TOP_N]


def count_truncated(reranker, query: str, texts: list[str]) -> int:
    """How many (query, text) pairs exceed the reranker's input limit and are cut."""
    limit = config.RERANKER_MAX_LENGTH
    enc = reranker.tokenizer([query] * len(texts), texts, add_special_tokens=True, truncation=False)
    return sum(1 for ids in enc["input_ids"] if len(ids) > limit)
