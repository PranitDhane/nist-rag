"""Output contract (RETRIEVAL_PLAN Section 5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    display_text: str
    metadata: dict
    rerank_score: float | None = None        # None for siblings that were not candidates
    rerank_rank: int | None = None           # 1-based position after reranking
    added_as_sibling: bool = False
    sibling_of: str | None = None            # chunk_id of the selected part that caused the addition

    @property
    def citation_label(self) -> str | None:
        return self.metadata.get("citation_label")


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    relevant: bool
    best_score: float | None
    threshold: float | None
    candidate_count: int
    timings_ms: dict = field(default_factory=dict)
    config_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
