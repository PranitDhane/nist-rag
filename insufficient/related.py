"""Stage I2: related_sections: the closest places searched, with NO answer content (IC-D3, IC-D4, IC-D5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from attribution import format as fmt
from retrieval.models import RetrievedChunk

MAX_RELATED = 5                      # IC-D4: top 5, always (fewer only when fewer chunks exist)
MODES = ("gate", "prompt")


@dataclass(frozen=True)
class RelatedEntry:
    """No chunk text, no summary and (Q3) no score: only where to look."""
    doc: str
    section: str
    pages: str
    citation_label: str
    citation_url: str
    kind: str

    def to_dict(self) -> dict:
        return asdict(self)


def entry_for(chunk: RetrievedChunk) -> RelatedEntry:
    m = chunk.metadata
    return RelatedEntry(doc=m["display_name"], section=fmt.section_text(m), pages=fmt.pages_text(m),
                        citation_label=fmt.citation_label(m), citation_url=fmt.citation_url(m), kind=fmt.kind(m))


def build_related_sections(chunks: list[RetrievedChunk], mode: str, limit: int = MAX_RELATED) -> list[RelatedEntry]:
    """Closest sections for an insufficient answer.

    mode "gate":   the reranked candidates in reranker order (best first), BEFORE sibling expansion.
    mode "prompt": the chunks that were in the generation prompt, in prompt order. Sibling parts added to the prompt
                   are not listed (Q2): they are further parts of a table that is already listed.
    Deduplicated by citation_label (Q1; the higher-ranked chunk wins), at most `limit` entries."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    ranked = [c for c in chunks if not c.added_as_sibling]
    if mode == "gate":
        ranked = sorted(ranked, key=lambda c: (c.rerank_rank is None, c.rerank_rank if c.rerank_rank is not None else 0))
    out: list[RelatedEntry] = []
    seen: set[str] = set()
    for chunk in ranked:
        entry = entry_for(chunk)
        if entry.citation_label in seen:
            continue
        seen.add(entry.citation_label)
        out.append(entry)
        if len(out) == limit:
            break
    return out
