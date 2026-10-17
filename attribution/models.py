"""SourceEntry: the API shape of one Sources list entry (ATTRIBUTION_PLAN Section 4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceEntry:
    keys: list[str]        # e.g. ["C1", "C3"], sorted by number
    label: str             # 'NIST AI 100-1, §5.1 Govern, pp. 22–23'
    url: str               # '<source_url>#page=<pdf_page_start>'
    kind: str              # prose | table | figure-ocr | footnote

    def to_dict(self) -> dict:
        return asdict(self)
