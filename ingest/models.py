"""Dataclasses for the manifest (Stage 1); record dataclasses are added per stage.

Why dataclasses: plain, typed, and easy to explain; no framework needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageTypeRange:
    """Expected page_type for an inclusive PDF page range (validation only)."""
    first: int
    last: int
    page_type: str


@dataclass(frozen=True)
class Validation:
    """Expected results for one document. Read ONLY by validate.py so the checks
    stay independent of the logic they test."""
    page_types: list[PageTypeRange]
    section_ids: list[str]
    printed_page_spot_checks: dict[int, str | None]
    expected_figures: int | None = None
    expected_footnotes: int | None = None
    subcategory_tables: int | None = None
    unique_action_ids: int | None = None
    forbidden_lines: list[str] = field(default_factory=list)          # whole lines that must be gone (running headers)
    forbidden_line_patterns: list[str] = field(default_factory=list)  # regexes for lines that must be gone (footers)


@dataclass(frozen=True)
class DocumentSpec:
    """One source document from sources.yaml."""
    doc_id: str
    display_name: str
    url: str
    filename: str
    sha256: str
    expected_pages: int
    expected_title: str
    page_number_strategy: str  # "page_label" | "first_line_offset"
    validation: Validation
    offset: int | None = None  # only for first_line_offset


@dataclass(frozen=True)
class Manifest:
    documents: list[DocumentSpec] = field(default_factory=list)

    def get(self, doc_id: str) -> DocumentSpec:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        raise KeyError(f"Unknown doc_id {doc_id!r}; known: {[d.doc_id for d in self.documents]}")
