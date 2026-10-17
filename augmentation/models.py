"""Data classes of the augmentation phase."""
from __future__ import annotations

from dataclasses import dataclass, field

KeyMap = dict[str, str]            # citation key ("C1") -> chunk_id; kept in graph state, never shown to the LLM


@dataclass
class ParsedAnswer:
    """Result of parse_answer() (AUGMENTATION_PLAN Section 5). Pure data: the parser never calls an LLM."""
    case: str                                              # "A_or_B" | "D" | "invalid"
    is_ambiguous_format: bool
    cited_keys: list[str] = field(default_factory=list)    # ordered, de-duplicated, e.g. ["C2", "C1"]
    unknown_keys: list[str] = field(default_factory=list)  # keys used that are not in the key map (invented)
    uncited_sentences: list[str] = field(default_factory=list)
    word_count: int = 0                                    # words, excluding citation markers
    violations: list[str] = field(default_factory=list)
    clean_text: str = ""                                   # text with unknown keys removed (nothing else rewritten)

    @property
    def ok(self) -> bool:
        return not self.violations
