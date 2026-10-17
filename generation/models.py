"""Result and event types of the generation phase (GENERATION_PLAN Sections 6 and 7)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

ANSWERED, INSUFFICIENT, ERROR = "ANSWERED", "INSUFFICIENT", "ERROR"


class GenerationError(Exception):
    """Generation failed; `code` is one of: empty_context, budget, rate_limit, server_error, timeout, ttft_timeout,
    total_timeout, midstream, length, unavailable, config."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class LLMOutput:
    """What one Groq call returned (no reasoning text, ever)."""
    text: str
    usage: dict = field(default_factory=dict)          # prompt, completion, reasoning, total tokens
    finish_reason: str | None = None
    ttft_ms: float | None = None
    total_ms: float = 0.0
    attempts: int = 1


@dataclass
class GenerationResult:
    outcome: str                                       # ANSWERED | INSUFFICIENT | ERROR
    final_text: str | None
    streamed_text: str | None = None
    corrected: bool = False
    cited_keys: list[str] = field(default_factory=list)
    key_map: dict = field(default_factory=dict)
    is_ambiguous_format: bool = False
    fixes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    timings_ms: dict = field(default_factory=dict)
    model: str = ""
    prompt_version: str = ""
    reasoning_effort: str = ""
    temperature: float = 0.0
    error: dict | None = None                          # {code, message}

    def to_dict(self) -> dict:
        return asdict(self)
