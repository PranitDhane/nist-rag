"""Stage A3: build the [system, user] messages. No conversation history, ever (G-D9)."""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from augmentation.context_builder import build_context
from augmentation.models import KeyMap
from augmentation.prompt_loader import Prompts, get_prompts
from retrieval.models import RetrievalResult

_QUESTION_TAG = re.compile(r"<(?=/?question)", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\{(sources_block|standalone_query)\}")


def neutralize_question_tags(query: str) -> str:
    """Literal <question>/</question> inside the query must not close our tag (< -> ‹). Nothing else changes."""
    return _QUESTION_TAG.sub("‹", query)


def fill_template(template: str, sources_block: str, standalone_query: str) -> str:
    """Single-pass substitution, so text inside a source that looks like a placeholder is never expanded."""
    values = {"sources_block": sources_block, "standalone_query": neutralize_question_tags(standalone_query)}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def build_messages(standalone_query: str, result: RetrievalResult, prompts: Prompts | None = None) -> tuple[list, KeyMap]:
    """(messages, key_map). Exactly two messages: the system prompt file, and the user template filled with the
    sources block and the standalone query. The only inputs are the query and the retrieval result."""
    prompts = prompts or get_prompts()
    sources_block, key_map = build_context(result)
    user = fill_template(prompts.user_template, sources_block, standalone_query)
    return [SystemMessage(content=prompts.system), HumanMessage(content=user)], key_map


def render_prompt(messages: list) -> str:
    """Plain-text rendering of an assembled prompt (used by the snapshot test and for inspection)."""
    return "\n".join(f"===== {m.type.upper()} =====\n{m.content}" for m in messages)
