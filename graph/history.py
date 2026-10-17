"""Stage R4: the history window the router sees: the last N turns before the current message (Q9: any kind of turn).

1 turn = 1 user message + 1 assistant reply. Assistant turns contribute only their answer text (no sources).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def build_history_window(messages: list[BaseMessage], turns: int = 2) -> list[tuple[str, str]]:
    """[(role, text), ...] oldest first. `messages` must NOT include the current user message."""
    if turns <= 0:
        return []
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for m in messages:
        if isinstance(m, HumanMessage):
            pending_user = str(m.content)
        elif isinstance(m, AIMessage) and pending_user is not None:
            pairs.append((pending_user, str(m.content)))
            pending_user = None
    window: list[tuple[str, str]] = []
    for user, assistant in pairs[-turns:]:
        window += [("User", user), ("Assistant", assistant)]
    return window


def format_history(window: list[tuple[str, str]]) -> str:
    if not window:
        return "(no previous messages)"
    return "\n".join(f"{role}: {text}" for role, text in window)
