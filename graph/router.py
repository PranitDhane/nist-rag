"""Stage R4: the combined classify-and-rewrite node (ONE LLM call per message that reaches it)."""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import Settings
from graph.history import build_history_window
from graph.prompts import ROUTER_SYSTEM_PROMPT, build_user_prompt
from llm.groq_client import GroqClient, LLMUnavailableError

log = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output of the routing call."""
    route: Literal["greeting", "nist_question", "out_of_scope"]
    standalone_query: str = Field(description="Self-contained question for nist_question; empty string otherwise")
    reason: str = Field(description="One short sentence")


def validate_decision(decision: RouteDecision | None) -> RouteDecision | None:
    """None if unusable. A nist_question needs a standalone query; other routes never carry one."""
    if decision is None:
        return None
    reason = decision.reason.strip()
    if decision.route == "nist_question":
        query = decision.standalone_query.strip()
        return RouteDecision(route="nist_question", standalone_query=query, reason=reason) if query else None
    return RouteDecision(route=decision.route, standalone_query="", reason=reason)


def build_router_messages(window: list[tuple[str, str]], message: str) -> list:
    return [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=build_user_prompt(window, message))]


def make_classify_node(client: GroqClient, settings: Settings):
    async def classify_and_rewrite(state: dict) -> dict:
        calls = state.get("llm_calls", 0)
        window = build_history_window(state["messages"][:-1], settings.history_turns)   # last message = current one
        messages = build_router_messages(window, state["user_message"])
        for attempt in (1, 2):
            try:
                result = await client.structured(RouteDecision, messages)
            except LLMUnavailableError as e:        # never guess a route on an outage
                log.error("router: LLM unavailable (%s) after %d attempts", e.reason, e.attempts)
                return {"error": "llm_unavailable", "retry_after": e.retry_after, "llm_calls": calls}
            calls += result.llm_calls
            decision = validate_decision(result.parsed)
            if decision is not None:
                return {"route": decision.route, "route_source": "llm", "route_reason": decision.reason,
                        "standalone_query": decision.standalone_query or None, "llm_calls": calls}
            log.warning("router: invalid output on attempt %d (%s)", attempt, result.parsing_error or "empty standalone_query")
            log.debug("router raw output: %s", result.raw_text)
        log.warning("router: invalid twice; using fallback route %s", settings.router_fallback_route)
        return {"route": settings.router_fallback_route, "route_source": "fallback", "route_reason": "router output invalid twice",
                "standalone_query": None, "llm_calls": calls}
    return classify_and_rewrite
