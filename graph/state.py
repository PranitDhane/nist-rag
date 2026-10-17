"""Graph state (API_ROUTING_PLAN Section 5). Only `messages` accumulates; everything else is reset every message."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["greeting", "nist_question", "out_of_scope"]
RouteSource = Literal["rule", "llm", "fallback"]


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # full conversation, persisted by the checkpointer
    user_message: str                                     # current raw message (the only input the API sends)
    route: Route | None
    route_source: RouteSource | None
    standalone_query: str | None
    route_reason: str | None
    answer: str | None
    sources: list                                         # attribution SourceEntry dicts; [] unless answered
    llm_calls: int
    error: dict | None                                    # {code, message, node} (GRAPH_INTEGRATION_PLAN Section 5)
    retry_after: float | None                             # seconds, set with error (extension of the plan's state)
    # --- graph integration (GRAPH_INTEGRATION_PLAN Section 4). Everything below is plain data (dicts, lists, str,
    # numbers): never dataclasses (LangGraph warns they will be blocked on checkpoint load) and never live objects.
    retrieval: dict | None                                # full retrieval data, see graph.serial (GI-D3)
    generation: dict | None                               # full GenerationResult as a dict; no reasoning exists in it
    cited_keys: list                                      # from the final parse
    key_map: dict                                         # C# -> chunk_id
    insufficient: dict | None                             # InsufficientReply as a dict (insufficient phase)
    related_sections: list                                # [] unless insufficient
    insufficient_reason: str | None
    outcome: str | None                                   # answered | insufficient | greeting | out_of_scope | error
    path: list                                            # node names in order (debug)
    timings_ms: dict                                      # node name -> duration
    usage: dict                                           # total tokens over this message's LLM calls


# Fields reset at the start of every message so nothing leaks from the previous one.
PER_MESSAGE_DEFAULTS: dict = {
    "route": None, "route_source": None, "standalone_query": None, "route_reason": None,
    "answer": None, "sources": [], "llm_calls": 0, "error": None, "retry_after": None,
    "retrieval": None, "generation": None, "cited_keys": [], "key_map": {}, "insufficient": None,
    "related_sections": [], "insufficient_reason": None, "outcome": None, "path": [], "timings_ms": {},
    "usage": {},
}
