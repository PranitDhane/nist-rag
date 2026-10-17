"""Builds and compiles the main graph once at startup (GRAPH_INTEGRATION_PLAN Section 3, Stage G4).

    START -> reset_turn -> rule_greeting_check --pure greeting--> greeting_reply ------------------+
                              `--else--> classify_and_rewrite (LLM #1)                              |
                                  |- greeting      -> greeting_reply ------------------------------+
                                  |- out_of_scope  -> out_of_scope_reply ---------------------------+
                                  |- nist_question -> rag (compiled SUBGRAPH, graph.rag_subgraph) --+--> finalize -> END
                                  `- error         -> error_finalize -> END                         |
                              rag --error--> error_finalize                                         |

`rag_node` is the RAG subgraph (or, in tests, any node standing in for it). There is no stub in the production graph.
"""
from __future__ import annotations

import copy
import logging
import time
import uuid
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from attribution.build import strip_markers
from graph import errors, usage
from graph.replies import greeting_reply, out_of_scope_reply
from graph.state import PER_MESSAGE_DEFAULTS, GraphState

log = logging.getLogger(__name__)

ClassifyNode = Callable[[dict], Awaitable[dict]]


def timed(name: str, fn, *, outcome: str | None = None, fresh: bool = False):
    """Wrap a simple node: append its name to `path`, record its duration, optionally set the terminal `outcome`.
    `fresh=True` (reset_turn only) starts `path` and `timings_ms` from scratch: the state it receives still holds the
    PREVIOUS message's values, because the reset it performs has not been applied yet."""
    async def node(state: dict) -> dict:
        started = time.perf_counter()
        update = dict(await fn(state))
        prior_path, prior_timings = ([], {}) if fresh else (state.get("path") or [], state.get("timings_ms") or {})
        update["path"] = [*prior_path, name]
        update["timings_ms"] = {**prior_timings, name: round((time.perf_counter() - started) * 1000, 1)}
        if outcome:
            update["outcome"] = outcome
        return update
    node.__name__ = name
    return node


def make_reset_turn():
    async def reset_turn(state: dict) -> dict:
        """First node of every message: reset every per-message field (deep copy: the defaults hold lists and dicts that
        nodes must never share across turns) and store the user message. Only `messages` accumulates."""
        text = state["user_message"]
        return {**copy.deepcopy(PER_MESSAGE_DEFAULTS), "messages": [HumanMessage(content=text, id=str(uuid.uuid4()))]}
    return timed("reset_turn", reset_turn, fresh=True)


def make_rule_node(is_pure_greeting: Callable[[str], bool]):
    async def rule_greeting_check(state: dict) -> dict:
        """Rules first: a message that is ENTIRELY a greeting/thanks/goodbye never reaches the LLM."""
        if is_pure_greeting(state["user_message"]):
            return {"route": "greeting", "route_source": "rule"}
        return {}
    return timed("rule_greeting_check", rule_greeting_check)


def make_classify_wrapper(classify_node: ClassifyNode):
    """Wrap the router node (module internals untouched): path/timing, and its string error becomes {code, message, node}."""
    async def classify_and_rewrite(state: dict) -> dict:
        try:
            with usage.collect() as router_tokens:
                update = dict(await classify_node(state))
            update["usage"] = usage.total(sum(router_tokens), 0)
        except Exception:                                      # noqa: BLE001 - recorded in state, never propagated (GI-D4)
            log.exception("node classify_and_rewrite failed")
            return {"error": errors.make_error(errors.INTERNAL, "classify_and_rewrite")}
        if isinstance(update.get("error"), str):
            code = errors.LLM_UNAVAILABLE if update["error"] == "llm_unavailable" else errors.INTERNAL
            update["error"] = errors.make_error(code, "classify_and_rewrite")
        return update
    return timed("classify_and_rewrite", classify_and_rewrite)


async def finalize(state: dict) -> dict:
    """Store the reply (template replies included) so the history stays complete; route goes in the metadata.
    [C#] markers are local to one answer, so they are stripped from the stored text (ATTRIBUTION_PLAN Q3).
    sources, related_sections, retrieval and generation are NEVER written into `messages`."""
    msg = AIMessage(content=strip_markers(state["answer"]), additional_kwargs={"route": state["route"]}, id=str(uuid.uuid4()))
    return {"messages": [msg]}


async def error_finalize(state: dict) -> dict:
    """The one error exit (GI-D4). History rule (generation Q7 / graph Q5): roll the turn back, so the next follow-up never
    sees a question that was never answered."""
    update: dict = {"answer": None, "outcome": "error"}
    msgs = state.get("messages") or []
    if msgs and isinstance(msgs[-1], HumanMessage):
        update["messages"] = [RemoveMessage(id=msgs[-1].id)]
    return update


def after_rule(state: dict) -> str:
    return "greeting_reply" if state.get("route") == "greeting" else "classify_and_rewrite"


def after_classify(state: dict) -> str:
    if state.get("error"):
        return "error_finalize"
    return {"greeting": "greeting_reply", "out_of_scope": "out_of_scope_reply", "nist_question": "rag"}[state["route"]]


def after_rag(state: dict) -> str:
    return "error_finalize" if state.get("error") else "finalize"


def build_graph(classify_node: ClassifyNode, greeting_check: Callable[[str], bool], rag_node, checkpointer=None):
    g = StateGraph(GraphState)
    g.add_node("reset_turn", make_reset_turn())
    g.add_node("rule_greeting_check", make_rule_node(greeting_check))
    g.add_node("classify_and_rewrite", make_classify_wrapper(classify_node))
    g.add_node("greeting_reply", timed("greeting_reply", greeting_reply, outcome="greeting"))
    g.add_node("out_of_scope_reply", timed("out_of_scope_reply", out_of_scope_reply, outcome="out_of_scope"))
    g.add_node("rag", rag_node)
    g.add_node("finalize", timed("finalize", finalize))
    g.add_node("error_finalize", timed("error_finalize", error_finalize))
    g.add_edge(START, "reset_turn")
    g.add_edge("reset_turn", "rule_greeting_check")
    g.add_conditional_edges("rule_greeting_check", after_rule, ["greeting_reply", "classify_and_rewrite"])
    g.add_conditional_edges("classify_and_rewrite", after_classify,
                            ["greeting_reply", "out_of_scope_reply", "rag", "error_finalize"])
    g.add_conditional_edges("rag", after_rag, ["finalize", "error_finalize"])
    for node in ("greeting_reply", "out_of_scope_reply"):
        g.add_edge(node, "finalize")
    g.add_edge("finalize", END)
    g.add_edge("error_finalize", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def build_app_graph(settings, client, checkpointer=None, rag_node=None, components=None):
    """The production graph: real greeting rule + real combined router + the real RAG subgraph.
    Components (vector store, BM25, reranker, generation client) are loaded ONCE here, at startup. Tests inject a
    `rag_node` (or fake `components`) instead of loading models."""
    from graph.greeting_rules import is_pure_greeting
    from graph.router import make_classify_node
    client = usage.TokenTrackingClient(client)           # the router's tokens count toward meta.usage
    if rag_node is None:
        from graph.components import load_components
        from graph.rag_subgraph import build_rag_subgraph
        rag_node = build_rag_subgraph(components or load_components(settings))
    return build_graph(make_classify_node(client, settings), is_pure_greeting, rag_node, checkpointer)
