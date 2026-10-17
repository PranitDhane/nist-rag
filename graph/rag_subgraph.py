"""The compiled RAG subgraph (GRAPH_INTEGRATION_PLAN Stage G3, GI-D1/GI-D2): used as ONE node of the main graph.

    START -> hybrid_search -> rerank -> gate --relevant=false--> gate_failure_reply --------------------> END
                                         `--relevant=true---> expand_context -> generate
                                                                  generate --INSUFFICIENT--> finalize_model_refusal -> END
                                                                  generate --ANSWERED------> attribute ------------> END
    any node with `error` set -> END (the main graph's edge after `rag` routes to error_finalize)

No loops, no retries. State interface = shared keys with the main state (Q1 a): the subgraph can also be invoked on its
own with {"standalone_query": "..."}.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from generation.node import make_generate_node
from graph import errors, usage
from graph.components import Components
from graph.rag_nodes import instrumented, make_rag_nodes
from graph.serial import retrieval_result_from_state


class RagState(TypedDict, total=False):
    """Subset of the main GraphState used by the subgraph (shared keys)."""
    standalone_query: str | None
    llm_calls: int
    retrieval: dict | None
    generation: dict | None
    answer: str | None
    sources: list
    cited_keys: list
    key_map: dict
    insufficient: dict | None
    related_sections: list
    insufficient_reason: str | None
    outcome: str | None
    error: dict | None
    path: list
    timings_ms: dict
    usage: dict


def make_generate(comps: Components):
    """Wrap generation.node.make_generate_node: the retrieval dict in state becomes the RetrievalResult the module expects
    (never stored in state); the module's string error becomes {code, message, node}."""
    inner = make_generate_node(comps.generation_client)

    @instrumented("generate", errors.GENERATION_FAILED)
    async def generate(state: dict, config) -> dict:
        update = await inner({"standalone_query": state["standalone_query"],
                              "retrieval_result": retrieval_result_from_state(state["retrieval"]),
                              "llm_calls": state.get("llm_calls", 0)})
        if update.get("error"):
            gen = update.get("generation") or {}
            code = (gen.get("error") or {}).get("code", "")
            return {"generation": gen or None, "answer": None, "error": errors.from_generation_code(code, "generate")}
        prior = state.get("usage") or {}
        return {"generation": update["generation"], "answer": update["answer"], "cited_keys": update["cited_keys"],
                "key_map": update["key_map"], "llm_calls": update["llm_calls"],
                "usage": usage.total(prior.get("router", 0), update["generation"]["usage"].get("total", 0))}
    return generate


# ------------------------------------------------------------------ conditional edges (error is always checked first)
def after_hybrid_search(state: dict) -> str:
    return END if state.get("error") else "rerank"


def after_rerank(state: dict) -> str:
    return END if state.get("error") else "gate"


def after_gate(state: dict) -> str:
    if state.get("error"):
        return END
    return "expand_context" if (state.get("retrieval") or {}).get("relevant") else "gate_failure_reply"


def after_expand(state: dict) -> str:
    return END if state.get("error") else "generate"


def after_generate(state: dict) -> str:
    if state.get("error"):
        return END
    return "finalize_model_refusal" if (state.get("generation") or {}).get("outcome") == "INSUFFICIENT" else "attribute"


def build_rag_subgraph(comps: Components, nodes: dict | None = None):
    nodes = dict(nodes or make_rag_nodes(comps))
    nodes.setdefault("generate", make_generate(comps))
    g = StateGraph(RagState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "hybrid_search")
    g.add_conditional_edges("hybrid_search", after_hybrid_search, ["rerank", END])
    g.add_conditional_edges("rerank", after_rerank, ["gate", END])
    g.add_conditional_edges("gate", after_gate, ["expand_context", "gate_failure_reply", END])
    g.add_conditional_edges("expand_context", after_expand, ["generate", END])
    g.add_conditional_edges("generate", after_generate, ["finalize_model_refusal", "attribute", END])
    for terminal in ("gate_failure_reply", "finalize_model_refusal", "attribute"):
        g.add_edge(terminal, END)
    return g.compile(name="rag")
