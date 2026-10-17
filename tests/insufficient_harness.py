"""Test-only graph for the API tests: the REAL main graph (graph.build.build_graph) with an injected `rag` node.

The insufficient-context and attribution API tests feed their own fake rag nodes (below) into the real routing, state,
`finalize`, error handling and checkpointer, so they exercise the API contract without loading any model. (This used to
build its own graph with an extra state schema; the real state now has those fields.)
"""
from __future__ import annotations

from graph.build import build_graph
from insufficient.reply import finalize_model_refusal, gate_failure_reply


def build_harness_graph(classify_node, rag_node, greeting_check=lambda t: False):
    return build_graph(classify_node, greeting_check, rag_node)


# ---- rag nodes standing in for item 5's wiring ------------------------------------------------------------------
def answered_node(text="GOVERN is a cross-cutting function [C1].", sources=None):
    async def node(state):
        return {"answer": text, "sources": sources or [], "outcome": "answered", "insufficient": None}
    return node


def gate_failure_node(retrieval_result):
    async def node(state):                              # gate failure: no generation call, llm_calls unchanged (routing only)
        reply = gate_failure_reply(retrieval_result)
        return {"answer": reply.answer, "sources": reply.sources, "outcome": reply.outcome, "insufficient": reply.to_dict()}
    return node


def model_refusal_node(generation_result, retrieval_result, streaming_model=None):
    """Optionally streams `streaming_model` tokens first (what the user would see while the model generates)."""
    async def node(state):
        if streaming_model is not None:
            await streaming_model.ainvoke("q")
        reply = finalize_model_refusal(generation_result, retrieval_result)
        return {"answer": reply.answer, "sources": reply.sources, "outcome": reply.outcome, "insufficient": reply.to_dict(),
                "llm_calls": int(state.get("llm_calls") or 0) + 1}      # the generation call happened before this node
    return node
