"""Stage N6: the LangGraph generation node, ready to wire (item 5 does the wiring).

State it reads:   standalone_query (str), retrieval_result (retrieval.models.RetrievalResult, non-empty chunks)
State it writes:  answer          final_text, or None on ERROR      -> graph.build.finalize stores it as the assistant message
                  generation      GenerationResult as a dict (no reasoning exists in it; streamed_text is for logs only)
                  cited_keys      from the final parse (item 4 builds Sources from these)
                  key_map         C# -> chunk_id
                  llm_calls       +1 when the model call succeeded
                  error           "generation_failed" on ERROR (else untouched)  -> graph.build.error_finalize rolls the turn back (Q7)
In the integrated graph (graph/rag_subgraph.py `make_generate`) this node is wrapped: the retrieval dict in state becomes the
RetrievalResult it expects, and its string error becomes {code, message, node}; the main graph then exits through
`error_finalize`, which removes the unanswered user message.

Streaming: the node calls the model through generate_stream(), i.e. chat_model.astream(), tagged `answer_generation`
(generation.config.ANSWER_TAG), so LangGraph's messages stream can be filtered to this node's content tokens only.
Reasoning never reaches the stream (hidden at the API, filtered again in the client). If the final text differs from
what was streamed, state["generation"]["corrected"] is True and ["final_text"] is the text to show (the `correction` event).
"""
from __future__ import annotations

import logging

from app.config import Settings
from generation.client import GenerationClient
from generation.generate import generate_stream
from generation.models import ERROR, GenerationError

log = logging.getLogger(__name__)

GENERATION_FIELDS = ("retrieval_result", "generation", "cited_keys", "key_map")


def make_generate_node(client: GenerationClient | None = None, settings: Settings | None = None):
    state_client: dict = {"client": client}

    async def generate_node(state: dict) -> dict:
        query = state["standalone_query"]
        retrieval_result = state.get("retrieval_result")
        if retrieval_result is None:
            raise GenerationError("empty_context", "state has no retrieval_result")
        from generation.generate import resolve_max_tokens
        if state_client["client"] is None:
            state_client["client"] = GenerationClient(settings or Settings.from_env(), max_tokens=resolve_max_tokens(False))
        result = None
        async for event in generate_stream(query, retrieval_result, client=state_client["client"]):
            if event["type"] == "result":
                result = event["result"]
            elif event["type"] == "error":
                log.error("generation node failed: %s", event["code"])
                from generation.generate import error_result
                result = error_result(GenerationError(event["code"], event["message"]), {}, state_client["client"])
        if result is None or result.outcome == ERROR:
            return {"answer": None, "generation": result.to_dict() if result else None, "cited_keys": [], "key_map": {},
                    "error": "generation_failed"}
        return {"answer": result.final_text,                 # ONLY the validated final text: never streamed_text
                "generation": result.to_dict(), "cited_keys": list(result.cited_keys), "key_map": dict(result.key_map),
                "llm_calls": int(state.get("llm_calls") or 0) + 1}

    return generate_node
