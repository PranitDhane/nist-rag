"""A stand-in for the RAG subgraph in tests that exercise routing / the API only (the production graph has no stub)."""
from __future__ import annotations

STUB_PREFIX = "[RAG pipeline not implemented yet] Standalone query: "


async def stub_rag(state: dict) -> dict:
    return {"answer": f"{STUB_PREFIX}{state.get('standalone_query')}", "sources": []}
