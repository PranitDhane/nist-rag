"""Fixed template replies: text lives in code, no LLM call (A-D10). Wording approved in Stage R5 (Q11)."""
from __future__ import annotations

GREETING_REPLY = ("Hello! I can answer questions about the NIST AI Risk Management Framework and its "
                  "Generative AI Profile. What would you like to know?")
OUT_OF_SCOPE_REPLY = "Sorry, we don't have this answer."


async def greeting_reply(state: dict) -> dict:
    return {"answer": GREETING_REPLY, "sources": []}


async def out_of_scope_reply(state: dict) -> dict:
    return {"answer": OUT_OF_SCOPE_REPLY, "sources": []}
