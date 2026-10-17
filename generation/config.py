"""Generation settings (GENERATION_PLAN Stage N1). Retry/backoff settings are the router's (Q8): app.config.Settings."""
from __future__ import annotations

import os

GEN_MODEL = os.environ.get("GEN_MODEL", "openai/gpt-oss-20b")       # GN-D1 / Q1: verified listed, ctx 131072
TEMPERATURE = 0.2                                                    # GN-D3
REASONING_EFFORT = "medium"                                          # GN-D2
REASONING_FORMAT = "hidden"                                          # Q2: reasoning removed at the API level
                                                                     # (also filtered in the stream: defence in depth)

# Account limits for GEN_MODEL, read from x-ratelimit-* response headers on 2026-10-17 (Q1).
TPM_LIMIT = 8000                                                     # tokens per minute: the binding limit
REQUEST_LIMIT_HEADER = 1000                                          # x-ratelimit-limit-requests (daily cap by Groq convention; unconfirmed)

# Output limit INCLUDING reasoning tokens. Chosen in Stage N4 from 27 live requests (live runs refuse to run if None).
MAX_COMPLETION_TOKENS: int | None = 3500                           # Q3 (revised for graph integration): 2500 was too tight for broad questions (reasoning varies 1.6k-2.5k+)
MEASUREMENT_MAX_COMPLETION_TOKENS = 3000                             # generous temporary limit for measurement mode only
LENGTH_POLICY: str | None = "error"                                  # Q4: finish_reason == "length" -> ERROR ("insufficient" | "keep" | "error")

TTFT_TIMEOUT_S = 20.0                                                # Q6: time to first CONTENT token (reasoning comes first)
TOTAL_TIMEOUT_S = 60.0

# Rough prompt-size estimate for the budget pre-check; calibrated against Groq's reported prompt_tokens in N4.
CHARS_PER_TOKEN = 4.0
ESTIMATE_MARGIN = 1.0                 # was 1.2: measured 4.0 chars/token, and the 1.2 margin rejected 14% of real NIST turns (graph integration)

ANSWER_TAG = "answer_generation"                                     # tag on the LLM call, for the API's stream filter
