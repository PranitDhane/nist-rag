"""End-to-end harness settings (E2E_EVALUATION_PLAN Section 6). Limits live here, not as literals in e2e.py."""
from __future__ import annotations

DEFAULT_BASE_URL = "http://127.0.0.1:8000"     # API plan Q7
PACE_SECONDS = 35.0                            # Q5 (user): minimum gap after a request that reached generation.
                                               # ~4,000 tokens per answered turn against 8,000 tokens/minute.
REQUEST_TIMEOUT_S = 120.0                      # generation alone may take up to its 60 s total timeout, plus routing and retrieval
HEALTH_TIMEOUT_S = 5.0
RATE_LIMIT_RETRIES = 1                         # plan: wait and retry that turn ONCE, recording it
RATE_LIMIT_CODES = ("llm_unavailable",)        # the API's code for rate limits / temporary Groq errors (503 + Retry-After)
DEFAULT_RETRY_AFTER_S = 15.0                   # used when a 503 carries no usable Retry-After header
RUNS_DIR_NAME = "e2e_runs"
EXIT_HARNESS_FAILURE = 2                       # the only non-zero exit (E-D6): server unreachable, bad --from-run, bad cases
WARMUP_MESSAGE = "Which of the four AI RMF functions is cross-cutting?"   # --warmup only; never part of the results
STREAM_TURNS = ("S1", "S7")                    # Q4 (user): the streaming subset = one answered turn + one near-scope turn
STREAM_PATH = "/chat/stream"
SUPPORT_THRESHOLD: float | None = 0.70         # Q2 (user): chosen from the V4 calibration of run 20261019-105606 (0% of negatives, 100% of positives pass)
RESULTS_MD = "docs/E2E_RESULTS.md"             # generated; never edited by hand (CLAUDE.md section 5)
