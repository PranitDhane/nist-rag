# API & Routing Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for the **API and routing phase**. Work stage by stage (Section 7). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. A stage is finished only when every "Done when" item is true **and** the user has confirmed the Checkpoint. Questions in Section 10 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

Build the HTTP app and the LangGraph graph that take a user's message, decide what kind of message it is, and either answer it directly with a fixed template or hand it to the RAG pipeline, keeping the conversation so follow-up questions work.

- **In scope:**
  - FastAPI app with a JSON endpoint and a streaming (SSE) endpoint;
  - request and response formats;
  - server-generated conversation IDs and in-memory conversation history;
  - the Groq client with rate-limit handling;
  - the LangGraph graph structure;
  - the rule-based greeting check, the combined LLM routing call, the template replies;
  - a **stub** RAG node;
  - tests and a routing evaluation.
- **Out of scope (next phase):** the inside of the RAG pipeline: retrieval, grading, query retry, answer generation, and citation building. In this phase the RAG node is a stub with a fixed, clearly marked placeholder answer, so everything around it can be built and tested first.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| A-D1 | Interface | **FastAPI** HTTP app (tested with Postman/curl) |
| A-D2 | Response modes | **Both**: `POST /chat` returns one JSON response; `POST /chat/stream` returns Server-Sent Events |
| A-D3 | LLM provider | **Groq, free tier** |
| A-D4 | Conversation storage | **In-memory** LangGraph checkpointer; conversations are lost on server restart |
| A-D5 | Conversation ID | **Server creates `thread_id`** on the first message and returns it; the client sends it back on later messages |
| A-D6 | Authentication | **None** (local app) |
| A-D7 | Routing method | **Rules first** for pure greetings (no LLM call), then **one combined LLM call** that returns the route **and** a standalone version of the question |
| A-D8 | Routes | **Three:** `greeting`, `nist_question` (includes follow-ups), `out_of_scope`. There is no separate "about the assistant" route; such questions are `out_of_scope`. |
| A-D9 | Out-of-scope behaviour | **Never answer.** Reply with a fixed apology. User's wording: "Sorry, we don't have this answer." |
| A-D10 | Greeting and out-of-scope replies | **Fixed template text in code, no LLM call** |
| A-D11 | History window | The combined routing call sees the **last 2 turns** (1 turn = 1 user message + 1 assistant reply) plus the current message |

## 3. Key requirements that follow from these decisions

1. **Mixed messages are questions.** A message containing a greeting **and** anything else ("Hi again! Which one is cross-cutting?") is **never** caught by the greeting rule. Only messages that are **entirely** a greeting, thanks, or goodbye are.
2. **Follow-ups need history.** "Which one is cross-cutting?" alone looks out of scope. The combined call must use the history window to recognise it as a NIST follow-up, and produce a standalone query (e.g., "Which of the four AI RMF functions is cross-cutting?").
3. **Out-of-scope never reaches retrieval or generation.** Nothing in that path may call the LLM or answer from general knowledge.
4. **The router's JSON must never be streamed to the client.** Only text produced for the user is streamed. The routing call's output is internal.
5. **LLM calls per message are counted and reported:** pure greeting → 0; greeting / out-of-scope via the routing call → 1; NIST question → 1 + whatever the RAG pipeline uses (next phase).
6. **Every reply, including template replies, is stored in the conversation**, so the history stays complete.

## 4. Request flow

```
HTTP request
  → validate body; create thread_id if missing
  → graph.invoke / graph.astream with config {"configurable": {"thread_id": ...}}
       START
         → rule_greeting_check ──(pure greeting)──────────→ greeting_reply ─┐
         │                                                                  │
         └─(anything else)→ classify_and_rewrite (1 LLM call)               │
                               ├─ greeting      → greeting_reply ───────────┤
                               ├─ out_of_scope  → out_of_scope_reply ───────┤
                               └─ nist_question → rag_pipeline (STUB) ──────┤
                                                                            ↓
                                                                        finalize → END
  → JSON response, or SSE events
```

## 5. Graph state (initial definition; the RAG phase will extend it)

| Field | Type | Purpose |
|---|---|---|
| `messages` | list of messages, with LangGraph's `add_messages` reducer | Full conversation (persisted by the checkpointer) |
| `user_message` | str | The current raw message |
| `route` | `"greeting"` \| `"nist_question"` \| `"out_of_scope"` \| null | Routing result |
| `route_source` | `"rule"` \| `"llm"` \| `"fallback"` | How the route was decided |
| `standalone_query` | str \| null | Rewritten self-contained question (NIST route only) |
| `route_reason` | str \| null | Short reason from the routing call (for logs/debug, not shown to users) |
| `answer` | str \| null | Final text for the user |
| `sources` | list | Empty in this phase; filled by the RAG phase |
| `llm_calls` | int | LLM calls used for this message |
| `error` | str \| null | Set if something failed (e.g., rate limit exhausted) |

Per-message fields must be **reset at the start of every message**, so values from the previous message never leak into the next one. Only `messages` accumulates.

## 6. Project layout additions

```
app/
├── __init__.py
├── main.py              ← FastAPI app, startup (build graph once), uvicorn entry
├── config.py            ← settings from .env (GROQ_API_KEY, model names, limits, host/port)
├── schemas.py           ← Pydantic request/response models
├── api.py               ← /chat, /chat/stream (+ extra endpoints per Q5)
└── sse.py               ← SSE event formatting
graph/
├── __init__.py
├── state.py             ← graph state definition
├── build.py             ← builds and compiles the graph with the checkpointer
├── greeting_rules.py    ← pure-greeting rule (no LLM)
├── router.py            ← combined classify + rewrite node
├── replies.py           ← template reply nodes (fixed texts)
├── rag_stub.py          ← placeholder RAG node (replaced next phase)
├── history.py           ← builds the 2-turn history window
└── prompts.py           ← routing prompt text
llm/
├── __init__.py
└── groq_client.py       ← ChatGroq setup, retries/backoff, call counting
tests/
├── test_greeting_rules.py
├── test_history_window.py
├── test_router_parsing.py     ← mocked LLM
├── test_graph_routes.py       ← mocked LLM, real graph + checkpointer
├── test_api.py                ← FastAPI TestClient, mocked LLM
├── test_sse.py
└── test_live_groq.py          ← real Groq; skipped without GROQ_API_KEY
eval/
└── routing_cases.yaml         ← labelled routing examples (Stage R8)
```

---

## 7. Stages

### Stage R0 — Prerequisites, dependencies, model choice

**Purpose:** settle what's needed before any code.

**Steps**
1. Confirm with the user that indexing is complete (or that the user wants this phase built in parallel; ask). Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.
2. **Ask Q1:** which Groq model to use for the routing call. Before asking, check Groq's **current** models page and rate-limits page, because models and free-tier limits change. Present 2–3 candidates with:
   - support for structured output (tool calling / JSON mode);
   - the free-tier limits (requests per minute and per day, tokens per minute and per day);
   - a short note on speed and quality.
3. **Ask Q2:** pin package versions. Propose `fastapi`, `uvicorn`, `langgraph`, `langchain-groq`, `pydantic`, `pydantic-settings` or `python-dotenv` (whichever the project already uses), `httpx` (for tests), and the SSE approach from Q3.
4. **Ask Q3:** SSE implementation: FastAPI's built-in `StreamingResponse` with hand-formatted events, or the `sse-starlette` package.
5. `GROQ_API_KEY` goes in `.env` (never committed); add it to `.env.example` with an empty value.

**Done when:** the answers are recorded; dependencies install in a fresh venv.

**Checkpoint:** show the model comparison the user chose from, the updated `requirements.txt`, and `.env.example`.

### Stage R1 — Configuration and Groq client

**Purpose:** one place for settings, and one LLM client that handles rate limits safely.

**Steps**
1. `app/config.py`:
   - `GROQ_API_KEY` (required; fail at startup with a clear message if missing);
   - `ROUTER_MODEL` (Q1);
   - `HISTORY_TURNS = 2`;
   - `MAX_MESSAGE_CHARS` (Q6);
   - host and port (Q7);
   - retry settings (Q8).
2. `llm/groq_client.py`:
   - builds `ChatGroq` with `temperature=0` for the routing call (classification should be stable);
   - **retry with exponential backoff** on rate-limit and temporary server errors, honouring a `retry-after` header when present, with a maximum number of attempts (Q8);
   - when retries are exhausted, raise a specific `LLMUnavailableError`, which the API turns into the HTTP status chosen in Q8;
   - increment `llm_calls` for each call that **succeeds** (failed retries are logged separately).
3. Check how the installed `langchain-groq` version exposes structured output (`with_structured_output` and its method options). **Report what you find; don't guess.**

**Done when:**
- unit tests pass for: retry on a simulated 429 then success; giving up after the maximum attempts; the call counter;
- a live ping to Groq succeeds (skipped without a key).

**Checkpoint:** show the retry test output, the live ping result and latency, and the structured-output method found.

### Stage R2 — Graph skeleton with checkpointer

**Purpose:** a working graph with all nodes as simple stubs, conversation memory included, before any real logic.

**Steps**
1. `graph/state.py` with the fields from Section 5, using `add_messages` for `messages`.
2. Nodes as stubs:
   - `rule_greeting_check` always says "not a greeting";
   - `classify_and_rewrite` returns a fixed route set by the test;
   - `greeting_reply` and `out_of_scope_reply` return their templates;
   - `rag_pipeline` is the stub from R5;
   - `finalize` appends the AI reply to `messages`.
3. Conditional edges exactly as in Section 4.
4. Compile with the **in-memory checkpointer** (LangGraph's `MemorySaver` / `InMemorySaver`; check the name in the installed version). The graph is built **once** at app startup, not per request.
5. Reset the per-message fields at the start of each run (Section 5).

**Done when:**
- tests prove each route reaches the right node;
- two messages with the same `thread_id` share history;
- different `thread_id`s don't see each other's history;
- per-message fields from message 1 are not present while message 2 is processed.

**Checkpoint:** show the graph diagram (LangGraph can export Mermaid; include the text) and the test results.

### Stage R3 — Pure-greeting rule

**Purpose:** answer plain greetings with zero LLM calls, without ever catching a real question.

**Steps**
1. `graph/greeting_rules.py` → `is_pure_greeting(text) -> bool`.
   - Normalize: lowercase, trim, collapse spaces, strip trailing punctuation and emoji.
   - Return true **only if the whole normalized message** matches the approved greeting / thanks / goodbye list (Q4), optionally followed by a short name like "Claude" or "bot".
2. **Ask Q4:** show the proposed list for approval (e.g., hi, hello, hey, good morning/afternoon/evening, thanks, thank you, thx, bye, goodbye, see you). Store the approved list in a config or data file, not scattered through the code.
3. Tests, all required:
   - **positives:** `hi`, `Hello!`, `hey there`, `Good morning`, `thanks!`, `Thank you so much`, `bye`;
   - **negatives (must be false):** `Hi, what is GOVERN?`, `Hi again! Which one is cross-cutting?`, `hello, explain confabulation`, `thanks, and what about MAP?`, `hey can you list the 12 risks`, `hello world program in python`, `good`, `thanks for nothing, what is NIST?`.

**Done when:** all positive and negative tests pass.

**Checkpoint:** show the approved list and the test table (message → result).

### Stage R4 — Combined classify-and-rewrite call

**Purpose:** with one LLM call, decide the route using recent history, and produce a standalone question for NIST messages.

**Steps**
1. **History window** (`graph/history.py`): take the last **2 turns** before the current message from `messages`. For assistant turns, include only the answer text (no sources list). **Ask Q9** whether greeting and out-of-scope turns count toward the 2 turns, or are skipped so the window holds the last 2 *NIST* turns. Show an example of each.
2. **Output schema** (Pydantic):
   - `route`: one of `greeting`, `nist_question`, `out_of_scope`;
   - `standalone_query`: str, required when route is `nist_question`, otherwise empty;
   - `reason`: one short sentence.
3. **Prompt** (`graph/prompts.py`) must state:
   - **the scope:** the NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1) and its Generative AI Profile (NIST AI 600-1), and AI risk management topics they cover;
   - **the three routes with definitions**: a message with a greeting plus a question is classified by its question; a follow-up that only makes sense with the history and refers to NIST content is `nist_question`; questions about the assistant itself are `out_of_scope` (A-D8);
   - **rewrite rules:** resolve pronouns and references from history ("which one" → "which of the four AI RMF functions"), keep the user's intent, **add no facts**, keep IDs exactly as written (e.g., `GV-1.7-001`);
   - **3–6 short examples**, including the "Hi again! Which one is cross-cutting?" case and an out-of-scope case;
   - "respond only in the required structure".

   Show the prompt to the user for approval before using it.
4. **Use the structured-output method** verified in R1, with `temperature=0`.
5. **Failure handling:** if the output is invalid, retry once; if it still fails, use the fallback route chosen in **Q10** and set `route_source = "fallback"`. Log the raw output.
6. If the LLM is unavailable after retries (R1), set `error` and stop. The API returns the status from Q8. **Never guess a route on an LLM outage.**

**Done when:**
- parsing tests with a mocked LLM pass (valid output, invalid-then-valid, invalid twice → fallback, outage → error);
- the live test passes on these 5 cases:
  1. "What are the four core functions of the AI RMF?" → `nist_question`
  2. after (1) and its answer, "Which one is cross-cutting?" → `nist_question` with a standalone query naming the AI RMF functions
  3. "Hi again! Which one is cross-cutting?" (same history) → `nist_question`
  4. "What's the capital of France?" → `out_of_scope`
  5. "Good morning, hope you're doing well" → `greeting`

**Checkpoint:** show the approved prompt, and for each live case: route, standalone query, reason, and latency.

### Stage R5 — Template replies, RAG stub, finalize

**Purpose:** finish every path through the graph.

**Steps**
1. **Templates** (`graph/replies.py`), fixed strings, no LLM. **Ask Q11** to approve the exact texts. Proposals:
   - greeting: "Hello! I can answer questions about the NIST AI Risk Management Framework and its Generative AI Profile. What would you like to know?"
   - out of scope: the user's wording, "Sorry, we don't have this answer." (Q11 also asks whether to append the one-line scope note.)
2. **RAG stub** (`graph/rag_stub.py`): sets `answer` to `[RAG pipeline not implemented yet] Standalone query: <standalone_query>` and `sources = []`, with **no LLM call**. It must be obviously a placeholder; it is replaced in the next phase.
3. **finalize:** append the AI message with the answer text to `messages`. Store `route` in the message metadata (it may be useful to the history window, depending on Q9).

**Done when:**
- each route produces the right reply;
- the out-of-scope path makes **zero** LLM calls after routing, and its reply is exactly the template;
- pure greetings make zero LLM calls in total.

**Checkpoint:** show a 4-message conversation going through all routes, with `llm_calls` per message.

### Stage R6 — `POST /chat` (JSON)

**Purpose:** the main HTTP endpoint.

**Request:**

| Field | Type | Rule |
|---|---|---|
| `message` | str | Required, non-empty after trimming, ≤ `MAX_MESSAGE_CHARS` (Q6) |
| `thread_id` | str \| null | Omit on the first message |

**Response (200):**

| Field | Type |
|---|---|
| `thread_id` | str (new UUID4 if the request had none) |
| `route` | `greeting` \| `nist_question` \| `out_of_scope` |
| `answer` | str |
| `sources` | list (empty until the RAG phase) |
| `standalone_query` | str \| null (NIST route only) |
| `meta` | `{llm_calls, route_source, latency_ms}` |

**Errors:**
- 422: validation (FastAPI default format);
- `thread_id` unknown: behaviour per **Q12**;
- LLM unavailable: status and body per **Q8**;
- unexpected error: 500 with a generic message and the details logged (never return stack traces).

**Steps**
1. Implement with the graph built at startup.
2. **Ask Q13** about concurrent requests on the same `thread_id` (reject with 409, or queue them).
3. Log one structured line per request: `thread_id`, route, `route_source`, `llm_calls`, `latency_ms`. **Never log the full message text** at INFO level; DEBUG only.

**Done when:** API tests pass:
- first message returns a new `thread_id`;
- a follow-up with that `thread_id` is routed using history;
- validation errors;
- unknown-`thread_id` behaviour (Q12);
- LLM-outage behaviour (Q8);
- out-of-scope reply is exactly the template.

**Checkpoint:** show Postman-style example requests and responses for each route, and one error case.

### Stage R7 — `POST /chat/stream` (SSE)

**Purpose:** the same behaviour as `/chat`, streamed.

**Steps**
1. Same request body and validation as `/chat`.
2. **Ask Q14:** approve the event format. Proposal:
   - `meta`: `{thread_id, route}`, sent **as soon as routing is done**;
   - `token`: `{text}`, repeated;
   - `sources`: `{sources}`, once, after the answer;
   - `done`: `{llm_calls, latency_ms}`;
   - `error`: `{code, message}`, which ends the stream.
3. **Template replies** (greeting / out of scope) are sent as **one** `token` event with the full text, then `done`.
4. **NIST route:** in this phase the stub's text is sent as one `token` event. Design the streaming code so the next phase can stream real LLM tokens from **only** the answer-generation node. Use LangGraph's message streaming filtered by node name or tag, so the routing call's JSON can **never** appear in the stream. Add a test that proves the routing call's output is not streamed.
5. If the client disconnects, stop processing that request cleanly and log it.

**Done when:**
- SSE tests pass: event order, templates as one token, the no-router-output-leak test, error event on LLM outage;
- `curl -N` shows events arriving.

**Checkpoint:** show a raw SSE transcript for a greeting, an out-of-scope message, and a NIST question.

### Stage R8 — Routing evaluation and report

**Purpose:** measure routing quality with real Groq calls, separately from the unit tests.

**Steps**
1. `eval/routing_cases.yaml` with **at least 25 labelled cases**. Each case has optional history (up to 2 turns), the message, the expected route, and for NIST cases the key terms the standalone query must contain. Include:
   - pure greetings (should be caught by the rule);
   - greetings the rule misses but the LLM should catch;
   - direct NIST questions about both documents;
   - follow-ups that need history ("Which one…", "What about MAP?", "Tell me more about the second risk");
   - mixed greeting + question;
   - out-of-scope general knowledge, coding requests, and questions about the assistant;
   - **tricky near-scope cases** (e.g., "What is the EU AI Act?" → out of scope, since it isn't in these documents).

   **Ask Q15** for the user to review and approve the cases.
2. Script `python -m eval.routing` that runs every case, respecting rate limits (pauses between calls), and reports:
   - accuracy per route;
   - a confusion table;
   - the standalone-query term check;
   - LLM calls used;
   - every failure in full.
3. Results go to `eval/routing_report.json`. **Misses do not fail the build.** They go to the user, who decides whether to adjust the prompt.

**Done when:** the evaluation runs end to end and the report is produced.

**Checkpoint:** show the accuracy table and every failed case.

---

## 8. Testing rules

- Unit and API tests use a **mocked LLM** and need no network or API key.
- Live tests (`test_live_groq.py`, the routing evaluation) are **skipped** without `GROQ_API_KEY`, and are kept small to respect free-tier limits.
- Never weaken the greeting-rule negatives or the out-of-scope "zero calls" assertion to make tests pass.

## 9. README additions from this phase

- How to set `GROQ_API_KEY` and start the server (`uvicorn` command, host and port).
- Example Postman/curl calls for `/chat` and `/chat/stream`, including how to continue a conversation with `thread_id`.
- Design notes: routing approach and why (rate limits), the three routes, template replies, the 2-turn window, in-memory history (lost on restart).

## 10. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Groq model for the routing call (current models and free-tier limits checked first) | R0 |
| Q2 | Exact package versions | R0 |
| Q3 | SSE via `StreamingResponse` or `sse-starlette` | R0 |
| Q4 | Approved pure-greeting / thanks / goodbye list | R3 |
| Q5 | Extra endpoints: `GET /health`? `GET /threads/{id}` (history)? `DELETE /threads/{id}`? | R6 |
| Q6 | Maximum message length (characters) | R1 |
| Q7 | Host and port (e.g., 127.0.0.1:8000) | R1 |
| Q8 | Retry attempts and backoff, and the HTTP status and body when Groq stays unavailable (e.g., 503 with a retry message) | R1 |
| Q9 | Do greeting and out-of-scope turns count toward the 2-turn window, or only NIST turns? | R4 |
| Q10 | Fallback route if the routing output is invalid twice (e.g., `out_of_scope` as the safe default, or `nist_question`) | R4 |
| Q11 | Exact template texts (greeting; out of scope, and whether to add a scope line) | R5 |
| Q12 | Unknown or expired `thread_id` (e.g., after a restart): return 404, or silently start a new conversation and return a new ID? | R6 |
| Q13 | Two simultaneous requests on the same `thread_id`: reject (409) or queue? | R6 |
| Q14 | SSE event names and payloads | R7 |
| Q15 | Approval of the routing evaluation cases | R8 |
| Q16 | Any difference between this plan and the installed library behaviour | Whenever found |

## 11. Handoff to the RAG phase (not to implement now)

- Replace `rag_stub` with the real RAG pipeline (a subgraph or a node sequence). It receives `standalone_query`, and must return `answer` and `sources`.
- The out-of-scope route is decided **before** retrieval. A NIST question for which retrieval finds nothing relevant is the RAG phase's "insufficient context" case, a different path with its own design.
- Stream answer tokens only from the generation node (Stage R7 design).
- Load the indexes once at startup using `indexing.loaders`.
