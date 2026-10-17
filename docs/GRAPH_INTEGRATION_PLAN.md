# Graph Integration Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for **graph integration**: connecting routing, retrieval, the relevance gate, generation, the insufficient-context path and attribution into the final LangGraph workflow, and exposing it through the existing API. Work stage by stage (Section 8). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. Questions in Section 11 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

- **In scope:**
  - the main graph and the RAG subgraph;
  - state design and serializability;
  - wiring every built module as nodes;
  - error routing to one error node;
  - streaming through the subgraph;
  - assembling the API response, including the optional `debug` block;
  - optional LangSmith tracing;
  - graph diagrams;
  - integration tests;
  - measuring state size and memory.
- **Out of scope:** changing the internals of the modules being wired (routing, retrieval, augmentation, generation, insufficient path, attribution). If wiring reveals a problem in a module, **stop and report it**; don't patch it silently. End-to-end quality evaluation is item 6.
- `rag_stub` is **removed** at the end of this phase.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| GI-D1 | Structure | **Main graph + a compiled RAG subgraph** used as one node of the main graph |
| GI-D2 | Retrieval granularity | **Separate nodes:** `hybrid_search → rerank → gate` (plus `expand_context` for sibling expansion and ordering, which retrieval does after the gate) |
| GI-D3 | State contents | **Keep the full retrieval and generation results in state** (reasoning text excluded, as the generation plan requires) |
| GI-D4 | Node failures | **Errors are written into state** and routed to **one `error_finalize` node**; exceptions don't propagate to the API |
| GI-D5 | Tracing | **LangSmith, optional**, enabled only by environment variables; **off by default** |
| GI-D6 | Debug output | **`meta.debug` only when the request has `debug=true`** |

The existing structured logs (API, generation, insufficient phases) stay as they are. LangSmith is additional.

## 3. Graph structure

**Main graph:**

```
START
 → reset_turn                      (per-message field reset, existing rule)
 → rule_greeting_check ──(pure greeting)─────────────→ greeting_reply ───────┐
 └→ classify_and_rewrite (LLM #1)                                            │
      ├─ greeting      → greeting_reply ─────────────────────────────────────┤
      ├─ out_of_scope  → out_of_scope_reply ─────────────────────────────────┤
      ├─ nist_question → rag (SUBGRAPH) ─────────────────────────────────────┤
      └─ error         → error_finalize ─────────────────────────────────────┤
                                                                             ↓
                                                     finalize (history) → END
error_finalize → END
```

**RAG subgraph:**

```
START
 → hybrid_search            (EnsembleRetriever: vector k=10 with query prefix + BM25 k=10, weights 0.6/0.4)
 → rerank                   (bge-reranker-base, top 5, scores kept)
 → gate                     (threshold)
      ├─ relevant=false → gate_failure_reply   (no LLM) ──────────────┐
      └─ relevant=true  → expand_context       (siblings + best-first ordering)
                           → generate          (LLM #2; tagged answer_generation)
                                ├─ outcome=INSUFFICIENT → finalize_model_refusal ─┤
                                ├─ outcome=ANSWERED     → attribute ──────────────┤
                                └─ outcome=ERROR        → (error exit) ───────────┤
                                                                                  ↓
                                                                                 END
```

- **Every node can produce an error** (Section 5). Each conditional edge checks `error` first.
- There are **no loops** (no retries anywhere), so the graph is a decision tree: easy to explain, test and diagram.
- `finalize` writes the assistant message to history (Section 6). `error_finalize` follows the history-on-error rule (Q5).

## 4. State design

**Main state** (extends the API phase's state):

| Field | Set by | Notes |
|---|---|---|
| `messages` | `finalize` | Accumulates (the only field that does) |
| `user_message`, `route`, `route_source`, `route_reason`, `standalone_query` | routing | Existing |
| `retrieval` | `hybrid_search`, `rerank`, `gate`, `expand_context` | **Full** retrieval data (GI-D3): candidates with text, rerank scores, gate result, siblings, final ordered chunks |
| `generation` | `generate` | **Full** `GenerationResult` without reasoning (GI-D3) |
| `answer` | reply nodes / `generate` | Final user-facing text |
| `sources` | `attribute` | `[]` unless answered |
| `related_sections` | insufficient nodes | `[]` unless insufficient |
| `outcome` | the terminal node of each path | `answered` / `insufficient` / `greeting` / `out_of_scope` / `error` |
| `insufficient_reason` | insufficient nodes | Null otherwise |
| `error` | any node | `{code, message, node}` or null |
| `path` | every node | Node names in order; used for `debug` |
| `timings_ms` | every node | Per-node durations |
| `llm_calls` | LLM nodes | Existing counter |

**Rules**
1. All per-message fields are **reset** at the start of every message (`reset_turn`), so the previous message's results never leak into the next (existing rule, now covering the new fields).
2. Everything in state must be **serializable by the checkpointer**. Stage G1 verifies how the installed checkpointer serializes values; retrieval and generation results are stored as Pydantic models or plain dicts accordingly. **Never store** live objects (retrievers, models, clients) in state.
3. The **subgraph's state interface** (shared keys vs a separate schema with explicit input/output mapping) is **Q1**.

## 5. Error handling (GI-D4)

- Each node catches its own failures and writes `error = {code, message, node}`. **It never swallows an error silently:** every caught error is logged with the node name and the stack trace at ERROR level.
- **Error codes** (approve the list and HTTP mapping in **Q4**):

| Code | Typical cause | Proposed HTTP status (`/chat`) |
|---|---|---|
| `LLM_UNAVAILABLE` | Groq retries exhausted (routing or generation) | as decided in API plan Q8 (e.g., 503) |
| `RETRIEVAL_FAILED` | vector store or BM25 failure | 500 |
| `RERANK_FAILED` | reranker failure | 500 |
| `GENERATION_FAILED` | non-rate-limit generation failure, timeout | 500, or the Q8 status if timeout-related (Q4) |
| `INTERNAL` | anything else | 500 |

- Inside the subgraph, an error leads to the subgraph's exit, and the main graph's edge after `rag` routes to `error_finalize`.
- `error_finalize` sets `outcome = error` and applies the history rule (Q5). The API maps the code to an HTTP status and a **generic** message; the response never contains stack traces or internal details. In streaming, an `error` event is sent with the code and the generic message.
- A **mid-stream** generation failure is already handled by the generation phase (error event, partial answer discarded). The graph only has to make sure no `sources`, `related` or `done` events follow it.

## 6. Finalize and history

- `finalize` appends **one** assistant message per turn: the final `answer` text, with `[C#]` markers stripped or kept per **attribution plan Q3** (ask here if it wasn't answered there; this is **Q6**).
- `sources`, `related_sections`, `retrieval` and `generation` are **never** written into `messages`, so the 2-turn routing window only ever sees conversation text.
- After an error, the rule from **generation plan Q7** applies (ask here if unanswered; this is **Q5**): keep the unanswered user message, or roll the turn back.

## 7. Streaming, response assembly, debug, tracing

**Streaming through the subgraph:**
- Stream the main graph with subgraph output enabled (check the installed LangGraph's option name and event format, and report).
- Forward to the client **only** content tokens from LLM calls tagged `answer_generation`.
- Then forward the generation phase's `correction` (if any), then `sources` **or** `related`, then `done`, in the orders approved in earlier phases.
- Template replies (greeting, out of scope, gate failure) are sent as one `token` event.

**Leak tests:** the router's output, the reasoning text, rerank internals and chunk texts must **never** appear in the stream. Test this with fakes that inject recognisable marker strings into each of them.

**`/chat` response** gains these `meta` fields, always present:
- `outcome`, `insufficient_reason`, `llm_calls`, `latency_ms` (existing);
- `model`, `prompt_version` (when generation ran);
- `fixes`, `warnings` (when generation ran);
- `usage` (total tokens across LLM calls).

**`debug=true`** (a request body field, default false) adds `meta.debug`. Proposed contents, approved in **Q3**:
- `path` (node sequence) and per-node `timings_ms`;
- `route`, `route_source`, `standalone_query`;
- retrieval: `candidate_count`; up to 20 candidates as `{chunk_id, label, rerank_score}`; `best_score`; `threshold`; `relevant`; `siblings_added`;
- generation: `model`, `prompt_version`, `reasoning_effort`, `temperature`, `usage` (including reasoning tokens), `finish_reason`, `fixes`, `warnings`, `corrected`;
- `config_version` hashes (retrieval and indexing).

It **never** contains chunk text, prompt text or reasoning text. For streaming, the same block is attached to the `done` event when `debug=true`.

**LangSmith (GI-D5):**
- **Off unless** the LangSmith environment variables are set (tracing flag, API key, project name; check the current variable names in the installed version).
- When on, runs are tagged with `thread_id`, `prompt_version`, and the config versions.
- The README states that enabling it sends prompts and retrieved text to LangSmith's servers.
- The app must start and run with **no** LangSmith variables and **no network access** to LangSmith; a test proves this.

**Async rule:** the API is async. LLM calls use the async methods. The CPU-heavy steps (`hybrid_search`'s embedding and BM25 work, and `rerank`) run in a worker thread (e.g., `asyncio.to_thread`) so they never block the event loop. A test proves that two simultaneous requests on **different** threads both progress.

**Components** (vector store, BM25, reranker, embedding model, Groq clients, prompts) are loaded **once at startup** and handed to nodes without being stored in state. The mechanism (closures at build time vs the installed LangGraph's runtime-context feature) is **Q2**.

---

## 8. Stages

### Stage G0 — Prerequisites

1. Confirm with the user that all module phases are built and their tests pass (routing, retrieval, augmentation, generation, insufficient, attribution). Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.
2. List every open question from earlier plans that affects wiring and is still unanswered (especially generation Q7 and attribution Q3). Ask them now (Q5, Q6).

**Checkpoint:** show the list and the answers.

### Stage G1 — State and serialization

**Steps**
1. Define the main state and the RAG subgraph interface per Section 4. **Ask Q1** (shared keys vs separate schema), showing the trade-off briefly with a code sketch of each.
2. Check how the installed checkpointer serializes values. Store a sample full `retrieval` and `generation` value through a real checkpoint write/read **round trip**, and confirm equality.
3. Extend `reset_turn` to every new per-message field.
4. Tests:
   - round trip of the state through the checkpointer;
   - reset leaves no field from the previous message;
   - no live objects in state (a test that walks the state and fails on non-data types).

**Done when:** tests pass.

**Checkpoint:** show the state definition and the round-trip result.

### Stage G2 — Components and async rules

**Steps**
1. **Ask Q2:** how nodes receive the shared components (closures when building the graph, or LangGraph runtime context, if the installed version offers it). Report what the installed version supports.
2. Implement the component container, loaded once at startup (reusing existing loaders).
3. CPU-heavy node work runs in a worker thread (Section 7).
4. Tests:
   - components load once (a counter);
   - two concurrent requests on different threads both complete, and neither waits for the other's reranking to finish before starting.

**Done when:** tests pass.

**Checkpoint:** show startup timing and the concurrency test result.

### Stage G3 — RAG subgraph

**Steps**
1. Nodes, each a thin wrapper around existing module functions:
   - `hybrid_search`, `rerank`, `gate` (retrieval internals);
   - `expand_context` (retrieval sibling expansion and ordering);
   - `generate` (generation `generate_node`, tagged `answer_generation`);
   - `attribute` (`build_sources`);
   - `gate_failure_reply` and `finalize_model_refusal` (insufficient path).
2. Every node appends its name to `path`, records its timing, and writes `error` on failure (Section 5).
3. Conditional edges exactly as in Section 3, with `error` checked first.
4. Tests, using fake components (no network, no models):
   - **each path:** answered, gate failure, model refusal, fixer-forced refusal, and an error at each node;
   - `path` contents for each;
   - `outcome` and `insufficient_reason` per path;
   - the gate path makes no generation call;
   - the subgraph can be invoked **on its own** with a standalone query.

**Done when:** tests pass.

**Checkpoint:** show the subgraph diagram (Mermaid export) and the path table (scenario → path → outcome).

### Stage G4 — Main graph wiring

**Steps**
1. Replace `rag_stub` with the RAG subgraph node; **delete `rag_stub`** and its temporary hooks.
2. Add `error_finalize` and the error edges, including after `classify_and_rewrite` (`LLM_UNAVAILABLE`) and after `rag`.
3. `finalize` and `error_finalize` follow Section 6.
4. Export the diagrams of the main graph and, expanded, the subgraph as Mermaid into `docs/graph.md` for the README.
5. Tests:
   - all routes end to end with a mocked LLM (greeting by rule, greeting by LLM, out of scope, answered, gate failure, model refusal, router outage, generation outage);
   - history after each (one assistant message per turn, the Q5/Q6 rules applied);
   - a 3-turn conversation where turn 3 is a follow-up that depends on turn 1.

**Done when:** tests pass; `rag_stub` no longer exists.

**Checkpoint:** show `docs/graph.md` and the route test table.

### Stage G5 — Streaming through the subgraph

**Steps**
1. Enable subgraph streaming in the API's stream handler (report the option used).
2. Forward events per Section 7, in the approved orders: `meta → token… → [correction] → sources|related → done`, or `→ error`.
3. **Leak tests** with marker strings injected into the router output, reasoning chunks, rerank internals and chunk texts: none may appear in any event.
4. **Order tests** for each path, including a mid-stream failure (no `sources`, `related` or `done` after `error`).

**Done when:** tests pass.

**Checkpoint:** show raw SSE transcripts for an answered, a corrected, a gate-failure and an error case.

### Stage G6 — Response assembly, `debug`, error mapping

**Steps**
1. **Ask Q3** (debug contents) and **Q4** (error codes and HTTP mapping).
2. Add the `meta` fields (Section 7) and the `debug=true` block (in `/chat` and in the stream's `done` event).
3. Map error codes to HTTP statuses and generic messages.
4. Tests:
   - `debug` absent by default and present with `debug=true`;
   - the `debug` block never contains chunk, prompt or reasoning text (marker strings again);
   - each error code gives the agreed status and body;
   - no stack traces in any response.

**Done when:** tests pass.

**Checkpoint:** show a `debug=true` response for an answered question and a gate failure.

### Stage G7 — Optional LangSmith tracing

**Steps**
1. **Ask Q7:** project name, tags and metadata; confirm it's off by default.
2. Enable tracing purely through environment variables (current names checked in the installed version). Add them, empty, to `.env.example` with a comment explaining what gets sent.
3. Tests:
   - with no LangSmith variables, the app runs and makes **no** LangSmith network calls (mock the network, or check that no client is created);
   - with variables set, runs carry the agreed tags (a live check, skipped without a key).

**Done when:** tests pass.

**Checkpoint:** a screenshot or link of one trace, if the user enables it; otherwise just the off-by-default test.

### Stage G8 — Integration checks and memory measurement

**Steps**
1. Live run through the API (real Groq, real indexes) of one message per path: greeting by rule, out of scope, answered, gate failure, and model refusal if one occurs naturally, or a known Case D query. Show path, outcome, `llm_calls`, latency and tokens.
2. **Memory measurement** (GI-D3 consequence): run a 20-message conversation and report the checkpoint size per message and in total, plus process memory before and after. **Ask Q8** only if growth looks problematic (e.g., a cap on stored checkpoints or trimming old ones).
3. Write the numbers to `eval/integration_report.json` and a short Markdown summary for the README.

**Done when:** the report exists.

**Checkpoint:** show the per-path table and the memory numbers.

---

## 9. Testing rules

- Unit and graph tests use fake components and mocked LLMs; no network and no API keys.
- Live checks are skipped without `GROQ_API_KEY` and respect rate limits.
- **Never patch a module's internals to make wiring tests pass.** Report the mismatch and wait.

## 10. README / DESIGN additions

- The two diagrams (main graph, RAG subgraph) and a walk through each path.
- Why a subgraph (conversation handling vs answering, each testable alone), why separate retrieval nodes (the graph shows the retrieval strategy), why no loops (no retries).
- State design: full results kept for traceability, with the measured memory cost and the note that in-memory history clears on restart.
- Error handling: one error node, generic error messages, codes.
- How to use `debug=true`, and how to enable LangSmith (what it sends).

## 11. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | RAG subgraph state interface: shared keys with the main state, or a separate schema with explicit input/output mapping? | G1 |
| Q2 | How nodes receive shared components: closures at build time, or LangGraph runtime context (if available)? | G2 |
| Q3 | Approve the contents of `meta.debug` | G6 |
| Q4 | Approve the error codes and their HTTP status mapping | G6 |
| Q5 | History after an error (generation plan Q7), if still unanswered | G0 |
| Q6 | Strip `[C#]` markers from history text (attribution plan Q3), if still unanswered | G0 |
| Q7 | LangSmith project name, tags and metadata; confirm off by default | G7 |
| Q8 | Memory control (e.g., cap checkpoints), only if Stage G8 shows a problem | G8 |
| Q9 | Any difference between this plan and the behaviour of the built modules or the installed LangGraph | Whenever found |

## 12. Handoff to item 6 (end-to-end evaluation)

- The full system is reachable through `/chat` and `/chat/stream`, with `debug=true` exposing the path, scores and usage needed for evaluation.
- `docs/graph.md` and `eval/integration_report.json` feed the README.
