# Generation Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for the **generation phase**: calling the Groq model with the augmented prompt, streaming the answer, validating it with `parse_answer()`, fixing what code can fix, and handling failures. Work stage by stage (Section 8). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. Questions in Section 11 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

- **In scope:**
  - the generation model and its settings;
  - the generation call (streaming and non-streaming);
  - hiding reasoning output;
  - the code fixer that applies the violation policy;
  - the correction event for streams;
  - mid-stream failure handling;
  - token usage and output-limit measurement;
  - the behaviour evaluation with the real model;
  - a generation node function ready to be wired into the graph.
- **Out of scope (later items):** what happens when retrieval says `relevant = false` (item 3); building the API's Sources list from `cited_keys` (item 4); wiring retrieve → generate → cite into the graph and replacing `rag_stub` (item 5); end-to-end conversation evaluation (item 6).
- Generation is only called with a **non-empty** context. Called with zero chunks, it raises a clear error (the `relevant = false` path is handled before it, in item 3).

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| GN-D1 | Model | A **smaller, faster Groq model**: `openai/gpt-oss-20b` **if it is listed** on Groq's current models page. If it isn't listed, stop and ask (Q1). |
| GN-D2 | Reasoning effort | **Medium** |
| GN-D3 | Temperature | **0.2** |
| GN-D4 | Streaming vs validation | **Stream live.** When the full answer is validated and the fixes change the text, send a **correction event** with the final text. |
| GN-D5 | Violation policy | **No retry call.** Code fixes what it can; otherwise the answer becomes **Case D**. Details in Section 5. |
| GN-D6 | Uncited sentence | **Keep it, log a warning** |
| GN-D7 | Over the length limit | **Keep it, log a warning** |
| GN-D8 | Rate limit | **No fallback model.** Retry with backoff, then return an error. |
| GN-D9 | Mid-stream failure | **Error event**; the client **discards** the partial answer |

## 3. Facts and constraints from Groq (verify at Stage N0)

These come from Groq's docs and third-party summaries, and **must be verified** against Groq's current models page and the **account's own limits page**:

1. **Model list:** `llama-3.3-70b-versatile` was reportedly decommissioned in August 2026, and Groq recommends the `openai/gpt-oss-*` family. Confirm the exact ID of the 20B model.
2. **gpt-oss models are reasoning models.** Three consequences:
   - **Reasoning tokens count toward the token-per-minute limit** and toward the maximum output setting.
   - **The maximum output (`max_completion_tokens` or equivalent) includes reasoning tokens.** If it is set too low, the visible answer gets cut off (finish reason `length`).
   - **Reasoning text must never reach the user**: not streamed, not returned, not stored in history.
3. **Free-tier limits** are reportedly about 30 requests per minute and only about **6,000–8,000 tokens per minute** per model, plus daily caps. With about 4,000 tokens per NIST answer, **tokens per minute is the binding limit** (about 2 answers per minute). Read the real values from the account limits page and record them in config.
4. **How reasoning effort is set and how reasoning output is hidden** depends on the Groq API and the installed `langchain-groq` version (parameter names have changed between versions). **Check and report; don't guess.** Prefer hiding reasoning **at the API level**, and **also** filter it in the stream (defence in depth).

## 4. Generation flow

```
generate(standalone_query, retrieval_result)
  → build_messages()                     (augmentation phase; no history)
  → budget pre-check                      (prompt tokens + max output ≤ per-minute token limit, else fail fast)
  → Groq call (stream or not)             (temperature 0.2, reasoning effort medium, reasoning hidden)
       ├─ retries with backoff ONLY before the first content token
       ├─ content tokens → streamed to the client as `token` events (stream mode)
       └─ failure after the first token → `error` event; partial text discarded (GN-D9)
  → full text → parse_answer()            (augmentation phase)
  → apply_fixes()                         (Section 5)
  → final text differs from streamed text? → `correction` event (stream mode)
  → GenerationResult
```

## 5. Violation policy (applied by `apply_fixes`, a pure function)

Applied **in this order**. After all fixes, the text is parsed again to recompute citations and warnings.

| Violation (from `parse_answer`) | Action | Outcome |
|---|---|---|
| `D_WITH_EXTRA_TEXT` | Replace the whole text with the exact Case D sentence | `INSUFFICIENT` |
| `EMPTY` | Case D sentence | `INSUFFICIENT` |
| `MALFORMED_CITATION` | Normalize: `[C1, C3]` / `[C1,C3]` → `[C1][C3]`; `(C1)` → `[C1]`. Leave unrecognised forms untouched and add a warning. | fixed |
| `UNKNOWN_CITATION` | Remove the invented markers | fixed |
| `HAS_SOURCES_LIST` | Remove the trailing "Sources:" / "References:" section | fixed |
| `HAS_URL` | Remove URLs (and empty brackets or parentheses left behind) | fixed |
| `HAS_HEADING_OR_CODE` | Remove heading markers (`#`) and code-fence lines; keep the text inside | fixed |
| `NO_CITATIONS` (re-checked **after** the fixes above; e.g., all citations were invented and removed) | Case D sentence | `INSUFFICIENT` |
| `UNCITED_SENTENCE` | **Keep**; add warning (GN-D6) | warning |
| `OVER_LENGTH` | **Keep**; add warning (GN-D7) | warning |

Rules:
- **The fixer never adds content and never rewords sentences.** It only removes or normalizes markup, or replaces everything with the Case D sentence.
- A model answer that is exactly Case D needs no fix; its outcome is `INSUFFICIENT`.
- Every fix and warning is recorded by code name in `GenerationResult.fixes` and `.warnings`, and logged.

## 6. Output: `GenerationResult`

| Field | Type | Notes |
|---|---|---|
| `outcome` | `ANSWERED` \| `INSUFFICIENT` \| `ERROR` | `INSUFFICIENT` = Case D (from the model or a fix) |
| `final_text` | str \| null | Validated, fixed text (null on ERROR) |
| `streamed_text` | str \| null | What was streamed (stream mode), for comparison and logging |
| `corrected` | bool | `final_text != streamed_text` (stream mode) |
| `cited_keys` | list[str] | From the final parse; used by item 4 for Sources |
| `key_map` | dict | `C# → chunk_id` (from augmentation) |
| `is_ambiguous_format` | bool | Case B |
| `fixes`, `warnings` | list[str] | Violation codes handled |
| `usage` | dict | prompt, completion, **reasoning** and total tokens (as reported by Groq) |
| `finish_reason` | str | e.g., `stop`, `length` |
| `timings_ms` | dict | `ttft` (time to first content token), `total` |
| `model`, `prompt_version`, `reasoning_effort`, `temperature` | | Traceability |
| `error` | `{code, message}` \| null | On ERROR |

## 7. Streaming protocol additions

The API phase defined the events `meta`, `token`, `sources`, `done`, `error`. This phase adds **`correction`**, whose name and payload need approval (Q5). Proposal:

- `correction`: `{"text": "<final_text>", "reasons": ["UNKNOWN_CITATION", ...]}`. The client **replaces** everything it showed for this answer with `text`.
- It is sent **only** when `final_text != streamed_text`, **after** the last `token` and **before** `sources`.
- If the final outcome is `INSUFFICIENT` because of a fix, the correction text is the exact Case D sentence.

Event order for a successful stream:
`meta → token × n → [correction] → sources → done`.

On a mid-stream failure: `meta → token × k → error`. No `correction`, `sources` or `done`. The client discards the partial text (GN-D9).

**JSON `/chat` mode:** no streaming. The response contains only `final_text`, and the violations are listed in `meta` (`fixes`, `warnings`).

---

## 8. Stages

### Stage N0 — Prerequisites and Groq verification

**Steps**
1. Confirm with the user that the augmentation phase is complete and its tests pass. Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.
2. **Ask Q1:** check Groq's current models page. Is the 20B gpt-oss model listed, and what is its exact ID? Read the **account's** limits page for that model: requests per minute and per day, tokens per minute and per day. Present them to the user. If the model isn't listed, present the available smaller models and ask.
3. **Q2:** find how the installed Groq / `langchain-groq` version sets the reasoning effort and hides or excludes reasoning output, in both streaming and non-streaming modes. Report the exact parameters and show a raw response, with any reasoning fields highlighted.

**Done when:** model ID, limits and reasoning parameters are confirmed and recorded.

**Checkpoint:** show the limits table and the raw-response inspection.

### Stage N1 — Configuration and client

**Steps**
1. `generation/config.py`:
   - `GEN_MODEL` (Q1), `TEMPERATURE = 0.2`, `REASONING_EFFORT = "medium"`;
   - the reasoning-hiding parameters (Q2);
   - `MAX_COMPLETION_TOKENS = None` until Stage N4 (refuse to run live without it, except in measurement mode);
   - per-model limits from Q1 (`TPM_LIMIT`, etc.);
   - timeouts (Q6);
   - retry settings (Q8).
2. A **separate client instance** for generation (different model and settings from the router), built through the existing `llm/groq_client.py` helpers, so retries, backoff and call counting are reused.
3. **Usage capture:** read prompt, completion and **reasoning** token counts from the response or stream metadata. Check where the installed version reports them (some only send usage at the end of a stream, or need an option to include it). Report.

**Done when:** a live non-streaming call returns content plus a full usage breakdown; the reasoning text is absent from the content.

**Checkpoint:** show the config and one usage breakdown.

### Stage N2 — Core `generate()` (non-streaming) and the fixer

**Steps**
1. `generation/fixer.py` → `apply_fixes(raw_text, key_map) -> FixResult(final_text, outcome, fixes, warnings, parsed)`, implementing Section 5 exactly, in order, re-parsing at the end.
2. `generation/generate.py` → `generate(standalone_query, retrieval_result) -> GenerationResult`, non-streaming:
   - empty context → raise;
   - budget pre-check (Section 4);
   - call;
   - `finish_reason == "length"` → the policy from **Q4**;
   - parse, fix, return.
3. Unit tests with a **mocked LLM** (no network). One test for every row of Section 5, plus:
   - a correct answer passes unchanged;
   - exact Case D → `INSUFFICIENT`, no fixes;
   - all citations invented → removed → `NO_CITATIONS` → Case D;
   - `[C1, C3]` normalized;
   - URL removed without leaving `()`;
   - the fixer never adds words (assert that the output's word set is a subset of the input's word set plus the Case D sentence);
   - empty context raises;
   - the budget pre-check fails fast when the prompt plus maximum output exceeds the per-minute limit.

**Done when:** all tests pass.

**Checkpoint:** show the fixer test table (input → fixes → outcome).

### Stage N3 — Streaming `generate_stream()`

**Steps**
1. `generate_stream(...)` yields events: `token` for **content tokens only** (reasoning chunks are filtered out, even if the API-level hiding fails), then a final internal `result` carrying the `GenerationResult`.
2. **Retries only before the first content token.** A 429 or connection error at the start is retried with backoff; after the first token has been sent, any failure → `error` event and stop (GN-D9).
3. After the stream ends:
   - assemble `streamed_text`;
   - `apply_fixes`;
   - if `final_text != streamed_text`, produce a `correction` (Section 7).
4. **Timeouts** (Q6): time to first token and total time. Exceeding either → `error`.
5. Tests with a **fake streaming LLM** (deterministic chunks):
   - normal stream with no correction;
   - stream with an invented citation → correction removes it;
   - stream ending in `D_WITH_EXTRA_TEXT` → correction to the Case D sentence;
   - **reasoning chunks mixed into the fake stream never appear in `token` events or in `streamed_text`**;
   - 429 before the first token → retried, then succeeds;
   - failure after 3 tokens → `error`, no correction, no result text;
   - TTFT timeout → `error`.

**Done when:** all tests pass.

**Checkpoint:** show the event sequences for: a normal stream, a corrected stream, and a mid-stream failure.

### Stage N4 — Token measurement and output limit

**Steps**
1. **Measurement mode:** run the augmentation phase's behaviour cases (and about 10 retrieval-evaluation queries) live, with a generous temporary output limit, pausing between calls to respect the per-minute limit. Record prompt, reasoning, visible-answer and total tokens, `finish_reason`, TTFT and total latency.
2. Report minimum, median, p95 and maximum for each, plus the **largest total tokens of a single request** compared with `TPM_LIMIT`.
3. **Ask Q3:** set `MAX_COMPLETION_TOKENS` from the measured reasoning plus answer tokens (the proposal shows p95 and max, with a suggested value and margin).
4. **Ask Q4:** what to do when `finish_reason == "length"` despite the limit. Options: Case D (`INSUFFICIENT`), keep the truncated answer with a warning, or `ERROR`.

**Done when:** values are chosen and recorded; `eval/generation_usage_report.json` exists.

**Checkpoint:** show the usage table and the chosen values.

### Stage N5 — Behaviour evaluation (real model)

**Steps**
1. Run all approved behaviour cases (augmentation Stage A5) through `generate()` with the final settings, pausing for rate limits.
2. For each case, report: expected case vs actual case, fixes, warnings, cited keys, word count, `corrected` (in a streaming rerun of a subset), tokens and latency.
3. **Summary metrics:**
   - case accuracy (A/B/D);
   - **uncited-sentence rate** (% of answers with ≥1 `UNCITED_SENTENCE`, and % of sentences uncited);
   - over-length rate;
   - invented-citation rate;
   - correction rate;
   - `INSUFFICIENT`-by-fix rate;
   - average tokens and latency.
4. **Ask Q9:** the user reviews the results and decides whether the rates are acceptable, or whether a prompt `v2` is wanted. Any prompt change needs the user's approval and a new version file, with `v1` kept.

**Done when:** `eval/generation_behaviour_report.json` and a Markdown summary `docs/process/GENERATION_RESULTS.md` exist.

**Checkpoint:** show the summary metrics and every case where the actual case differed from the expected one.

### Stage N6 — Generation node, ready to wire (item 5 does the wiring)

**Steps**
1. `generation/node.py` → a LangGraph node function `generate_node(state)` that:
   - reads `standalone_query` and the retrieval result from state;
   - calls `generate()` (or the streaming variant, depending on how item 5 streams);
   - writes `answer` (= `final_text`), `generation` (the `GenerationResult` without reasoning), `cited_keys` and `key_map`;
   - increments `llm_calls`.
   - **Tags its LLM calls** (e.g., tag `answer_generation`) so the API's streaming filter streams **only** this node's content tokens.
2. **History rule:** only `final_text` is stored as the assistant message, never `streamed_text`, never reasoning, never a partial answer. **Ask Q7** what happens to the user's message in history when generation ends in `ERROR`: keep it without a reply, or roll the turn back so the next follow-up isn't confused by an unanswered question.
3. Tests:
   - the node writes the expected fields;
   - the history contains `final_text` after a correction (not the streamed version);
   - nothing from reasoning appears anywhere in the state;
   - the error behaviour follows Q7.

**Done when:** tests pass; the node is documented for item 5.

**Checkpoint:** show the node's state changes for one answered case, one corrected case and one error case.

---

## 9. Testing rules

- Unit tests use mocked or fake-streaming LLMs; no network, no API key.
- Live tests (measurement, behaviour evaluation) are skipped without `GROQ_API_KEY`, pause between calls, and record usage.
- Never weaken the policy in Section 5 to improve metrics; report honestly. The user decides about prompt changes.

## 10. README / DESIGN additions

- Model choice and why: smaller and faster, with the per-minute token limit as the binding constraint, shown with the measured numbers.
- Reasoning model handling: medium effort; reasoning hidden at the API and filtered in the stream; the output limit includes reasoning tokens.
- Streaming with a correction event: why (fast first token, guaranteed-valid final answer) and how a client should handle it.
- The violation policy table, and the **honest note** that uncited sentences and over-length answers are kept with warnings, with the measured rates from Stage N5.
- No retries and no fallback model: why (free-tier token budget, consistent model behaviour), and the error a user sees.

## 11. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Is the 20B gpt-oss model listed? Exact ID, and the account's limits for it. If not listed, which model instead? | N0 |
| Q2 | Exact parameters for reasoning effort and for hiding/excluding reasoning (streaming and non-streaming) in the installed versions | N0 |
| Q3 | `MAX_COMPLETION_TOKENS` value, from the measured p95/max of reasoning plus answer tokens | N4 |
| Q4 | Policy when `finish_reason == "length"`: Case D, keep with warning, or error | N4 |
| Q5 | Approval of the `correction` event name, payload and position | N3 |
| Q6 | Timeouts: time to first token, and total | N1 |
| Q7 | History on `ERROR`: keep the unanswered user message, or roll the turn back | N6 |
| Q8 | Retry attempts and backoff for generation: same as the router's settings (API plan Q8) or separate? | N1 |
| Q9 | Are the measured rates (uncited, over-length, case accuracy) acceptable, or is a prompt v2 wanted? | N5 |
| Q10 | Optional: a local token-budget limiter to avoid hitting 429s at all. Only if Stage N4/N5 show frequent 429s. | After N5 |
| Q11 | Any difference between this plan and the real API or library behaviour | Whenever found |

## 12. Handoff

- **Item 3 (insufficient context):** `outcome == INSUFFICIENT` (Case D) and retrieval `relevant == false` should lead to the same user-facing reply. Item 3 decides the details.
- **Item 4 (source attribution):** builds Sources from `cited_keys` → `key_map` → chunk metadata (`citation_label`, `citation_url`), and sends the `sources` event after any `correction`.
- **Item 5 (graph integration):** wires retrieve → [relevance decision] → `generate_node` → cite, and streams only events tagged `answer_generation`.
