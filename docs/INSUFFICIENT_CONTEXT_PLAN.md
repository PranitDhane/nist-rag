# Insufficient-Context Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for the **insufficient-context path**: what happens when a NIST question can't be answered from the documents. Work stage by stage (Section 8). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. Questions in Section 11 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

When a question routed as `nist_question` can't be answered, the user receives:
1. the exact sentence `The NIST documents provided do not contain this information.`,
2. a list of the **closest sections** that were found (which do **not** answer the question), and
3. in `meta`, the **reason** the answer was insufficient.

- **In scope:** detecting both failure sources, the reason codes, the `related_sections` builder, the gate-failure reply (no LLM), post-processing a model refusal, the API and SSE contract additions, and tests.
- **Out of scope:** the normal Sources list for answered questions (item 4, which shares one label/URL formatter with this phase); wiring the nodes into the graph (item 5); end-to-end evaluation (item 6).
- **No retries anywhere in this path.** No query rewriting, no second retrieval, no second generation.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| IC-D1 | Gate failure (`relevant = false`) | Reply with the Case D sentence **straight away**: no query rewrite, no retry, **no generation call** |
| IC-D2 | Model or fixer ends in Case D | **Final.** Reply with the Case D sentence; no retry |
| IC-D3 | User sees | The Case D sentence **plus a list of the closest sections** found, with **no answer content** |
| IC-D4 | How many closest sections | **Top 5, always** (no score floor) |
| IC-D5 | Where they go | A **separate field `related_sections`** in `/chat`, and a separate **SSE event `related`**. **Never** inside `sources`. |
| IC-D6 | Reason visible to the client | **Yes**: `meta.insufficient_reason` |
| IC-D7 | Exact sentence | `The NIST documents provided do not contain this information.` (same for both failure sources) |

## 3. The two failure sources

| Source | Detected by | LLM calls for this message | Reason codes |
|---|---|---|---|
| **Gate failure** | Retrieval returns `relevant = false` | Routing only (1) | `gate_low_score`, or `no_candidates` if retrieval returned nothing |
| **Model refusal** | `GenerationResult.outcome == INSUFFICIENT` | Routing + generation (2) | see below |

**Model-refusal reason codes**, derived from `GenerationResult` (generation phase, Section 5):

| Situation | `insufficient_reason` |
|---|---|
| The model wrote exactly the Case D sentence | `model_case_d` |
| The fixer forced Case D because of `D_WITH_EXTRA_TEXT` | `fix_partial_answer` |
| The fixer forced Case D because of `NO_CITATIONS` (after removing invented citations) | `fix_no_citations` |
| The fixer forced Case D because of `EMPTY` | `fix_empty_output` |
| `finish_reason == "length"`, **only if** generation Q4 chose "Case D" for that | `fix_truncated` |

If several apply, the first matching row in the order above wins. The mapping is a pure function with a test for every row.

**Honest consequence of IC-D4, to put in the README:** with "top 5, always" and no score floor, an unanswerable question (e.g., "What penalties does the EU AI Act set?") will still list 5 sections, and some may be only loosely related. The wording that goes with the list (Q4) must make clear these are the *closest matches searched*, not supporting evidence.

## 4. `related_sections`

**Which chunks they come from:**
- **Gate failure:** the reranked candidates from `RetrievalResult`, in **reranker order** (best first), **before** sibling expansion.
- **Model refusal:** the chunks that were **in the generation prompt**, in prompt order (reranker order). Whether siblings that were added to the prompt are included is **Q2**.

**How the list is built:**
1. Walk the chosen chunks in order.
2. **Deduplicate** per Q1 (proposal: by `citation_label`, so two chunks from the same section and pages appear once).
3. Stop at **5 entries**, or when the chunks run out (fewer than 5 is allowed only if fewer exist).

**Entry fields:**

| Field | Source |
|---|---|
| `doc` | `display_name` |
| `section` | `section_path` joined with ` > ` (plus subcategory or category for tables) |
| `pages` | printed page range, e.g., `pp. 22–23` |
| `citation_label` | chunk metadata (the **shared formatter** with item 4) |
| `citation_url` | `source_url#page={pdf_page_start}` |
| `kind` | `prose` / `table` / `figure-ocr` / `footnote` |
| `score` | rerank score, **only if Q3 says to include it** |

There is **no chunk text, no answer and no summary** in `related_sections` (IC-D3).

**One label/URL formatter** (`attribution/format.py`, or wherever item 4 places it) is used by both `sources` and `related_sections`, so they never format differently. If item 4 isn't built yet, create it here, and item 4 reuses it.

## 5. API contract additions

**`/chat` JSON response when insufficient:**

| Field | Value |
|---|---|
| `route` | `nist_question` |
| `answer` | the exact Case D sentence |
| `sources` | `[]` (nothing supports an answer) |
| `related_sections` | up to 5 entries (Section 4) |
| `meta.outcome` | `insufficient` (answered responses: `answered`) |
| `meta.insufficient_reason` | a code from Section 3 (null when answered) |

The `related_sections` field is **always present** in NIST responses (an empty list when the question was answered), so clients have a stable schema. Greeting and out-of-scope responses are unchanged.

**SSE (`/chat/stream`)**, a new `related` event: `{"note": "<Q4 wording>", "sections": [...]}`. Proposed orders, to be approved (Q5):
- **Gate failure:** `meta → token (the sentence, as one event) → related → done`
- **Model refusal, sentence written by the model:** `meta → token × n → related → done`
- **Model refusal forced by the fixer:** `meta → token × n → correction (the sentence) → related → done`
- Whether an empty `sources` event is also sent in insufficient cases is part of Q5.

`done` carries `outcome` and `insufficient_reason`.

**History:** the assistant message stored is the Case D sentence only. `related_sections` is **not** stored in history, so it never enters the routing window.

## 6. Logging

One structured line per insufficient outcome with:
- `thread_id`, `insufficient_reason`, `best_score`, `threshold`;
- the number of `related_sections`;
- `llm_calls`;
- for model refusals, the `fixes` list.

Never log the full message text at INFO.

## 7. Project layout additions

```
insufficient/
├── __init__.py
├── reasons.py        ← reason codes + mapping from GenerationResult / RetrievalResult
├── related.py        ← build_related_sections(...)
└── reply.py          ← gate-failure reply function/node (no LLM) + model-refusal post-processing
attribution/
└── format.py         ← shared citation_label / citation_url formatter (created here if item 4 isn't built yet)
tests/
├── test_insufficient_reasons.py
├── test_related_sections.py
├── test_insufficient_reply.py
└── test_insufficient_api.py
```

---

## 8. Stages

### Stage I0 — Prerequisites

1. Confirm with the user that retrieval and generation are built and their tests pass. Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.
2. Check whether generation Q4 (`finish_reason == "length"` policy) was answered "Case D". This decides whether `fix_truncated` exists.

**Checkpoint:** report both.

### Stage I1 — Reason codes

**Steps**
1. `reasons.py`: an enum of the codes in Section 3, plus:
   - `reason_from_retrieval(result) -> code | None` (`no_candidates` if there are no candidates; `gate_low_score` if `relevant` is false);
   - `reason_from_generation(gen_result) -> code | None`, mapping per the Section 3 table, first match wins.
2. Tests: one per table row, including the priority order when several fixes apply, and `None` for answered results.

**Done when:** tests pass.

**Checkpoint:** show the mapping test table.

### Stage I2 — Shared formatter and `related_sections`

**Steps**
1. The shared formatter (`attribution/format.py`): `citation_label(meta)` and `citation_url(meta)`. If item 4 hasn't defined a label format yet, **ask Q6** to approve one. Proposal: `NIST AI 100-1, §5.1 Govern, p. 21` (or `pp. 22–23`). For tables: `NIST AI 600-1, §3 GOVERN 1.7, p. 17`. For figures: add `, Figure 5 (OCR)`.
2. **Ask Q1** (deduplication rule), **Q2** (include prompt siblings for model refusals?) and **Q3** (include scores?).
3. `related.py` → `build_related_sections(chunks, mode) -> list[Entry]`, where `mode` is `gate` (reranked candidates before siblings) or `prompt` (the prompt's chunks).
4. Tests:
   - order is preserved (best first);
   - deduplication per Q1;
   - at most 5 entries; fewer only when fewer chunks exist;
   - every field is present;
   - `kind` covers all chunk types;
   - **no chunk text appears anywhere in the output**;
   - the `citation_url` page uses `pdf_page_start`, not the printed page.

**Done when:** tests pass.

**Checkpoint:** show `related_sections` built from real retrieval for "What penalties does the EU AI Act set?" and for one answerable question run through the gate path with a temporarily high threshold.

### Stage I3 — Gate-failure reply and model-refusal post-processing

**Steps**
1. `reply.py` → `gate_failure_reply(retrieval_result) -> InsufficientReply`: the exact sentence, `related_sections` (mode `gate`), the reason, `sources = []`, `llm_calls_added = 0`. **No LLM call, enforced by a test with an LLM mock that fails if called.**
2. `reply.py` → `finalize_model_refusal(gen_result, retrieval_result) -> InsufficientReply`: the sentence (already `final_text` after the fixes), `related_sections` (mode `prompt`), the reason from `reason_from_generation`, `sources = []`.
3. Both return the same `InsufficientReply` shape, so the API and graph treat them the same way.
4. Tests:
   - the gate path never calls an LLM;
   - the model-refusal path never adds an LLM call;
   - both produce the identical sentence;
   - `sources` is always empty when insufficient;
   - the reason codes are correct.

**Done when:** tests pass.

**Checkpoint:** show one `InsufficientReply` from each path.

### Stage I4 — API and SSE contract

**Steps**
1. **Ask Q4:** approve the `note` wording that goes with `related_sections`. Proposal: `"These were the closest sections found. They do not answer the question, but you may want to check them."`
2. **Ask Q5:** approve the SSE orders in Section 5, and whether an empty `sources` event is sent in insufficient cases.
3. Add `related_sections`, `meta.outcome` and `meta.insufficient_reason` to the `/chat` response schema (always present for the NIST route; `related_sections` is empty when answered).
4. Add the `related` event to `/chat/stream`, and `outcome` and `insufficient_reason` to `done`.
5. The history stores only the sentence (test).
6. Because the graph wiring is item 5, test this stage by calling the reply functions through a thin test harness, or a temporary hook in `rag_stub` **if the user approves** (Q7).

**Done when:**
- schema tests pass for answered, gate-failure, model-refusal and fixer-forced cases;
- SSE order tests pass;
- the history test passes.

**Checkpoint:** show example `/chat` JSON and the raw SSE transcript for a gate failure and for a fixer-forced refusal.

### Stage I5 — Checks with real data

**Steps**
1. Run every `answerable: false` case from the retrieval evaluation set through the gate path, and 3 answerable cases through the full path (live generation; respect rate limits).
2. Report, per case: outcome, reason, `best_score` vs threshold, and the 5 related sections (labels only).
3. Flag for the user any unanswerable case where related sections look clearly unrelated, so the user can judge the IC-D4 consequence (Section 3) with real examples.

**Done when:** `eval/insufficient_report.json` exists.

**Checkpoint:** show the report table.

---

## 9. Testing rules

- Unit tests need no network. The "no LLM call" guarantees are enforced with mocks that fail when called.
- Never put chunk text, summaries or partial answers into an insufficient response.
- Never reuse the `sources` field for related sections.

## 10. README / DESIGN additions

- The two failure sources and why there are no retries (free-tier token budget; truly unanswerable questions waste any retry).
- Why closest sections are shown (no dead end; the user can verify), why they're kept separate from `sources`, and the honest note that "top 5 always" can list loosely related sections.
- The reason codes, and how to read `meta.insufficient_reason`.

## 11. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Deduplication rule for related sections: by `citation_label` (proposal), by section, or none | I2 |
| Q2 | For model refusals, include sibling parts that were added to the prompt? | I2 |
| Q3 | Include the rerank `score` in each related entry? | I2 |
| Q4 | Wording of the `note` shown with `related_sections` | I4 |
| Q5 | SSE event order for insufficient cases; send an empty `sources` event too? | I4 |
| Q6 | Citation label format (only if item 4 hasn't fixed it yet) | I2 |
| Q7 | Temporary hook in `rag_stub` for API testing before item 5, or a test harness only? | I4 |
| Q8 | Any difference between this plan and the actual behaviour of the built phases | Whenever found |

## 12. Handoff to item 4 and item 5

- **Item 4** reuses `attribution/format.py` for `sources`, so both lists use one format.
- **Item 5** wires: retrieve → (`relevant`? → `generate_node` : `gate_failure_reply`) → (`outcome == INSUFFICIENT`? → `finalize_model_refusal` : attribution) → finalize.
