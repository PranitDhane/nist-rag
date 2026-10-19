# End-to-End Evaluation Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for the **end-to-end evaluation**: checking that the complete system behaves correctly **as a user experiences it**, through the real HTTP API, including multi-turn conversations. Work stage by stage (Section 8). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. Questions in Section 11 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

- **In scope:**
  - a small evaluation set (single-turn cases plus conversations);
  - a harness that calls a running server over HTTP;
  - deterministic checks (routes, outcomes, fixed texts, gold key facts, citations, links, the insufficient path);
  - a **local** groundedness score using the reranker, calibrated with controls;
  - a small streaming check;
  - a report (JSON + Markdown) for the README.
- **Out of scope:** changing the system. The evaluation **observes and reports**. If it reveals a bug, it is reported to the user, and fixing it is a separate, approved task.
- **This evaluation makes no LLM calls of its own.** The only LLM usage is the system's own routing and generation calls for each turn.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| E-D1 | Size | **Small: about 15 cases** |
| E-D2 | Split | **About 10 single-turn cases + 2 conversations of 3 turns each** (16 requests in total) |
| E-D3 | Groundedness | **Local only:** citation rules plus a **reranker sentence-support score** (no LLM judge) |
| E-D4 | Correctness | **Gold key facts** per question: deterministic must-contain checks (with variants), plus must-not-contain traps and expected cited sections |
| E-D5 | How it runs | **Real HTTP** against a **running server** (the way a reviewer would run it) |
| E-D6 | Result | **Report only.** No pass/fail; the script exits 0 unless the harness itself fails (e.g., the server is unreachable) |

## 3. Proposed case composition (approved at Stage E1, Q1)

**Single-turn (10):**

| # | Type | Example question | Expected |
|---|---|---|---|
| S1 | AI RMF answerable | "What are the four core functions of the NIST AI RMF?" (the assignment's own example) | answered; facts: govern, map, measure, manage; cites AI RMF §5 |
| S2 | AI RMF answerable | "What characteristics make an AI system trustworthy according to the AI RMF?" | answered; cites §3 |
| S3 | AI RMF answerable (appendix) | "How do AI risks differ from traditional software risks?" | answered; cites Appendix B |
| S4 | GenAI answerable (risk) | "What is confabulation in the Generative AI Profile?" | answered; cites §2.2 |
| S5 | GenAI answerable (action table) | "What does the GenAI Profile suggest for decommissioning GAI systems?" | answered; cites GOVERN 1.7 |
| S6 | Exact ID | "What is action GV-1.7-001?" | answered; cites GOVERN 1.7 |
| S7 | Unanswerable, near scope | "What penalties does the EU AI Act set?" | insufficient (Case D sentence, 5 related sections, `sources` empty) |
| S8 | Unanswerable, partly answerable two-part question | "What are the four functions, and when will NIST publish AI RMF 2.0?" | insufficient (partial → refuse entirely) |
| S9 | Out of scope | "What is the capital of France?" | out-of-scope apology, exact text |
| S10 | Greeting | "Hello!" | greeting template, 0 LLM calls |

**Conversation C1 (the assignment's example, 3 turns, same `thread_id`):**
1. "What are the four functions of the AI RMF?" → answered (govern, map, measure, manage)
2. "Which one is cross-cutting?" → answered; fact: govern; standalone query names the AI RMF functions
3. "What categories does it include?" → answered; GOVERN categories (e.g., GOVERN 1–6); standalone query refers to GOVERN

**Conversation C2 (3 turns, same `thread_id`):**
1. "What is confabulation?" → answered; GenAI §2.2
2. "Hi again! Which trustworthy AI characteristics is it related to?" → **must route as `nist_question`** (the mixed greeting trap); answered from the risk's "Trustworthy AI Characteristic" line
3. "What's the weather in Pune today?" → out-of-scope apology (topic switch inside a conversation)

The gold facts, variants, traps and expected sections are written per turn in the case file.

## 4. Checks per turn

Each check gives **pass / fail / not applicable**, with a reason. They are **reported, never gated** (E-D6).

**A. Routing and outcome (all turns)**
- A1: `route` equals the expected route.
- A2: `meta.outcome` equals the expected outcome.
- A3: **fixed texts are exact**: greeting template, out-of-scope apology, Case D sentence.
- A4: follow-ups only: `standalone_query` (from `debug`) contains the expected key terms.
- A5: `llm_calls` as expected: 0 for a rule greeting, 1 for out of scope and gate failure, 2 for an answered question.

**B. Content correctness (answered turns, E-D4)**
- B1: every **must-contain** fact is present; each fact can have variants (case-insensitive, hyphen/space-insensitive).
- B2: no **must-not-contain** trap is present.
- B3: the answer is in English; word count is reported (≤ ~150 expected, but reported, not failed).

**C. Citations (answered turns)**
- C1: every `[C#]` in the answer appears in `sources` and vice versa (the attribution consistency rule, re-checked from outside).
- C2: **expected cited sections**: at least one source label matches a gold section (by doc + section/subcategory). Sources from an unexpected document are listed.
- C3: **link check:** for each cited chunk, the page at its `url` contains the chunk's text (reusing the attribution plan's link-check logic, applied to cited chunks only).
- C4: generation `fixes` and `warnings` (from `debug`) are reported, especially `UNCITED_SENTENCE`.

**D. Groundedness (answered turns, E-D3; Section 5)**
- D1: each fact-stating sentence's **support score** against its cited chunk(s) (the maximum over its citations).
- D2: % of sentences above the calibrated threshold (Q2).
- D3: uncited sentences are counted separately (they can't be scored).

**E. Insufficient path (insufficient turns)**
- E1: `answer` is exactly the Case D sentence.
- E2: `sources` is empty.
- E3: `related_sections` has 5 entries (or fewer only if fewer existed).
- E4: `insufficient_reason` is present and recorded.

**F. Operational (all turns)**
- F1: HTTP status 200 (errors are recorded with their code).
- F2: latency, tokens (from `debug.usage`), and `llm_calls` are recorded.

## 5. Local groundedness scoring (E-D3)

**Scorer:** the local `BAAI/bge-reranker-base` cross-encoder, loaded **in the evaluation process** (not through the API). It scores the pair *(answer sentence with `[C#]` markers removed, cited chunk text)*.

**Getting the chunk text:** the API never returns chunk text. The harness maps each `[C#]` → `chunk_id` using **`key_map` from `debug`**, then reads the text from the local `chunks.jsonl`. If the `debug` block doesn't include `key_map`, **ask Q3** (it means a small addition to the graph phase's debug contents; `key_map` holds IDs only, no text).

**Calibration (before choosing any threshold):**
- **Positive pairs:** each answer sentence vs its own cited chunk(s).
- **Negative controls:** each answer sentence vs 3 random chunks from **different sections** (a fixed random seed, so it's repeatable).
- **Sanity controls:** a sentence copied verbatim from a chunk vs that chunk (expected to score very high).
- Report all three score distributions (min, median, p10, p90, max) and the overlap between positive pairs and negative controls. **Ask Q2:** the user chooses the "supported" threshold from this table, the same way the retrieval threshold was chosen.

**Honest limitation (README):** the same model is used for retrieval reranking, so it isn't a fully independent judge. A sentence can score high on topic overlap even with a subtle factual error. Gold key facts (B) and the citation checks (C) complement it.

## 6. Harness behaviour

- **Base URL** of the running server, from a CLI argument or env var (default `http://127.0.0.1:8000`). Before starting, check the server is reachable; if it isn't, exit non-zero with a clear message (the only non-zero exit).
- Every request is sent with `debug=true`.
- **Conversations:** the first turn has no `thread_id`; the harness stores the returned ID and sends it with later turns.
- **Pacing:** wait between requests that reach generation, to respect the Groq tokens-per-minute limit (interval in **Q5**). If the server returns the rate-limit error, wait and retry that turn **once**, recording it.
- **Raw records:** every request and response (JSON) is saved to `eval/e2e_runs/<timestamp>/raw/`, so the checks can be re-run on saved responses without spending tokens again (`--from-run <dir>`).
- **Streaming subset:** a few NIST turns are also sent to `/chat/stream` (how many is **Q4**). Checks:
  - event order is valid;
  - no `token` content after `error`;
  - after applying any `correction`, the final text equals the `/chat` answer for the **same** run; the comparison is on structure (outcome, cited keys present), because temperature 0.2 can change the wording between two calls.
- **Repeats:** each case runs once by default; repeating for variance is **Q6**.

## 7. Report

- `eval/e2e_runs/<timestamp>/report.json`: every check per turn, plus aggregates.
- `docs/E2E_RESULTS.md`: the Markdown summary for the README.

**Aggregates:**
- route accuracy;
- outcome accuracy;
- exact-text accuracy;
- follow-up resolution (A4);
- key-fact coverage (% of facts found) and trap hits;
- citation consistency;
- expected-section hit rate;
- link-check pass rate;
- support rate (% of sentences above the threshold);
- uncited-sentence rate;
- over-length count;
- insufficient-path correctness;
- latency p50/p95;
- tokens and LLM calls per turn type.

**Failures:** every failed check is listed with the case, the turn, what was expected, and what happened, without being softened or excluded.

---

## 8. Stages

### Stage V0 — Prerequisites

1. Confirm with the user that graph integration is complete and the server runs. Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.
2. Check that `debug=true` provides: `path`, `route`, `standalone_query`, `usage`, `fixes`, `warnings`, `llm_calls`, and **`key_map`**. Report anything missing (Q3).

**Checkpoint:** one `debug=true` response, abbreviated.

### Stage V1 — Case file

**Steps**
1. Write `eval/e2e_cases.yaml` following Section 3. Each turn has: `id`, `message`, `expected_route`, `expected_outcome`, optional `expected_standalone_terms`, `must_contain` (facts with variants), `must_not_contain`, `expected_sections` (doc + section/subcategory), and a short `notes` line.
2. Gold facts **must be confirmed against the documents**: for each fact, record the chunk or page where it appears (from `chunks.jsonl`), so the gold data itself is verifiable.
3. **Ask Q1:** the user reviews and approves the cases, facts, traps and sections.

**Done when:** the case file is approved.

**Checkpoint:** show the case table with gold facts and their document locations.

### Stage V2 — Harness

**Steps**
1. `eval/e2e.py` with: the base-URL check, single-turn and conversation runners, `debug=true`, pacing (Q5), one retry on rate-limit errors, raw saving, and `--from-run`.
2. Tests **against a fake local server** (a tiny stub app), so the harness is tested without Groq:
   - conversation `thread_id` handling;
   - pacing waits;
   - rate-limit retry once;
   - raw files saved;
   - `--from-run` reads saved responses;
   - unreachable server → clear non-zero exit.

**Done when:** harness tests pass.

**Checkpoint:** show a dry run against the fake server.

### Stage V3 — Deterministic checks

**Steps**
1. Implement checks A, B, C1, C2, C4, E and F (Section 4) as pure functions over saved responses.
2. C3 (link check) reuses the attribution plan's checker, applied only to cited chunks.
3. Unit tests with hand-made responses: one passing and one failing example for every check, including the fact-variant matching (`cross-cutting` vs `cross cutting`) and a trap hit.

**Done when:** tests pass.

**Checkpoint:** show the check results on the fake-server dry run.

### Stage V4 — Groundedness scorer and calibration

**Steps**
1. Implement sentence extraction (reuse the augmentation parser's sentence splitting), marker removal, `key_map` lookup, chunk text loading, and reranker scoring.
2. Run the **first live evaluation** (Stage V5 can then reuse its saved run). Compute the positive, negative-control and sanity-control distributions (Section 5).
3. **Ask Q2:** the user chooses the support threshold from the distribution table.

**Done when:** the threshold is chosen and recorded in `eval/config`.

**Checkpoint:** show the three distributions and the overlap.

### Stage V5 — Full run and streaming subset

**Steps**
1. **Ask Q4** (streaming subset size) and **Q6** (repeats) before the live run; state the estimated time and tokens for each option.
2. Start the server (the user starts it, or the harness documents the command). Run all 16 requests (plus the streaming subset), paced.
3. Apply all checks, including D with the chosen threshold.

**Done when:** a complete run is saved with all checks applied.

**Checkpoint:** show the per-turn result table.

### Stage V6 — Report

**Steps**
1. Generate `report.json` and `docs/E2E_RESULTS.md` (Section 7).
2. **Ask Q7:** the user approves the Markdown summary format for the README.

**Done when:** both files exist.

**Checkpoint:** show the Markdown summary and every failure.

---

## 9. Testing rules

- The harness and checks are tested against a fake server and hand-made responses; no Groq calls in tests.
- **Never edit gold data after seeing results to improve scores**, unless the gold data is shown to be wrong against the document, and then only with the user's approval and a note in the report.
- Never filter failed cases out of the report.

## 10. README / DESIGN additions

- How to run: start the server, then `python -m eval.e2e --base-url …`, the time it takes, and the token use.
- The results table, from `docs/E2E_RESULTS.md`.
- The method: gold key facts plus citation checks plus a local support score, with the calibration table and the honest limitation of the support score.
- Explicit evidence for the assignment's follow-up requirement (conversation C1 turns 2–3) and for insufficient and out-of-scope handling.

## 11. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Approval of the cases, gold facts, traps and expected sections | V1 |
| Q2 | Support-score threshold, chosen from the calibration distributions | V4 |
| Q3 | Add `key_map` (IDs only) to the `debug` block, if missing | V0 |
| Q4 | Streaming subset size: none, 2 turns, or all NIST turns (time and tokens shown) | V5 |
| Q5 | Pacing interval between generation requests (from the measured tokens-per-minute use) | V2 |
| Q6 | Repeat each case for variance (e.g., 3×), or run once | V5 |
| Q7 | Approval of the Markdown results format for the README | V6 |
| Q8 | Any difference between this plan and the actual system behaviour | Whenever found |
