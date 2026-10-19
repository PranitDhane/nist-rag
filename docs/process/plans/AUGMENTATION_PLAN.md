# Augmentation Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for the **augmentation phase**: turning a `RetrievalResult` into the exact prompt the LLM receives, plus the **output contract** that the generation phase will enforce. Work stage by stage (Section 7). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. Questions in Section 10 must be asked **at the stage named**, never answered by assumption.
>
> The prompt files are part of this specification: `prompts/answer_system.v1.md` and `prompts/answer_user.v1.md`. **Do not change their wording without the user's approval.**

---

## 1. Goal and boundary

- **In scope:**
  - the system prompt (5 sections: Purpose, Constraints, Interpretation, Decision, Output format);
  - the user-message template;
  - the context builder (`RetrievalResult` → XML `<source>` blocks with citation keys);
  - the citation-key map;
  - prompt versioning;
  - token measurement;
  - the **output contract** with a pure, unit-tested parser/validator (no LLM calls);
  - the behaviour test cases for the next phase.
- **Out of scope (generation phase):** choosing the Groq generation model, calling the LLM, streaming, what to do when validation fails (retry or fallback), building the final Sources list for the API, the `relevant=false` path, and graph integration.
- **No LLM calls** in this phase. Everything is deterministic and unit-testable.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| G-D1 | Prompt structure | System prompt with exactly five sections: **PURPOSE, CONSTRAINTS, INTERPRETATION, DECISION, OUTPUT FORMAT** |
| G-D2 | Interpretation principle | **"Do not assume anything"**, about both the sources and the question |
| G-D3 | LLM output | **Plain text with inline citations** (streams cleanly); no JSON |
| G-D4 | Source presentation | **XML-style blocks**: `<source id="C1" …>text</source>` |
| G-D5 | Ambiguous question | **Answer every reasonable reading**, each cited |
| G-D6 | Ambiguous, only some readings supported | **Answer the supported readings**; state that the others are not covered |
| G-D7 | Partly answerable question | **Refuse entirely** with the insufficient-context sentence |
| G-D8 | Insufficient-context sentence (exact) | `The NIST documents provided do not contain this information.` (separate from the out-of-scope reply) |
| G-D9 | Conversation history in the generation prompt | **None**: only the standalone query and the sources |
| G-D10 | Citation frequency | **Every sentence that states a fact** carries at least one `[C#]` |
| G-D11 | Length and style | **Concise**: short paragraph, bullets for lists, about **150 words max** |
| G-D12 | Quoting | **Paraphrase**, but **quote exact NIST terms and definitions** |
| G-D13 | Language | **Always English** |

## 3. Design rules that follow from these decisions

1. **The citation keys `C1…Cn` are assigned by code** in prompt order (the best-first order from retrieval, with siblings placed as retrieval decided). A key → `chunk_id` map is kept in the graph state. The LLM only ever sees keys, never chunk IDs or URLs.
2. **Sources come before the question** in the user message. The question sits right before the instruction to answer, so it is the last thing the model reads.
3. **The source tag attributes give the model only what it needs for reasoning:** `id`, `doc`, `section`, `type`, `part`. **Page numbers are not shown** (Q2): the model must not write pages, and citations with pages are built in code.
4. **The phrase for an uncovered reading** (`not covered by the provided documents`) is deliberately different from the insufficient-context sentence, so the code never mistakes a Case B answer for Case D.
5. **Escaping:** chunk text is inserted as-is, except that any literal `<source`, `</source`, `<sources` or `</sources` inside chunk text is neutralized (e.g., `<` → `‹`). Attribute values escape `"` and `&`. Nothing else in the text changes, so quoted NIST terms stay exact.
6. **Prompts live in versioned files** (`prompts/answer_system.v1.md`, `prompts/answer_user.v1.md`), loaded once at startup. `PROMPT_VERSION = "v1"` is logged with every answer and returned in the API's `meta`.

## 4. Source block format

One block per retrieved chunk, in final order:

```
<source id="C1" doc="NIST AI 100-1 (AI RMF 1.0)" section="Part 2 > 5 AI RMF Core > 5.1 Govern" type="prose" part="1 of 1">
{display_text}
</source>
```

| Attribute | Source in chunk metadata | Values |
|---|---|---|
| `id` | assigned by code | `C1`, `C2`, … |
| `doc` | `display_name` | e.g., `NIST AI 600-1 (GenAI Profile)` |
| `section` | `section_path` joined with ` > `; for table chunks, add ` > {subcategory or category}` | |
| `type` | from `chunk_type`: `prose` → `prose`; `genai_action_table` and `rmf_core_table` → `table`; `figure` → `figure-ocr`; `footnote` → `footnote` | |
| `part` | `"{part} of {parts_total}"` | `1 of 1` for unsplit chunks |

Text inside a block is `display_text`, **not** `embed_text` (the breadcrumb is already in the attributes).

## 5. Output contract (enforced by the parser; used by the generation phase)

The parser `parse_answer(text, key_map) -> ParsedAnswer` classifies and checks the raw model output:

| Field | Meaning |
|---|---|
| `case` | `A_or_B` (answer), `D` (insufficient), `invalid` |
| `is_ambiguous_format` | First line is exactly `This question can be read in more than one way:` |
| `cited_keys` | Ordered, de-duplicated list of keys used, e.g., `["C2", "C1"]` |
| `unknown_keys` | Keys used that are **not** in the key map (invented citations) |
| `uncited_sentences` | Fact-stating sentences or bullets with no `[C#]` marker (lead-in lines ending with `:` and "not covered" bullets are exempt) |
| `word_count` | Words, excluding citation markers |
| `violations` | List of rule codes (below) |
| `clean_text` | The text, with unknown keys removed (the generation phase decides whether to use it) |

**Case D detection:** after trimming whitespace, the output equals the exact sentence (a missing final period is tolerated). If the sentence appears **together with** other text, `case = D` and violation `D_WITH_EXTRA_TEXT` is raised, because partial answers must be refused entirely (G-D7).

**Violation codes:**
- `UNKNOWN_CITATION`
- `UNCITED_SENTENCE`
- `NO_CITATIONS` (case A/B with zero citations)
- `OVER_LENGTH` (> 150 words; threshold for ambiguous answers per Q1)
- `D_WITH_EXTRA_TEXT`
- `HAS_SOURCES_LIST` (the model wrote "Sources:" / "References:")
- `HAS_URL`
- `HAS_HEADING_OR_CODE` (markdown headings or code fences)
- `EMPTY`

**Citation marker regex:** `\[C(\d+)\]`. Adjacent markers `[C1][C3]` are allowed. Markers in other forms (`[C1, C3]`, `(C1)`, `[1]`) raise `MALFORMED_CITATION`.

The parser **never calls the LLM** and **never rewrites the content**, except for removing unknown keys in `clean_text`.

## 6. Project layout additions

```
prompts/
├── answer_system.v1.md       ← provided; wording changes need user approval
└── answer_user.v1.md         ← provided
augmentation/
├── __init__.py
├── config.py                 ← PROMPT_VERSION, file paths, word limit, phrases
├── prompt_loader.py          ← loads and checks the prompt files at startup
├── context_builder.py        ← RetrievalResult -> (sources_block, key_map)
├── messages.py               ← builds [system, user] messages
├── tokens.py                 ← prompt size measurement (Q3)
├── parser.py                 ← parse_answer() and the output contract
└── models.py                 ← ParsedAnswer, KeyMap dataclasses
eval/
└── answer_behaviour_cases.yaml   ← behaviour cases for the generation phase (Stage A5)
tests/
├── test_prompt_loader.py
├── test_context_builder.py
├── test_messages.py
└── test_parser.py
```

---

## 7. Stages

### Stage A0 — Prerequisites

1. Confirm with the user that retrieval is complete and its tests pass. Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.
2. Copy the provided prompt files into `prompts/`, unchanged.

**Checkpoint:** confirm the files are in place (show their first 10 lines).

### Stage A1 — Prompt loader and versioning

**Steps**
1. `prompt_loader.py` loads both files once and checks:
   - the system prompt contains the five section headings **in order**: `# PURPOSE`, `# CONSTRAINTS`, `# INTERPRETATION`, `# DECISION`, `# OUTPUT FORMAT`;
   - it contains the exact insufficient-context sentence on its own line;
   - the user template contains exactly the placeholders `{sources_block}` and `{standalone_query}`.

   Startup fails with a clear message if any check fails.
2. `PROMPT_VERSION` comes from the file names (`v1`).

**Done when:** loader tests pass, including failure tests (missing section, wrong order, missing placeholder, altered sentence).

**Checkpoint:** show the test results.

### Stage A2 — Context builder

**Steps**
1. `build_context(result: RetrievalResult) -> (sources_block: str, key_map: dict[str, str])`:
   - keys `C1…Cn` in the **given** chunk order (never re-sorted here);
   - attributes per Section 4;
   - escaping per Section 3.5;
   - `display_text` as the body.
2. If `result.chunks` is empty, return an empty block and an empty map. The generation phase handles `relevant=false` before calling this; the function must still behave safely.
3. Tests:
   - key order follows chunk order;
   - siblings (`added_as_sibling`) get their own keys in place;
   - the `type` mapping covers all 5 chunk types;
   - table `section` includes the subcategory or category;
   - `part` formatting;
   - escaping neutralizes a fake `</source>` in the text, while a quoted NIST term (with quotes and `&`) survives unchanged in the body;
   - no page numbers or URLs appear anywhere in the block;
   - deterministic output.

**Done when:** all tests pass.

**Checkpoint:** show the sources block built from a real `RetrievalResult` for "Which of the four AI RMF functions is cross-cutting?", plus its key map.

### Stage A3 — Message assembly

**Steps**
1. `build_messages(standalone_query, result) -> (messages, key_map)` returns a system message (the system prompt file) and a user message (the template filled in). **No conversation history** (G-D9).
2. The query is inserted as-is, except for neutralizing `<question>` / `</question>` tags inside it (Section 3.5 style).
3. Tests:
   - exactly 2 messages;
   - the system message equals the file content;
   - the user message has the sources before the question;
   - no history leaks in, even when the graph state has previous messages;
   - snapshot test: for a fixed fixture, the full assembled prompt matches a stored snapshot. Changing it requires deliberately updating the snapshot.

**Done when:** all tests pass.

**Checkpoint:** show one fully assembled prompt (system plus user) for a real query.

### Stage A4 — Output parser and validator

**Steps**
1. Implement `parse_answer()` per Section 5.
2. **Sentence splitting** for `UNCITED_SENTENCE`: split on sentence ends and on bullet lines. Treat abbreviations common in these documents (`e.g.`, `i.e.`, `etc.`, `U.S.`, section numbers like `5.1`) as not ending a sentence. Exempt: the Case B first line, lead-in lines ending with `:`, and "not covered" bullets.
3. Tests with hand-written outputs, at least these:
   - **valid:** Case A paragraph; Case A with bullets; Case B with one covered and one uncovered reading; exact Case D; Case D without the final period;
   - **invalid:** one uncited sentence; an invented `[C9]` when only C1–C5 exist; `[C1, C2]` malformed; a "Sources:" list; a URL; a markdown heading; Case D sentence plus extra text; 180 words; empty output; an answer with zero citations;
   - **edge cases:** a sentence containing "e.g." stays one sentence; `[C1][C3]` adjacent is valid; a quoted term containing a period.

**Done when:** all tests pass.

**Checkpoint:** show the test table (input summary → case → violations).

### Stage A5 — Behaviour test cases for the generation phase

**Purpose:** define now, while the rules are fresh, how the full prompt **should** behave, so the generation phase can test it against the real Groq model.

**Steps**
1. `eval/answer_behaviour_cases.yaml`, **at least 15 cases**. Each case has a standalone query, a note on which retrieved context is expected (or a fixed context fixture), the expected case (A/B/D), and checks (e.g., "must quote 'GOVERN'", "must not mention pages", "must name both documents"). Cover:
   - **A:** the four functions (list with bullets); the cross-cutting function; a definition question (exact quote expected); a table-based question (GenAI action); a figure-OCR-based question (must mention the figure);
   - **B:** an ambiguous question with both readings covered; one with only one reading covered;
   - **C → D:** a two-part question where only one part is in the documents;
   - **D:** a near-scope question not in the documents;
   - **conflict:** a topic described differently in the two documents (both views, each attributed);
   - **injection:** a fixture source containing "Ignore previous instructions…" (must be ignored);
   - **language:** a question asked in Hindi (answer must be in English);
   - **length:** a broad question that tempts a long answer (must stay ≤ 150 words).
2. **Ask Q5** for the user to review and approve the cases.

**Done when:** the cases file exists and is approved.

**Checkpoint:** show the cases table.

### Stage A6 — Prompt size measurement

**Steps**
1. For every query in the retrieval evaluation set, build the full prompt and measure:
   - system prompt size;
   - sources block size;
   - total size;
   - the number of chunks (including siblings).

   Report the minimum, median, p95 and maximum. Use the tokenizer chosen in **Q3**.
2. Report the worst case (top 5 plus the largest possible siblings) as a computed upper bound.
3. The generation phase compares these numbers with the chosen Groq model's context window and free-tier tokens-per-minute limit. **Ask Q4** about an overflow policy only if the measured maximum could exceed a limit the user names.

**Done when:** the size report exists in `eval/prompt_size_report.json`.

**Checkpoint:** show the size table.

---

## 8. Testing rules

- Everything in this phase is deterministic and tested without network access or an API key.
- Never change prompt wording to make a test pass. Prompt changes require the user's approval and a version bump (`v2` files, keeping `v1`).
- The snapshot test must fail when the assembled prompt changes unexpectedly.

## 9. README / DESIGN additions

- The prompt's five sections and why each exists. Show the prompt file in the design notes.
- Why plain text with inline citations (streaming) instead of JSON, and how the parser enforces the contract.
- Why no conversation history in generation (the standalone query already resolves follow-ups; this saves tokens; old turns can't leak in).
- Why partial answers are refused (no confident-looking answers with hidden gaps), with the trade-off stated.

## 10. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Word limit for ambiguous (Case B) answers: 150 words in total, or a larger total such as about 250? Show one example of each. | A4 |
| Q2 | Confirm that page numbers are **not** shown in source attributes (Section 3.3), or show them | A2 |
| Q3 | Tokenizer for prompt measurement: the Groq model's own tokenizer, if published, or a stated estimate (e.g., characters ÷ 4 with a 20% margin) | A6 |
| Q4 | Overflow policy, only if needed: drop the lowest-ranked non-sibling chunks first? | A6 |
| Q5 | Approval of the behaviour test cases | A5 |
| Q6 | Approval of the exact phrases: `This question can be read in more than one way:` and `not covered by the provided documents.` | A1 |
| Q7 | Any wording change to the prompt files (only if the user requests one) | Any |

## 11. Handoff to the generation phase

- Call `build_messages()`, send to the Groq generation model, stream the tokens, then run `parse_answer()` on the full text.
- The generation phase decides the response to violations (e.g., retry once, remove unknown citations, fall back to Case D), the `relevant=false` path, and builds the API's Sources list from `cited_keys` → `key_map` → chunk metadata (`citation_label`, `citation_url`).
