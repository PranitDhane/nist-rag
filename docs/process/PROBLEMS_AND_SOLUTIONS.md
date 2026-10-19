# Problems and solutions

Problems found while building the system and what was decided about each. It is a readable index into [`PROGRESS.md`](PROGRESS.md), the running log, where each item has its full entry (cited here as *date | phase | stage*). It records what happened, not metrics: current measurements are in [`../RESULTS.md`](../RESULTS.md). Status: **fixed** (changed in code), **handled** (worked around by design), **accepted** (a known limitation, documented), **open** (not decided or not done).

The plans in [`plans/`](plans/) say what each phase intended; this file says where reality differed.

## Ingestion (phase 1)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| Two footnotes on one GenAI page came out of PyMuPDF as a single text block | `bottom_blocks` are split per footnote | fixed | 10-17 Stage 4 |
| Footnotes 6–8 on another page start below the plan's 75 % page-height cut-off and were missed | Any block that contains a footnote-start line is kept as a footnote block | fixed | 10-17 Stage 4 fix |
| The converter puts the page number last (the plan said first) and footnotes as `>` blockquotes | The cleaning stage handles both | handled | 10-17 Stage 4, Stage 7 |
| The converter drops rotated text; the AI RMF page with the rotated figure came out empty | Stage 4 records the rotated text and a ninth cleaning rule appends it | fixed | 10-17 Stage 10 |
| Icon-font garbling leaves stray characters in action-ID cells (`MP-3.4-004   i      i`) and in "AI Actor Tasks" | A cleaning rule repairs the ID cells; the actor names are repaired later against the known vocabulary | fixed | 10-17 Stage 7 fix |
| One footnote (GenAI p.7) is garbled by the converter and cannot be matched | Left in place with a warning | accepted | 10-17 Stage 7 |
| Figure 2 OCR returns nothing (curved text) | Confidence threshold 60 chosen by the user; Figure 2 gets no chunk | accepted | 10-17 Stage 6 |
| The plan expected 212 action IDs in table rows; the PDF has 211 table-row IDs | The user set 211 for ingestion; chunking later found the 212th (see Chunking) | resolved | 10-17 Stage 8 |
| A subcategory heading was counted as a table start even when no table followed (AI RMF) | A heading counts only if an "Action ID" table follows | fixed | 10-17 Stage 8 |
| Some links have empty or cut-off anchor text; some bare URLs are garbled by a lost ligature | Reported as warnings; links are kept as found | accepted | 10-17 Stage 9 |
| Header and footer strings had to be forbidden in the output without hard-coding them in the pipeline | They live in the manifest's `validation:` block, read only by the validator | fixed | 10-17 Stage 10 |

## Chunking (phase 2)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| The AI RMF Core tables are mangled by the converter (cells split across rows, text interleaved) | Categories are found from the `FUNC N:` labels, not from the table cells; the MAP 1 and MAP 2 descriptions cannot be recovered, so those chunks carry the label only | handled | 10-17 Chunking C3-C4, C6 |
| The PDF prints one action ID as `GV4.3--001`, so the ingestion pattern skips it | Chunking repairs it, logs the repair, and the manifest records the 212 IDs chunking expects | fixed | 10-17 Chunking C6 |
| Very small sections (a heading and a line) make useless chunks | Sections of 60 tokens or fewer merge forward (chosen by the user from the profile) | decided | 10-17 Chunking C2, C5 |
| Large table pieces risk the 512-token limit | Table pieces are capped at 450 tokens, split only between rows, header repeated (chosen by the user) | decided | 10-17 Chunking C2, C6 |
| Three of four figure captions are not in the page Markdown, so the section is unknown | The section is taken at the end of the page and logged | accepted | 10-17 Chunking C7 |
| Chunks needed more than the plan's fields to avoid losing information (character spans of merged chunks) | `char_spans` added to the chunk record | decided | 10-17 Chunking C8-C9 |

## Embedding and indexing (phase 3)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| Chroma's default distance is L2, not cosine | Set through the collection configuration object and read back to verify | fixed | 10-17 Embedding E4 |
| Dense vectors miss exact identifiers (`GV-1.7-001`) | BM25 runs alongside; the tokenizer keeps IDs whole and never stems them | decided | 10-17 Embedding E5, E6 |
| BM25 had to be saved without pickle | Save token lists as JSON and rebuild the object at load time | decided | 10-17 Embedding E5 |
| `langchain-huggingface` replaces newlines with spaces before encoding | Checked: no effect on token counts for this model | accepted | 10-17 Embedding E2 |
| The index must never be queried with a different model or revision than it was built with | The manifest records them and the application checks at start-up | fixed | 10-17 Embedding E1, Retrieval T1 |

## API and routing (phase 4)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| The Llama models most guides mention are not offered to this Groq key | `openai/gpt-oss-120b` for routing and `gpt-oss-20b` for generation, verified against the live model list | decided | 10-17 API R0-R1, Generation N0 |
| `langchain-groq`'s default structured-output method is function calling | `json_schema` with `strict=True` is set explicitly | fixed | 10-17 API R0-R1 |
| The Groq async client is bound to the first event loop | Tests use one loop; irrelevant under uvicorn | handled | 10-17 API R0-R1 |
| An LLM outage would leave an unanswered user message in the history | A `fail` node removes it (later generalised to `error_finalize`) | fixed | 10-17 API R2, Graph G4 |
| Raw router output could contain user text | Logged at DEBUG only; a warning records invalid output | decided | 10-17 API R4 |
| The routing evaluation misses two cases: a question about AI vs software risks is routed out of scope, and one follow-up's standalone query misses a key term | Reported, not patched | open | 10-17 API R8 |

## Retrieval (phase 5)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| The plan's BM25 library choice conflicted with the retrieval plan | LangChain's `BM25Retriever`, built from the saved token lists with our tokenizer; it pads zero-score chunks when fewer than k match | decided | 10-17 Retrieval T0-T1, T2-T3 |
| LangChain's cross-encoder reranker drops the scores the gate needs | `sentence-transformers` `CrossEncoder` is used directly | decided | 10-17 Retrieval T4 |
| Some (query, chunk) pairs exceed the reranker's 512-token input | Accepted and measured | accepted | 10-17 Retrieval T4 |
| Two answerable paraphrases score below the chosen gate threshold, and the threshold is tuned on few negatives | Threshold 0.15 chosen by the user from the table | accepted | 10-17 Retrieval T6 |
| Only split tables should pull in sibling parts, not every multi-part chunk | Siblings are added for tables only, and are extra to the top 5 | decided | 10-17 Retrieval T5 |

## Augmentation, generation, insufficient context and attribution (phases 6-9)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| The plan's example document label differs from the real display name | The builder uses the real name unchanged | handled | 10-17 Augmentation A0-A4 |
| Reasoning tokens count toward `max_tokens` (a tiny limit gave empty text); the first limit was too tight for broad questions | Output cap raised to 3,500 by the user | fixed | 10-17 Generation N0-N1, N4, Graph G0-G1 |
| The budget pre-check over-estimated prompt size and rejected a share of real turns | Margin lowered from 1.2 to 1.0 (measured 4.0 characters per token) | fixed | 10-17 Graph G8 |
| The model writes non-breaking hyphens and spaces in IDs | Checks compare after normalisation | handled | 10-17 Generation N5 |
| The parser treated `... . [C1]` as an uncited sentence, inflating warnings | Known; the evaluation re-attaches such markers when scoring | accepted | 10-17 Generation N5; 10-19 E2E V4 |
| The real model sometimes ignores the prompt (cites `[G1]`, never uses the ambiguous-question format, hits the length cap) | Reported as behaviour findings; prompt v2 is not written | open | 10-17 Generation N5 |
| Gate-failure related sections can be clearly unrelated to the question | Kept (top 5, always) and documented; they are never mixed into `sources` | accepted | 10-17 Insufficient I5 |
| LangGraph silently drops node output keys that are not in the state schema | State gained `outcome` and `insufficient` | fixed | 10-17 Insufficient I4 |
| Citation labels and URLs are needed by both the insufficient-context phase (related sections) and attribution (sources) | One shared formatter (`attribution/format.py`), checked equal to the labels stored in all chunks; `citation_name` moved into the manifest | decided | 10-17 Insufficient I2, Attribution S0-S1 |
| 37 chunk links do not match the sampled text exactly (synthetic lines, table cell order) | Graded: explained mismatches are listed, a link to the wrong page still fails | accepted | 10-17 Attribution S4 |

## Graph integration (phase 10)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| Checkpoints holding the real dataclasses warn they will be blocked in a future LangGraph version | State holds plain dicts | fixed | 10-17 Graph G0-G1 |
| A test found the per-turn `path` list appended to the previous turn's stale value | `reset_turn` deep-copies the defaults | fixed | 10-17 Graph G4 |
| Per-step checkpoints are large (about fifteen per NIST turn) and `InMemorySaver` cannot prune | `durability="exit"`: one checkpoint per request | decided | 10-17 Graph G8 |
| Tracing was switched on by a test and tried to reach LangSmith | Tracing stays off unless enabled by environment; the tracing test file blocks sockets | fixed | 10-17 Graph G7 |
| Near-scope questions are usually caught by the router before the gate | Accepted; the gate remains the second line | accepted | 10-17 Graph G8 |
| A live LangSmith test fails when `LANGSMITH_API_KEY` is set, because its own file blocks the network | Left unchanged by the user's decision | open | 10-19 Deliverables D1 |
| Conversation history is in memory | Accepted; documented as a limitation | accepted | 10-17 API R2 |

## End-to-end evaluation (phase 11)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| `debug` lacked the `key_map` the groundedness and citation checks need | Added (IDs only), with tests | fixed | 10-18 E2E V0 |
| Draft gold data contained pseudo-quotes and a trap on a section that does not exist (the assignment's illustrative "§1.3") | Absence claims made explicit; the trap kept and flagged; gold quotes verified against the PDFs | fixed | 10-19 E2E V1 |
| Short trap words over-matched ("fee" in "feedback") | Variants tightened with the user's approval; matching is word-start prefix | fixed | 10-19 E2E V1 |
| The English heuristic wrongly failed a real answer with only two common words | Required stop words lowered from three to two, test added | fixed | 10-19 E2E V6 |
| Real answers contain Markdown bold although the output contract says plain text | Observed and reported; not changed | open | 10-19 E2E V6 |
| The first live run exposed two follow-ups refused with the insufficient-context sentence although retrieval scored them as highly relevant, three sentences left uncited, and one routing miss | Reported in full in `docs/E2E_RESULTS.md`; fixing them is a separate, approved task | open | 10-19 E2E V6 |

## Deliverables (phase 12)

| Problem | Decision | Status | Log entry |
|---|---|---|---|
| Generated data was tracked in git (the PDFs, chunk files, reports) | Untracked and `data/` ignored (history is not rewritten); the PDFs are downloaded | fixed | 10-19 Deliverables D0, D2 |
| `.env.example` lacked seven variables the code reads | Added, with a test that fails when a read variable is undocumented | fixed | 10-19 Deliverables D2 |
| Numbers in docs would drift from the reports | `docs/build_results.py` generates `RESULTS.md`, the README block and the evidence paragraphs of `DESIGN.md`; tests reject typed metrics | decided | 10-19 Deliverables D3, D4, D5 |
| The README had grown into per-phase notes with hand-typed numbers | Replaced; the old text is kept as [`phase_notes.md`](phase_notes.md) | fixed | 10-19 Deliverables D5 |
| This file did not exist although `CLAUDE.md` named it | Written from the progress log (this file) | fixed | 10-19 Deliverables D6 |

## What is still open

Prompt v2 (refused answerable follow-ups, uncited sentences, Markdown in answers, partial answers to two-part questions), the routing miss, the failing live LangSmith test, and persistent conversation history. They are also listed under "What I'd do with more time" in the README.
