# Chunking Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply here too. This file is the specification for the **chunking phase**. Work stage by stage (Section 6). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. A stage is finished only when every "Done when" item is true **and** the user has confirmed the Checkpoint. Questions in Section 9 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

Turn `data/processed/pages.jsonl` (ingestion output) into **retrieval-ready chunks** in `data/processed/chunks.jsonl`, plus a report `data/processed/chunk_report.json`.

- **In scope:** prerequisite ingestion changes (Stage C0), token counting, profiling, joining pages, section segmentation, prose / table / figure / footnote chunking, metadata, validation.
- **Out of scope:** computing embeddings, vector store, BM25, retrieval, LangGraph, UI. Do not add their dependencies, except the tokenizer from Stage C1.
- **Chunking never opens the PDFs.** Everything comes from `pages.jsonl`.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| D1 | Embedding model the chunks are sized for | `BAAI/bge-small-en-v1.5`, hard limit **512 tokens** |
| D2 | Prose chunk size | **250–350 tokens** of content |
| D3 | Prose overlap | **~10%**, only between pieces of the **same** section |
| D4 | Section breadcrumb | Added to the **embedded text only**; displayed text stays clean |
| D5 | Tiny / heading-only sections | **Merge forward** into the next chunk |
| D6 | Footnotes | **Their own small chunks**, cited to their page |
| D7 | Figure OCR text | **One chunk per figure**: caption + OCR text, labelled as OCR |
| D8 | Splitting method | Split by **section first**, using ingestion's Stage 8 headings (**not** LangChain's `MarkdownHeaderTextSplitter`); recursive splitting only for sections that are too long |
| D9 | Page tracking | **Page map** (character offsets) built when joining pages; **never** find pages by searching for chunk text; **never** insert page markers into the text |
| D10 | Tables | **Table-aware**: chunk unit = natural table unit; split only **between rows**; **repeat the header** in every piece |

## 3. Core principles

1. **Every chunk has exactly one section path.** Sections are cut before any size splitting, so no chunk spans two sections.
2. **Every chunk knows its pages exactly**, from character offsets and the page map.
3. **The embedded text is never longer than 512 tokens**, counted with the bge-small tokenizer, including the breadcrumb and special tokens. bge-small silently ignores anything past 512, so this check is mandatory.
4. **Structure decides chunk boundaries where it exists** (sections, tables, figures, footnotes). Size limits decide only inside a structural unit.
5. **No index-eligible text is lost.** Every non-whitespace character of every index-eligible page ends up in at least one chunk (verified by offsets).
6. **Deterministic:** the same `pages.jsonl` always produces the same chunks and the same chunk IDs.

## 4. Chunk types

| `chunk_type` | Unit | Size rule |
|---|---|---|
| `prose` | A section's text (excluding its tables) | 250–350 content tokens target; longer sections split recursively with ~10% overlap |
| `genai_action_table` | One GenAI subcategory table: subcategory line + action rows + "AI Actor Tasks" line, joined across pages | Whole table if it fits; otherwise split between action rows, repeating the subcategory line, column header and "AI Actor Tasks" line in each piece |
| `rmf_core_table` | One AI RMF Core category (e.g., GOVERN 1 with GOVERN 1.1–1.7) from Tables 1–4, joined across pages | Whole category if it fits; otherwise split between subcategories, repeating the category text in each piece |
| `figure` | One figure: caption + OCR text, labelled | Only if OCR text exists; if too long, split recursively, repeating the caption and label |
| `footnote` | One footnote | Normally tiny; never merged with other chunks |

## 5. Worked example (page map)

Section 5.1 Govern in the joined AI RMF text. The numbers are illustrative.

| Page | Characters | Starts at | Ends before |
|---|---|---|---|
| 22 | 2,400 | 0 | 2,400 |
| 23 | 2,100 | 2,400 | 4,500 |
| 24 | 1,800 | 4,500 | 6,300 |

A chunk covering offsets 2,150–2,950 starts on page 22 (2,150 < 2,400) and ends on page 23 (2,400 ≤ 2,950 < 4,500), so it is cited as pp. 22–23. If the section starts at offset 300 and the recursive splitter reports a piece starting at 1,900 within the section, that piece's absolute start is 300 + 1,900 = 2,200.

---

## 6. Stages

### Stage C0 — Prerequisites and ingestion changes

**Purpose:** make sure ingestion output contains everything chunking needs, and change ingestion where it doesn't.

**Steps**
1. Confirm with the user that ingestion is complete and `python -m ingest --rebuild` passes all hard checks. Ask the user to approve updating the "Current phase" line in `CLAUDE.md` to the chunking phase, and pointing it at this file.
2. **Change ingestion Stage 8** so each heading record also has `char_offset`: the position of the heading line's first character in the page's **cleaned** `markdown`.
3. **Change ingestion Stage 8** so each table marker also has `char_offset` (where the table unit starts in the page's cleaned `markdown`) and a `kind` (`genai_subcategory` or `rmf_table`).
4. Add tests for both: for a sample page, slicing `markdown` at `char_offset` must start with the heading or marker text.
5. Re-run ingestion; all its hard checks must still pass.

**Done when:** the new fields exist on every heading and marker; the new tests pass; ingestion validation passes.

**Checkpoint:** show three heading records and three table-marker records with their offsets, plus the matching slice of `markdown` for each.

### Stage C1 — Tokenizer and configuration

**Purpose:** count tokens exactly as the embedding model will, and keep all limits in one place.

**Steps**
1. **Ask the user** (open question Q1) which package to use for the bge-small tokenizer. Do not add a dependency before the answer.
2. Create `chunking/config.py` with:
   - model name `BAAI/bge-small-en-v1.5`
   - `HARD_MAX_EMBED_TOKENS = 512` (including special tokens)
   - `PROSE_TARGET_MIN = 250`, `PROSE_TARGET_MAX = 350`, `PROSE_OVERLAP ≈ 10%` of the piece size
   - placeholders for the values decided after profiling: `TINY_SECTION_MAX_TOKENS` and `TABLE_PIECE_MAX_TOKENS` (Q3, Q4); no defaults until the user answers
3. Create `chunking/tokens.py` with one function, `count_tokens(text) -> int`, counting **with special tokens**, plus a test that checks a known short string's token count.

**Done when:** `count_tokens` works offline after the first model download; the test passes.

**Checkpoint:** show the token counts for 3 sample strings (one prose paragraph, one table row, one breadcrumb) and where the tokenizer files are cached.

### Stage C2 — Profiling (measure; produce no chunks)

**Purpose:** measure the real sizes, so the remaining numeric decisions are made from data.

**Steps**
1. Join the pages (using Stage C3's logic, implemented here first or as a shared helper) and segment them into sections and table units (Stage C4's logic).
2. Report token sizes (count, min, median, max, and a histogram in 50-token buckets) for:
   - prose per section, and the number of sections over 350 tokens;
   - every section under 60 tokens (listed by name, with its text), as candidates for "tiny";
   - every GenAI subcategory table (expected 49; median about 240, max about 610 by rough estimate), and how many exceed 350, 400, 450 and 480 tokens;
   - every AI RMF Core category (Tables 1–4), and how many exceed the same thresholds;
   - each figure's OCR text (if present);
   - all footnotes (count, min, max).
3. **Check whether a table continued on the next page repeats its column header row** in the Markdown (e.g., `|Action ID|Suggested Action|GAI Risks|` at the top of a continued page). Report each occurrence.
4. Write all of this to `chunk_report.json` under `profile`.

**Done when:** the profile covers both documents and all five chunk types.

**Checkpoint:** show the profile summary. **Ask Q3** (tiny-section threshold) and **Q4** (maximum tokens for a table piece) using the measured numbers. Record the answers in `config.py`.

### Stage C3 — Join pages and build the page map

**Purpose:** create one continuous text per document for section cutting, without losing page information.

**Steps**
1. For each document, take the **index-eligible** pages in `pdf_page` order. Join their cleaned `markdown` with a fixed separator of `\n\n`.
2. While joining, record for each page: `pdf_page`, `printed_page`, `start` offset, `end` offset (exclusive). The separator belongs to no page.
3. Provide `pages_for_span(start, end)`, which returns the first and last page (both `pdf_page` and `printed_page`) touched by a span. An offset inside a separator is assigned to the page **before** it.
4. Convert every heading's and table marker's page-local `char_offset` to an absolute offset: page `start` + `char_offset`.

**Done when:** unit tests pass for:
- a span inside one page;
- a span crossing one page boundary;
- a span crossing two page boundaries;
- a span starting inside a separator;
- the worked example in Section 5.

For every heading, the joined text at its absolute offset must begin with the heading text.

**Checkpoint:** show the page map for the first 5 pages of each document, and the heading-offset check result.

### Stage C4 — Section and unit segmentation

**Purpose:** cut each joined document into structural units before any size splitting.

**Steps**
1. **Sections:** each heading starts a section that ends at the next heading, at any level. The section's `section_path` comes from ingestion's Stage 8 hierarchy (e.g., `["Part 2", "5 AI RMF Core", "5.1 Govern"]`). Text before the first heading of a document belongs to a document-level section.
2. **Table units inside sections:**
   - **GenAI subcategory table:** starts at a `genai_subcategory` marker and runs through the following Markdown table lines (lines starting with `|`), continuing across page separators, until the next marker, the next heading, or the first non-table, non-blank line. Remove a repeated column-header row from the continued part **only if** Stage C2 found them, and log each removal.
   - **AI RMF Core table:** from an `rmf_table` marker (the `Table N:` caption) through its Markdown table lines, continuing across pages. Within it, **category units**: a category starts at a row whose first cell is non-empty (e.g., `GOVERN 1:`), and the following rows with an empty first cell belong to it.
3. **Prose segments:** a section's text with its table units removed. Prose before a table and prose after it are separate segments of the same section, kept in reading order.
4. Give every unit a `sequence` number in reading order, per document.

**Done when:**
- every non-whitespace character of each joined document belongs to exactly one unit (apart from logged header-row removals);
- 49 GenAI subcategory table units;
- AI RMF categories: report the count per function and confirm the expected count with the user;
- no unit crosses a section boundary.

**Checkpoint:** show the unit list for AI RMF Section 5.1 and GenAI Section 3 (first 5 units): type, section path, pages, and the first 80 characters.

### Stage C5 — Prose chunks (including tiny-section merging)

**Purpose:** turn prose segments into right-sized chunks.

**Steps**
1. **Ask the user** (open question Q2) whether to use `langchain-text-splitters`' `RecursiveCharacterTextSplitter` or a small custom recursive splitter. Either way:
   - length is measured with `count_tokens` (not characters);
   - separators in priority order: blank line, line break, sentence end (`. `), space;
   - the start offset of every piece is known (with LangChain, `add_start_index=True`).
2. A segment of ≤ 350 tokens becomes **one** chunk. A longer one is split recursively to 250–350 tokens with ~10% overlap. The last piece may be shorter than 250.
3. **Tiny-section merge (D5):** a prose segment whose section has ≤ `TINY_SECTION_MAX_TOKENS` (Q3) of its own text, or only a heading, is **prepended** to the next **prose** chunk in the same document.
   - The merged chunk keeps the **next** section's `section_path`, and records the merged section in `merged_sections`.
   - Its page span is the union of both spans.
   - If there is no next prose chunk in the same document, merge **backward** into the previous prose chunk instead.
   - Never merge into or from table, figure or footnote chunks.
   - If the merged chunk would exceed the limit, do not merge; keep it as a small chunk and log it.
   - Log every merge.
4. Each chunk records `char_start` and `char_end` (absolute offsets), and its pages from `pages_for_span`.

**Done when:**
- every prose chunk's content is ≤ 350 tokens (merged chunks may exceed this only up to the hard limit, and are listed);
- overlap never crosses a section;
- tiny-merge tests pass (forward, backward fallback, would-exceed-limit case);
- every prose character is covered.

**Checkpoint:** show the token-size histogram of prose chunks, every merge performed, and two full examples: a section split into several pieces (showing the overlap), and a merged tiny section.

### Stage C6 — Table chunks

**Purpose:** chunk tables by their natural units without ever separating a row from its header.

**Steps**
1. **Ask the user** (open question Q5) how table text should appear inside chunks. Show the same GenAI table (GOVERN 1.7) and the same AI RMF category (GOVERN 1) rendered both ways:
   - (a) the original Markdown table;
   - (b) plain labelled lines, one per row (e.g., `GV-1.7-001 — Suggested action: … — GAI risks: …`).

   Report the token count of each version.
2. **GenAI subcategory table:** if the whole unit fits within `TABLE_PIECE_MAX_TOKENS` (Q4), it becomes one chunk. Otherwise, split **between action rows** into the fewest pieces that fit. Every piece starts with the subcategory line (and column header, if Markdown is kept) and ends with the "AI Actor Tasks" line. Pieces are labelled `part i of n`.
3. **AI RMF category:** same rule. Split **between subcategories** (the `**GOVERN 1.6:**` boundaries inside a cell), and repeat the category text (e.g., `GOVERN 1: Policies, processes…`) in every piece.
4. **Guard:** if one row or one subcategory alone exceeds the limit with its header, stop and report it to the user. Never split inside a row.
5. Metadata per chunk:
   - GenAI: `function` (GOVERN/MAP/MEASURE/MANAGE), `subcategory` (e.g., `GOVERN 1.7`), `action_ids`, `gai_risks`, `ai_actor_tasks`, `part`, `parts_total`.
   - AI RMF: `function`, `category` (e.g., `GOVERN 1`), `subcategories`, `part`, `parts_total`.

**Done when:**
- every one of the **212** action IDs appears in **exactly one** table chunk's `action_ids`;
- every GenAI subcategory (49) has at least one chunk;
- every piece contains its repeated header and the "AI Actor Tasks" line;
- no row is split;
- every table chunk fits the limit.

**Checkpoint:** show GOVERN 1.7 (expected to be one chunk), MEASURE 2.11 (expected to be split), and one split AI RMF category if any, all in full, with token counts.

### Stage C7 — Figure and footnote chunks

**Purpose:** make the separately stored figure OCR text and footnotes retrievable, clearly labelled.

**Steps**
1. **Figures (D7):** for each figure record with non-null OCR `text`, create one chunk:
   - display text: the caption, then a label line `Text extracted by OCR from Figure N (may contain recognition errors):`, then the OCR text;
   - `section_path`: the section in effect at the figure's position on its page;
   - pages: the figure's page;
   - if it exceeds the limit, split recursively and repeat the caption and label in each piece.

   Figures with null OCR text get **no** chunk (their caption is already in the prose).
2. **Footnotes (D6):** one chunk per footnote:
   - display text: `Footnote N: <text>`;
   - pages: the footnote's page;
   - `links`: the footnote's links;
   - `section_path`: **ask the user** (open question Q6) whether it should be the section in effect at the **end** of the page, or the page's `section_path_start`. The marker was removed during ingestion, so the exact sentence is unknown. Either way, set `section_path_is_page_level: true`.
3. Figure and footnote chunks are **never** merged with other chunks.

**Done when:** the number of figure chunks equals the number of figures with OCR text (or more, if split); the number of footnote chunks equals the total number of footnotes in `pages.jsonl`; every one has a page.

**Checkpoint:** show every figure chunk (truncated to 300 characters) and 5 footnote chunks.

### Stage C8 — Breadcrumb, IDs and metadata assembly

**Purpose:** build the final chunk records.

**Steps**
1. **Breadcrumb (D4)**, prepended to the **embedded text only**, followed by a blank line. Proposed format:
   - `{display_name} > {section_path joined with " > "}`
   - for table pieces, add ` > {subcategory or category} [part i of n]`
   - for figures, add ` > Figure N (OCR)`
   - for footnotes, add ` > Footnote N`

   **Show 5 examples (one per chunk type) and get the user's approval (Q7)** before finalizing.
2. `display_text` = the chunk's clean content with no breadcrumb. `embed_text` = breadcrumb + blank line + `display_text`.
3. **Hard check:** `count_tokens(embed_text) ≤ 512` for every chunk. If one fails, stop and report it; do not truncate silently.
4. **`chunk_id`** (deterministic): `{doc_id}:{chunk_type}:{sequence:04d}:{part}`. The same input must produce the same IDs.
5. **Links:** attach to each chunk the links whose position falls inside the chunk's span, taken from the page records (body links). Footnote chunks get their footnote's links. Figure chunks get none.
6. **Citation fields:**
   - `pdf_page_start`, `pdf_page_end`, `printed_page_start`, `printed_page_end`;
   - `source_url`;
   - `citation_url` = `source_url#page={pdf_page_start}`;
   - `citation_label`: show a proposed format to the user for approval (e.g., `NIST AI 100-1, §5.1 Govern, pp. 22–23`).

**Done when:** every chunk has every required field (Section 7); all `embed_text` values are ≤ 512 tokens; running twice produces identical IDs.

**Checkpoint:** show one complete record per chunk type, and confirm the breadcrumb and citation-label formats with the user.

### Stage C9 — Validation and report

**Purpose:** prove the chunks are complete, correct and within limits.

**Hard checks (the run exits non-zero if any fail):**
1. Every `embed_text` is ≤ 512 tokens (bge-small tokenizer, special tokens included).
2. **Coverage:** every non-whitespace character of every index-eligible page is inside at least one prose or table chunk span (apart from logged repeated header-row removals).
3. No prose chunk spans two sections; overlap exists only between pieces of the same section.
4. 212 action IDs, each in exactly one table chunk; 49 GenAI subcategories each with at least one chunk.
5. Every table piece contains its repeated header; no table row is split.
6. Every chunk has a `doc_id`, `section_path`, page range, `citation_url` and non-empty `display_text`.
7. No `display_text` contains the breadcrumb line.
8. Footnote chunk count equals the total footnotes; figure chunks exist only for figures with OCR text.
9. Determinism: two consecutive runs produce identical `chunks.jsonl` (byte-for-byte or by sorted comparison).

**Soft warnings:** prose chunks under 100 tokens (other than merged or last pieces); merged chunks above 350 tokens; figures with OCR confidence near the threshold; footnotes shorter than 5 words.

**Report** `chunk_report.json`:
- chunk counts by document and type;
- token histograms per type;
- merges, splits and header-row removals;
- the profile from Stage C2;
- check results and warnings.

**Done when:** all hard checks pass on `python -m chunking --rebuild`.

**Checkpoint:** show the console summary and the counts table.

---

## 7. `chunks.jsonl` schema (one object per chunk)

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | str | Deterministic |
| `doc_id`, `display_name` | str | From the manifest |
| `chunk_type` | str | `prose` / `genai_action_table` / `rmf_core_table` / `figure` / `footnote` |
| `section_path` | list[str] | Exactly one section |
| `section_path_is_page_level` | bool | true for footnotes |
| `merged_sections` | list[str] | Tiny sections merged in (D5) |
| `sequence` | int | Reading order within the document |
| `part`, `parts_total` | int | For split units; 1/1 otherwise |
| `char_start`, `char_end` | int | Absolute offsets in the joined document (prose/table) |
| `pdf_page_start`, `pdf_page_end` | int | |
| `printed_page_start`, `printed_page_end` | str \| null | |
| `source_url`, `citation_url`, `citation_label` | str | |
| `display_text` | str | Shown to users; no breadcrumb |
| `embed_text` | str | Breadcrumb + display_text; ≤ 512 tokens |
| `token_count_embed` | int | |
| `function`, `category`, `subcategory` | str \| null | Table chunks |
| `subcategories`, `action_ids`, `gai_risks`, `ai_actor_tasks` | list[str] | Table chunks |
| `trustworthy_characteristics` | list[str] | Chunks inside a GenAI risk section (from Stage 8 entities) |
| `figure_number`, `ocr_confidence` | int / float \| null | Figure chunks |
| `footnote_number` | int \| null | Footnote chunks |
| `links` | list[{text, url}] | |

Lists stay real JSON lists in `chunks.jsonl`. Converting them into the flat values a vector store accepts is the embedding phase's job, not this one.

## 8. Layout, CLI and tests

```
chunking/
├── __init__.py
├── __main__.py      ← python -m chunking  (--rebuild, --only-stage C<n>)
├── config.py        ← all limits (Stage C1)
├── tokens.py        ← count_tokens
├── join.py          ← page map (C3)
├── segment.py       ← sections and units (C4)
├── prose.py         ← prose + tiny merge (C5)
├── tables.py        ← table chunking (C6)
├── extras.py        ← figure and footnote chunks (C7)
├── assemble.py      ← breadcrumb, IDs, metadata (C8)
└── validate.py      ← checks and report (C9)
tests/
├── test_join.py     ← page-map cases from C3
├── test_segment.py  ← unit boundaries, continued tables, header-row removal
├── test_prose.py    ← size bounds, overlap within a section only, tiny merge cases
├── test_tables.py   ← row-boundary splitting, header repetition, row-too-big guard
└── test_chunking_integration.py  ← full run on real pages.jsonl; skipped if absent
```

Unit tests use small inline fixtures and must not need the real files. Expected values used in validation (49, 212) are read from the manifest's `validation:` block, never hardcoded in pipeline code.

## 9. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Which package loads the bge-small tokenizer (e.g., `tokenizers` vs `transformers`)? | C1 |
| Q2 | Recursive splitter: `langchain-text-splitters` or a small custom one? | C5 |
| Q3 | Tiny-section threshold (tokens), chosen from the C2 list | C2 |
| Q4 | Maximum tokens for a table piece (e.g., 350 like prose, or larger up to the hard limit), chosen from C2 numbers | C2 |
| Q5 | Table rendering in chunks: Markdown table or plain labelled lines? Shown side by side with token counts | C6 |
| Q6 | Footnote `section_path`: section at the end of the page, or the page's start section? | C7 |
| Q7 | Approval of the breadcrumb and citation-label formats from examples | C8 |
| Q8 | Any difference between this plan and what the real `pages.jsonl` contains | Whenever found |

## 10. Handoff notes for the embedding phase (not to implement now)

- bge models support an optional instruction prefix on **queries** (never on documents); v1.5 was designed to work reasonably without it. Decide and test this in the embedding/retrieval phase.
- Embed `embed_text`, show `display_text`, cite with `citation_label` + `citation_url`.
- Flatten list metadata for the vector store at that stage.
