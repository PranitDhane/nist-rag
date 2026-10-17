# Source Attribution Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for **source attribution**: turning the citation keys in a validated answer into the user-facing Sources list. Work stage by stage (Section 8). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. Questions in Section 11 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

For an answered NIST question, build `sources`: one entry per cited source, each showing which `[C#]` markers point to it, a citation label, and a clickable link to the exact PDF page.

- **In scope:**
  - the shared citation formatter (label and URL), also used by `related_sections`;
  - `build_sources()`;
  - merging, ordering, the `kind` field;
  - consistency checks between the answer text and the list;
  - the `sources` field and SSE payload;
  - the history rule for markers;
  - checks with real data, including that every link opens a page containing the cited text.
- **Out of scope:** the insufficient path (already planned; there `sources` is always `[]`); graph wiring (item 5); end-to-end evaluation (item 6).
- **No LLM calls.** Attribution is deterministic code.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| SA-D1 | Markers in the answer | **Keep the `[C#]` keys as written.** No renumbering, no removal, so attribution never triggers a stream correction. |
| SA-D2 | Each list entry shows its keys | e.g., `[C1][C3]` |
| SA-D3 | Ordering | **Document, then page** |
| SA-D4 | Same label cited by several keys | **Merge into one entry** listing all the keys |
| SA-D5 | Entry contents | **Label + page link only** (plus the keys from SA-D2 and the `kind` field from SA-D6). No excerpt, no external links. |
| SA-D6 | Figures (OCR) and footnotes | Marked by a **separate `kind` field only**; the label text has no "(OCR)" or "Footnote" suffix |
| SA-D7 | Label format | `NIST AI 100-1, §5.1 Govern, pp. 22–23` |

**This also settles open questions in other plans:** insufficient-context plan Q6 (label format) is answered by SA-D7 and SA-D6. Record it in `docs/PROGRESS.md`.

## 3. Label and URL specification (the shared formatter)

**Label:** `{citation_name}, {section_part}, {page_part}`

| Part | Rule | Example |
|---|---|---|
| `citation_name` | A short document name from the manifest. **Add `citation_name` to each manifest entry** (`NIST AI 100-1`, `NIST AI 600-1`); never derive it from `display_name` by string slicing. | `NIST AI 100-1` |
| `section_part`, numbered section | `§` + the **leaf** of `section_path` (number and title) | `§5.1 Govern` |
| `section_part`, unnumbered section | The leaf title, no `§` | `Executive Summary` |
| `section_part`, appendix | `Appendix X` (+ title per Q1) | `Appendix B` |
| `section_part`, table chunk | Per **Q1**. Proposal: `§3 GOVERN 1.7` (GenAI); `§5.1 GOVERN 1` (AI RMF category) | `§3 GOVERN 1.7` |
| `section_part`, figure / footnote chunk | Same rule as prose (the section where it appears); the kind shows only in `kind` | `§5 AI RMF Core` |
| `page_part`, one page | `p. {printed}` | `p. 21` |
| `page_part`, range | `pp. {start}–{end}`, with an **en dash** | `pp. 22–23` |
| `page_part`, printed page missing (null) | `PDF p. {pdf_page}` (shouldn't happen for indexed pages; logged as a warning) | `PDF p. 26` |

**URL:** `{source_url}#page={pdf_page_start}`. It always uses the **PDF page index**, not the printed number, so the AI RMF label `p. 21` links to `#page=26`. The README must explain this, so a reviewer isn't confused by the different numbers.

**`kind`:** `prose` / `table` / `figure-ocr` / `footnote` (the same mapping as the augmentation `type` attribute and `related_sections`).

**Single source of truth:** the formatter computes labels from the structured metadata at runtime. Chunking may have already stored a `citation_label` in each chunk's metadata (from its own format approval). Stage S1 compares the two, and **Q2** decides what to do on a mismatch.

## 4. Building the Sources list

`build_sources(final_text, cited_keys, key_map, chunk_lookup) -> list[SourceEntry]`:

1. Take `cited_keys` from the **final** parse (after the fixer), so they always match `final_text`.
2. Map each key → `chunk_id` → metadata. A key with no mapping is an internal error: raise in tests, log ERROR in production.
3. Compute `label`, `url` and `kind` with the shared formatter.
4. **Merge** entries with the same `label` (SA-D4). Keys inside an entry are sorted by number (`C1` before `C3`). If merged chunks have different `kind` values, **ask Q4** at stage S2 when it first occurs. Proposal: `kind` becomes a list.
5. **Order** (SA-D3): document order from the manifest (AI RMF 1.0 first, then GenAI Profile), then `pdf_page_start`, then `pdf_page_end`, then chunk `sequence` (deterministic tie-break).
6. Only **cited** chunks appear. Chunks that were in the prompt but never cited are not listed.

**`SourceEntry`** (the API shape):

| Field | Example |
|---|---|
| `keys` | `["C1", "C3"]` |
| `label` | `NIST AI 100-1, §5.1 Govern, pp. 22–23` |
| `url` | `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf#page=27` |
| `kind` | `prose` |

**Consistency checks** (tests always; at runtime, log ERROR on failure):
- every `[C#]` in `final_text` appears in exactly one entry;
- every key in the entries appears in `final_text`;
- no entry has an empty key list;
- labels are unique after merging.

**Case B (ambiguous) answers** use the same rules: keys from all readings, one list.

## 5. API contract

- **`/chat`:** `sources: list[SourceEntry]`. It is empty for insufficient answers (insufficient plan) and for greeting and out-of-scope responses.
- **`/chat/stream`:** the `sources` event `{"sources": [...]}`. It is sent after the last `token` and after any `correction`, and before `done` (the order the API and generation plans already set). The payload shape needs approval (Q5).
- Sources are built **once**, from the final text, and are identical in both endpoints.

## 6. History rule

The assistant message stored in history is the answer text. `[C#]` keys are **local to one answer** (C2 in one answer isn't C2 in the next), so markers left in history could confuse the routing call's 2-turn window. **Q3** decides whether to strip the markers from the stored history text. Proposal: strip them; the user-facing response keeps them either way.

## 7. Project layout additions

```
attribution/
├── __init__.py
├── format.py        ← shared formatter (label, url, kind); used by sources and related_sections
├── build.py         ← build_sources() and consistency checks
└── models.py        ← SourceEntry
tests/
├── test_format.py
├── test_build_sources.py
└── test_sources_api.py
eval/
└── link_check.py    ← Stage S4: do links open pages containing the cited text?
```

---

## 8. Stages

### Stage S0 — Prerequisites

1. Confirm with the user that the generation and insufficient-context phases are built. Check whether `attribution/format.py` already exists (the insufficient plan may have created it). If it does, **extend it; don't duplicate it.**
2. Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.

**Checkpoint:** report what exists.

### Stage S1 — Formatter finalization

**Steps**
1. Add `citation_name` to both manifest entries (`NIST AI 100-1`, `NIST AI 600-1`).
2. **Ask Q1:** show the label the formatter would produce for one chunk of every kind and situation:
   - numbered section;
   - unnumbered section (Executive Summary);
   - appendix;
   - GenAI table (single part and part 1 of 2);
   - AI RMF core table;
   - figure;
   - footnote;
   - single page vs range.

   The user approves or adjusts the table, appendix and unnumbered formats.
3. Implement `label(meta)`, `url(meta)` and `kind(meta)` per Section 3.
4. **Compare** the formatter's labels with any `citation_label` already stored in chunk metadata, for **all** chunks. Report the matches and mismatches with examples. **Ask Q2:** on a mismatch, either (a) use the formatter at runtime and ignore the stored labels, or (b) re-run chunking and indexing so the stored labels match.
5. Tests:
   - every row of the Section 3 table;
   - the en dash in ranges;
   - `p.` vs `pp.`;
   - the `PDF p.` fallback;
   - the URL uses `pdf_page_start`;
   - `kind` mapping for all chunk types;
   - no "(OCR)" or "Footnote" text in any label (SA-D6).

**Done when:** tests pass; Q1 and Q2 are answered.

**Checkpoint:** show the approved example table and the comparison result.

### Stage S2 — `build_sources()`

**Steps**
1. Implement Section 4.
2. Tests:
   - single citation;
   - several citations to different documents (order: AI RMF before GenAI, then by page);
   - **two keys with the same label merge** into one entry with sorted keys;
   - a prompt chunk that wasn't cited is not listed;
   - a sibling part cited separately from part 1 (different pages, so separate entries; or same label, so merged);
   - a Case B answer;
   - the consistency checks catch: a key in the text missing from the list; a key in the list missing from the text; a key with no mapping;
   - deterministic output.

**Done when:** tests pass.

**Checkpoint:** show the sources built for 3 real answers (from the generation behaviour run), including one with a merge.

### Stage S3 — API, SSE and history

**Steps**
1. **Ask Q5:** approve the `sources` event payload (Section 5).
2. Add `sources` to the `/chat` response and the `sources` event to the stream, built once from the final text.
3. **Ask Q3:** whether to strip `[C#]` markers from the history text. Implement the answer, with a test.
4. Tests:
   - JSON and SSE carry identical sources for the same answer;
   - `sources` comes after `correction` in the stream;
   - insufficient, greeting and out-of-scope responses have `sources = []`;
   - after a correction that removed a key, the sources match the **corrected** text.

**Done when:** tests pass.

**Checkpoint:** show a `/chat` JSON response and the matching SSE transcript for one answer.

### Stage S4 — Link and label checks with real data

**Purpose:** prove every citation can be verified, by checking that each link opens a page that contains the cited text.

**Steps**
1. `eval/link_check.py`: for every chunk in `chunks.jsonl` (not just cited ones), open the PDF from `data/raw/` and extract the text of pages `pdf_page_start`–`pdf_page_end`. Normalize both texts (lowercase, collapse whitespace, remove hyphenation and ligatures as ingestion does). Check that a meaningful sample of the chunk's text (e.g., the first and last 60 normalized characters of `display_text`) appears in that page text.
2. Skip figure-OCR chunks for the text match (OCR text isn't in the text layer); check their page instead by confirming the figure caption appears on it.
3. Also confirm that each label's printed page matches the page's own printed number (from ingestion's `printed_page`).
4. Report: chunks checked, passed, failed (with chunk ID, label, URL and what failed).

**Done when:** the report exists in `eval/link_check_report.json`; any failure is shown to the user. Failures are **reported, not fixed silently**; a failure usually points to an ingestion or chunking bug.

**Checkpoint:** show the summary and every failure.

---

## 9. Testing rules

- Everything is deterministic and testable without network access or an API key (Stage S4 needs only the local PDFs).
- Never change `final_text` here: attribution reads the text; it never edits it.
- Never list uncited chunks as sources.

## 10. README / DESIGN additions

- How to read an answer: `[C#]` markers, and the Sources list with the same keys.
- Why the keys are kept rather than renumbered (renumbering would require correcting almost every streamed answer).
- Why the label shows the printed page while the link uses the PDF page index (`p. 21` → `#page=26`).
- The link check: every citation link was verified to open a page containing the cited text (show the pass rate).

## 11. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Label format for tables, appendices and unnumbered sections (shown with real examples) | S1 |
| Q2 | If stored `citation_label` values differ from the formatter: use the formatter at runtime, or re-run chunking and indexing? | S1 |
| Q3 | Strip `[C#]` markers from the text stored in conversation history? | S3 |
| Q4 | If merged chunks have different `kind` values: make `kind` a list, or something else? | S2 (only if it occurs) |
| Q5 | Approval of the `sources` SSE payload | S3 |
| Q6 | Any difference between this plan and the behaviour of the built phases | Whenever found |

## 12. Handoff to item 5

- The graph's attribution step calls `build_sources()` after `generate_node` when `outcome == ANSWERED`, and stores `sources` in the state for the API.
- `related_sections` (insufficient path) uses the same `format.py`.
