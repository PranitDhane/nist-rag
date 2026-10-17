# Ingestion Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first. It contains the working rules. This file is the specification. Work stage by stage (Section 6). Each stage has the same blocks: **Purpose → Input → Output → Steps → Done when → Checkpoint**. A stage is finished only when every "Done when" item is true **and** the user has confirmed the Checkpoint.

---

## 1. Goal and boundary

Turn the two official NIST PDFs into **validated, cleaned, page-level records** in `data/processed/pages.jsonl`, plus a report `data/processed/ingest_report.json`.

- **In scope:** acquire, profile, extract, classify, figure OCR, clean, structure annotation, links, assemble, validate.
- **Out of scope:** chunking, embeddings, vector store, retrieval, LangGraph, UI.
- Chunking (a later phase) will read **only** `pages.jsonl` and will never open the PDFs.

## 2. Core principles

1. **The PDFs are the source of truth.** Every page number and link must trace back to the official PDF.
2. **Markdown is a regenerated intermediate format.** It is never edited by hand.
3. **Every correction is a named, tested rule in code.** No manual fixes; the reviewer runs everything from a fresh download.
4. **Page metadata (page numbers, links, fonts) comes from PyMuPDF, never from the Markdown converter.**
5. **Work page by page.** Never merge pages into one Markdown file.
6. **Everything in `data/processed/` can be regenerated** with one command.

## 3. Environment

| Item | Value |
|---|---|
| Python | **OPEN QUESTION: ask the user (3.11 or 3.12)** |
| Environment | `pip` + `venv` |
| Dependencies | `pymupdf`, `pymupdf4llm`, `pytesseract`, `pillow`, `pyyaml`, `python-dotenv`, `requests`, `pytest` |
| Version pinning | Exact versions (`==`) in `requirements.txt`. The investigation used `pymupdf==1.28.2` and `pymupdf4llm==1.28.2`; propose these and confirm with the user. |
| Tesseract | **Optional** system program. If it is missing, ingestion still runs, with a warning (Stage 6). |

## 4. Project layout

```
nist-rag/
├── CLAUDE.md
├── docs/
│   ├── INGESTION_PLAN.md
│   └── PROGRESS.md
├── data/                      ← all git-ignored
│   ├── raw/                   ← PDFs (source of truth)
│   └── processed/
│       ├── stages/            ← per-stage cache files (Section 5.3)
│       ├── debug/figures/     ← rendered figure PNGs for visual checks
│       ├── pages.jsonl        ← FINAL output
│       └── ingest_report.json
├── ingest/
│   ├── __init__.py
│   ├── __main__.py            ← CLI: python -m ingest
│   ├── sources.yaml           ← document manifest + validation expectations
│   ├── config.py              ← paths/settings, .env overridable
│   ├── models.py              ← dataclasses for records
│   ├── download.py            ← Stage 2
│   ├── profile.py             ← Stage 3
│   ├── extract.py             ← Stage 4
│   ├── classify.py            ← Stage 5
│   ├── figures.py             ← Stage 6
│   ├── clean.py               ← Stage 7 (one function per rule)
│   ├── structure.py           ← Stage 8
│   ├── links.py               ← Stage 9
│   └── validate.py            ← Stage 10
├── tests/
│   ├── test_clean.py
│   ├── test_links.py
│   ├── test_structure.py
│   └── test_integration.py    ← skipped if PDFs absent
├── .env.example
├── .gitignore                 ← .venv/, data/, __pycache__/, .env
├── requirements.txt
└── README.md
```

## 5. Reference data

### 5.1 Verified facts: NIST AI 100-1 (AI RMF 1.0)

Measured on local copies. **Re-verify after the official download** and report any difference.

| Property | Value |
|---|---|
| doc_id / filename | `ai_rmf_1_0` / `NIST.AI.100-1.pdf` |
| URL | `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf` |
| Pages | 48 |
| PDF Title metadata | `Artificial Intelligence Risk Management Framework (AI RMF 1.0)` |
| Page labels | Present. `page.get_label()` gives the printed number: PDF p.6 → `1`, p.25 → `20`, p.40 → `35`. PDF p.4 → `i`, p.5 → `ii`. |
| Running header | Two lines, exactly `NIST AI 100-1` and `AI RMF 1.0`, among the **first 3 non-empty lines** of PDF pp.5–47. **Order differs** between raw text and Markdown, so match either order. |
| Running footer | Last non-empty line, matching `^Page [0-9ivx]+$` (pp.5–47); bare `i` on p.4. |
| Footnotes | **None.** Zero superscript spans in the whole document. |
| Bookmarks | Unreliable (Appendices A–C all point to one page). **Do not use.** |
| Figures (image-only text) | See table below |
| Tables 1–4 | GOVERN/MAP/MEASURE/MANAGE categories and subcategories; span pages; leftovers: `Continued on next page`, captions ending `(Continued)`, a plain-text line `Categories Subcategories` |
| Appendix headings | Split over two lines (`Appendix A:` then the title on the next line) |
| Links | 21 external (12 unique), 82 internal (named destinations), 2 bare URLs |

**AI RMF figures** (bbox in PDF points, from `page.get_image_info()`):

| Figure | PDF page | Printed | Image bbox | Rotation |
|---|---|---|---|---|
| Fig. 1 (harms) | 10 | 5 | [67, 25, 556, 285] | none |
| Fig. 2 (lifecycle wheel) | 15 | 10 | [112, 71, 500, 371] | none (text curves along arcs) |
| Fig. 3 (AI actors table) | 16 | 11 | [67, 50, 556, 683] | **90°**: transform `[0, -632.41, 488.87, 0]` |
| Fig. 4 (trustworthiness) | 17 | 12 | [79, 257, 533, 380] | none |
| Fig. 5 (core functions) | 25 | 20 | [77, 185, 535, 550] | none |

Detect rotation from the image transform matrix: non-zero `b`/`c` with zero `a`/`d` means a 90° rotation.

**AI RMF page types (expected; used only for validation):**

| PDF pages | page_type | Notes |
|---|---|---|
| 1 | cover | |
| 2–3 | front_matter | title page; notice + "Update Schedule and Versions" |
| 4–5 | toc | Table of Contents + List of Tables (p.4); List of Figures (p.5) |
| 6–39 | body | Executive Summary … Section 6 |
| 40–47 | appendix | Appendices A–D |
| 48 | back_cover | |

**AI RMF expected section IDs** (validation only): `Executive Summary`, `Part 1`, `1`, `1.1`, `1.2`, `1.2.1`, `1.2.2`, `1.2.3`, `1.2.4`, `2`, `3`, `3.1`–`3.7`, `4`, `Part 2`, `5`, `5.1`, `5.2`, `5.3`, `5.4`, `6`, `Appendix A`, `Appendix B`, `Appendix C`, `Appendix D`.

### 5.2 Verified facts: NIST AI 600-1 (GenAI Profile)

| Property | Value |
|---|---|
| doc_id / filename | `genai_profile` / `NIST.AI.600-1.pdf` |
| URL | `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf` (the assignment links a landing page; this is the PDF) |
| Pages | 64 |
| PDF Title metadata | `Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile` |
| Page labels | **None** (all empty) |
| Printed page | First non-empty line of PDF pp.5–64 is a bare number. **printed = pdf_page − 4.** Spot checks: p.5 → 1, p.9 → 5, p.31 → 27, p.58 → 54, p.64 → 60. PDF pp.1–4 → `null`. |
| Running header/footer | Only the page-number line at the top. Remove it **only if** it equals the computed printed page. |
| Fonts | Body text 11.0pt; table text 10.0pt; footnote markers in text 7.0pt; footnote numbers at the page bottom 6.6pt; footnote text 10.0pt |
| Footnote markers | Raised digits attached to words (`AI Actor14`, `contexts.9,10`). In pymupdf4llm Markdown they appear as `<sup>9,10</sup>`. |
| **Superscript-flag trap** | On PDF p.17, a normal **10pt table cell** also has the superscript flag (`flags & 1`) set and appears in Markdown as `<sup>Protocols are put in place…</sup>`. **The flag alone is not reliable.** |
| Footnote blocks | Bottom of page (e.g., p.10: numbers at y≈670 and y≈695 of 792) |
| Bookmarks | Contain junk entries (`1.`, `2.`). **Do not use.** |
| Ligatures | Present in raw text (`ﬁ`) |
| Section 2 | Risks 2.1–2.12; each ends with a `Trustworthy AI Characteristic:` line |
| Section 3 | Suggested-action tables: **49 subcategory tables** (header like `GOVERN 1.7: …`), **212 unique action IDs** (`(GV|MP|MS|MG)-\d+\.\d+-\d{3}`). Each table: subcategory header → rows (Action ID / Suggested Action / GAI Risks) → `AI Actor Tasks:` line. About **19 pages begin mid-table** (crude measurement; confirm). |
| Action IDs in prose | The intro on PDF p.17 mentions `GV-1.1-001` and `GV-1.1-002` as examples, split across a line break (`GV-` / `1.1-001`). So an ID can appear **more than once** in the text. |
| Garbled cell | Markdown contains `AI Actor Tasks   i` (should read `AI Actor Tasks:`) |
| Links | 255 external (142 unique), 16 internal, 107 bare URLs; 104 links are in Appendix B References |

**GenAI page types (expected; used only for validation):**

| PDF pages | Printed | page_type | Notes |
|---|---|---|---|
| 1 | – | cover | |
| 2–3 | – | front_matter | title page; "About AI at NIST", policies, publication history, contact |
| 4 | – | toc | |
| 5–50 | 1–46 | body | Sections 1–3 |
| 51–57 | 47–53 | appendix | Appendix A |
| 58–64 | 54–60 | references | Appendix B References — **not index-eligible** |

There is no back cover.

**GenAI expected section IDs** (validation only): `1`, `2`, `2.1`–`2.12`, `3`, `Appendix A`, `A.1`, `A.1.1`–`A.1.8`, `Appendix B`.

### 5.3 pymupdf4llm behaviour (observed with v1.28.2)

- Headings are detected, but **levels are inconsistent** (GenAI risks come out as `####`, Appendix A as `###`). Hierarchy must come from section numbers.
- Headings are often wrapped in bold: `#### **2.2. Confabulation**`.
- Tables become Markdown tables with `<br>` inside cells; hyphenated line breaks survive inside cells (`or-<br>ganizational`).
- **262** `<u>`/`<sup>` tags in the GenAI output.
- The running header/footer text stays in the output.
- **It runs Tesseract OCR automatically** on image pages (about 20 s per document). This must be turned off.
- `to_markdown` shows only `*args, **kwargs`. **Look up the real option names** for "per-page output" and "disable OCR" in the installed version's documentation or source before using them, and tell the user what you found.

### 5.4 Per-stage cache files

Each stage writes its output to `data/processed/stages/NN_<stage>.jsonl` (or `.json` for document-level data), and the next stage reads it. This makes `--from-stage` possible and each stage inspectable.

| Stage | Cache file |
|---|---|
| 3 Profile | `03_profile.json` |
| 4 Extract | `04_extract.jsonl` |
| 5 Classify | `05_classify.jsonl` |
| 6 Figures | `06_figures.jsonl` |
| 7 Clean | `07_clean.jsonl` |
| 8 Structure | `08_structure.jsonl` |
| 9 Links | `09_links.jsonl` |
| 10 Assemble | `pages.jsonl` + `ingest_report.json` |

---

## 6. Stages

### Stage 1 — Manifest and configuration

**Purpose:** define the documents and settings in one place, so no other code depends on filenames or paths.

**Input:** none. **Output:** `ingest/sources.yaml`, `ingest/config.py`, `.env.example`, `requirements.txt`, `.gitignore`, package skeleton, `docs/PROGRESS.md`.

**Steps**
1. `sources.yaml` has a `documents:` list. Each entry has: `doc_id`, `display_name`, `url`, `filename`, `sha256` (empty for now), `expected_pages`, `expected_title`, `page_number_strategy` (`page_label` for AI RMF; `first_line_offset` with `offset: 4` for GenAI).
2. Each entry has a `validation:` block holding the expected page-type ranges, expected section IDs, and spot-check page numbers from Section 5. This block is **read only by `validate.py`**.
3. `config.py` loads paths (`DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `STAGES_DIR`) with `.env` overrides, and loads the manifest into dataclasses from `models.py`.
4. Create the empty modules from Section 4 with docstrings describing each module's job.

**Done when:** `python -c "from ingest.config import load_manifest; print(load_manifest())"` prints both documents; `requirements.txt` is pinned; `pytest -q` runs (no tests yet is fine).

**Checkpoint:** show `sources.yaml`, `requirements.txt`, and the folder tree. Ask the Python version question if not answered yet.

### Stage 2 — Acquire

**Purpose:** get the official PDFs into `data/raw/` and prove they are the right files.

**Input:** manifest. **Output:** PDFs in `data/raw/`, acquisition results in the report.

**Steps**
1. **Download mode** (default): stream from `url` with a timeout and a User-Agent header, writing to a temp file. Move it into place only after the checks pass.
2. **Local mode** `--from-dir <path>`: find each file by its manifest filename **or** a case-insensitive match on the `100-1` / `600-1` part of the name (the user's copies are named `nist_ai_100-1.pdf` and `NIST_AI_600-1.pdf`). Copy it to the manifest filename.
3. Checks, in order:
   a. the first 4 bytes are `%PDF`;
   b. the page count equals `expected_pages`;
   c. the PDF Title metadata equals `expected_title`;
   d. SHA-256: if the manifest value is empty, print the computed hash and tell the user to pin it; if present, it must match.
4. **Idempotent:** if a file already exists and passes a–d, skip it and log "already present".
5. On failure: one retry for downloads, then stop with a clear message (which check failed, expected vs actual).

**Done when:** both modes work; running twice skips the second time; a fake file (e.g., an HTML file renamed to `.pdf`) fails check (a).

**Checkpoint:** show the console output of both modes, the computed SHA-256 values (for the user to pin), and the result of the fake-file test.

### Stage 3 — Profile and environment check

**Purpose:** record each document's properties and the machine's OCR capability, so later stages make decisions from data rather than assumptions.

**Input:** PDFs. **Output:** `stages/03_profile.json`.

**Steps**
1. **Tesseract:** detect it through `pytesseract`; record `{"available": bool, "version": str|null}`. If unavailable, log a WARNING saying figure OCR will be skipped.
2. **For each document:** page count, Title, text layer present (non-empty text on body pages), page labels present (any non-empty `get_label()`), and the configured page-number strategy.
3. **Images per page:** from `page.get_image_info(xrefs=True)`, record `bbox`, `transform`, `rotated` (true if transform `a == 0 and d == 0`), and the nearest caption below or beside it whose text starts with `Fig. N.` (record `figure_number` and the caption text).
4. Skip tiny images (both sides < 50pt) such as logos, but log how many were skipped.

**Done when:** the profile matches Section 5 for AI RMF figures (5 figures, Fig. 3 rotated, captions found); the GenAI image findings are reported, whatever they are.

**Checkpoint:** show a summary table of figures per document (page, figure number, bbox, rotated, caption start), plus the Tesseract status.

### Stage 4 — Page extraction

**Purpose:** get each page's Markdown, plus the page facts that the Markdown cannot be trusted for.

**Input:** PDFs, profile. **Output:** `stages/04_extract.jsonl`, one record per page.

**Steps**
1. **Markdown:** use pymupdf4llm **per page**, with its **OCR disabled**. Look up the real option names first (Section 5.3) and tell the user which you used.
2. **Page facts** from PyMuPDF:
   - `pdf_page` (1-based).
   - `printed_page`: AI RMF uses `get_label()`, or null if empty. GenAI uses the first non-empty line **if** it is digits **and** equals `pdf_page − 4`, otherwise null.
   - `page_height`.
3. **Font summary** from `page.get_text("dict")`:
   - `body_font_size`: the most common span size weighted by character count.
   - `marker_candidates`: spans whose text matches `^\d+(,\d+)*$` and whose size is ≤ 0.75 × `body_font_size`, with bbox and size.
   - `bottom_blocks`: text blocks whose top y is > 75% of `page_height`, with bbox, first-span text, first-span size, and full text.
4. **Raw links:** everything from `page.get_links()` (kind, uri, rect). Do not filter yet.

**Done when:** all 112 pages are present; printed pages match the spot checks in Section 5; GenAI p.10 has marker candidates `9,10` and bottom blocks starting with `9` and `10`; the whole of AI RMF has zero marker candidates.

**Checkpoint:** show one full record (abbreviated Markdown) for AI RMF p.25, GenAI p.10, and GenAI p.21, plus the printed-page spot-check results.

### Stage 5 — Page classification

**Purpose:** label every page so chunking knows what to index. Nothing is deleted.

**Input:** `04_extract.jsonl`. **Output:** `stages/05_classify.jsonl` (adds `page_type`, `index_eligible`).

**Steps**
1. Classify with **content rules** (do not read the `validation:` ranges here):
   - **cover**: page 1.
   - **toc**: page contains `Table of Contents`, `List of Tables` or `List of Figures` as a heading.
   - **front_matter**: pages before the first body heading that are not cover/toc.
   - **body**: from the first body heading (`Executive Summary` for AI RMF; `1. Introduction` for GenAI) until the first appendix heading.
   - **appendix**: from the first `Appendix` heading onward.
   - **references**: from a heading `Appendix B. References` onward (GenAI).
   - **back_cover**: the last page, if it has fewer than 30 words and no heading.
2. `index_eligible = page_type in {"body", "appendix"}`.

**Done when:** the result matches the expected ranges in Section 5 for both documents (verified with a quick comparison against the `validation:` block — comparison only, the classification must not read it).

**Checkpoint:** show a compact table: doc, page range, page_type, index_eligible.

### Stage 6 — Figure OCR (optional at runtime)

**Purpose:** recover the text that exists only inside figure images, clearly labelled as OCR.

**Input:** profile, `05_classify.jsonl`. **Output:** `stages/06_figures.jsonl` (figure records per page).

**Steps**
1. Only figures on pages with `index_eligible = true`.
2. **If Tesseract is unavailable:** write a figure record with `caption`, `text: null`, `method: "caption_only"`, `ocr_status: "skipped_unavailable"`, and stop here.
3. **If available:** render **only the image bbox** (`page.get_pixmap(clip=bbox, dpi=300)`). If `rotated`, rotate the image so text is horizontal (pick the direction from the transform signs). Save the PNG to `data/processed/debug/figures/<doc_id>_fig<N>.png`.
4. Run Tesseract and compute mean word confidence (ignore confidence −1).
5. Record `figure_number`, `pdf_page`, `caption`, `text`, `method: "ocr"`, `confidence`, `tesseract_version`, `ocr_status: "ok"`.
6. **Threshold:** for this first run, record the confidences but **do not filter**. The user chooses the threshold at the Checkpoint. Afterwards, below threshold → `text: null`, `ocr_status: "below_threshold"`.
7. Figure text **never** goes into page Markdown.

**Done when:** 5 AI RMF figure records exist; the debug PNGs exist and are upright; the run works with Tesseract hidden (simulate it as unavailable).

**Checkpoint:** show for each figure: confidence, the first ~200 characters of OCR text, and the PNG path. **Ask the user to choose the confidence threshold.** Expected quality: Figs 1, 3, 4 and 5 readable; Fig. 2 probably poor.

### Stage 7 — Cleaning

**Purpose:** remove noise from page Markdown using named, tested rules, applied in a fixed order.

**Input:** `05_classify.jsonl` (Markdown + font summary). **Output:** `stages/07_clean.jsonl` (cleaned `markdown`, `footnotes`, `cleaning_log`).

Apply the rules **in this exact order**. Each is its own function `rule(page) -> page`, appends its name to `cleaning_log` when it changes something, and has unit tests.

| # | Rule | Behaviour |
|---|---|---|
| 1 | `normalize_unicode` | NFKC (ligatures `ﬁ` → `fi`), non-breaking and other unusual spaces → normal space. |
| 2 | `strip_running_headers_footers` | AI RMF: remove lines that are exactly `NIST AI 100-1` or `AI RMF 1.0` (after stripping `#`, `*` and spaces) **only within the first 3 non-empty lines**; remove the last non-empty line if it matches `^Page [0-9ivx]+$`. GenAI: remove the first non-empty line **only if** it is digits equal to `printed_page`. |
| 3 | `extract_footnotes` | GenAI: a footnote block is a `bottom_blocks` entry whose first span is digits with size ≤ 0.75 × body size. Move each block's text (number + text) into `footnotes: [{number, text}]` and remove it from the Markdown by matching its text (normalize whitespace on both sides). If a match fails, keep the text in place and log a WARNING with page and footnote number. |
| 4 | `strip_footnote_markers` | Remove `<sup>…</sup>` **only if** its inner text matches `^\s*\d+(\s*,\s*\d+)*\s*$`. Cross-check: the number removed should equal the page's `marker_candidates` count, otherwise WARNING. |
| 5 | `unwrap_markup_tags` | Remove the remaining `<u>`, `</u>`, `<sup>`, `</sup>` tags, **keeping their inner text** (fixes the false-superscript table cell on PDF p.17). |
| 6 | `fix_line_hyphenation` | Joins across `-<br>` and `-\n`. If the character before the hyphen is a **letter** and the next character is a **lowercase letter** → join and **drop** the hyphen (`or-<br>ganizational` → `organizational`), **unless** the joined word is on the protect-list, in which case keep the hyphen. If the next character is a **digit or uppercase letter** → join and **keep** the hyphen (`GV-` + `1.1-001` → `GV-1.1-001`; `human-` + `AI` → `human-AI`). |
| 7 | `strip_table_leftovers` | Remove lines `Continued on next page`, repeated table captions ending with `(Continued)`, and the **plain-text** line `Categories Subcategories`. **Keep real Markdown table header rows.** Normalize the garbled `AI Actor Tasks   i` cell to `AI Actor Tasks:`. Before writing this rule, grep the Stage 5 output for these patterns and show the user the exact matches and counts. |
| 8 | `collapse_whitespace` | Collapse 3+ blank lines into 2; trim trailing spaces. |

**Protect-list for rule 6:** before finalizing, scan both documents for every `letter-<br>letter` / `letter-\nletter` join. Show the user the full list of candidate words with a proposed keep/drop decision, and let the user approve it. Store the approved list in `ingest/hyphen_protect.txt`.

**Done when:** all rule unit tests pass, including these negative cases:
- `ISO 31000`, `Act of 2020`, `GPT-4` and `2.12` are untouched by rules 4 and 6;
- `GV-` + newline + `1.1-001` becomes `GV-1.1-001`;
- the p.17 table cell text survives rule 5 without tags;
- AI RMF pages show **zero** changes from rules 3 and 4.

No header/footer text remains on any page.

**Checkpoint:** show rule hit counts per document; before/after diffs for GenAI p.10 (footnotes), GenAI p.17 (false superscript plus the `GV-` join), and AI RMF p.28 (table leftovers); the extracted footnotes for p.10; and any WARNINGs.

### Stage 8 — Structure annotation

**Purpose:** label sections, tables and key values on each page. It labels structure only; it does not chunk.

**Input:** `07_clean.jsonl`. **Output:** `stages/08_structure.jsonl`.

**Steps**
1. **Heading detection:** take Markdown heading lines, plus bold-only lines that look like headings. Strip `#`, `*` and extra spaces, then match:
   - numbered: `^(\d+(?:\.\d+)*)\.?\s+(.+)$` → `section_id` = the number (no trailing dot).
   - appendix: `^Appendix ([A-D])[.:]\s*(.*)$` → `section_id` = `Appendix X`. If the title is empty (AI RMF splits it over two lines), take the next non-empty line as the title.
   - sub-appendix: `^([A-D]\.\d+(?:\.\d+)*)\.\s+(.+)$`.
   - named: `Executive Summary`, `Part 1: …` → `Part 1`, `Part 2: …` → `Part 2`.
   - **Ignore** numbered-looking lines inside tables and numbered list items, e.g. `1. Be risk-based…` (Appendix D) or `4. Presenting AI system…` (Appendix C). Accept a numbered heading only if it came from a Markdown heading line or a bold-only line.
2. **Level:** derived from the `section_id`: `2` → 1, `2.2` → 2, `1.2.3` → 3; `Appendix X` → 1, `A.1` → 2, `A.1.1` → 3; `Part N` → 0; `Executive Summary` → 1.
3. **`section_path_start`:** the section path in effect at the **top** of the page, carried from the previous page (e.g., `["Part 2", "5 AI RMF Core", "5.1 Govern"]`).
4. **Tables:**
   - `table_markers` — GenAI: lines matching `^(GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+:` → subcategory table starts. AI RMF: `Table N:` captions that do **not** end with `(Continued)`.
   - `continues_table` — true if the first content line of the page is a table row or an action-ID row, with no table marker before it.
5. **Entities:** `action_ids` (only IDs that are the **first cell of a table row**); `subcategories` (e.g., `GOVERN 1.7`); `gai_risks` (from the GAI Risks column, split on `;`); `trustworthy_characteristics` (from `Trustworthy AI Characteristic:` lines); `captions` (Fig./Table captions).

**Done when:**
- every expected section ID in Section 5 is detected **exactly once** per document, and no unexpected ones appear (any extras are listed for the user);
- GenAI subcategory table markers = **49**;
- the set of table-row action IDs = **212** unique, each appearing as a table-row ID exactly once;
- the number of `continues_table` pages is reported (expected about 19).

**Checkpoint:** show the detected heading tree per document; counts of table markers, action IDs and `continues_table` pages; and any extra or missing section IDs.

### Stage 9 — Links

**Purpose:** produce clean, safe external links with readable anchor text, tagged by where they appear.

**Input:** `08_structure.jsonl` (raw links, footnotes, page_type). **Output:** `stages/09_links.jsonl`.

**Steps**
1. Keep links whose URI scheme is `http`, `https` or `mailto`. Drop internal links (goto/named) and **reject every other scheme**, logging each rejected one.
2. **Anchor text:** the words from `page.get_text("words")` whose bbox lies at least 60% inside the link rect (by area), in reading order.
3. **Merge** consecutive link annotations with the same URI (a link split over two lines) into one link with joined anchor text.
4. **Deduplicate** per page by (uri, anchor text).
5. **Bare URLs** in the Markdown that have no annotation: strip trailing `.,;:)` and rejoin URLs broken across a line break. If a bare URL matches an annotation's URI (ignoring trailing punctuation), the **annotation URI wins**.
6. **Location:** `references` if page_type is `references`; `footnote` if the rect lies within a footnote block's bbox; otherwise `body`. Also record `section_path` at the link's position.
7. Record each link as `{text, url, location, section_path, pdf_page}`.

**Done when:** link tests pass (two-line merge, punctuation stripping, `javascript:` rejection); counts per document per location are reported; no anchor text is empty for `body` links (empty ones are listed).

**Checkpoint:** show link counts per document by location; 10 sample body links with anchor text; and every link whose anchor text looks cut off (starts or ends mid-word).

### Stage 10 — Assemble and validate

**Purpose:** write the final output and prove it is correct.

**Input:** all stage caches. **Output:** `data/processed/pages.jsonl`, `data/processed/ingest_report.json`, a console summary.

**`pages.jsonl` schema (one object per page):**

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Manifest ID |
| `display_name` | str | For citations |
| `source_url` | str | Official PDF URL (for `#page=` links) |
| `pdf_page` | int | 1-based physical page |
| `printed_page` | str \| null | Printed number (`"20"`, `"ii"`) |
| `page_type` | str | Stage 5 |
| `index_eligible` | bool | Stage 5 |
| `section_path_start` | list[str] | Stage 8 |
| `headings` | list[{section_id, title, level}] | Stage 8 |
| `markdown` | str | Cleaned text (no footnotes, no figure OCR) |
| `footnotes` | list[{number, text, links}] | Stage 7 + 9 |
| `figures` | list[{figure_number, caption, text, method, confidence, ocr_status}] | Stage 6 |
| `continues_table` | bool | Stage 8 |
| `table_markers` | list[str] | Stage 8 |
| `entities` | {action_ids, subcategories, gai_risks, trustworthy_characteristics, captions} | Stage 8 |
| `links` | list[{text, url, location, section_path}] | Stage 9 |
| `cleaning_log` | list[str] | Stage 7 |

**Hard checks (the run exits non-zero if any fail):**
1. Page counts: 48 and 64; 112 records in total.
2. Printed-page spot checks from Section 5 pass.
3. Page types match the expected ranges in Section 5.
4. No page contains the AI RMF header/footer strings or a leftover GenAI page-number line.
5. All expected section IDs are detected exactly once per document.
6. 49 GenAI subcategory table markers; 212 unique table-row action IDs, each exactly once as a table-row ID.
7. AI RMF has 5 figure records, each with a caption.
8. No `body` page has empty `markdown`.
9. Every link URL starts with `http://`, `https://` or `mailto:`.
10. GenAI pages 58–64 have `index_eligible = false`.
11. AI RMF has zero footnotes and zero footnote-marker removals.

**Soft warnings (reported; the run continues):** OCR skipped or below threshold; footnote match failures; marker-count mismatches; body pages under 80 words; leftover `<`…`>` tags; anchor texts that look cut off.

**Report:** the top of `ingest_report.json` holds the OCR status (available, version, per-figure status), then per-document counts, check results, warnings, rule hit counts, and run time.

**Done when:** all hard checks pass on a clean run (`python -m ingest --rebuild`), and the integration test passes.

**Checkpoint:** show the console summary and the first 40 lines of the report.

---

## 7. CLI

| Command | Behaviour |
|---|---|
| `python -m ingest` | Run all stages; acquisition skips files that pass its checks |
| `python -m ingest --rebuild` | Ignore all cached stage outputs (PDFs that pass checks are kept) |
| `python -m ingest --from-dir <path>` | Acquire from local PDFs |
| `python -m ingest --from-stage <3-10>` | Rerun from that stage using cached earlier outputs; error clearly if a needed cache is missing |
| `python -m ingest --only-stage <n>` | Run one stage (for development) |

Exit code is 0 on success, non-zero on any hard-check failure.

## 8. Testing

- `tests/test_clean.py`: every rule in Stage 7, with positive **and** negative cases (Stage 7 "Done when").
- `tests/test_links.py`: two-line merge, punctuation stripping, annotation-wins, scheme rejection.
- `tests/test_structure.py`: heading regexes (including the two-line appendix title and numbered list items that are **not** headings), level derivation, table-row action-ID extraction.
- `tests/test_integration.py`: full pipeline on the real PDFs, asserting the hard checks; **skipped** if the PDFs are absent; the OCR part **skipped** if Tesseract is absent.
- Unit tests use small inline text fixtures. They must not need the PDFs or Tesseract.

## 9. README sections this phase must produce

Setup (Python version, venv, pip); optional Tesseract install per OS and what changes without it; how to download and process (`python -m ingest`, `--from-dir`); ingestion design decisions (Section 11); known limitations (Section 12).

## 10. Open questions — ask, do not assume

1. Python version (3.11 or 3.12)? Ask in Stage 1.
2. Confirm the dependency versions. Stage 1.
3. Pin the SHA-256 values. Stage 2.
4. OCR confidence threshold. Stage 6.
5. Hyphenation protect-list approval. Stage 7.
6. Any difference between Section 5 and the official downloads. Whenever found.

## 11. Decision log (feeds DESIGN.md)

| Decision | Choice | Reason |
|---|---|---|
| Source of truth | Official NIST PDFs | Citations and page links must match the published documents |
| Markdown | Regenerated, per page | Heading structure and table rows; never hand-edited; keeps page numbers |
| PDF location | `data/raw/`, git-ignored, downloaded by script | Assignment requires download as part of the app; reproducible |
| Document identity | Manifest: doc_id, checksum, page count, title | No filename dependence; detects revised documents |
| Page numbers | Both `pdf_page` and `printed_page` | Printed for human citations; PDF index for `#page=` links |
| Sections | Headings plus section numbering, not bookmarks | Bookmarks in both PDFs contain errors |
| Figures | Targeted OCR, stored separately, labelled; Tesseract optional | Recovers figure text without a hard system dependency |
| Footnotes | Markers removed (font size + digits, not the flag alone); text moved to a field | Clean text flow; the superscript flag gives false positives |
| References appendix | Parsed, kept, not index-eligible | Bibliography titles match many queries but answer none |
| Internal links | Dropped | Navigation only |
| Cleaning | Ordered, named, tested rules | Reproducible; explainable |
| Validation | Expectations in the manifest, used only by the validator | Checks stay independent of the logic they test |
| Environment | pip + venv, pinned | Simplest for a reviewer |

## 12. Known limitations

- Figure 2's curved text will probably not OCR usefully.
- Figure text in answers depends on whether Tesseract is installed (and its version).
- Footnote markers are no longer linked to their footnote text.
- The References appendix is not retrievable.
- The cleaning rules and expectations are tuned to these two documents; a new document needs its own manifest entry and possibly new rules.
