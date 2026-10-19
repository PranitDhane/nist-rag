# Embedding & Indexing Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for the **embedding and indexing phase**. Work stage by stage (Section 6). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. A stage is finished only when every "Done when" item is true **and** the user has confirmed the Checkpoint. Questions in Section 9 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

Turn `data/processed/chunks.jsonl` (chunking output) into two **persisted, validated search indexes** built from the **same chunk set**:

1. a **Chroma** vector index (dense search), and
2. a **BM25** keyword index (sparse search),

plus an **index manifest** that records exactly how they were built, and **loader functions** that the retrieval phase will reuse.

- **In scope:** embedding model setup, metadata preparation, writing Chroma, building and saving BM25, loaders, validation, and smoke checks.
- **Out of scope:** retrieval logic (hybrid fusion, re-ranking, query rewriting), LangGraph, LLM calls, UI. Smoke checks here only confirm that the indexes respond sensibly; they are **not** retrieval evaluation.
- This phase never opens the PDFs or `pages.jsonl`; everything comes from `chunks.jsonl`.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| E-D1 | Embedding model | `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token input limit) |
| E-D2 | Embedding library | LangChain `HuggingFaceEmbeddings` (package `langchain-huggingface`, which uses `sentence-transformers` underneath) |
| E-D3 | Vector store | **Chroma**, persistent, local |
| E-D4 | Chroma access | Through the **LangChain wrapper** (`langchain-chroma`) |
| E-D5 | Collection layout | **One collection** for both documents; `doc_id` is a metadata field used as a filter |
| E-D6 | Keyword index | **BM25**, built in this phase from the same chunks |
| E-D7 | BM25 persistence | Save the **tokenized corpus as JSON**; **rebuild** the BM25 object at load time. **No pickle.** |
| E-D8 | Text that gets embedded | `embed_text` (breadcrumb + content); `display_text` is stored for display |

## 3. Facts and requirements that follow from these decisions

These are properties of the chosen tools, not new choices. Claude Code must still **verify each one against the installed versions** and report what it finds.

1. **Normalization:** the bge model card recommends normalized embeddings with cosine similarity. `HuggingFaceEmbeddings` does **not** normalize unless told to (`encode_kwargs={"normalize_embeddings": True}`). Verify that stored vectors have a norm ≈ 1.0.
2. **Distance metric:** Chroma's default distance is **L2**, not cosine. The collection must be **created** with cosine distance, because it can't be changed afterwards. The way to set it differs between Chroma versions (older: collection metadata `hnsw:space`; newer: a configuration object). **Check the installed version's documentation and tell the user which mechanism was used.** Verify it by reading the collection's settings back.
3. **Silent truncation:** `sentence-transformers` cuts input at the model's `max_seq_length` without any warning. Chunking already guarantees `embed_text` ≤ 512 tokens. This phase **rechecks** each chunk's token count with the model's own tokenizer before embedding, and **stops** on any violation (never truncates silently). It also reports the loaded model's `max_seq_length`.
4. **Chroma metadata limits:** values must be `str`, `int`, `float` or `bool`. **`None` is not allowed**, and lists may not be supported in the installed version. Check this and report it (Stage E3).
5. **Model consistency:** queries at retrieval time **must** use exactly the same model, revision and normalization as the documents. The index manifest records them, and the loader refuses to load on a mismatch.
6. **Query instruction prefix:** bge models support an optional instruction prefix for **queries only**. This phase embeds **documents only, with no prefix**. Whether queries use a prefix is a retrieval-phase decision; the manifest records `document_prefix: none`.
7. **Single source of chunks:** Chroma and BM25 must contain **identical sets of `chunk_id`s**. Validated in Stage E6.

## 4. Project layout additions

```
indexing/
├── __init__.py
├── __main__.py          ← CLI: python -m indexing
├── config.py            ← model, paths, collection name, batch size, device, BM25 settings
├── embedder.py          ← builds the HuggingFaceEmbeddings instance (E2)
├── metadata.py          ← chunk → Chroma metadata conversion (E3)
├── vector_store.py      ← create/write the Chroma collection (E4)
├── bm25_index.py        ← tokenizer, build, save, load (E5)
├── loaders.py           ← load_vector_store(), load_bm25(), load_chunk_lookup(); used by retrieval later
├── manifest.py          ← writes and checks index_manifest.json
└── validate.py          ← checks, smoke checks, report (E6)
data/
├── chroma/              ← Chroma persistence directory (git-ignored)
└── index/               ← git-ignored
    ├── bm25_corpus.json
    ├── index_manifest.json
    └── index_report.json
tests/
├── test_metadata.py
├── test_bm25_tokenizer.py
├── test_bm25_roundtrip.py
└── test_indexing_integration.py   ← skipped if chunks.jsonl absent
```

## 5. Index manifest (`data/index/index_manifest.json`)

Records what was built, so retrieval can check compatibility and a reviewer can reproduce it:

- `embedding_model`, `model_revision` (pinned, Q3), `dimension` (384), `normalized` (true), `max_seq_length`, `document_prefix` ("none")
- `vector_store` ("chroma"), `chroma_version`, `collection_name`, `distance` ("cosine"), `persist_dir`
- `bm25`: library and version, tokenizer version string, parameters (`k1`, `b`), stopword/stemming settings
- `chunks_file_sha256` (hash of the `chunks.jsonl` that was indexed), `chunk_count`, `chunk_ids_sha256` (hash of the sorted IDs)
- `built_at` (UTC), `build_seconds`, `device`

**Loader rule:** `load_vector_store()` and `load_bm25()` read the manifest first. If the configured model, revision, normalization or distance doesn't match it, **raise a clear error** ("index was built with X, config says Y; rebuild with `python -m indexing --rebuild`").

---

## 6. Stages

### Stage E0 — Prerequisites and remaining decisions

**Purpose:** confirm the inputs exist, and settle the decisions that need the user.

**Steps**
1. Confirm with the user that chunking is complete and `python -m chunking --rebuild` passes all hard checks.
2. Ask the user to approve updating the "Current phase" line in `CLAUDE.md` to this phase and file.
3. Ask **Q1** (rebuild strategy), **Q2** (device) and **Q3** (pinning the model revision and package versions). Record the answers in `indexing/config.py` and `requirements.txt`.
4. Add dependencies **only after** Q3: `langchain-huggingface`, `sentence-transformers`, `langchain-chroma`, `chromadb`, and the BM25 library chosen in Q5 (asked in E5; add it then).

**Done when:** the answers are recorded; `pip install -r requirements.txt` succeeds in a fresh venv.

**Checkpoint:** show the updated `requirements.txt` and `indexing/config.py`.

### Stage E1 — Configuration and manifest scaffolding

**Purpose:** put every setting in one place and prepare the manifest writer and checker.

**Steps**
1. `indexing/config.py`:
   - model name, revision (Q3), device (Q2);
   - `NORMALIZE = True`;
   - batch size (proposed **32**; confirm with the user at the checkpoint);
   - `CHROMA_DIR = data/chroma`; collection name (proposed `nist_ai_rmf`; confirm);
   - `INDEX_DIR = data/index`;
   - BM25 settings (filled in E5).

   All paths are overridable via `.env`.
2. `indexing/manifest.py`: `write_manifest(...)` and `check_compatibility(config) -> None` (raises on mismatch), with unit tests for match and mismatch.

**Done when:** the config loads; the manifest tests pass.

**Checkpoint:** show the config values; confirm the batch size and collection name.

### Stage E2 — Embedder

**Purpose:** create the embedding function exactly as specified, and prove it behaves as expected.

**Steps**
1. `embedder.py` → `get_embeddings()` returns a `HuggingFaceEmbeddings` with: the model name and revision (Q3), the device (Q2), `encode_kwargs={"normalize_embeddings": True, "batch_size": <config>}`. **Check the installed `langchain-huggingface` version's parameter names** before writing this, and report them.
2. Report the loaded model's `max_seq_length` (expected 512) and the output dimension (expected 384).
3. Load the bge-small tokenizer (the same tokenizer chunking used) for the token recheck in E4.
4. Sanity tests:
   - embed 3 sample texts: each vector has length 384 and norm within 1.0 ± 1e-3;
   - two near-identical sentences have higher cosine similarity than two unrelated ones;
   - embedding the same text twice gives the same vector (within 1e-5).

**Done when:** all sanity tests pass; the model loads offline from the cache after the first download.

**Checkpoint:** show the dimension, `max_seq_length`, the norms, the 3 similarity values, and the model cache location.

### Stage E3 — Metadata preparation

**Purpose:** convert each chunk into Chroma-compatible metadata **without losing information**.

**Steps**
1. **Check the installed Chroma version:** does it accept list-valued metadata? Report the result. If it does, **ask Q4** whether to use native lists. Unless the user chooses native lists, use the default below.
2. **Default conversion** (`metadata.py` → `to_chroma_metadata(chunk) -> dict`):
   - **Scalars kept as-is** (these are the fields useful for filtering): `chunk_id`, `doc_id`, `display_name`, `chunk_type`, `sequence`, `part`, `parts_total`, `pdf_page_start`, `pdf_page_end`, `printed_page_start`, `printed_page_end`, `function`, `category`, `subcategory`, `figure_number`, `ocr_confidence`, `footnote_number`, `section_path_is_page_level`, `token_count_embed`, `source_url`, `citation_url`, `citation_label`.
   - **Section path:** `section_path_str` = path joined with ` > ` (readable), plus `section_top` (first element) and `section_leaf` (last element) as scalars for filtering.
   - **Lists stored as JSON strings** (exact round-trip): `section_path_json`, `merged_sections_json`, `subcategories_json`, `action_ids_json`, `gai_risks_json`, `ai_actor_tasks_json`, `trustworthy_characteristics_json`, `links_json`.
   - **Null values are left out** of the metadata dictionary (Chroma doesn't allow `None`). The loader treats a missing key as null.
3. The **document text** stored in Chroma is `embed_text` (what was embedded). `display_text` is stored in metadata as `display_text`. **Q6** asks whether that is the right arrangement.
4. `from_chroma_metadata(meta) -> dict` reverses the conversion (parses the JSON fields, restores missing keys as null).
5. Tests:
   - round trip `chunk → metadata → chunk` returns identical values for every field, on one chunk of each of the 5 types;
   - no `None` values in the output;
   - every value is a `str`, `int`, `float` or `bool`.

**Done when:** the round-trip tests pass for all chunk types.

**Checkpoint:** show the converted metadata for one `genai_action_table` chunk and one `footnote` chunk.

### Stage E4 — Write the Chroma collection

**Purpose:** embed all chunks and persist them in one cosine-distance collection.

**Steps**
1. **Token recheck:** for every chunk, `count_tokens(embed_text) ≤ 512` using the model's tokenizer. Stop on any violation and list the chunk IDs.
2. **Create the collection with cosine distance** using the mechanism verified in Section 3.2, through the LangChain wrapper. If the wrapper can't set it in the installed version, **stop and report** before trying any workaround.
3. **Add documents** with `ids = chunk_id` (deterministic IDs from chunking), `documents = embed_text`, and `metadatas` from E3, in batches of the configured size, with logged progress.
4. **Rebuild behaviour** follows Q1:
   - (a) full rebuild: delete and recreate the collection on every run; or
   - (b) incremental: upsert by `chunk_id` and delete IDs no longer present.

   Either way, a second run must **never create duplicates**.
5. Record `build_seconds` and the device.

**Done when:**
- the collection count equals the number of chunks;
- the set of stored IDs equals the set of chunk IDs;
- the distance setting reads back as cosine;
- 5 randomly sampled stored vectors have norm ≈ 1.0;
- running the stage twice leaves the count unchanged.

**Checkpoint:** show the count, the distance setting as read back, the build time, and one stored record (ID, first 150 characters of the document, metadata keys).

### Stage E5 — BM25 keyword index

**Purpose:** build an exact-term index that complements the vectors (action IDs, subcategory IDs, rare terms like "confabulation").

**Steps**
1. **Ask Q5:** which BM25 library (`rank_bm25` used directly, or LangChain's `BM25Retriever`, which uses `rank_bm25` underneath)? Then add the dependency.
2. **Ask Q7:** which text to index for BM25, `embed_text` (includes the breadcrumb) or `display_text`?
3. **Ask Q8:** stopword removal and stemming. Show the user the token output for 3 sample chunks under each option before deciding.
4. **Tokenizer** (`bm25_index.py` → `tokenize(text) -> list[str]`). These requirements are fixed regardless of Q8:
   - lowercase;
   - **keep identifiers whole:** action IDs (`GV-1.7-001` → `gv-1.7-001`), dotted section numbers (`2.12`, `1.2.3`), and subcategory references (`GOVERN 1.7` → tokens `govern`, `1.7`, **and** a combined token `govern_1.7`);
   - split everything else on non-alphanumeric characters;
   - **the same function must be used for queries at retrieval time.** It gets a `TOKENIZER_VERSION` string, stored in the manifest.
5. **Save** `data/index/bm25_corpus.json` with:
   - `chunk_ids` (ordered list);
   - `tokens` (a list of token lists, in the same order);
   - `tokenizer_version`;
   - BM25 parameters `k1` and `b` (library defaults unless the user says otherwise);
   - `library` and `version`.
6. **Load** (`loaders.py` → `load_bm25()`): read the JSON, check `tokenizer_version` and the compatibility with the manifest, rebuild the BM25 object, and return it together with the ordered `chunk_ids`.
7. Tests:
   - tokenizer keeps `GV-1.7-001`, `2.12` and `GOVERN 1.7` as specified;
   - normal words are lowercased and split;
   - save → load round trip gives identical scores for 3 sample queries;
   - a mismatched `tokenizer_version` raises a clear error.

**Done when:**
- the BM25 chunk IDs exactly equal the Chroma IDs;
- the query `GV-1.7-001` ranks the chunk containing it **first**;
- the round-trip tests pass.

**Checkpoint:** show the tokens for one GenAI table chunk, the top-3 BM25 results (chunk ID, citation label, score) for `GV-1.7-001` and for `confabulation`, and the size of `bm25_corpus.json`.

### Stage E6 — Validation, smoke checks and report

**Purpose:** prove both indexes are complete, consistent and loadable.

**Hard checks (the run exits non-zero if any fail):**
1. Chroma count = BM25 corpus size = chunk count from `chunks.jsonl`.
2. Chroma ID set = BM25 ID set = chunk ID set.
3. The `chunks_file_sha256` in the manifest equals the hash of the current `chunks.jsonl` (the indexes are not stale).
4. Dimension is 384; sampled norms are ≈ 1.0; the distance is cosine.
5. The metadata round trip for 10 random chunks matches `chunks.jsonl` exactly.
6. `load_vector_store()` and `load_bm25()` work in a **fresh Python process** (persistence really works).
7. `check_compatibility` raises an error when the configured model name is deliberately changed (test).
8. A second full run produces no duplicates and the same counts.

**Smoke checks (soft, reported for the user to review; not a retrieval evaluation):** for each query below, report the top-5 from **vector** search and from **BM25** separately (chunk ID, citation label, score). The expected chunks are stored in `indexing/config.py` under `SMOKE_CHECKS`, never in pipeline code.

| Query | Expected to appear in the top 5 (vector and/or BM25) |
|---|---|
| "What are the four core functions of the AI RMF?" | AI RMF §5 AI RMF Core (prose) |
| "Which function is cross-cutting?" | AI RMF §5.1 Govern, or §5 with the Fig. 5 caption |
| "What is confabulation?" | GenAI §2.2 Confabulation |
| "decommissioning GAI systems" | GenAI GOVERN 1.7 table chunk |
| "GV-1.7-001" | GenAI GOVERN 1.7 table chunk (BM25 rank 1) |
| "How do AI risks differ from traditional software risks?" | AI RMF Appendix B |

The report notes which ones hit, but **misses do not fail the run**. They are input for the retrieval phase.

**Report** `data/index/index_report.json`:
- counts by `doc_id` and `chunk_type`;
- build time and device;
- model and Chroma settings as read back;
- BM25 vocabulary size and average document length;
- the results of every check;
- the smoke-check results.

**Done when:** all hard checks pass on `python -m indexing --rebuild`.

**Checkpoint:** show the console summary and the smoke-check table.

---

## 7. CLI

| Command | Behaviour |
|---|---|
| `python -m indexing` | Build or update according to Q1 |
| `python -m indexing --rebuild` | Always delete and rebuild both indexes |
| `python -m indexing --only bm25` / `--only chroma` | Rebuild one index (development); the ID-equality check still runs |
| `python -m indexing --validate` | Run Stage E6 only, against the existing indexes |

Exit code is 0 on success, non-zero on any hard-check failure.

## 8. README additions from this phase

- First run downloads the model from Hugging Face (internet needed once; later runs work offline from the cache).
- Install note: `sentence-transformers` brings in PyTorch, which is a large download. This is a known cost of the chosen library.
- How to build the indexes, and the design decisions (Section 2) with their reasons.

## 9. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Rebuild strategy: delete and recreate every run, or incremental upsert plus deletion of removed IDs? | E0 |
| Q2 | Device: CPU only, or use a GPU if available? | E0 |
| Q3 | Pin the model revision (a specific Hugging Face commit) and the exact package versions? Propose specific values. | E0 |
| Q4 | If the installed Chroma supports list metadata, use native lists instead of JSON strings? | E3 |
| Q5 | BM25 library: `rank_bm25` directly, or LangChain `BM25Retriever`? | E5 |
| Q6 | Store `embed_text` as the Chroma document and `display_text` in metadata, or the reverse? | E3 |
| Q7 | BM25 indexes `embed_text` (with breadcrumb) or `display_text`? | E5 |
| Q8 | BM25 stopword removal and/or stemming? Shown on samples. | E5 |
| Q9 | Confirm chunking-plan Q1: count tokens with bge-small's tokenizer from `transformers` (already installed through `sentence-transformers`) | E0 |
| Q10 | Any difference between this plan and the installed library behaviour | Whenever found |

## 10. Handoff notes for the retrieval phase (not to implement now)

- Use `load_vector_store()`, `load_bm25()` and the shared `tokenize()` for queries; never rebuild indexes at query time.
- Hybrid fusion (e.g., reciprocal rank fusion), result counts, relevance thresholds, and the query instruction prefix are retrieval-phase decisions.
- Fetch citations from metadata (`citation_label`, `citation_url`); show `display_text`; parse the `*_json` fields with `from_chroma_metadata`.
