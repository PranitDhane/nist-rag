# Retrieval Plan — NIST AI RMF Agentic RAG

> **For Claude Code:** read `CLAUDE.md` first; its working rules apply. This file is the specification for the **retrieval phase**: retrieving candidates, fusing, reranking, adding sibling table parts, ordering, and signalling relevance. Work stage by stage (Section 7). Each stage has: **Purpose → Input → Output → Steps → Done when → Checkpoint**. A stage is finished only when every "Done when" item is true **and** the user has confirmed the Checkpoint. Questions in Section 10 must be asked **at the stage named**, never answered by assumption.

---

## 1. Goal and boundary

Build a retrieval module that takes a **standalone query** (from the routing phase) and returns the **final, ordered context chunks** with scores, plus a **relevance signal** saying whether anything relevant was found.

- **In scope:**
  - query embedding with the bge prefix;
  - vector and BM25 retrievers;
  - weighted-RRF fusion with LangChain's `EnsembleRetriever`;
  - cross-encoder reranking;
  - sibling table-part expansion;
  - final ordering;
  - the relevance threshold;
  - a diagnostic mode;
  - a retrieval evaluation set, configuration comparisons, and threshold tuning.
- **Out of scope (next phase):** LLM grading, query-retry loop, answer generation, citations in the answer, the insufficient-context reply text. This phase only produces the `relevant: true/false` signal; the next phase decides what to do with it.
- Indexes are **loaded once** with `indexing.loaders`; nothing is rebuilt at query time.

## 2. Decisions already made by the user (do not change)

| # | Decision | Value |
|---|---|---|
| T-D1 | Fusion | LangChain **`EnsembleRetriever`** (weighted Reciprocal Rank Fusion) |
| T-D2 | Ensemble weights | **0.6 vector / 0.4 BM25** |
| T-D3 | Candidates per retriever | **10** from vector search, **10** from BM25 |
| T-D4 | Reranking | **Local cross-encoder**, model **`BAAI/bge-reranker-base`** |
| T-D5 | Rerank scope | **Rerank all** fused candidates (10–20 unique chunks) |
| T-D6 | Final count | **Top 5** after reranking (before sibling expansion, T-D8) |
| T-D7 | Query prefix | **Use** the bge-small query instruction prefix, on **queries only** (documents were embedded without it) |
| T-D8 | Split table pieces | **Always add missing sibling parts** of a selected split table |
| T-D9 | Prompt order | **Best first**, by reranker score |
| T-D10 | "Nothing relevant" detection | **Reranker score threshold**, tuned on an evaluation set |

## 3. Facts and requirements that follow from these decisions

Claude Code must **verify each against the installed versions** and report.

1. **BM25 must be a LangChain retriever** for `EnsembleRetriever`. Use LangChain's `BM25Retriever` with **our versioned `tokenize()`** from `indexing.bm25_index` as its preprocessing function, built from the saved tokenized corpus (`bm25_corpus.json`) without re-tokenizing, if the installed version allows building it from a prebuilt BM25 object. **Check how to construct it and report.** If the class can't be built from saved tokens, stop and ask before re-tokenizing at load time. This resolves indexing-plan Q5 in favour of `BM25Retriever`; record that in `docs/PROGRESS.md`.
2. **Duplicate detection inside the ensemble:** `EnsembleRetriever` merges the same chunk found by both retrievers either by a configurable key or by comparing `page_content`. **Check whether the installed version supports an ID key** and use `chunk_id`. If it doesn't, both retrievers must return **identical `page_content`** (`embed_text`) for the same chunk; add a test proving a chunk found by both appears **once**.
3. **The order of retrievers and weights must match:** retrievers `[vector, bm25]` with weights `[0.6, 0.4]`. Use the RRF constant at its library default (60) and record it in config.
4. **EnsembleRetriever returns no fusion scores** and doesn't say which retriever found each chunk. A separate **diagnostic mode** (Stage T3) runs both retrievers independently to provide that information. Diagnostic mode is for evaluation and debugging, never the normal query path.
5. **Query prefix:** verify the exact prefix text for `bge-small-en-v1.5` on its model card (expected: `Represent this sentence for searching relevant passages: `). Apply it **only** when embedding queries (e.g., through the query-specific encoding options of `HuggingFaceEmbeddings`; check the installed version). Add a test that **document** embeddings are unchanged.
6. **Reranker scores:** `bge-reranker-base` produces a raw score per (query, chunk) pair; libraries may or may not apply a sigmoid to map it into 0–1 by default. **Check the installed behaviour**, choose **one** scale, record it in config (`RERANK_SCORE_SCALE`), and use it everywhere, including the threshold.
7. **Reranker input limit:** query plus chunk text together are cut at 512 tokens. Report how many candidate pairs would be truncated on the evaluation set.
8. **Scores are needed** for the threshold (T-D10) and for best-first ordering (T-D9). LangChain's cross-encoder compressor may return only the reordered documents without scores. **Q3** decides the implementation; the scores must survive either way.
9. **Sibling expansion can push the count above 5.** For example, top 5 includes MEASURE 2.11 part 1 of 2, so part 2 is added and 6 chunks go to the LLM. The ordering and counting rules for siblings are decided in **Q5/Q6**.

## 4. Pipeline

```
standalone_query
  → [vector retriever: query embedded WITH prefix → Chroma cosine, k=10]
  → [BM25 retriever: our tokenize() → k=10]
  → EnsembleRetriever (weights 0.6 / 0.4, RRF) → 10–20 unique candidates, deduplicated by chunk_id
  → cross-encoder bge-reranker-base scores every (query, candidate) pair
  → sort by rerank score → keep top 5
  → relevance gate: best score vs threshold (+ per-chunk rule per Q7)
  → sibling expansion: add missing parts of selected split tables (fetched by metadata)
  → final ordering: best first (sibling placement per Q5)
  → RetrievalResult
```

## 5. Output contract: `RetrievalResult`

| Field | Type | Notes |
|---|---|---|
| `query` | str | The standalone query received |
| `chunks` | list[RetrievedChunk] | Final ordered context |
| `relevant` | bool | Relevance gate result |
| `best_score` | float \| null | Highest rerank score (in `RERANK_SCORE_SCALE`) |
| `threshold` | float | Threshold used |
| `candidate_count` | int | Unique candidates after fusion |
| `timings_ms` | dict | `vector`, `bm25`, `fusion`, `rerank`, `siblings`, `total` |
| `config_version` | str | Short hash of the retrieval config, for traceability |

`RetrievedChunk`:

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | str | |
| `display_text` | str | For the LLM prompt and for display |
| `metadata` | dict | Restored with `from_chroma_metadata` (citation fields, pages, section path, etc.) |
| `rerank_score` | float \| null | null for added siblings that were not candidates |
| `rerank_rank` | int \| null | 1-based position after reranking |
| `added_as_sibling` | bool | true if added by T-D8 |
| `sibling_of` | str \| null | chunk_id of the selected part that caused the addition |

## 6. Project layout additions

```
retrieval/
├── __init__.py
├── config.py          ← all retrieval settings (k, weights, model, prefix, threshold, scale)
├── components.py      ← builds the vector retriever, BM25Retriever, EnsembleRetriever, reranker (once)
├── fusion.py          ← ensemble call + chunk_id dedupe guard
├── rerank.py          ← cross-encoder scoring, top-N selection
├── siblings.py        ← sibling-part lookup and insertion
├── ordering.py        ← final ordering rules
├── gate.py            ← relevance threshold logic
├── pipeline.py        ← retrieve(query) -> RetrievalResult
├── diagnostics.py     ← per-retriever ranks, fused order, rerank order side by side
└── models.py          ← RetrievalResult, RetrievedChunk dataclasses
eval/
├── retrieval_cases.yaml
├── retrieval_eval.py      ← python -m eval.retrieval_eval
└── retrieval_report.json  (generated)
tests/
├── test_query_prefix.py
├── test_fusion.py
├── test_rerank.py
├── test_siblings.py
├── test_ordering.py
├── test_gate.py
└── test_retrieval_integration.py   ← needs the real indexes; skipped if absent
```

---

## 7. Stages

### Stage T0 — Prerequisites and dependencies

**Steps**
1. Confirm with the user that indexing is complete and `python -m indexing --validate` passes. Ask the user to approve updating the "Current phase" line in `CLAUDE.md`.
2. Identify the packages needed for `EnsembleRetriever`, `BM25Retriever` and the cross-encoder **in the installed LangChain version** (these classes have moved between packages in different LangChain releases; check, don't assume), plus `rank_bm25`. **Ask Q1** to approve the exact packages and versions.
3. **Ask Q2:** device for the reranker (CPU, or a GPU if available).

**Done when:** dependencies install in a fresh venv; imports work.

**Checkpoint:** show the package list with versions and the import locations found.

### Stage T1 — Retrieval configuration

**Steps**
1. `retrieval/config.py`:
   - `VECTOR_K = 10`, `BM25_K = 10`;
   - `WEIGHTS = [0.6, 0.4]` (order: vector, bm25);
   - `RRF_C` = library default (record it);
   - `RERANKER_MODEL = "BAAI/bge-reranker-base"`;
   - `TOP_N = 5`;
   - `QUERY_PREFIX` = the verified text;
   - `RERANK_SCORE_SCALE` (Stage T4);
   - `THRESHOLD = None` until Stage T6 (the pipeline must refuse to run the gate without it, except in evaluation mode);
   - device (Q2).
2. `config_version` = short hash of all these values.
3. A startup check compares the embedding model, revision and normalization with the index manifest (reuse `check_compatibility`).

**Done when:** the config loads; the compatibility check passes; a test proves a mismatched embedding model raises an error.

**Checkpoint:** show the config and the verified query prefix with its source (model card section).

### Stage T2 — Individual retrievers

**Purpose:** build and verify each retriever on its own before fusing.

**Steps**
1. **Vector retriever:** the Chroma store from `load_vector_store()`, with an embedding function that adds the prefix to **queries only**, as a retriever with `k = VECTOR_K`. Results carry full metadata including `chunk_id`.
2. **BM25 retriever:** `BM25Retriever` built from the saved corpus with our `tokenize()` (Section 3.1), `k = BM25_K`. Each result is a document whose metadata includes `chunk_id`, and whose `page_content` follows Section 3.2.
3. Tests:
   - **prefix test:** query embedding ≠ embedding of the same text without the prefix; a document embedding is identical to what's stored in Chroma for that chunk;
   - **BM25 tests:** query `GV-1.7-001` returns the GOVERN 1.7 table chunk at rank 1; query `confabulation` returns a §2.2 chunk in the top 3;
   - **vector test:** "What are the four core functions of the AI RMF?" returns an AI RMF §5 chunk in the top 5.

**Done when:** all tests pass, using the real indexes.

**Checkpoint:** for 3 sample queries, show each retriever's top 10 (chunk ID, citation label).

### Stage T3 — Fusion with `EnsembleRetriever` + diagnostic mode

**Steps**
1. `EnsembleRetriever([vector, bm25], weights=[0.6, 0.4])`, using the ID-key setting from Section 3.2 if available.
2. **Dedupe guard** (`fusion.py`): after the ensemble, check that no `chunk_id` appears twice. If one does, raise an error in tests and log ERROR in production. **Never silently drop.**
3. **Diagnostic mode** (`diagnostics.py`): for a query, return a table with:
   - each candidate's vector rank, BM25 rank, and fused position (from the ensemble result order);
   - an independently computed weighted-RRF value using the same formula, to confirm the ensemble's order is understood (report any mismatch; don't "fix" the ensemble).
4. Tests:
   - a chunk found by both retrievers appears once;
   - fused list size is between 10 and 20;
   - the weights order is honoured: for a constructed case where the retrievers disagree, the vector-preferred chunk ranks higher when ranks are otherwise symmetric.

**Done when:** tests pass; the diagnostic table works.

**Checkpoint:** show the diagnostic table for "Which function is cross-cutting?" and for "GV-1.7-001".

### Stage T4 — Cross-encoder reranking

**Steps**
1. **Ask Q3:** implement reranking with (a) LangChain's cross-encoder compressor/reranker, **only if** it can return scores, or (b) the `sentence-transformers` `CrossEncoder` directly in `rerank.py`. Show what the installed LangChain class returns before asking.
2. **Ask Q4:** what text the reranker reads for each chunk: `embed_text` (includes the breadcrumb, e.g., "GOVERN 1.7") or `display_text`. Show 2 examples of the scores under each option.
3. Score every fused candidate. Record the score scale found (Section 3.6) in `RERANK_SCORE_SCALE`.
4. Sort by score, descending; ties broken by fused position, then by `chunk_id` (deterministic). Keep `TOP_N = 5`.
5. Report the number of truncated pairs (Section 3.7).

**Done when:**
- scores are attached to every candidate;
- ordering is deterministic (two runs, same order);
- reranking 20 candidates on the chosen device is timed and reported.

**Checkpoint:** for 3 queries, show the fused order next to the reranked order with scores, plus the reranker latency.

### Stage T5 — Sibling expansion and final ordering

**Steps**
1. **Siblings** (`siblings.py`): for each selected chunk with `parts_total > 1`, find its other parts by metadata (same `doc_id`, same `chunk_type`, same `subcategory` (GenAI) or `category` (AI RMF), every `part` from 1 to `parts_total`), using the vector store's metadata lookup (check the method in the installed wrapper). Add the parts that are missing, with `added_as_sibling = true` and `sibling_of` set. **Never add duplicates.**
2. **Ask Q5 (placement):** where added siblings go in the best-first order:
   - (a) a split table is kept together in part order (part 1, part 2), and the group takes the position of its highest-scoring selected part; or
   - (b) siblings are appended at the end.

   Show one example of each.
3. **Ask Q6 (counting):** whether siblings count toward the 5. Either:
   - (a) the limit is 5 *selections* and siblings are extra, so the total may exceed 5; or
   - (b) the total must stay 5, so lower-ranked chunks are dropped to make room.

   Report the largest possible total under (a), given the largest `parts_total` in the corpus.
4. **Final ordering** (`ordering.py`): best first by `rerank_score`, with siblings placed per Q5.
5. Tests:
   - a selected part 1 of 2 brings in part 2;
   - both parts already selected means nothing is added;
   - non-table chunks are never expanded;
   - ordering follows the Q5/Q6 answers exactly.

**Done when:** tests pass; the MEASURE 2.11 case works end to end with the real indexes (query: "How should fairness and bias be evaluated in GAI systems?", or another query that selects one MEASURE 2.11 part).

**Checkpoint:** show the final ordered list for that query (chunk ID, part, score, `added_as_sibling`).

### Stage T6 — Retrieval evaluation and threshold tuning

**Purpose:** measure retrieval quality, compare configurations, and choose the relevance threshold from data.

**Steps**
1. **Evaluation set** (`eval/retrieval_cases.yaml`), **at least 30 cases**, labelled by **structure, not by chunk ID** (chunk IDs change if chunking changes). Each answerable case lists one or more gold targets: `doc_id` plus a section ID, subcategory, action ID, or printed page range. A retrieved chunk "hits" if it matches any gold target. Include:
   - the assignment's examples ("What are the four core functions of the NIST AI RMF?", "Which one is cross-cutting?" in its standalone form);
   - questions answered in each document, and some answered in both;
   - exact-ID questions (`GV-1.7-001`, "GOVERN 1.7");
   - questions whose answer is in a table, a figure (OCR) or a footnote;
   - paraphrased questions that don't share words with the text (tests the vector side);
   - **at least 8 unanswerable but near-scope questions** (e.g., "What penalties does the EU AI Act set?", "What does NIST say about quantum computing risks?"), labelled `answerable: false`.

   **Ask Q8** for the user to review and approve the set before running.
2. **Configurations compared**, all with the same cases:
   - (1) vector only, top 5;
   - (2) BM25 only, top 5;
   - (3) ensemble only, top 5 (no reranker);
   - (4) ensemble + reranker, top 5 (the chosen design).
3. **Metrics** for answerable cases: **Hit@5** (at least one gold match in the top 5), **MRR** (based on the rank of the first gold match), and **Recall@20 of the fused candidates** (whether the reranker even had a chance). The report says, per case, which configuration found it.
4. **Threshold tuning** (configuration 4): for each candidate threshold across the observed score range (e.g., 20 evenly spaced values), report:
   - % of answerable cases whose best score is ≥ threshold (kept);
   - % of unanswerable cases whose best score is < threshold (correctly rejected);
   - the best and worst scores of each group.

   **Ask Q7** for the user to choose the threshold **and** whether it applies only to the best score (a gate) or also removes individual chunks below it from the top 5. Record the answer in config.
5. The evaluation respects determinism: same config, same results. No LLM calls are needed in this phase.

**Done when:** the report includes the configuration comparison table, the threshold table, and every failed case; the user has chosen the threshold.

**Checkpoint:** show the comparison table (Hit@5, MRR, Recall@20 per configuration), the threshold table, the failures, and the user's chosen threshold.

### Stage T7 — Relevance gate and pipeline assembly

**Steps**
1. `gate.py`: apply the threshold per Q7. `relevant = best_score ≥ threshold`. If Q7 says to remove individual chunks, chunks below the threshold are removed **before** sibling expansion.
2. **If `relevant` is false, still return the (possibly empty) chunks and scores**, so the next phase can log and explain the decision; the next phase decides the reply.
3. `pipeline.py` → `retrieve(query) -> RetrievalResult`, with timings per step and `config_version`.
4. **Ask Q9:** connect `retrieve()` to the LangGraph graph now, with the `rag_stub` from the API phase showing the retrieved citation labels and the `relevant` flag (useful for testing through Postman before generation exists), or leave it to the next phase.
5. **Ask Q10:** an acceptable latency target for `retrieve()` on the chosen device. Report the measured p50/p95 over the evaluation set either way.

**Done when:**
- gate tests pass: above threshold, below threshold, removal behaviour per Q7, empty result handling;
- integration tests pass on 5 evaluation cases;
- timings are reported.

**Checkpoint:** show `RetrievalResult` (abbreviated) for one answerable and one unanswerable case, plus the latency numbers.

### Stage T8 — Final report

The report holds the configuration comparison, the threshold choice, latency, the truncated-pair count, and the known retrieval weaknesses (the failed cases). It is written to `eval/retrieval_report.json`, with a short Markdown summary in `docs/RETRIEVAL_RESULTS.md` for the README and DESIGN.md.

**Checkpoint:** show the Markdown summary.

---

## 8. Testing rules

- Unit tests use small fixtures and mocked components where possible. Integration tests and the evaluation need the real indexes and are skipped if absent.
- Never adjust gold labels or loosen metrics to make results look better. Report honestly; the user decides.
- The threshold is **only** chosen by the user from the Stage T6 table. Never hardcode it from a guess.

## 9. README / DESIGN additions

- Why hybrid search (meaning vs exact terms), why RRF (no score calibration needed), why a cross-encoder (reads the query and chunk together), why best-first order, why sibling expansion, how the threshold was chosen.
- The configuration comparison table from Stage T6: this is the evidence for "retrieval strategy".
- Known limitations: the reranker's 512-token input limit; the threshold is tuned on a small set; EnsembleRetriever hides fusion scores (diagnostic mode exists for this).

## 10. Open questions — ask at the stage named, never assume

| # | Question | Ask at |
|---|---|---|
| Q1 | Exact packages and versions for `EnsembleRetriever`, `BM25Retriever`, the cross-encoder, `rank_bm25` | T0 |
| Q2 | Reranker device: CPU, or a GPU if available? | T0 |
| Q3 | Reranker implementation: LangChain class (only if it returns scores) or `sentence-transformers` `CrossEncoder` directly | T4 |
| Q4 | Reranker reads `embed_text` (with breadcrumb) or `display_text`? | T4 |
| Q5 | Sibling placement: keep the table group together at its best part's position, or append at the end? | T5 |
| Q6 | Do siblings count toward the 5, or are they extra? | T5 |
| Q7 | Threshold value, and whether it gates only on the best score or also removes individual low-scoring chunks | T6 |
| Q8 | Approval of the retrieval evaluation cases | T6 |
| Q9 | Connect `retrieve()` to the graph now (stub shows retrieved citations) or in the next phase? | T7 |
| Q10 | Latency target for `retrieve()` | T7 |
| Q11 | Metadata filters (e.g., restrict to `doc_id` when the user names a document) or an exact-ID shortcut: **not planned now.** Raise only if the evaluation shows a clear need, with evidence. | After T6 |
| Q12 | Any difference between this plan and the installed library behaviour | Whenever found |

## 11. Handoff to the next phase (grading and generation)

- Input: `RetrievalResult`. If `relevant` is false, the next phase decides the reply (and whether to rewrite the query and retry once, as discussed at the start of the project).
- The prompt receives `chunks` **in the given order**, each with its `chunk_id` as the citation key; citations are built from `metadata` in code, never by the LLM.
