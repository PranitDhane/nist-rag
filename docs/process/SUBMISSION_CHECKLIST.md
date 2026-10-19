# Submission checklist

Every requirement of the assignment (*Agentic RAG System Using LangGraph*), every deliverable it lists, and every item under "What We Evaluate", with where it is shown. Paths are relative to the repository root. Measurements are never quoted here: they are generated in [`docs/RESULTS.md`](../RESULTS.md) and [`docs/E2E_RESULTS.md`](../E2E_RESULTS.md).

## 1. Requirements

| Requirement | Where it is met | Evidence |
|---|---|---|
| **Document ingestion:** load both documents | `ingest/` (download with pinned SHA-256, extraction, cleaning, structure, validation); `make download`, `make ingest` | `ingest/sources.yaml`; 11 hard validation checks; [`docs/RESULTS.md`](../RESULTS.md#corpus-chunks-and-indexes) |
| Split into searchable chunks | `chunking/` (section-first, table-aware, footnote and figure chunks); [`DESIGN.md` 2.1](../../DESIGN.md#21-how-the-documents-were-split-and-why) | `tests/test_prose.py`, `tests/test_tables.py`, `tests/test_segment.py`, chunk validation checks |
| Generate embeddings and store them in a vector store | `indexing/` (`bge-small-en-v1.5`, Chroma, plus BM25); [`DESIGN.md` 2.2](../../DESIGN.md#22-why-this-embedding-model-and-vector-store) | `make index`; index validation checks; `tests/test_vector_store.py` |
| Chunk metadata: document name, page number, section | Every chunk carries `doc_id`, `display_name`, `section_path`, pdf and printed pages, `citation_label`, `citation_url` | `tests/test_metadata.py`; every chunk link opened and checked against the PDF ([`RESULTS.md`](../RESULTS.md#citation-links)) |
| Briefly explain chunking, embedding and store choices in the README | README "Important design decisions" table, linked to `DESIGN.md` | [`README.md`](../../README.md#important-design-decisions) |
| **Question answering** grounded in retrieved content | `retrieval/`, `augmentation/`, `generation/`; the model sees only the standalone query and the sources | Gold-fact and support checks in [`docs/E2E_RESULTS.md`](../E2E_RESULTS.md); `eval/generation_behaviour_report.json` |
| The example question "What are the four core functions…" | Case S1 of the end-to-end set; README example 1 | Real answer shown in the README and in `eval/e2e_runs/*/raw/S1.json` |
| **LangGraph:** workflow implemented with LangGraph | `graph/` (main graph + RAG subgraph); diagrams in [`docs/graph.md`](../graph.md) | `tests/test_graph_*.py`; a test fails if `docs/graph.md` is stale |
| Which nodes are needed | Main graph: `reset_turn`, `rule_greeting_check`, `classify_and_rewrite`, reply nodes, `rag`, `finalize`, `error_finalize`; RAG subgraph: `hybrid_search`, `rerank`, `gate`, `expand_context`, `generate`, `attribute`, reply nodes | [`DESIGN.md` 2.4](../../DESIGN.md#24-how-the-langgraph-workflow-is-structured) |
| What information the graph state maintains | Per-message fields reset every turn; only `messages` accumulates; no models or clients in state | `graph/state.py`; [`DESIGN.md` 2.4](../../DESIGN.md#24-how-the-langgraph-workflow-is-structured) |
| When retrieval takes place | Only for `nist_question`; greetings and out-of-scope messages never retrieve | Route table in README; routing evaluation |
| How insufficient or irrelevant results are handled | Router (`out_of_scope`), relevance gate, prompt rule and fixer; reason codes; five related sections | [`DESIGN.md` 2.5](../../DESIGN.md#25-how-insufficient-or-irrelevant-context-is-handled); [`RESULTS.md`](../RESULTS.md#insufficient-context-path) |
| Whether the query is modified or retrieval retried | The query **is** modified (a standalone rewrite of follow-ups). Retrieval is **not** retried: there are no loops, to bound cost on the free tier | [`DESIGN.md` 2.4 and 2.5](../../DESIGN.md#24-how-the-langgraph-workflow-is-structured) |
| **Source attribution:** references that identify and verify the source | `[C#]` markers; `sources` built in code (`attribution/`) as label + `#page=N` link | Link check: every chunk link opens a page containing the cited text ([`RESULTS.md`](../RESULTS.md#citation-links)); `tests/test_build_sources.py`. The assignment's "Section 1.3" is only illustrative: the real answer cites §5 |
| **Follow-up questions** (at least two) | Server `thread_id`, in-memory checkpointer, standalone-query rewriting | Case C1 (three turns, the assignment's own example) and C2 in the end-to-end set; README example 2; [`DESIGN.md` 2.6](../../DESIGN.md#26-how-conversational-context-is-maintained) |
| **Technologies:** Python, LangGraph, a vector store, an embedding model, an LLM | Python 3.12, LangGraph, Chroma, `bge-small-en-v1.5`, Groq `gpt-oss` models; LangChain used only for supporting components | `requirements.txt` (exact pins) |
| **Rate limits handled** | Token-per-minute budget check, retry with backoff only for rate-limit or temporary errors and only before the first streamed token, no fallback model | [`DESIGN.md` 4](../../DESIGN.md#4-free-tier-constraints); `tests/test_llm*.py`, `tests/test_generate*.py` |

## 2. Deliverables

| Deliverable | Status | Where |
|---|---|---|
| **Source code**: ingestion, retrieval, LangGraph workflow, question answering, conversation handling | Complete and runnable from a fresh clone | `ingest/ chunking/ indexing/ retrieval/ augmentation/ generation/ insufficient/ attribution/ graph/ app/ llm/`; fresh-clone acceptance test in [`RESULTS.md`](../RESULTS.md#fresh-clone-acceptance-test) |
| **README**: installation | Done | README "Installation" |
| Configuration | Done (every variable, with defaults, checked against `.env.example` by a test) | README "Configuration" |
| How to download and process the documents | Done (`make build`, offline `FROM_DIR`, rebuild) | README "Downloading and processing the documents" |
| How to run the application | Done (`make serve`, endpoints, request and response, streaming, errors) | README "Running the application" |
| Example questions | Done (the assignment's own first; curl commands verified in the fresh clone) | README "Example questions" |
| Important design decisions | Done | README table linking to `DESIGN.md` |
| Known limitations | Done | README "Known limitations" |
| **Design explanation**: how the documents were split and why | Done | [`DESIGN.md` 2.1](../../DESIGN.md#21-how-the-documents-were-split-and-why) |
| Why this embedding model and vector store | Done | [`DESIGN.md` 2.2](../../DESIGN.md#22-why-this-embedding-model-and-vector-store) |
| How retrieval works | Done | [`DESIGN.md` 2.3](../../DESIGN.md#23-how-retrieval-works) |
| How the LangGraph workflow is structured | Done | [`DESIGN.md` 2.4](../../DESIGN.md#24-how-the-langgraph-workflow-is-structured) |
| How insufficient or irrelevant context is handled | Done | [`DESIGN.md` 2.5](../../DESIGN.md#25-how-insufficient-or-irrelevant-context-is-handled) |
| How conversational context is maintained | Done | [`DESIGN.md` 2.6](../../DESIGN.md#26-how-conversational-context-is-maintained) |

## 3. "What We Evaluate"

| Criterion | Where it is shown | Honest note |
|---|---|---|
| **1. Understanding of RAG** | The whole pipeline is explained with reasons in [`DESIGN.md`](../../DESIGN.md): chunking, hybrid retrieval, reranking, grounding, refusal. Retrieval is measured, not assumed ([`RESULTS.md`](../RESULTS.md#retrieval)) | The reranker lifts the hybrid result; the evidence is generated |
| **2. LangGraph implementation** | `graph/` and [`docs/graph.md`](../graph.md): main graph + RAG subgraph, one error exit, state reset per turn, streaming filter | No loops by design |
| **3. Retrieval strategy** | Vector + BM25 fused with weighted RRF, cross-encoder over all candidates, top 5, sibling table parts, relevance gate; four configurations compared ([`RESULTS.md`](../RESULTS.md#retrieval)) | Weights, k and the threshold are tuned on a small labelled set |
| **4. Handling of insufficient or irrelevant context** | Three layers (router, gate, prompt and fixer), exact fixed texts, related sections, reason codes | The model does not always follow the "refuse partly answerable questions" rule, and in the end-to-end runs it sometimes refused answerable questions; both are reported |
| **5. Source attribution** | Citations built in code, merged and ordered, with verifiable page links; every chunk link checked against the PDF | Some sentences stay uncited (kept with a warning, rate measured) |
| **6. Conversation handling** | `thread_id`, checkpointer, standalone-query rewriting, no history in generation, rollback of failed turns | History is in memory and lost on restart |
| **7. Code quality** | Small single-purpose modules, config modules for every limit, type hints, docstrings that explain why; about a thousand tests (offline, with positive and negative cases, mutation-checked guards for docs and reports); `make test` passes in a fresh clone | One live LangSmith test fails only when its key is set; one timing-based test is occasionally flaky (both reported) |
| **8. Technical reasoning and trade-offs** | [`DESIGN.md`](../../DESIGN.md) (problem, decision, alternatives, trade-off, evidence per point); [`PROBLEMS_AND_SOLUTIONS.md`](PROBLEMS_AND_SOLUTIONS.md); the decision log [`PROGRESS.md`](PROGRESS.md) | Trade-offs are stated next to each decision, including the ones that cost quality |

## 4. Submission mechanics

- No secrets in the tree or in git history (scan clean at stage D2; `.env` is git-ignored and `.env.example` holds no values).
- `data/`, `.venv/` and `.env` are never committed or zipped; the PDFs are downloaded by `make build`.
- A fresh clone, following only the README, reaches a cited answer and a working three-turn conversation (acceptance test, [`RESULTS.md`](../RESULTS.md#fresh-clone-acceptance-test)).
- GitHub repository and zip: created by the author from the final commit. To produce the zip from that commit: `git archive --format=zip --prefix=nist-rag/ -o nist-rag.zip HEAD`.
