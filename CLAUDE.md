# CLAUDE.md — NIST AI RMF Agentic RAG

Read this file at the start of every session. It defines **how to work** on this project and the **rules that hold in every phase**. The detailed specification for each phase is in its plan file under `docs/`. **This file never replaces a plan; it tells you how to use them.**

---

## 1. Project in one paragraph

A take-home assignment: a question-answering system over two NIST documents, **AI RMF 1.0 (NIST AI 100-1)** and the **Generative AI Profile (NIST AI 600-1)**. It's built with **LangGraph** behind a **FastAPI** HTTP API, using **Groq (free tier)** as the LLM, `BAAI/bge-small-en-v1.5` embeddings in **Chroma**, a **BM25** keyword index, and a `BAAI/bge-reranker-base` cross-encoder. Answers must be **grounded in the documents**, cite their sources with verifiable page links, support **follow-up questions**, and handle **insufficient or irrelevant context** honestly. The reviewers value **clear reasoning and justified trade-offs** over complexity: every design choice must be explainable in an interview.

## 2. Phases and plan files

| # | Phase | Plan file | Module(s) |
|---|---|---|---|
| 1 | Ingestion | `docs/process/plans/INGESTION_PLAN.md` | `ingest/` |
| 2 | Chunking | `docs/process/plans/CHUNKING_PLAN.md` | `chunking/` |
| 3 | Embedding & indexing | `docs/process/plans/EMBEDDING_INDEXING_PLAN.md` | `indexing/` |
| 4 | API & routing | `docs/process/plans/API_ROUTING_PLAN.md` | `app/`, `graph/`, `llm/` |
| 5 | Retrieval | `docs/process/plans/RETRIEVAL_PLAN.md` | `retrieval/` |
| 6 | Augmentation (prompt construction) | `docs/process/plans/AUGMENTATION_PLAN.md` + `prompts/` | `augmentation/` |
| 7 | Generation | `docs/process/plans/GENERATION_PLAN.md` | `generation/` |
| 8 | Insufficient-context path | `docs/process/plans/INSUFFICIENT_CONTEXT_PLAN.md` | `insufficient/` |
| 9 | Source attribution | `docs/process/plans/ATTRIBUTION_PLAN.md` | `attribution/` |
| 10 | Graph integration | `docs/process/plans/GRAPH_INTEGRATION_PLAN.md` | `graph/` (main graph + RAG subgraph) |
| 11 | End-to-end evaluation | `docs/process/plans/E2E_EVALUATION_PLAN.md` | `eval/` |
| 12 | Deliverables (README, DESIGN.md, Makefile, packaging) | `docs/process/plans/DELIVERABLES_PLAN.md` | `Makefile`, `README.md`, `DESIGN.md`, `docs/` |

Supporting documents:
- `docs/process/PROBLEMS_AND_SOLUTIONS.md`: every problem found and the decision made (source material for `DESIGN.md`).
- `docs/process/PROGRESS.md`: the running status log (Section 4).
- `eval/e2e_cases.yaml`: the end-to-end test set (assignment examples plus coverage cases, gold facts verified against the PDFs). Approval is E2E plan Q1.

**Where the process documents live (moved in deliverables Stage D6):** all plan files are in `docs/process/plans/`; `PROGRESS.md`, `PROBLEMS_AND_SOLUTIONS.md` and the per-phase summaries are in `docs/process/`. `CLAUDE.md` stays at the repository root so Claude Code loads it (D6 Q6). The generated results (`docs/RESULTS.md`, `docs/E2E_RESULTS.md`), `docs/build_results.py`, `docs/graph.md` and `docs/architecture.html` stay in `docs/`. Append progress entries to `docs/process/PROGRESS.md`; its older entries keep the paths that were valid when they were written.

**Status reported by the user:** phases 1–11 are built (phase 6 completed 2026-10-17, phase 11 on 2026-10-19). Phase 12 is in progress.

**Current phase: 12 — Deliverables.** *(Confirm with the user at the start of every session. Change this line only with the user's approval.)*

## 3. How to work (every phase)

1. **Session start:**
   - read this file, `docs/process/PROGRESS.md`, and the **current phase's plan file in full**;
   - summarise in 3–5 bullets where things stand (last finished stage, open questions);
   - confirm the current phase and stage with the user.
2. **One stage at a time**, in the order the plan gives. Never start the next stage until the user confirms the current one.
3. **At the start of a stage:** restate in 3–5 bullets what you will build and which files you will create or change. Then build it.
4. **At the end of a stage:** do exactly what its **Checkpoint** says: run the stage, run its tests, and show the listed outputs. Then **stop and wait**.
5. **Open questions:** each plan has an "Open questions" table saying *when* to ask each one. Ask each question **at its stage**, present the options with evidence (measurements, examples, comparisons), and record the answer in the right config file and in `docs/process/PROGRESS.md`.
6. **If a plan is ambiguous, or reality contradicts it** (a file, a library, an API behaving differently): **stop and ask.** Don't guess and don't silently adjust the plan.
7. **The current phase's plan is authoritative for that phase.** Decisions recorded in the "Decisions already made by the user" table of any plan may not be changed.

## 4. Progress log (`docs/process/PROGRESS.md`)

Append one line per finished stage:

`YYYY-MM-DD | phase | stage | result | answers to open questions | open issues`

Also record here every answer to an open question, and every **cross-plan resolution** (Section 9). Never rewrite past entries; append corrections as new lines.

## 5. Hard rules (never break these)

**Integrity**
- **Never hardcode expected results to make a check pass.** No lists of expected IDs, page types, headings, links, labels or answers inside pipeline code. Expected values live only in validation config (e.g., the manifest's `validation:` block, eval case files) and are read only by validators and evaluations.
- **Never weaken, skip or delete a failing check or test.** Report the failure, explain the likely cause, propose a fix, and wait.
- **Never edit gold or evaluation data to improve results.** Change it only if it's shown to be wrong against the documents, with the user's approval and a note in the report.
- **Never filter failures out of a report.**

**Data and scope**
- **Never edit files in `data/`** by hand or with one-off scripts. Everything there is regenerated by pipeline code.
- **Never build parts outside the current phase**, and never add their dependencies early.
- **Never patch another phase's module to make the current phase work.** Report the mismatch and wait.

**Dependencies and libraries**
- **Never add a dependency** that isn't in the plan without asking. Pin exact versions in `requirements.txt`.
- **Never guess library parameters or class locations.** LangChain, LangGraph, Chroma, langchain-groq, pymupdf4llm and sentence-transformers have all changed between versions. Check the installed version's documentation or source, and **tell the user what you found**.
- **Verify external facts before relying on them:** Groq model IDs, free-tier limits, and model-card details (e.g., the bge query prefix). Plans mark these "verify".

**Prompts**
- **Never change the wording of files in `prompts/`** without the user's approval. A change means a **new version file** (`v2`); `v1` is kept.

**Secrets and git**
- Never commit `.env` or API keys. Keep `.env.example` current, with empty values and comments.
- Don't commit or push to git unless the user asks. Before any push, run the secret scan (deliverables Stage D2), including git history.

**Documentation (applies whenever docs are written, mainly in phase 12)**
- **No hand-typed numbers.** Every metric in README, DESIGN or RESULTS comes from report files through `docs/build_results.py`. Generated blocks (between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`) are never edited by hand.
- **Docs describe the system as built.** If docs and code disagree, the code is the truth; report the mismatch.
- **Every command in the README must have been run** in the fresh-clone acceptance test (deliverables Stage D7).

## 6. System invariants (true in every phase; plans give the details)

**Documents and data**
- The **PDFs are the source of truth.** Markdown is a regenerated intermediate format. Page numbers and links come from PyMuPDF, never from the converter.
- Documents are identified by **`doc_id`** (`ai_rmf_1_0`, `genai_profile`) from `ingest/sources.yaml`, never by filename.
- Every chunk has exactly **one section path** and exact pages (`pdf_page_*` and `printed_page_*`).
- `embed_text` (breadcrumb + content) ≤ **512 tokens**, counted with the bge-small tokenizer, including special tokens.

**Retrieval**
- Hybrid search: `EnsembleRetriever` with vector (k=10, **query prefix on queries only**) + BM25 (k=10, our versioned `tokenize()`), weights **0.6/0.4**.
- Rerank all candidates with `bge-reranker-base`; keep **top 5** best-first; always add missing **sibling table parts**.
- The relevance gate uses the reranker **threshold the user chose** from the evaluation.

**LLM and conversation**
- **Routing:** a strict pure-greeting rule (no LLM), then **one combined LLM call** (route + standalone query) seeing the **last 2 turns**.
- **Routes:** `greeting`, `nist_question`, `out_of_scope`.
- **Generation** sees **only** the standalone query and the sources. **No conversation history.**
- **Groq free tier: tokens per minute is the binding limit.** No retry calls for content problems, no fallback model. Retries with backoff happen only for rate-limit and temporary errors, and **only before the first streamed token**.
- **Reasoning output** (gpt-oss models) is **never** streamed, returned, stored or logged at INFO.

**Fixed texts (exact; the code depends on them)**
- Out of scope: `Sorry, we don't have this answer.`
- Insufficient context: `The NIST documents provided do not contain this information.`
- Ambiguous answer, first line: `This question can be read in more than one way:`
- Uncovered reading: `not covered by the provided documents.`
- The greeting template text, as approved in the API phase.

**Answers and citations**
- Answers are **plain text** with inline **`[C#]`** markers, a citation on **every fact-stating sentence**, at most **about 150 words**, **English only**.
- **The LLM never writes URLs, page numbers or source lists.** Citations are built **in code** from `key_map` → chunk metadata.
- **Sources:**
  - **only cited chunks** appear;
  - entries with the **same label are merged**;
  - order is **document, then page**;
  - each entry is `{keys, label, url, kind}`;
  - label format: `NIST AI 100-1, §5.1 Govern, pp. 22–23`;
  - the URL is `#page={pdf_page_start}`;
  - `kind` is a separate field.
- **Insufficient answers:** `sources = []`, **`related_sections`** holds 5 entries (separate field, separate SSE event `related`), and `meta.insufficient_reason` is set.

**Streaming (SSE)**
- Successful stream: `meta → token… → [correction] → sources | related → done`. On failure: `→ error`, and nothing follows.
- Only content tokens from calls tagged **`answer_generation`** are streamed. Router output, reasoning and chunk text never are.

**State, errors, debug**
- Per-message state fields are reset each turn; only `messages` accumulates.
- History stores only the **final** answer text, never sources, related sections, the streamed draft, reasoning or partial answers.
- Node errors are written into state as `{code, message, node}` and routed to `error_finalize`. Responses carry only a code and a generic message, **never stack traces**.
- `debug=true` exposes path, scores, labels, IDs and usage, **never** chunk text, prompt text or reasoning.
- LangSmith is **off unless** its environment variables are set.

## 7. Code conventions

- Python with **type hints everywhere**; small single-purpose functions; plain functions plus dataclasses or Pydantic models. No class hierarchies or plugin frameworks.
- **Docstrings explain *why***, not just what. The user must explain this code in an interview.
- Use `logging` (no `print` in library code). Every log line starts with the phase or stage name. Never log full user messages or chunk text at INFO.
- Error messages say **what** failed, **where** (document, page, node, stage) and **what to do** next.
- Components (models, indexes, clients, prompts) are **loaded once at startup** and never stored in graph state. CPU-heavy work (embedding, BM25, reranking) runs in a worker thread inside async code.
- All limits, thresholds, model names and phrases live in **config modules**, not scattered as literals.
- Determinism: the same input gives the same output (sorted tie-breaks, fixed seeds).

## 8. Testing

- **Unit tests need no network and no API keys.** Mock or fake LLMs, Groq, and slow models.
- **Live tests** (Groq calls, real indexes, evaluations) are **skipped** when `GROQ_API_KEY` or the required data is missing. They pause between calls to respect rate limits and record token usage.
- Every rule gets **positive and negative tests**. Negative cases (things that must *not* happen) are the most important; never remove them.
- **Snapshot tests** guard the assembled prompt; update a snapshot only deliberately.
- **Leak tests** with marker strings protect the stream and the `debug` output.
- Run `pytest -q` before declaring any stage done.

## 9. Cross-plan resolutions (already settled)

| Earlier open question | Settled by |
|---|---|
| Indexing Q5 (BM25 library) | Retrieval plan: LangChain `BM25Retriever` with our `tokenize()` |
| Chunking Q1 (tokenizer package) | Indexing plan Q9: the bge-small tokenizer via `transformers` (confirm if not yet confirmed) |
| Insufficient Q6 (label format) | Attribution plan SA-D6 / SA-D7 |
| Router output must not stream | API plan R7, extended to subgraphs by graph integration G5 |
| `key_map` in `debug` | Needed by the end-to-end evaluation (E2E Q3); add to the graph phase's debug contents when approved |
| E2E case-file fields | `eval/e2e_cases.yaml` uses fields beyond the E2E plan's list (`must_not_cite`, `acceptable_alternative`, `expected_route_source`, `min_facts`, `expected_text_key`); the harness (E2E Stages V1/V3) must support them |
| Makefile vs Python CLI | The user chose a Makefile; the README must pair every target with its Python command (Windows) |

Questions that one phase leaves for a later one (e.g., generation Q7 history-on-error, attribution Q3 marker stripping) are re-asked at graph integration Stage G0 if they're still open. Record every new resolution in `docs/process/PROGRESS.md`.

## 10. Commands

```bash
# environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# data pipeline (phases 1–3)
python -m ingest            # --rebuild | --from-dir <path> | --from-stage <n>
python -m chunking          # --rebuild | --only-stage C<n>
python -m indexing          # --rebuild | --validate | --only bm25|chroma

# API (phase 4+)
uvicorn app.main:app --host 127.0.0.1 --port 8000      # host/port as approved in API phase

# tests
pytest -q

# evaluations (live; need GROQ_API_KEY and a running server where noted)
python -m eval.routing
python -m eval.retrieval_eval
python -m eval.e2e --base-url http://127.0.0.1:8000     # phase 11
```

**Makefile** (phase 12; it assumes an activated venv; on systems without `make`, use the Python commands above):

```bash
make help        # list targets
make build       # ingest → chunk → index (stops at the first failure)
make rebuild     # same, with --rebuild
make serve       # start the API
make test        # pytest -q
make e2e BASE_URL=http://127.0.0.1:8000   # end-to-end evaluation (server must be running)
make results     # regenerate docs/RESULTS.md and the README results block
make clean       # remove processed data and indexes (never data/raw)
```

Every target prints the Python command it runs (`README.md` pairs them), so the Python commands above work without `make`.

## 11. Repository layout (overview)

```
CLAUDE.md  README.md  DESIGN.md  Makefile
docs/        ← RESULTS.md (generated), E2E_RESULTS.md (generated), build_results.py, graph.md, architecture.html
docs/process/        ← PROGRESS.md, PROBLEMS_AND_SOLUTIONS.md, phase_notes.md, per-phase result summaries, README.md
docs/process/plans/  ← the 12 plan files
prompts/     ← versioned prompt files (answer_system.v1.md, answer_user.v1.md)
ingest/  chunking/  indexing/  retrieval/  augmentation/  generation/  insufficient/  attribution/
app/  graph/  llm/
eval/        ← evaluation case files, scripts, reports
tests/
data/        ← git-ignored: raw/, processed/, chroma/, index/
.env.example  requirements.txt  README.md
```
