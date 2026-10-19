# Design explanation

How the NIST AI RMF question-answering system is built, and why. The six numbered points in section 2 follow the six topics the assignment asks for. The `Evidence` paragraphs are **generated** from the evaluation reports by `docs/build_results.py` (`make results`); everything else is reasoning. Full tables are in [`docs/RESULTS.md`](docs/RESULTS.md), diagrams are in [`docs/graph.md`](docs/graph.md) and [`docs/architecture.html`](docs/architecture.html).

The system answers questions about two documents, **NIST AI 100-1 (AI RMF 1.0)** and **NIST AI 600-1 (Generative AI Profile)**. It is a LangGraph app behind a FastAPI API, with Groq's free tier as the LLM, a local embedding model, a local reranker and local indexes. Every design choice was made so that an answer is grounded in the documents, cites a page a reader can open, and says so when the documents do not contain the answer.

## 1. Design at a glance

| Area | Decision | Why |
|---|---|---|
| Source of truth | The PDFs. Page numbers and links come from PyMuPDF; Markdown is only an intermediate format | A citation must open the right page; converters get page numbers wrong |
| Chunking | Section-first, table-aware; 250–350 token prose chunks; every chunk has one section path and exact pages | A chunk that mixes sections gives a misleading citation |
| Embedding and store | `bge-small-en-v1.5` (384-dim, normalised, cosine) in one local Chroma collection | Small and fast on a laptop, 512-token input, good retrieval quality for its size |
| Retrieval | Vector (k=10) + BM25 (k=10), weighted 0.6 / 0.4, fused by RRF, then a cross-encoder reranks all candidates; top 5 | Vectors handle paraphrases, BM25 handles identifiers such as `GV-1.7-001`, the reranker orders the result |
| Workflow | Main graph (route and converse) + a RAG subgraph (retrieve, answer); no loops | Every path is explicit and testable |
| Insufficient context | Three layers: router (`out_of_scope`), relevance gate (reranker threshold), and the answering prompt; partly answerable questions are refused | An honest "not in the documents" beats a confident partial answer |
| Conversation | Server-side `thread_id` with an in-memory checkpointer; one LLM call rewrites a follow-up into a standalone question; generation sees no history | Cheap, and old turns cannot leak into an answer |
| Grounding | Plain text with `[C#]` markers; sources are built in code from those markers | The LLM can never invent a URL, page number or source list |
| Free tier | Small answering model, tokens-per-minute budget checked before each call, no fallback model | The 8,000 tokens/minute limit, not request count, is the binding constraint |

## 2. The six points the assignment asks for

### 2.1 How the documents were split, and why

**Problem.** Questions are about specific sections, tables and action IDs, and each answer must cite exact pages. Naive fixed-size splitting would cut tables mid-row and merge sections.

**Decision.** Ingestion extracts every page with PyMuPDF (text, printed page number, links), converts it to Markdown, cleans running headers, footnotes and hyphenation, and finds the section headings. When chunking joins the pages it records a character-offset **page map**, so each chunk's pages are computed rather than searched for. Chunking then works **section first**: one chunk never spans two sections.
- **Prose:** 250–350 tokens of content, about 10 % overlap, and only between pieces of the same section. Sections that are only a heading or a few words (60 tokens or fewer) are merged forward into the next one.
- **Tables:** the unit is the natural table (a GenAI subcategory table, an AI RMF Core category). A table is split only **between rows**, into pieces of at most 450 tokens, and the header repeats in every piece.
- **Footnotes and figures:** footnotes become their own small chunks; each figure with OCR text becomes its own chunk (caption plus OCR text; a long one is split in two).
- **Text that is embedded** is the breadcrumb (document, section path) plus the content; the text that is *shown* stays clean. The hard limit of 512 tokens is counted with the embedding model's own tokenizer.

**Alternatives.** The chunker uses ingestion's own heading detection instead of LangChain's `MarkdownHeaderTextSplitter`, because that detection is validated against the expected section list of each document (each expected section must be found exactly once). Finding a chunk's pages by searching the PDF for its text was rejected in favour of the page map.

**Trade-offs.** The rules are tuned to these two documents (the cleaning rules, the table detection, a few repairs for icon-font garbling in the PDF). Footnote markers are not linked to the sentences that use them, and the References appendix is not retrievable. Figure 2 has no usable OCR text, so it has no chunk.

<!-- EVIDENCE:chunking:START -->
**Evidence** ([results](docs/RESULTS.md#corpus-chunks-and-indexes)): 208 chunks (107 prose, 63 genai action table, 19 rmf core table, 14 footnote, 5 figure). Prose chunks have a median of 325 embedded tokens (maximum 385); the longest chunk of any type is 478 tokens, under the 512-token limit. 14 GenAI action tables were split between rows, 4 tiny sections were merged forward, and 9/9 chunk validation checks pass.
<!-- EVIDENCE:chunking:END -->

### 2.2 Why this embedding model and vector store

**Decision.** `BAAI/bge-small-en-v1.5` with normalised vectors and cosine distance, stored in a **single persistent Chroma collection** for both documents (`doc_id` is a metadata field). BM25 (`rank_bm25`) runs alongside on the same chunks. The model revision is pinned, and an index manifest records the model, revision, normalisation, distance, tokenizer version, chunk count and a hash of the chunks file; at start-up the application refuses to run if the embedding model, revision, normalisation or distance differ from what the index was built with.

**Why.** The model is small (384 dimensions), runs on a laptop CPU or GPU, and has a 512-token input window, which the chunker is sized for. Chroma is local, persistent and needs no server, which keeps the project runnable from a fresh clone. One collection keeps cross-document questions simple.

**Why BM25 as well.** Dense vectors are weak on exact identifiers. The index build's own smoke test shows it: the vector index misses `GV-1.7-001` while BM25 ranks the right chunk first. The BM25 tokenizer therefore keeps action IDs whole and stems everything else, and the index is saved as a JSON token list and rebuilt at load time (no pickle).

**Alternatives and trade-offs.** A larger embedding model was not tried: it would cost download size and start-up time on every machine, and the reranker (next section) is what lifts the final ranking, as the evidence below shows. The query prefix that bge recommends is added to queries only, never to documents.

<!-- EVIDENCE:indexing:START -->
**Evidence** ([results](docs/RESULTS.md#corpus-chunks-and-indexes)): `BAAI/bge-small-en-v1.5` produces 384-dimensional normalised vectors stored in one cosine collection of 208 chunks; the BM25 index has a vocabulary of 2,408 terms (tokenizer `1-ids-whole+snowball-english`).
<!-- EVIDENCE:indexing:END -->

### 2.3 How retrieval works

1. The standalone query is embedded (with the bge query prefix) and searched in Chroma (k=10); the same query goes to BM25 (k=10).
2. LangChain's `EnsembleRetriever` fuses both lists with weighted Reciprocal Rank Fusion (weights 0.6 vector / 0.4 BM25) and removes duplicates by chunk ID.
3. A local cross-encoder, `bge-reranker-base`, scores **every** fused candidate against the query and the best 5 are kept, best first.
4. The relevance gate checks the best score against a threshold (section 2.5).
5. If a selected chunk is one part of a split table, the missing sibling parts are added next to it so the model never sees half a table. Siblings are extra: they do not count toward the 5.

The reranker is applied to all candidates, not just the top few, because the fusion step only knows ranks, not relevance; the cross-encoder reads query and chunk together. The threshold is applied to the reranker's score and not the fused score, because reciprocal-rank fusion scores depend only on ranks, not on how relevant a chunk is.

**Trade-offs.** The reranker's input limit is 512 tokens, so a few long table chunks are truncated when scored. `EnsembleRetriever` hides the fusion scores (a diagnostic mode reproduces them). Weights, k and the threshold were chosen on a small hand-labelled set, so they are reasonable defaults rather than tuned optima.

<!-- EVIDENCE:retrieval:START -->
**Evidence** ([results](docs/RESULTS.md#retrieval)), 33 answerable questions: Hit@5 is 0.879 for vector search alone, 0.879 for BM25 alone and 0.879 for the hybrid, and **1.000** with the reranker; MRR rises from 0.781 to 0.899. The correct chunk was within the first 20 fused candidates in 33/33 cases (Recall@20), which is what the reranker gets to see. Retrieval takes p50 755.1 ms and p95 934.5 ms; 27 of 745 reranked pairs exceed the reranker's input limit.
<!-- EVIDENCE:retrieval:END -->

### 2.4 How the LangGraph workflow is structured

```
reset_turn → rule_greeting_check ─ pure greeting ──────────────► greeting_reply ─┐
                    │ else                                                       │
                    ▼                                                            │
          classify_and_rewrite  (LLM #1: route + standalone query)               │
            ├ greeting ──────────────────────────────────────► greeting_reply ───┤
            ├ out_of_scope ──────────────────────────────────► out_of_scope_reply┤
            ├ nist_question ──► RAG subgraph ────────────────────────────────────┤──► finalize
            └ error ─────────────────────────────────────────► error_finalize    │
RAG subgraph:  hybrid_search → rerank → gate ─ fail ─► gate_failure_reply
                                          └ pass ► expand_context → generate ─ answered ► attribute
                                                                          └ Case D ► finalize_model_refusal
```

- **Two graphs.** The main graph handles routing and conversation; the RAG subgraph is compiled once and used as one node of the main graph. Retrieval is split into separate nodes (`hybrid_search`, `rerank`, `gate`, `expand_context`) so each step records its own timing and can fail on its own.
- **No loops.** There is no retry or re-query loop: an answer that fails validation is repaired by deterministic code or becomes the insufficient-context reply. This keeps the cost per question bounded at two LLM calls, which matters on the free tier.
- **State.** Per-message fields (route, retrieval result, generation result, sources, timings) are reset at the start of every turn; only the message history accumulates. Models, indexes, prompts and the LLM client are loaded once at start-up and captured by the nodes, never stored in graph state, so state stays plain, serialisable data. CPU-heavy work (embedding, BM25, reranking) runs in worker threads so the event loop is not blocked.
- **Errors.** Each node catches its own failure and writes `{code, message, node}` into state; the graph routes to a single `error_finalize` node, which rolls the unanswered user message out of the history. The API turns the code into an HTTP status with a generic message and never returns a stack trace.
- **Streaming.** Only tokens from the call tagged `answer_generation` are streamed. Router output, reasoning text and chunk text are never sent. If validation changes the text after streaming, a `correction` event carries the final text.

<!-- EVIDENCE:routing:START -->
**Evidence** ([results](docs/RESULTS.md#routing)): the router passed 33/35 cases (97.1% route accuracy); 3 of 35 were answered by the greeting rule without an LLM call. Follow-ups passed 5/6. Failures: `nist-6` (nist_question → out_of_scope); `follow-4` (right route, standalone-query terms missing).
<!-- EVIDENCE:routing:END -->

### 2.5 How insufficient or irrelevant context is handled

There are three independent layers, from cheapest to most expensive, and no retries at any of them.

1. **Irrelevant to the documents (router).** The combined routing call classifies a message as `greeting`, `nist_question` or `out_of_scope`. Out-of-scope messages get the fixed reply *Sorry, we don't have this answer.* with no retrieval and no generation. A strict rule catches a message that is *entirely* a greeting before any LLM call.
2. **Relevant topic, nothing relevant retrieved (gate).** If the best reranker score is below the threshold (0.15), the system replies with *The NIST documents provided do not contain this information.* straight away, without calling the answering model, and returns the five closest sections in a separate `related_sections` field (never inside `sources`) plus a machine-readable `insufficient_reason`.
3. **Retrieved but not answerable (prompt and fixer).** The answering prompt tells the model to use only the sources and to give the insufficient-context sentence when they do not contain the answer. A **partly** answerable question should be refused entirely rather than answered partially, so a confident answer never hides a gap; the prompt asks for this, but following it is up to the model (see the evidence). If the model mixes the sentence with other text, answers with no valid citation, or returns nothing, the code replaces the whole output with the same sentence.

The threshold (0.15) was chosen from the retrieval evaluation. A question such as *"What penalties does the EU AI Act set?"* is stopped by layer 1 or 2 depending on how the router sees it; both are acceptable and both are reported.

**Trade-offs.** The gate is tuned on a small set of negative questions, so it can wrongly reject an unusual paraphrase and can pass an off-topic question that happens to use the documents' vocabulary. The related sections are simply the five best-scoring chunks, so they can be loosely related. Strictness has a price: in the end-to-end run the model sometimes refuses a question the documents can answer.

<!-- EVIDENCE:gate:START -->
**Evidence** ([results](docs/RESULTS.md#retrieval)): on 10 unanswerable questions the best reranker score was at most 0.0729; on answerable ones it ranged from 0.0054 to 0.9998. At a threshold near 0.15 (table row 0.1579) 100% of unanswerable questions are rejected and 93.9% of answerable ones are kept, so the threshold was tuned on only 10 negative cases.
<!-- EVIDENCE:gate:END -->

<!-- EVIDENCE:insufficient:START -->
**Evidence** ([results](docs/RESULTS.md#insufficient-context-path)): 10/10 unanswerable questions received the insufficient-context reply, 10 of them from the relevance gate alone (no generation call); the five related sections looked unrelated for 8 of them. The cost of strictness shows in the end-to-end run: 2 questions the documents can answer were refused (C1-T3, C2-T2). Compliance with the 'refuse partly answerable questions' rule is up to the model: in the integration run the two-part question was answered partially (outcome `answered`).
<!-- EVIDENCE:insufficient:END -->

### 2.6 How conversational context is maintained

- **Identity.** The server creates a `thread_id` on the first message and returns it; the client sends it back on later messages. An unknown or expired id is a 404, and a second simultaneous request on the same thread is a 409.
- **Storage.** A LangGraph in-memory checkpointer keeps the thread's messages. The history contains only the **final answer text** with its `[C#]` markers stripped: never sources, related sections, the streamed draft or reasoning text. A failed turn is rolled back so the history never ends with an unanswered question. Checkpoints are written once per request (`durability="exit"`) rather than once per graph step.
- **Follow-ups.** The routing call sees the last 2 turns and returns both the route and a self-contained **standalone query** ("Which one is cross-cutting?" becomes a question that names the AI RMF functions). Retrieval and generation use only that query.
- **No history in generation.** The answering model sees the standalone query and the sources and nothing else. This saves tokens and means an old turn cannot leak into, or contradict, a grounded answer.

**Trade-offs.** History is lost when the server restarts, and memory grows with conversation length. A persistent checkpointer and history trimming are the obvious next steps. If the routing call misreads a follow-up, retrieval looks for the wrong thing; the standalone query is therefore returned in the response (and in `debug`) so it can be inspected.

<!-- EVIDENCE:conversation:START -->
**Evidence** ([results](docs/RESULTS.md#end-to-end-evaluation)): in the end-to-end run 3/3 follow-up questions were rewritten into a standalone query containing the expected terms. Checkpointing once per request (`durability="exit"`) costs 86,927 bytes per NIST turn, against 611,065 with LangGraph's per-step default over the same 20-turn conversation.
<!-- EVIDENCE:conversation:END -->

## 3. Grounding and citations

- **Prompt.** The system prompt has five sections: purpose, constraints, interpretation ("do not assume anything"), decision (answer / answer each reading of an ambiguous question / insufficient), and output format. The loader refuses to start if a section, the exact insufficient-context sentence or a placeholder is missing. Sources reach the model as `<source id="C1" ...>` blocks with no page numbers or URLs, and text inside them is declared to be data, not instructions.
- **Output.** Plain text with `[C1]` markers on every sentence that states a fact, about 150 words at most, English only. Plain text was chosen over JSON because it streams cleanly token by token.
- **Citations are built in code.** The parser extracts the markers; a key map links each `C#` to a chunk; the response's `sources` list (label, `#page=N` link, kind) is built from chunk metadata, merged by label and ordered by document and page. The model cannot invent a URL or a page.
- **Validation without retries.** A deterministic fixer normalises malformed markers, removes invented keys, source lists, URLs and headings, and falls back to the insufficient-context sentence when nothing valid is left. It never rewrites sentences. An **uncited sentence or an over-length answer is kept and logged as a warning**, not rejected: a retry would cost another call on a tight token budget, and the warning rate is measured instead.
- **Verifiable links.** A dedicated check opens every chunk's link in the PDF and confirms the cited text is on that page.

<!-- EVIDENCE:grounding:START -->
**Evidence** ([results](docs/RESULTS.md#citation-links)): all 208 chunk links open a PDF page containing the cited text (0 failed). In the end-to-end run 11/11 cited sentences scored at least 0.7 against their cited chunk, and 3 of 14 fact-stating sentences carried no citation. In the 17 generation behaviour cases the invented-citation rate was 0%, the over-length rate 0% and 17.6% of sentences were uncited.
<!-- EVIDENCE:grounding:END -->

## 4. Free-tier constraints

Groq's free tier limits tokens per minute (8,000), not just requests, so every design choice that costs tokens was weighed against that.
- The cheap, fast model (`gpt-oss-20b`) writes answers; the larger one (`gpt-oss-120b`) only classifies and rewrites with low reasoning effort. Reasoning output is hidden at the API level and never streamed, stored or logged.
- A budget check estimates prompt plus the output cap before each call and fails fast instead of being rate-limited.
- Retries with backoff happen only for rate-limit or temporary errors and only before the first streamed token. There is no fallback model and no retry for content problems.
- A question costs at most two LLM calls, and many (greetings, out-of-scope, gate failures) cost zero or one.

The practical consequence is low throughput on the free tier (the evidence below gives the figure), which is why the end-to-end harness paces its requests.

<!-- EVIDENCE:freetier:START -->
**Evidence** ([results](docs/RESULTS.md#generation)): a generation call used a median of 2,925 tokens (p95 4,602, maximum 4,939) against the 8,000 tokens-per-minute limit; with the router call an answered end-to-end turn averaged 4,227 tokens. Prompts are 3,123 tokens at the median (4,378 at most). At that average the limit allows about 1.9 answered questions per minute, so the end-to-end harness paces 35 s after each generation turn.
<!-- EVIDENCE:freetier:END -->

## 5. Evaluation approach

| Evaluation | What it measures | Main limitation |
|---|---|---|
| Retrieval (`eval.retrieval_eval`) | Hit@5 and MRR of vector, BM25, hybrid and hybrid + reranker, against structural gold labels (document and section, never chunk IDs) | Hand-written cases; the threshold is tuned on few negatives |
| Routing (`eval.routing`) | Route accuracy and follow-up rewriting, including greetings mixed with questions | Small set; the router is an LLM |
| Generation behaviour and usage | Whether the real model follows the prompt (case type, citations, length) and what it costs in tokens and time | Small set; temperature 0.2 gives wording variance |
| Insufficient path | Whether unanswerable questions are stopped, and what the related sections look like | Only near-scope questions |
| Link check | Every chunk's citation link against the PDF text | Layout-based matching has some explained mismatches |
| Graph integration | Real Groq and real indexes, per path: latency, tokens, memory growth | One run per path |
| End to end (`eval.e2e`) | The case file's requests through the running HTTP API (two 3-turn conversations included): routes, outcomes, exact texts, gold key facts, citations and links, a local support score, and the streaming protocol | Small sample; the support score uses the same reranker as retrieval, so it is not an independent judge |

The end-to-end evaluation only observes: it makes no LLM calls of its own, it never gates, and every failed check is listed. Gold data is itself verified against the PDFs before use.

<!-- EVIDENCE:e2e:START -->
**Evidence** ([results](docs/RESULTS.md#end-to-end-evaluation)): 16 requests through the running API; route accuracy 15/16, outcome accuracy 13/16, key-fact coverage 29/45, 19 failed checks (all listed in `docs/E2E_RESULTS.md`).
<!-- EVIDENCE:e2e:END -->

## 6. Known limitations that follow from these decisions

The README lists them with the others; the ones that come straight from the design are: in-memory history; strict refusal of partly answerable questions; uncited sentences and over-length answers kept with a warning; related sections that can be loosely related; thresholds and weights tuned on small sets; cleaning and table rules tuned to two documents; English answers only; and low throughput on the free tier.
