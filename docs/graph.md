# Graph

The diagrams below are exported from the real compiled graphs; `tests/test_graph_docs.py` fails if this file goes out of date.
Regenerate with `python -m tests.test_graph_docs`.

## Main graph (conversation handling)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	reset_turn(reset_turn)
	rule_greeting_check(rule_greeting_check)
	classify_and_rewrite(classify_and_rewrite)
	greeting_reply(greeting_reply)
	out_of_scope_reply(out_of_scope_reply)
	rag(rag)
	finalize(finalize)
	error_finalize(error_finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> reset_turn;
	classify_and_rewrite -.-> error_finalize;
	classify_and_rewrite -.-> greeting_reply;
	classify_and_rewrite -.-> out_of_scope_reply;
	classify_and_rewrite -.-> rag;
	greeting_reply --> finalize;
	out_of_scope_reply --> finalize;
	rag -.-> error_finalize;
	rag -.-> finalize;
	reset_turn --> rule_greeting_check;
	rule_greeting_check -.-> classify_and_rewrite;
	rule_greeting_check -.-> greeting_reply;
	error_finalize --> __end__;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Every message starts at `reset_turn` (all per-message state fields reset, user message stored), then `rule_greeting_check`
(a message that is *entirely* a greeting/thanks/goodbye never reaches an LLM). Everything else goes to `classify_and_rewrite`
(LLM call #1: route + standalone question). Four ways out:

| Route | Path | LLM calls |
|---|---|---|
| Pure greeting (rule) | `reset_turn → rule_greeting_check → greeting_reply → finalize` | 0 |
| Greeting (LLM) | `… → classify_and_rewrite → greeting_reply → finalize` | 1 |
| Out of scope | `… → classify_and_rewrite → out_of_scope_reply → finalize` | 1 |
| NIST question | `… → classify_and_rewrite → rag (subgraph) → finalize` | 1 + 1 if it reaches generation |
| Any failure | `… → error_finalize` (the one error node: outcome `error`, the unanswered user message is rolled back) | |

`finalize` appends exactly one assistant message per turn (the final answer text with `[C#]` markers stripped). Sources,
related sections, retrieval and generation data are never written into `messages`.

## RAG subgraph (answering), expanded

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	reset_turn(reset_turn)
	rule_greeting_check(rule_greeting_check)
	classify_and_rewrite(classify_and_rewrite)
	greeting_reply(greeting_reply)
	out_of_scope_reply(out_of_scope_reply)
	finalize(finalize)
	error_finalize(error_finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> reset_turn;
	classify_and_rewrite -.-> error_finalize;
	classify_and_rewrite -.-> greeting_reply;
	classify_and_rewrite -.-> out_of_scope_reply;
	classify_and_rewrite -.-> rag\3ahybrid_search;
	greeting_reply --> finalize;
	out_of_scope_reply --> finalize;
	rag\3a__end__ -.-> error_finalize;
	rag\3a__end__ -.-> finalize;
	reset_turn --> rule_greeting_check;
	rule_greeting_check -.-> classify_and_rewrite;
	rule_greeting_check -.-> greeting_reply;
	error_finalize --> __end__;
	finalize --> __end__;
	subgraph rag
	rag\3ahybrid_search(hybrid_search)
	rag\3arerank(rerank)
	rag\3agate(gate)
	rag\3aexpand_context(expand_context)
	rag\3agate_failure_reply(gate_failure_reply)
	rag\3afinalize_model_refusal(finalize_model_refusal)
	rag\3aattribute(attribute)
	rag\3agenerate(generate)
	rag\3a__end__(<p>__end__</p>)
	rag\3aexpand_context -.-> rag\3a__end__;
	rag\3aexpand_context -.-> rag\3agenerate;
	rag\3agate -.-> rag\3a__end__;
	rag\3agate -.-> rag\3aexpand_context;
	rag\3agate -.-> rag\3agate_failure_reply;
	rag\3agenerate -.-> rag\3a__end__;
	rag\3agenerate -.-> rag\3aattribute;
	rag\3agenerate -.-> rag\3afinalize_model_refusal;
	rag\3ahybrid_search -.-> rag\3a__end__;
	rag\3ahybrid_search -.-> rag\3arerank;
	rag\3arerank -.-> rag\3a__end__;
	rag\3arerank -.-> rag\3agate;
	rag\3aattribute --> rag\3a__end__;
	rag\3afinalize_model_refusal --> rag\3a__end__;
	rag\3agate_failure_reply --> rag\3a__end__;
	end
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

| Scenario | Subgraph path | Outcome |
|---|---|---|
| Answered | `hybrid_search → rerank → gate → expand_context → generate → attribute` | `answered` |
| Gate failure (best rerank score below the threshold, or no candidates) | `hybrid_search → rerank → gate → gate_failure_reply` (no generation call) | `insufficient` (`gate_low_score` / `no_candidates`) |
| Model refusal (the model wrote the Case D sentence) | `… → generate → finalize_model_refusal` | `insufficient` (`model_case_d`) |
| Fixer-forced refusal | `… → generate → finalize_model_refusal` | `insufficient` (`fix_partial_answer`, `fix_no_citations`, `fix_empty_output`) |
| Any node failure | the failing node, then the subgraph exits; the main graph routes to `error_finalize` | `error` |

There are no loops and no retries anywhere: the graph is a decision tree.
