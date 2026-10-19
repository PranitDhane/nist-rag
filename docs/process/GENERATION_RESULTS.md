# Generation results

Live runs against Groq `openai/gpt-oss-20b` (reasoning effort medium, temperature 0.2, reasoning hidden at the API and filtered in the stream), prompt `v1`. Reports: `eval/generation_usage_report.json` (Stage N4), `eval/generation_behaviour_report.json` (Stage N5). Reproduce with `python -m eval.generation_eval measure|behaviour`; `rescore` recomputes checks from saved answers without API calls.

## Limits and token usage (Stage N4: 27 live requests)

Account limits for the model (response headers): **8,000 tokens per minute** (binding), 1,000 requests (daily cap by Groq convention, unconfirmed). Context window 131,072.

| | min | median | p95 | max |
|---|---|---|---|---|
| Prompt tokens | 1,161 | 2,509 | 3,212 | 3,500 |
| Reasoning tokens | 78 | 229 | 1,244 | 1,608 |
| Visible answer tokens | 20 | 79 | 195 | 275 |
| Total tokens | 1,275 | 2,925 | 4,602 | 4,939 |
| TTFT (ms) | 553 | 1,155 | 2,512 | 2,659 |
| Total latency (ms) | 587 | 1,213 | 2,600 | 2,907 |

The largest single request used 4,939 of the 8,000 per-minute tokens, so about 1.6 requests per minute fit at the worst case and about 2.7 at the median. The `chars / 4 x 1.2` estimator used by the budget pre-check over-estimated the real prompt by about 20% (measured 4.0 chars per token).

**Reasoning tokens vary a lot between runs of the same input.** The broad "explain everything" case used 1,608 reasoning tokens in the measurement run and 2,498 in the behaviour run, where it hit the 2,500-token output limit (`finish_reason = length`) and became an `ERROR` under the chosen length policy. The measured maximum (1,883) was therefore not a safe ceiling.

## Behaviour (Stage N5: 17 cases, final settings, streamed)

| Metric | Result |
|---|---|
| Case accuracy (A/B/D) | **13/17 (76.5%)** |
| Answers with at least one uncited sentence | 30.0% (3 of 10 answered); 17.6% of sentences (3 of 17) |
| Over-length answers (>150 words) | 0% (the one broad case errored before finishing) |
| Invented-citation rate | 0% |
| Correction rate | 12.5% (2 of 16 completed) |
| `INSUFFICIENT` produced by a fix | 6.2% (1 of 16) |
| Average tokens / latency | 2,021 tokens / 978 ms |
| Case checks passing | 33/47 strict, 35/47 after normalizing typographic characters |

### Cases where the actual case differed from the expected one

| Case | Expected | Actual | Cause |
|---|---|---|---|
| `a02-cross-cutting` | A | D | The model answered correctly but cited `[G1][G2]` instead of `[C1][C2]`. The parser sees no valid citation, so the fixer applied `NO_CITATIONS` and replaced a correct answer with the Case D sentence. First run of the same case was fine. |
| `b01-both-readings-covered` | B | A | The model merged the two readings into one paragraph and did not use the Case B format. Seen in both runs. |
| `b02-only-one-reading-covered` | B | A | The model answered one reading as a bullet list and did not state the other is not covered. Seen in both runs. |
| `n01-broad-question-stays-short` | A | ERROR | Output limit reached (2,498 of 2,500 tokens were reasoning). See above. |

### Other check failures

- `a01`: four functions written as one sentence, not bullets (check wants at least 4 bullets).
- `a03`: the model quoted a different passage of the confabulation text than the check expects.
- `a04`, `a05`: the model wrote `GV‑1.7‑001` with non-breaking hyphens and `Figure 5` with a narrow no-break space. The strict check fails; the normalized check passes. Exact IDs with these characters will not match a plain search or copy-paste.
- `a02`: the downstream effect of the `[G1]` citation (answer lost).

### Honest notes

- **Three of the four "uncited sentence" warnings are a parser artifact,** not missing citations: the model wrote `... systems. [C1]` (marker after the period). `augmentation.parser` treats the sentence before the marker as uncited. The fourth is the `[G1]` case. I did not change the parser (it belongs to the augmentation phase) or the policy.
- **Uncited sentences and over-length answers are kept with warnings** (GN-D6, GN-D7). Over this small run the real rates were: uncited about 1 of 17 sentences, over-length 0.
- The case file was run as provided and was not edited. Results are from one run per case at temperature 0.2, so single-case differences between runs are expected (a02 and n01 changed between the two runs).
- No 429 occurred in 44 live requests; calls were paced under the per-minute token limit, so the optional local limiter (Q10) is not needed yet.

## Decisions recorded

Q1 model `openai/gpt-oss-20b`; Q2 `reasoning_format="hidden"` plus stream filter; Q3 `MAX_COMPLETION_TOKENS = 2500` (to be reviewed, see above); Q4 `finish_reason = length` becomes `ERROR`; Q5 `correction` event `{text, reasons}` approved; Q6 timeouts TTFT 20 s / total 60 s; Q7 roll the turn back on `ERROR`; Q8 retries same as the router.

## Changes made during graph integration

- `MAX_COMPLETION_TOKENS` raised from 2,500 to **3,500** (the output limit includes hidden reasoning, which varied from ~1.6k to 2.5k+ on the broad case), at the user's request.
- The budget pre-check's estimator margin went from 1.2 to **1.0** (measured 4.0 chars per token): with 3,500 and 1.2 it rejected 2 of 14 real NIST turns whose real prompts (~3.8k and ~4.3k tokens) fit under the 8,000-token limit.
- Live finding, not patched: with real retrieval the model answered "functions + penalties" partly and noted what was missing instead of refusing entirely (the two-part Case C behaviour the prompt asks for); the fixture version of that case passed earlier.
