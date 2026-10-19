# Process documents

These files record **how the system was built**, so the reasoning behind the code can be inspected. They are not needed to run the project; start with the [README](../../README.md) and [DESIGN.md](../../DESIGN.md).

The project was built in twelve phases with an AI coding assistant working from a written plan per phase, one stage at a time. After each stage the assistant stopped, showed the result, and waited for confirmation; every open design question in a plan was asked at the stage named and its answer recorded.

| File | What it is | How it was used |
|---|---|---|
| [`plans/`](plans/) | One plan per phase: ingestion, chunking, embedding and indexing, API and routing, retrieval, augmentation, generation, insufficient context, attribution, graph integration, end-to-end evaluation, deliverables | The specification for the phase: goal, decisions already made by the user, stages with checkpoints, tests, and an "open questions" table. The assistant could not change a recorded decision or build outside the current phase |
| [`PROGRESS.md`](PROGRESS.md) | The running log, one line per finished stage: date, phase, stage, result, answers to open questions, open issues | Appended after every stage, never rewritten (corrections are new lines). Older entries keep the file paths that were valid when they were written |
| [`PROBLEMS_AND_SOLUTIONS.md`](PROBLEMS_AND_SOLUTIONS.md) | Every problem found while building and the decision taken, by phase, with its status | Readable index into `PROGRESS.md`; source material for `DESIGN.md` |
| [`phase_notes.md`](phase_notes.md) | The project README as it stood during the build (per-phase notes) | Kept as a historical record; its numbers were typed by hand then and may be out of date |
| [`RETRIEVAL_RESULTS.md`](RETRIEVAL_RESULTS.md), [`GENERATION_RESULTS.md`](GENERATION_RESULTS.md) | Per-phase result summaries written when those phases were finished | Superseded by the generated [`docs/RESULTS.md`](../RESULTS.md) |

The root [`CLAUDE.md`](../../CLAUDE.md) holds the working rules that applied in every phase (never hard-code expected results, never weaken a failing test, never change a recorded decision, ask at the stage a question is due, and so on). It stays at the repository root because that is where Claude Code loads it from.

Rules the plans enforced, and where to see the evidence: tests with positive and negative cases for every rule (`tests/`), evaluations that report failures instead of hiding them (`eval/`, [`docs/RESULTS.md`](../RESULTS.md), [`docs/E2E_RESULTS.md`](../E2E_RESULTS.md)), and documents whose numbers are generated from the reports (`docs/build_results.py`).
