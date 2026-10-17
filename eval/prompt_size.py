"""Stage A6: prompt size measurement. `python -m eval.prompt_size [--out PATH]`.

No LLM calls and no API key: runs the real retrieval for every query of the retrieval evaluation set, builds the
real prompt with build_messages(), and reports sizes. Tokens are an ESTIMATE (Q3): characters / CHARS_PER_TOKEN
(generation.config), shown with and without the 20% safety margin. Gate failures are included on purpose: they
never reach the model, but their chunks show how large a prompt *would* be."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from augmentation.context_builder import build_context
from augmentation.messages import build_messages
from augmentation.prompt_loader import get_prompts
from generation import config as gen_config
from retrieval import config as ret_config
from retrieval.models import RetrievalResult, RetrievedChunk

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = EVAL_DIR / "prompt_size_report.json"
MARGIN = 1.2                       # Q3: the stated 20% safety margin for the estimate
MAX_CHUNKS = 2 * ret_config.TOP_N  # retrieval plan Q6: top N plus siblings, at most 10 chunks


def est_tokens(chars: int, margin: float = 1.0) -> int:
    return math.ceil(chars / gen_config.CHARS_PER_TOKEN * margin)


def stats(values: list[int | float]) -> dict:
    """min / median / p95 / max (nearest-rank p95, so the number is one that actually occurred)."""
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {"min": ordered[0], "median": statistics.median(ordered), "p95": p95, "max": ordered[-1]}


def measure(query: str, result: RetrievalResult) -> dict:
    """Sizes of one real prompt (characters, then estimated tokens)."""
    messages, _ = build_messages(query, result)
    system_chars, user_chars = len(messages[0].content), len(messages[1].content)
    sources_chars = len(build_context(result)[0])
    total = system_chars + user_chars
    return {"chunks": len(result.chunks), "siblings": sum(c.added_as_sibling for c in result.chunks),
            "system_chars": system_chars, "sources_chars": sources_chars, "total_chars": total,
            "total_tokens_est": est_tokens(total), "total_tokens_est_margin": est_tokens(total, MARGIN)}


def chunk_from_record(rec: dict) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=rec["chunk_id"], display_text=rec["display_text"], metadata=rec)


def worst_case(lookup: dict[str, dict]) -> dict:
    """Computed upper bound: the MAX_CHUNKS chunks whose rendered source blocks are largest, in one prompt, with
    the longest query the API accepts. Real prompts cannot be larger unless retrieval limits change."""
    from app.config import get_settings
    sized = sorted(lookup.values(), key=lambda r: -len(build_context(_one(r))[0]))
    chosen = sized[:MAX_CHUNKS]
    result = RetrievalResult(query="", chunks=[chunk_from_record(r) for r in chosen], relevant=True,
                             best_score=None, threshold=None, candidate_count=len(chosen))
    out = measure("x" * get_settings().max_message_chars, result)
    out["chunk_ids"] = [r["chunk_id"] for r in chosen]
    return out


def _one(rec: dict) -> RetrievalResult:
    return RetrievalResult(query="", chunks=[chunk_from_record(rec)], relevant=True, best_score=None,
                           threshold=None, candidate_count=1)


def run() -> dict:
    from eval.retrieval_eval import load_cases
    from retrieval.components import get_components
    from retrieval.pipeline import retrieve, warmup

    warmup()
    cases = load_cases()
    rows = []
    for case in cases:
        result = retrieve(case["query"], evaluation_mode=True)
        rows.append({"id": case["id"], "answerable": case["answerable"], "relevant": result.relevant,
                     **measure(case["query"], result)})
    report: dict = {"prompt_version": get_prompts().version, "queries": len(rows),
                    "estimator": f"characters / {gen_config.CHARS_PER_TOKEN}; margin column x{MARGIN} (Q3)",
                    "limits": {"tpm": gen_config.TPM_LIMIT, "max_completion_tokens": gen_config.MAX_COMPLETION_TOKENS,
                              "model_context": 131072},
                    "summary": {k: stats([r[k] for r in rows]) for k in
                                ("chunks", "siblings", "system_chars", "sources_chars", "total_chars",
                                 "total_tokens_est", "total_tokens_est_margin")},
                    "worst_case": worst_case(get_components().chunk_lookup), "per_query": rows}
    budget = gen_config.TPM_LIMIT - (gen_config.MAX_COMPLETION_TOKENS or 0)
    report["prompt_budget_tokens"] = budget       # what is left of the TPM after reserving the output
    report["over_budget"] = {"observed_est": sum(r["total_tokens_est"] > budget for r in rows),
                             "observed_est_margin": sum(r["total_tokens_est_margin"] > budget for r in rows),
                             "worst_case_est": report["worst_case"]["total_tokens_est"] > budget}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()
    report = run()
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for k, v in report["summary"].items():
        print(f"{k:26s} {v}")
    w = report["worst_case"]
    print("worst case:", {k: w[k] for k in ("chunks", "total_chars", "total_tokens_est", "total_tokens_est_margin")})
    print("prompt budget:", report["prompt_budget_tokens"], "over_budget:", report["over_budget"])


if __name__ == "__main__":
    main()
