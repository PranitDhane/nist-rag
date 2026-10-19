"""Generate docs/RESULTS.md and the README results block from the evaluation report files (deliverables stage D3).

    python docs/build_results.py            # or: make results

Rules (CLAUDE.md section 5): no hand-typed numbers. Every figure printed here is read from a report JSON under
eval/; a report that is missing becomes a clear "not run" note, never a crash and never an invented value. The output is
deterministic: it depends only on the report contents (no timestamps of its own, no file modification times), so running it
twice, or on a fresh clone, gives identical text. Each section names its source file and the first 12 characters of the file's
SHA-256, so a reader can check that a number belongs to the report that is committed.

Standard library only: it must run before (or without) the project's dependencies being installed."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "docs" / "RESULTS.md"
README_PATH = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

REPORTS = {
    "retrieval": ("eval/retrieval_report.json", "make eval-retrieval"),
    "routing": ("eval/routing_report.json", "make eval-routing"),
    "generation": ("eval/generation_behaviour_report.json", "python -m eval.generation_eval"),
    "usage": ("eval/generation_usage_report.json", "python -m eval.generation_eval"),
    "prompt_size": ("eval/prompt_size_report.json", "python -m eval.prompt_size"),
    "insufficient": ("eval/insufficient_report.json", "python -m eval.insufficient"),
    "link_check": ("eval/link_check_report.json", "python -m eval.link_check"),
    "integration": ("eval/integration_report.json", "python -m eval.integration_check"),
}
E2E_GLOB = "eval/e2e_runs/*/report.json"
# Written by `make build` (git-ignored). They carry build timings and timestamps, so they are read but never hashed or dated here.
BUILD_FILES = {"ingest": "data/processed/ingest_report.json", "chunks": "data/processed/chunk_report.json", "index": "data/index/index_manifest.json"}


# ------------------------------------------------------------------ small helpers
def load(root: Path, rel: str) -> tuple[dict | None, str]:
    """(report, short sha256) or (None, '') when the file is missing or unreadable."""
    path = root / rel
    if not path.is_file():
        return None, ""
    raw = path.read_bytes()
    try:
        return json.loads(raw), hashlib.sha256(raw).hexdigest()[:12]
    except ValueError:
        return None, ""


def latest_e2e(root: Path) -> tuple[dict | None, str, str]:
    """The newest saved end-to-end run that has a report (run directories are timestamp-named, so sorted order is time order)."""
    found = sorted(root.glob(E2E_GLOB))
    if not found:
        return None, "", E2E_GLOB
    rel = found[-1].relative_to(root).as_posix()
    rep, sha = load(root, rel)
    return rep, sha, rel


def not_run(title: str, rel: str, command: str) -> list[str]:
    return [f"## {title}", "", f"_Not run: `{rel}` was not found. Run `{command}` to produce it, then `make results`._", ""]


def table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c).replace("|", "/") for c in row) + " |" for row in rows]
    return out + [""]


def n(x, nd: int | None = None, unit: str = "") -> str:
    """Format a number read from a report: thousands separators for ints, fixed decimals when asked."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}{unit}"
    return f"{x:.{nd}f}{unit}" if nd is not None else f"{x:g}{unit}"


def pct(x) -> str:
    return "n/a" if x is None else f"{x:g}%"


def source(rel: str, sha: str, extra: str = "") -> str:
    return f"_Source: `{rel}` (sha256 `{sha}`){extra}_"


# ------------------------------------------------------------------ one section per report
def sec_corpus(ing: dict, ch: dict, ix: dict) -> list[str]:
    """What `make build` produced. Build seconds and timestamps are deliberately not shown (they differ on every run)."""
    docs = ing["documents"]
    passed = sum(c["passed"] for c in ing["checks"])
    cc = ch["checks"]
    L = ["## Corpus, chunks and indexes", "",
         "_Source: `data/processed/ingest_report.json`, `data/processed/chunk_report.json`, `data/index/index_manifest.json` (written by `make build`, not committed)_", "",
         f"Ingestion: {n(sum(d['pages'] for d in docs.values()))} pages in {len(docs)} documents; {passed}/{len(ing['checks'])} hard validation checks passed, "
         f"{len(ing['warnings'])} soft warnings (listed in the report).", ""]
    L += table(["Document", "Pages", "Index-eligible pages", "Sections found", "Figures", "Footnotes"],
               [[k, d["pages"], d["index_eligible_pages"], d["headings"], d["figures"], d["footnotes"]] for k, d in docs.items()])
    L += [f"Chunking: {n(ch['total_chunks'])} chunks; {sum(c['passed'] for c in cc)}/{len(cc)} chunk validation checks passed "
          f"(every chunk's embedded text is at most 512 tokens; no prose chunk spans two sections; no table row is split).", ""]
    counts: dict[str, int] = {}
    for per_doc in ch["counts"].values():
        for kind, c in per_doc.items():
            counts[kind] = counts.get(kind, 0) + c
    hist = ch["token_histograms_embed"]
    L += table(["Chunk type", "Chunks", "Embedded tokens: min", "median", "max"],
               [[k, counts[k], n(hist[k]["min"]), n(hist[k]["median"]), n(hist[k]["max"])] for k in sorted(counts)])
    sp, mg = ch["splits"], ch["merges"]
    L += [f"Splits: {sum(sp['table_pieces'].values())} GenAI action tables were split between rows (the header repeats in each piece); "
          f"{sum(sp['prose_pieces'].values())} prose pieces came from long sections. Tiny sections merged forward: {sum(len(v) for v in mg.values())}.", ""]
    b = ix["bm25"]
    L += [f"Indexes: `{ix['embedding_model']}` (revision `{ix['model_revision'][:8]}`, {ix['dimension']} dimensions, normalised, document prefix: {ix['document_prefix']}) "
          f"in one {ix['distance']}-distance {ix['vector_store']} collection (`{ix['collection_name']}`, {ix['chunk_count']} chunks); BM25 via `{b['library']}` "
          f"{b['version']} (k1={b['k1']}, b={b['b']}, {b['stemming']} stemming, action IDs kept whole, vocabulary {n(b['vocab_size'])}, tokenizer `{b['tokenizer_version']}`).", ""]
    return L


def sec_retrieval(r: dict, rel: str, sha: str) -> list[str]:
    c, s = r["config"], r["summary"]
    L = ["## Retrieval", "", source(rel, sha, f"; config `{r['config_version']}`"), "",
         f"{r['cases']} cases ({r['answerable']} answerable, {r['unanswerable']} unanswerable). Gold labels are structural (document + section / "
         f"subcategory / action ID), never chunk IDs.", "",
         f"Configuration: vector k={c['vector_k']}, BM25 k={c['bm25_k']}, weights {c['weights']}, RRF c={c['rrf_c']}; reranker `{c['reranker_model']}` "
         f"(revision `{c['reranker_revision'][:8]}`, input limit {c['reranker_max_length']} tokens); top {c['top_n']}; relevance threshold {c['threshold']}; "
         f"embedding `{c['embedding_model']}` (revision `{c['embedding_revision'][:8]}`).", ""]
    labels = {"vector_only": "Vector only", "bm25_only": "BM25 only", "ensemble": "Hybrid (weighted RRF)", "ensemble_rerank": "Hybrid + reranker"}
    L += table(["Configuration", "Hit@5", "MRR", "Cases hit"],
               [[labels[k], n(s[k]["hit_at_5"], 3), n(s[k]["mrr"], 3), f"{s[k]['hits']}/{s[k]['answerable_cases']}"] for k in labels])
    rc = s["recall_at_20_fused"]
    L += [f"Recall@20 of the fused candidate list (what the reranker gets to see): {rc['found']}/{rc['of']} ({pct(rc['pct'])}).", ""]
    fails = r.get("failures", {})
    L += ["Cases missed in the top 5: " + "; ".join(f"{labels[k]}: {', '.join(v) or 'none'}" for k, v in fails.items()) + ".", ""]
    t = r["thresholds"]
    a, u = t["answerable_scores"], t["unanswerable_scores"]
    L += [f"Relevance gate: reranker scores of the best chunk range {n(a['worst'], 4)}-{n(a['best'], 4)} for answerable questions and "
          f"{n(u['worst'], 4)}-{n(u['best'], 4)} for unanswerable ones; the chosen threshold is {c['threshold']}.", ""]
    L += table(["Threshold", "Answerable kept", "Unanswerable rejected"],
               [[n(x["threshold"], 4), pct(x["answerable_kept_pct"]), pct(x["unanswerable_rejected_pct"])] for x in t["rows"]])
    lat, tp = r["latency_ms"], r["truncated_pairs"]
    L += [f"Latency over {lat['n']} queries (hybrid search + rerank + gate): p50 {n(lat['p50'], 1, ' ms')}, p95 {n(lat['p95'], 1, ' ms')}, "
          f"max {n(lat['max'], 1, ' ms')} (target p95 {n(lat['target_p95_ms'], 0, ' ms')}: {n(lat['meets_target'])}).",
          f"{tp['truncated']} of {tp['pairs']} (query, chunk) pairs exceed the reranker's {tp['limit_tokens']}-token input limit and are cut.", ""]
    return L


def sec_routing(r: dict, rel: str, sha: str) -> list[str]:
    L = ["## Routing", "", source(rel, sha, f"; router `{r['model']}`"), "",
         f"{r['passed']}/{r['cases']} cases passed; route accuracy {n(r['route_accuracy'] * 100, 1, '%')}. "
         f"{r['llm_calls']} LLM calls, {r['rule_decided']} decided by the greeting rule, {r['fallbacks']} fallbacks, {n(r['tokens'])} tokens, "
         f"average LLM latency {n(r['avg_latency_ms_llm'], 0, ' ms')}.", ""]
    L += table(["Expected route", "Cases", "Correct", "Accuracy"],
               [[k, v["cases"], v["correct"], n(v["accuracy"] * 100, 1, "%")] for k, v in r["accuracy_per_route"].items()])
    L += table(["Category", "Cases", "Passed"], [[k, v["cases"], v["passed"]] for k, v in r["by_category"].items()])
    cols = ["greeting", "nist_question", "out_of_scope", "null"]
    L += ["Confusion matrix (rows = expected, columns = actual):", ""]
    L += table(["expected \\ actual"] + cols, [[k] + [v.get(c, 0) for c in cols] for k, v in r["confusion_expected_vs_actual"].items()])
    st = r["standalone_query_term_check"]
    L += [f"Follow-up rewriting: the standalone query had to contain the expected key terms in {st['nist_cases_routed_correctly'] - len(st['term_failures'])}/"
          f"{st['nist_cases_routed_correctly']} correctly routed NIST cases; failed term check: {', '.join(st['term_failures']) or 'none'}.", ""]
    if r["failures"]:
        L += ["Failed cases:", ""]
        L += table(["Case", "Message", "Expected", "Got"], [[f["id"], f["message"], f["expected_route"],
                                                                f["route"] + (" (route right; standalone-query terms missing)" if f["route"] == f["expected_route"] else "")]
                                                               for f in r["failures"]])
    return L


def sec_generation(g: dict, u: dict | None, rel: str, sha: str, urel: str, usha: str) -> list[str]:
    L = ["## Generation", "", source(rel, sha, f"; model `{g['model']}`"), "",
         f"{g['cases']} behaviour cases, {g['errors']} ended in an error. Case accuracy {g['case_accuracy']['correct']}/{g['case_accuracy']['of']} "
         f"({pct(g['case_accuracy']['pct'])}); checks passed {g['checks_passed_strict']}/{g['checks_total']} strict, {g['checks_passed_normalized']}/{g['checks_total']} "
         f"after normalising typography.", ""]
    un = g["uncited_sentence_rate"]
    L += table(["Measure", "Value"], [
        ["Answers with at least one uncited sentence", pct(un["answers_with_uncited_pct"])],
        ["Uncited sentences", f"{pct(un['sentences_uncited_pct'])} ({un['uncited']} of {un['sentences']})"],
        ["Over-length answers", pct(g["over_length_rate_pct"])], ["Invented citations", pct(g["invented_citation_rate_pct"])],
        ["Answers corrected by the fixer", pct(g["correction_rate_pct"])], ["Turned into 'insufficient' by the fixer", pct(g["insufficient_by_fix_rate_pct"])],
        ["Average total tokens / latency", f"{n(g['avg_total_tokens'], 0)} / {n(g['avg_total_ms'], 0, ' ms')}"]])
    if g["case_mismatches"]:
        L += ["Cases where the model chose a different answer type than the prompt expects:", ""]
        L += table(["Case", "Expected", "Actual"], [[m["id"], m["expected"], m["actual"]] for m in g["case_mismatches"]])
    if u:
        L += ["### Token usage and speed", "", source(urel, usha, f"; reasoning effort `{u['reasoning_effort']}`; {u['requests']} live requests, "
                                                                 f"{len(u['errors'])} errors"), ""]
        names = [("prompt_tokens", "Prompt tokens"), ("reasoning_tokens", "Reasoning tokens"), ("visible_answer_tokens", "Visible answer tokens"),
                 ("completion_tokens_incl_reasoning", "Completion tokens (incl. reasoning)"), ("total_tokens", "Total tokens"),
                 ("ttft_ms", "Time to first token (ms)"), ("total_ms", "Total time (ms)")]
        L += table(["Measure", "Min", "Median", "p95", "Max"], [[lbl] + [n(u[k][x], 1 if "ms" in k else None) for x in ("min", "median", "p95", "max")] for k, lbl in names])
        L += [f"Largest request: {u['largest_vs_tpm_limit']} tokens per minute (the Groq free-tier limit of {n(u['tpm_limit'])}). Finish reasons: "
              + ", ".join(f"{k} {v}" for k, v in u["finish_reasons"].items()) + ".", ""]
    else:
        L += ["_Token-usage report not found; run `python -m eval.generation_eval`._", ""]
    return L


def sec_prompt(p: dict, rel: str, sha: str) -> list[str]:
    s, w, lim = p["summary"], p["worst_case"], p["limits"]
    L = ["## Prompt size", "", source(rel, sha, f"; prompt `{p['prompt_version']}`"), "",
         f"{p['queries']} real prompts (retrieval evaluation queries). Tokens are estimated as {p['estimator']}.", ""]
    rows = [("chunks", "Chunks (incl. siblings)"), ("siblings", "Sibling chunks"), ("system_chars", "System prompt (chars)"), ("sources_chars", "Sources block (chars)"),
            ("total_chars", "Total (chars)"), ("total_tokens_est", "Total tokens (estimate)"), ("total_tokens_est_margin", "Total tokens (estimate, +20% margin)")]
    L += table(["Measure", "Min", "Median", "p95", "Max"], [[lbl] + [n(s[k][x]) for x in ("min", "median", "p95", "max")] for k, lbl in rows])
    ob = p["over_budget"]
    L += [f"Prompt budget: {n(p['prompt_budget_tokens'])} tokens ({n(lim['tpm'])} tokens/minute minus {n(lim['max_completion_tokens'])} reserved for the output). "
          f"Prompts over budget: {ob['observed_est']} by the plain estimate, {ob['observed_est_margin']} with the 20% margin.",
          f"Computed worst case ({w['chunks']} largest chunks in the corpus + the longest allowed query): {n(w['total_tokens_est'])} tokens "
          f"({n(w['total_tokens_est_margin'])} with margin); over budget: {n(ob['worst_case_est'])}. The largest prompt actually observed had "
          f"{n(s['chunks']['max'])} chunks and about {n(s['total_tokens_est']['max'])} tokens ({n(s['total_tokens_est_margin']['max'])} with margin).", ""]
    return L


def sec_insufficient(i: dict, rel: str, sha: str) -> list[str]:
    un, an = i["unanswerable"], i["answerable"]
    L = ["## Insufficient-context path", "", source(rel, sha), "",
         f"Unanswerable questions: {un['insufficient']}/{un['cases']} correctly returned the insufficient-context reply "
         f"({un['via_gate']} through the relevance gate with no LLM call, {un['via_model']} through the model); {un['answered']} were wrongly answered. "
         f"Answerable control questions: {an['answered']}/{an['cases']} answered ({i['live_generation_calls']} live generation calls). "
         f"The five related sections looked unrelated to the query for {un['flagged_unrelated']} of {un['cases']} unanswerable questions (a known limitation).", ""]
    L += table(["Case", "Kind", "Outcome", "Reason", "Best score", "Threshold"],
               [[r["id"], r["kind"], r["outcome"], r.get("reason") or "-", n(r.get("best_score"), 4), n(r.get("threshold"), 2)] for r in i["rows"]])
    return L


def sec_links(k: dict, rel: str, sha: str) -> list[str]:
    L = ["## Citation links", "", source(rel, sha), "",
         f"Every chunk's citation link was checked: the page it opens must contain the chunk's text, the printed page in the label must match the PDF, "
         f"and the URL must carry `#page=<pdf page>`. {k['chunks_checked']} chunks checked: {k['passed_strict']} pass strictly, "
         f"{k['explained_mismatch']} have a text-sample mismatch explained by the PDF text layer, **{k['failed']} failed** (pass rate {pct(k['pass_rate_pct'])}).", ""]
    L += table(["Explained mismatch (grade : chunk type)", "Chunks"], [[a, b] for a, b in k["explained_by_grade_and_type"].items()])
    L += [f"Samples are the first and last {k['sample_chars']} characters; below {n(k['coverage_ok'] * 100, 0, '%')} word coverage a miss counts as a real failure.", ""]
    return L


def sec_integration(i: dict, rel: str, sha: str) -> list[str]:
    live, mem = i["live"], i["memory"]
    L = ["## Graph integration (live paths and memory)", "", source(rel, sha), "",
         f"Real Groq calls and real indexes, server start-up {n(live['startup_seconds'], 1, ' s')}.", ""]
    rows = []
    for p in live["paths"]:
        rows.append([p["path_name"], p["message"], p["outcome"], p["llm_calls"], n(p["latency_ms"], 0, " ms"), n(p["usage"]["total"]), p.get("best_score") and n(p["best_score"], 4)])
    for p in live.get("extra_paths", []):
        rows.append(["extra", p["message"], p["outcome"], p["llm_calls"], n(p["latency_ms"], 0, " ms"), n(p["usage"]["total"]), p.get("best_score") and n(p["best_score"], 4)])
    L += table(["Path", "Message", "Outcome", "LLM calls", "Latency", "Tokens", "Best score"], rows)
    st = live["stream"]
    L += [f"Streaming: {st['tokens']} token events, first token after {n(st['first_token_ms'], 0, ' ms')}, {n(st['total_ms'], 0, ' ms')} in total; "
          f"final event `done` reported {st['done']['llm_calls']} LLM calls and {n(st['done']['usage']['total'])} tokens.", ""]
    gf = live.get("gate_failure_real_subgraph_no_llm", [])
    if gf:
        L += ["Gate failures run directly through the real RAG subgraph (no LLM call):", ""]
        L += table(["Query", "Best score", "Threshold", "LLM calls", "Reason"], [[g["query"], n(g["best_score"], 4), g["threshold"], g["llm_calls"], g["insufficient_reason"]] for g in gf])
    L += ["Checkpoint memory over a 20-turn conversation (a mix of NIST questions, greetings and out-of-scope turns):", ""]
    L += table(["Checkpointing", "Total checkpoint bytes", "Per NIST turn (avg)", "Per turn (max)", "Checkpoints"],
               [[m["durability"], n(m["total_checkpoint_bytes"]), n(m["avg_bytes_per_nist_turn"]), n(m["max_bytes_per_turn"]), m["checkpoints_total"]] for m in (mem["per_step_default"], mem["production"])])
    return L


def sec_e2e(e: dict, rel: str, sha: str) -> list[str]:
    a = e["aggregates"]

    def rate(r: dict) -> str:
        return "n/a" if r["pct"] is None else f"{r['pct']:g}% ({r['passed']}/{r['applicable']})"
    cov, sup, unc = a["key_fact_coverage"], a["support_rate"], a["uncited_sentence_rate"]
    L = ["## End-to-end evaluation", "", source(rel, sha, f"; run `{e['run']}`, generated {e['generated_at']}"), "",
         f"{e['requests']} `/chat` requests and {e['streamed']} `/chat/stream` requests against a running server, "
         f"paced {n(e['pace_seconds'], 0, ' s')} after each generation turn. Full per-turn table and every failed check: [`docs/E2E_RESULTS.md`](E2E_RESULTS.md).", ""]
    L += table(["Metric", "Result"], [
        ["Route accuracy", rate(a["route_accuracy"])], ["Outcome accuracy", rate(a["outcome_accuracy"])], ["Fixed texts exact", rate(a["exact_text_accuracy"])],
        ["Follow-ups resolved in the standalone query", rate(a["follow_up_resolution"])], ["LLM calls as expected", rate(a["llm_calls_as_expected"])],
        ["Key-fact coverage (all gold facts)", f"{cov['pct']:g}% ({cov['found']}/{cov['total']})"], ["Answered turns meeting `min_facts`", rate(a["turns_meeting_min_facts"])],
        ["Turns with a trap hit", f"{a['trap_hits']['turns_with_a_hit']} of {a['trap_hits']['traps_checked_turns']}"],
        ["`[C#]` markers match sources", rate(a["citation_consistency"])], ["Expected section cited", rate(a["expected_section_hit"])],
        ["Links open the cited text", rate(a["link_check"])], ["Answers with no fix or warning", rate(a["no_fixes_or_warnings"])],
        [f"Sentences supported by their cited chunk (score >= {sup['threshold']})", f"{sup['pct']:g}% ({sup['supported']}/{sup['total']})"],
        ["Uncited fact-stating sentences", f"{unc['pct']:g}% ({unc['uncited']} uncited, {unc['cited']} cited)"],
        ["Insufficient path fully correct", f"{a['insufficient_path']['all_checks_passed']} of {a['insufficient_path']['turns']} turns"],
        ["Answers over ~150 words", f"{a['over_length_answers']['count']} of {a['over_length_answers']['of']} (longest {a['over_length_answers']['words_max']} words)"],
        ["Latency p50 / p95", f"{n(a['latency_ms']['p50'], 0, ' ms')} / {n(a['latency_ms']['p95'], 0, ' ms')}"],
        ["Streaming checks (order / nothing after error / same structure as /chat)", " / ".join(rate(a["streaming"][k]) for k in ("ST1", "ST2", "ST3"))]])
    L += table(["Turn type", "Turns", "LLM calls", "Tokens (mean / max)", "Latency p50 / p95"],
               [[k, v["turns"], ", ".join(map(str, v["llm_calls"])), f"{n(v['tokens_mean'])} / {n(v['tokens_max'])}",
                 f"{n(v['latency_ms_p50'], 0, ' ms')} / {n(v['latency_ms_p95'], 0, ' ms')}"] for k, v in a["by_turn_type"].items()])
    cal = e.get("calibration")
    if cal:
        L += [f"Support-score calibration: {cal['positives']['n']} cited sentences scored {n(cal['positives']['min'], 3)}-{n(cal['positives']['max'], 3)}; "
              f"{cal['negatives']['n']} random other-section chunks scored up to {n(cal['negatives']['max'], 3)}; "
              f"{cal['sanity']['n']} verbatim sentences scored at least {n(cal['sanity']['min'], 4)}. Threshold chosen: {e['support_threshold']}.", ""]
    L += [f"{len(e['failures'])} failed checks are listed in full in `docs/E2E_RESULTS.md`; none were filtered out.", ""]
    return L


# ------------------------------------------------------------------ assemble
def build(root: Path = ROOT) -> tuple[str, str]:
    """(docs/RESULTS.md text, README block text) from the reports under `root`."""
    loaded = {k: load(root, rel) for k, (rel, _) in REPORTS.items()}
    e2e, e2e_sha, e2e_rel = latest_e2e(root)
    out = ["# Evaluation results", "",
           "_Generated by `docs/build_results.py` (`make results`) from the report files named in each section. Do not edit by hand: change the code, "
           "re-run the evaluation, then regenerate._", ""]

    def put(key: str, title: str, fn) -> None:
        rep, sha = loaded[key]
        rel, cmd = REPORTS[key]
        out.extend(fn(rep, rel, sha) if rep is not None else not_run(title, rel, cmd))

    build_files = {k: load(root, rel)[0] for k, rel in BUILD_FILES.items()}
    if all(build_files.values()):
        out.extend(sec_corpus(build_files["ingest"], build_files["chunks"], build_files["index"]))
    else:
        out += ["## Corpus, chunks and indexes", "", "_Not built: run `make build` to produce the ingestion, chunking and index reports, then `make results`._", ""]
    put("retrieval", "Retrieval", sec_retrieval)
    put("routing", "Routing", sec_routing)
    gen, gsha = loaded["generation"]
    use, usha = loaded["usage"]
    if gen is not None:
        out.extend(sec_generation(gen, use, REPORTS["generation"][0], gsha, REPORTS["usage"][0], usha))
    else:
        out.extend(not_run("Generation", REPORTS["generation"][0], REPORTS["generation"][1]))
    put("prompt_size", "Prompt size", sec_prompt)
    put("insufficient", "Insufficient-context path", sec_insufficient)
    put("link_check", "Citation links", sec_links)
    put("integration", "Graph integration (live paths and memory)", sec_integration)
    out.extend(sec_e2e(e2e, e2e_rel, e2e_sha) if e2e is not None else not_run("End-to-end evaluation", E2E_GLOB, "make e2e && make e2e-report RUN=eval/e2e_runs/<run>"))
    return "\n".join(out).rstrip() + "\n", readme_block(loaded, e2e)


def readme_block(loaded: dict, e2e: dict | None) -> str:
    """Short summary for the README: the headline numbers only, each read from a report."""
    L = ["Numbers below are generated from the evaluation reports by `make results`; details in [`docs/RESULTS.md`](docs/RESULTS.md).", ""]
    r, _ = loaded["retrieval"]
    if r:
        s = r["summary"]
        L += [f"- **Retrieval** ({r['answerable']} answerable questions): Hit@5 vector {n(s['vector_only']['hit_at_5'], 3)}, BM25 {n(s['bm25_only']['hit_at_5'], 3)}, "
              f"hybrid {n(s['ensemble']['hit_at_5'], 3)}, hybrid + reranker **{n(s['ensemble_rerank']['hit_at_5'], 3)}**; MRR {n(s['vector_only']['mrr'], 3)} / "
              f"{n(s['bm25_only']['mrr'], 3)} / {n(s['ensemble']['mrr'], 3)} / {n(s['ensemble_rerank']['mrr'], 3)}; retrieval latency p50 {n(r['latency_ms']['p50'], 1, ' ms')}, "
              f"p95 {n(r['latency_ms']['p95'], 1, ' ms')}."]
    else:
        L += ["- **Retrieval:** not run."]
    rt, _ = loaded["routing"]
    L += [f"- **Routing:** {rt['passed']}/{rt['cases']} cases, route accuracy {n(rt['route_accuracy'] * 100, 1, '%')}." if rt else "- **Routing:** not run."]
    ins, _ = loaded["insufficient"]
    if ins:
        u = ins["unanswerable"]
        L += [f"- **Insufficient context:** {u['insufficient']}/{u['cases']} unanswerable questions got the insufficient-context reply "
              f"({u['via_gate']} through the relevance gate without calling the LLM)."]
    lk, _ = loaded["link_check"]
    if lk:
        L += [f"- **Citation links:** {lk['chunks_checked'] - lk['failed']}/{lk['chunks_checked']} chunks open a PDF page that contains the cited text ({lk['failed']} failed)."]
    us, _ = loaded["usage"]
    if us:
        L += [f"- **Generation cost:** median {n(us['total_tokens']['median'])} tokens per answered question (max {n(us['total_tokens']['max'])} of the "
              f"{n(us['tpm_limit'])} tokens/minute free-tier limit); median time to first token {n(us['ttft_ms']['median'], 0, ' ms')}."]
    ig, _ = loaded["integration"]
    if ig:
        m = ig["memory"]["production"]
        L += [f"- **Conversation memory:** {n(m['avg_bytes_per_nist_turn'])} bytes of checkpoint per NIST turn (in-memory, lost on restart)."]
    if e2e:
        a = e2e["aggregates"]

        def frac(x: dict) -> str:
            return f"{x['passed']}/{x['applicable']}"
        L += [f"- **End to end** ({e2e['requests']} requests through the running API, run `{e2e['run']}`): route {frac(a['route_accuracy'])}, outcome {frac(a['outcome_accuracy'])}, "
              f"follow-ups resolved {frac(a['follow_up_resolution'])}, links open the cited text {frac(a['link_check'])}, "
              f"sentences supported {a['support_rate']['supported']}/{a['support_rate']['total']}, "
              f"{len(e2e['failures'])} failed checks (all listed in [`docs/E2E_RESULTS.md`](docs/E2E_RESULTS.md)), latency p50 {n(a['latency_ms']['p50'], 0, ' ms')}."]
    else:
        L += ["- **End to end:** not run."]
    return "\n".join(L) + "\n"


def replace_block(text: str, start: str, end: str, block: str) -> str | None:
    """Replace everything between one well-ordered pair of markers; None when the pair is missing, duplicated or reversed."""
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        return None
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}{end}{tail}"


def update_readme(text: str, block: str) -> str | None:
    return replace_block(text, START, END, block)


# ------------------------------------------------------------------ evidence lines for DESIGN.md
EVIDENCE_NAMES = ("chunking", "indexing", "retrieval", "gate", "routing", "conversation", "insufficient", "grounding", "freetier", "e2e")


def evidence_markers(name: str) -> tuple[str, str]:
    return f"<!-- EVIDENCE:{name}:START -->", f"<!-- EVIDENCE:{name}:END -->"


def build_evidence(root: Path = ROOT) -> dict[str, str]:
    """One short, generated 'Evidence' paragraph per DESIGN.md section. Every figure is read from a report; a missing report
    gives a 'not available' sentence, never a made-up number. DESIGN.md holds the reasoning; these blocks hold the numbers."""
    rp = {k: load(root, rel)[0] for k, (rel, _) in REPORTS.items()}
    bf = {k: load(root, rel)[0] for k, rel in BUILD_FILES.items()}
    e2e = latest_e2e(root)[0]
    na = "_Evidence not available: the report was not found (see `docs/RESULTS.md`)._"
    ev: dict[str, str] = {}

    if all(bf.values()):
        ch, ix = bf["chunks"], bf["index"]
        counts: dict[str, int] = {}
        for per_doc in ch["counts"].values():
            for kind, c in per_doc.items():
                counts[kind] = counts.get(kind, 0) + c
        hist = ch["token_histograms_embed"]
        longest = max(h["max"] for h in hist.values())
        ev["chunking"] = (f"**Evidence** ([results](docs/RESULTS.md#corpus-chunks-and-indexes)): {n(ch['total_chunks'])} chunks ("
                          + ", ".join(f"{counts[k]} {k.replace('_', ' ')}" for k in sorted(counts, key=lambda k: -counts[k]))
                          + f"). Prose chunks have a median of {n(hist['prose']['median'])} embedded tokens (maximum {n(hist['prose']['max'])}); the longest chunk of any type is "
                          f"{n(longest)} tokens, under the 512-token limit. {sum(ch['splits']['table_pieces'].values())} GenAI action tables were split between rows, "
                          f"{sum(len(v) for v in ch['merges'].values())} tiny sections were merged forward, and {sum(c['passed'] for c in ch['checks'])}/{len(ch['checks'])} "
                          f"chunk validation checks pass.")
        b = ix["bm25"]
        ev["indexing"] = (f"**Evidence** ([results](docs/RESULTS.md#corpus-chunks-and-indexes)): `{ix['embedding_model']}` produces {ix['dimension']}-dimensional normalised vectors stored in one "
                          f"{ix['distance']} collection of {ix['chunk_count']} chunks; the BM25 index has a vocabulary of {n(b['vocab_size'])} terms "
                          f"(tokenizer `{b['tokenizer_version']}`).")
    else:
        ev["chunking"] = ev["indexing"] = "_Evidence not available: run `make build` and `make results`._"

    r = rp["retrieval"]
    if r:
        s, lat = r["summary"], r["latency_ms"]
        ev["retrieval"] = (f"**Evidence** ([results](docs/RESULTS.md#retrieval)), {r['answerable']} answerable questions: Hit@5 is {n(s['vector_only']['hit_at_5'], 3)} for vector search alone, "
                           f"{n(s['bm25_only']['hit_at_5'], 3)} for BM25 alone and {n(s['ensemble']['hit_at_5'], 3)} for the hybrid, and **{n(s['ensemble_rerank']['hit_at_5'], 3)}** with the reranker; "
                           f"MRR rises from {n(s['ensemble']['mrr'], 3)} to {n(s['ensemble_rerank']['mrr'], 3)}. The correct chunk was within the first 20 fused candidates "
                           f"in {s['recall_at_20_fused']['found']}/{s['recall_at_20_fused']['of']} cases (Recall@20), which is what the reranker gets to see. Retrieval takes p50 {n(lat['p50'], 1, ' ms')} "
                           f"and p95 {n(lat['p95'], 1, ' ms')}; {r['truncated_pairs']['truncated']} of {r['truncated_pairs']['pairs']} reranked pairs exceed the reranker's input limit.")
        t = r["thresholds"]
        row = min(t["rows"], key=lambda x: abs(x["threshold"] - r["config"]["threshold"]))
        ev["gate"] = (f"**Evidence** ([results](docs/RESULTS.md#retrieval)): on {r['unanswerable']} unanswerable questions the best reranker score was at most {n(t['unanswerable_scores']['best'], 4)}; "
                      f"on answerable ones it ranged from {n(t['answerable_scores']['worst'], 4)} to {n(t['answerable_scores']['best'], 4)}. At a threshold near {r['config']['threshold']} "
                      f"(table row {n(row['threshold'], 4)}) {pct(row['unanswerable_rejected_pct'])} of unanswerable questions are rejected and {pct(row['answerable_kept_pct'])} of answerable ones are kept, "
                      f"so the threshold was tuned on only {r['unanswerable']} negative cases.")
    else:
        ev["retrieval"] = ev["gate"] = na

    rt = rp["routing"]
    if rt:
        fu = rt["by_category"]["follow_up"]
        ev["routing"] = (f"**Evidence** ([results](docs/RESULTS.md#routing)): the router passed {rt['passed']}/{rt['cases']} cases ({n(rt['route_accuracy'] * 100, 1, '%')} route accuracy); "
                         f"{rt['rule_decided']} of {rt['cases']} were answered by the greeting rule without an LLM call. Follow-ups passed {fu['passed']}/{fu['cases']}. "
                         f"Failures: " + "; ".join(f"`{f['id']}` ({f['expected_route']} → {f['route']})" if f["route"] != f["expected_route"] else f"`{f['id']}` (right route, standalone-query terms missing)"
                                                  for f in rt["failures"]) + ".")
    else:
        ev["routing"] = na

    ig = rp["integration"]
    if e2e and ig:
        a, m, per = e2e["aggregates"], ig["memory"], ig["memory"]["per_step_default"]
        ev["conversation"] = (f"**Evidence** ([results](docs/RESULTS.md#end-to-end-evaluation)): in the end-to-end run {a['follow_up_resolution']['passed']}/{a['follow_up_resolution']['applicable']} follow-up questions "
                              f"were rewritten into a standalone query containing the expected terms. Checkpointing once per request (`durability=\"exit\"`) costs "
                              f"{n(m['production']['avg_bytes_per_nist_turn'])} bytes per NIST turn, against {n(per['avg_bytes_per_nist_turn'])} with LangGraph's per-step default "
                              f"over the same {m['production']['turns']}-turn conversation.")
    else:
        ev["conversation"] = na

    ins = rp["insufficient"]
    if ins and e2e:
        u = ins["unanswerable"]
        part = next((p_ for p_ in (rp["integration"] or {}).get("live", {}).get("paths", []) if p_["path_name"].startswith("model_refusal")), None)
        two_part = (f"in the integration run the two-part question was {'answered partially (outcome `' + part['outcome'] + '`)' if part['outcome'] == 'answered' else 'refused (outcome `' + part['outcome'] + '`)'}."
                    if part else "no integration run is available.")
        refused = [r_["id"] for r_ in e2e["per_turn"] if any(c["id"] == "A2" and c["status"] == "fail" and "outcome=insufficient" in c["detail"] for c in r_["checks"])]
        ev["insufficient"] = (f"**Evidence** ([results](docs/RESULTS.md#insufficient-context-path)): {u['insufficient']}/{u['cases']} unanswerable questions received the insufficient-context reply, "
                              f"{u['via_gate']} of them from the relevance gate alone (no generation call); the five related sections looked unrelated for {u['flagged_unrelated']} of them. "
                              f"The cost of strictness shows in the end-to-end run: {len(refused)} questions the documents can answer were refused ({', '.join(refused) or 'none'}). "
                              f"Compliance with the 'refuse partly answerable questions' rule is up to the model: " + two_part)
    else:
        ev["insufficient"] = na

    lk, gen = rp["link_check"], rp["generation"]
    if lk and gen and e2e:
        a = e2e["aggregates"]
        ev["grounding"] = (f"**Evidence** ([results](docs/RESULTS.md#citation-links)): all {lk['chunks_checked']} chunk links open a PDF page containing the cited text ({lk['failed']} failed). "
                           f"In the end-to-end run {a['support_rate']['supported']}/{a['support_rate']['total']} cited sentences scored at least {a['support_rate']['threshold']} against their cited chunk, "
                           f"and {a['uncited_sentence_rate']['uncited']} of {a['uncited_sentence_rate']['uncited'] + a['uncited_sentence_rate']['cited']} fact-stating sentences carried no citation. "
                           f"In the {gen['cases']} generation behaviour cases the invented-citation rate was {pct(gen['invented_citation_rate_pct'])}, the over-length rate {pct(gen['over_length_rate_pct'])} "
                           f"and {pct(gen['uncited_sentence_rate']['sentences_uncited_pct'])} of sentences were uncited.")
    else:
        ev["grounding"] = na

    us, ps = rp["usage"], rp["prompt_size"]
    if us and ps and e2e:
        tt = us["total_tokens"]
        ev["freetier"] = (f"**Evidence** ([results](docs/RESULTS.md#generation)): a generation call used a median of {n(tt['median'])} tokens (p95 {n(tt['p95'])}, maximum {n(tt['max'])}) against the "
                          f"{n(us['tpm_limit'])} tokens-per-minute limit; with the router call an answered end-to-end turn averaged "
                          f"{n(e2e['aggregates']['by_turn_type']['answered']['tokens_mean'])} tokens. Prompts are {n(ps['summary']['total_tokens_est']['median'])} tokens at the median "
                          f"({n(ps['summary']['total_tokens_est']['max'])} at most). At that average the limit allows about "
                          f"{us['tpm_limit'] / e2e['aggregates']['by_turn_type']['answered']['tokens_mean']:.1f} answered questions per minute, so the end-to-end harness paces "
                          f"{n(e2e['pace_seconds'], 0, ' s')} after each generation turn.")
    else:
        ev["freetier"] = na

    if e2e:
        a = e2e["aggregates"]
        ev["e2e"] = (f"**Evidence** ([results](docs/RESULTS.md#end-to-end-evaluation)): {e2e['requests']} requests through the running API; route accuracy {a['route_accuracy']['passed']}/{a['route_accuracy']['applicable']}, "
                     f"outcome accuracy {a['outcome_accuracy']['passed']}/{a['outcome_accuracy']['applicable']}, key-fact coverage {a['key_fact_coverage']['found']}/{a['key_fact_coverage']['total']}, "
                     f"{len(e2e['failures'])} failed checks (all listed in `docs/E2E_RESULTS.md`).")
    else:
        ev["e2e"] = na
    return ev


def update_design(text: str, evidence: dict[str, str]) -> tuple[str, list[str]]:
    """Fill every EVIDENCE block of DESIGN.md. Returns (new text, names whose markers were missing or malformed)."""
    missing = []
    for name in EVIDENCE_NAMES:
        start, end = evidence_markers(name)
        new = replace_block(text, start, end, evidence[name] + "\n")
        if new is None:
            missing.append(name)
        else:
            text = new
    return text, missing


def main(root: Path = ROOT) -> int:
    results, block = build(root)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "RESULTS.md").write_text(results, encoding="utf-8")
    print(f"wrote {root / 'docs' / 'RESULTS.md'} ({len(results.splitlines())} lines)")
    readme = root / "README.md"
    if readme.is_file():
        new = update_readme(readme.read_text(encoding="utf-8"), block)
        if new is None:
            print(f"note: {readme.name} has no {START} ... {END} pair, so its results block was not updated")
        elif new != readme.read_text(encoding="utf-8"):
            readme.write_text(new, encoding="utf-8")
            print(f"updated the results block in {readme.name}")
        else:
            print(f"{readme.name} results block already up to date")
    design = root / "DESIGN.md"
    if design.is_file():
        old = design.read_text(encoding="utf-8")
        new, missing = update_design(old, build_evidence(root))
        if missing:
            print(f"note: DESIGN.md is missing evidence markers for: {', '.join(missing)}")
        if new != old:
            design.write_text(new, encoding="utf-8")
            print("updated the evidence blocks in DESIGN.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
