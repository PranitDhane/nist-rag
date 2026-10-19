"""Stage V4: local groundedness scoring (E2E plan Section 5, decision E-D3: no LLM judge).

A sentence of an answer is "supported" when the local `bge-reranker-base` cross-encoder scores the pair
(sentence without its [C#] markers, text of the chunk it cites) above a threshold chosen from a calibration table
(Q2). The score is the MAX over the sentence's cited chunks. The reranker is also the retrieval model, so this is NOT an
independent judge: it measures topical/lexical support, and a subtle factual error can still score high. The gold
facts (check B) and the citation checks (C) complement it.

Pure functions plus an injected `scorer(sentence, texts) -> list[float]`, so everything here is tested without a model.
`python -m eval.e2e_ground --from-run <dir>` prints the calibration table for a saved run and writes calibration.json."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from augmentation.parser import VALID_MARKER, split_units

Scorer = Callable[[str, list[str]], list[float]]       # (sentence, chunk texts) -> one score per text
NEGATIVES_PER_SENTENCE = 3
SANITY_MIN_WORDS = 8
THRESHOLD_STEPS = [round(i * 0.05, 2) for i in range(0, 20)]      # 0.00 .. 0.95
DEFAULT_SEED = 20261019


@dataclass
class Unit:
    """One fact-stating sentence: its text without markers, and the citation keys it carries."""
    text: str
    keys: list[str]


@dataclass
class Scored:
    unit: Unit
    scores: dict[str, float] = field(default_factory=dict)        # key -> score against that key's chunk

    @property
    def best(self) -> float | None:
        return max(self.scores.values()) if self.scores else None


LEADING_MARKERS = re.compile(r"^(?:\s*\[C\d+\])+")


def strip_markers(text: str) -> str:
    """Remove [C#] markers and the space they leave before punctuation ('cross-cutting [C1].' -> 'cross-cutting.')."""
    text = re.sub(r"\s+", " ", VALID_MARKER.sub("", text)).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def extract_units(answer: str) -> tuple[list[Unit], list[str]]:
    """(cited units, uncited sentences). Uses the augmentation parser's sentence splitting (bullets, abbreviations, quotes,
    exempt lead-ins). Markers written AFTER the full stop ('... cross-cutting. [C1] It is ...') are split off the front of
    the next sentence and given back to the sentence before them; a unit that is only markers does the same."""
    entries: list[list] = []                                       # [text, keys] in order
    for raw in split_units(answer):
        lead = LEADING_MARKERS.match(raw)
        if lead and entries:
            keys = [f"C{n}" for n in VALID_MARKER.findall(lead.group(0))]
            entries[-1][1] = list(dict.fromkeys(entries[-1][1] + keys))
            raw = raw[lead.end():]
        text = strip_markers(raw)
        if re.search(r"\w", text):
            entries.append([text, list(dict.fromkeys(f"C{n}" for n in VALID_MARKER.findall(raw)))])
    cited = [Unit(t, k) for t, k in entries if k]
    uncited = [t for t, k in entries if not k]
    return cited, uncited


def score_units(units: list[Unit], key_to_text: dict[str, str], scorer: Scorer) -> list[Scored]:
    """Score every unit against each of its resolvable cited chunks. A key with no text is skipped (reported by the caller)."""
    out = []
    for unit in units:
        keys = [k for k in unit.keys if k in key_to_text]
        scores = scorer(unit.text, [key_to_text[k] for k in keys]) if keys else []
        out.append(Scored(unit, dict(zip(keys, scores))))
    return out


# ------------------------------------------------------------------ calibration (plan Section 5)
def dist(values: list[float]) -> dict:
    """min / median / p10 / p90 / max (nearest rank, so each is a score that occurred) plus the count."""
    if not values:
        return {"n": 0}
    v = sorted(values)

    def rank(p: float) -> float:
        return v[min(len(v) - 1, max(0, int(-(-p * len(v) // 1)) - 1))]
    mid = len(v) // 2
    median = v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2
    return {"n": len(v), "min": v[0], "p10": rank(0.10), "median": median, "p90": rank(0.90), "max": v[-1]}


def negative_pool(chunks: dict[str, dict], cited_ids: list[str]) -> list[str]:
    """Chunks from sections the answer does NOT cite (different doc or different section path), sorted for determinism."""
    cited_sections = {(chunks[c]["doc_id"], tuple(chunks[c]["section_path"])) for c in cited_ids if c in chunks}
    return sorted(cid for cid, c in chunks.items() if (c["doc_id"], tuple(c["section_path"])) not in cited_sections)


def first_sentence(display_text: str) -> str | None:
    """A sentence copied verbatim from a chunk (sanity control): the first line/sentence with >= SANITY_MIN_WORDS words
    after dropping Markdown decoration. None if the chunk has none (very short chunks)."""
    for line in display_text.splitlines():
        line = re.sub(r"[|*#>_`]+", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if len(sent.split()) >= SANITY_MIN_WORDS:
                return sent
    return None


def threshold_table(positives: list[float], negatives: list[float]) -> list[dict]:
    """For each candidate threshold: the % of positive pairs it would call supported, and the % of negative controls it
    would wrongly call supported. The user picks the threshold from this table (Q2)."""
    def pct(vals: list[float], t: float) -> float | None:
        return round(100 * sum(x >= t for x in vals) / len(vals), 1) if vals else None
    return [{"threshold": t, "positives_supported_pct": pct(positives, t), "negatives_supported_pct": pct(negatives, t)}
            for t in THRESHOLD_STEPS]


def calibrate(turns: list[dict], chunks: dict[str, dict], scorer: Scorer, seed: int = DEFAULT_SEED) -> dict:
    """turns: [{id, answer, key_map}] for the answered turns. Returns the three score sets and their overlap.
      positives  max score of each cited sentence against its own cited chunk(s)
      negatives  each cited sentence against NEGATIVES_PER_SENTENCE random chunks from sections it does not cite (seeded)
      sanity     a sentence copied from each cited chunk against that chunk (should score very high)"""
    rng = random.Random(seed)
    positives, negatives, neg_max, sanity = [], [], [], []
    per_sentence, sanity_rows, unresolved = [], [], []
    for turn in turns:
        units, _ = extract_units(turn["answer"])
        key_to_id = {k: v for k, v in turn["key_map"].items()}
        key_to_text = {k: chunks[cid]["display_text"] for k, cid in key_to_id.items() if cid in chunks}
        unresolved += [f"{turn['id']}:{k}" for u in units for k in u.keys if k not in key_to_text]
        cited_ids = sorted({key_to_id[k] for u in units for k in u.keys if k in key_to_id})
        pool = negative_pool(chunks, cited_ids)
        for sc in score_units(units, key_to_text, scorer):
            if sc.best is None:
                continue
            positives.append(sc.best)
            picks = rng.sample(pool, min(NEGATIVES_PER_SENTENCE, len(pool)))
            neg = scorer(sc.unit.text, [chunks[c]["display_text"] for c in picks])
            negatives += neg
            neg_max.append(max(neg))
            per_sentence.append({"turn": turn["id"], "sentence": sc.unit.text, "keys": sc.unit.keys, "positive": round(sc.best, 4),
                                 "negatives": [{"chunk_id": c, "score": round(s, 4)} for c, s in zip(picks, neg)]})
        for cid in cited_ids:
            sent = first_sentence(chunks[cid]["display_text"])
            if sent:
                score = scorer(sent, [chunks[cid]["display_text"]])[0]
                sanity.append(score)
                sanity_rows.append({"turn": turn["id"], "chunk_id": cid, "sentence": sent, "score": round(score, 4)})
    pos_d, neg_d = dist(positives), dist(negatives)
    overlap = {}
    if positives and negatives:
        overlap = {"positives_at_or_below_negative_p90_pct": round(100 * sum(p <= neg_d["p90"] for p in positives) / len(positives), 1),
                   "negatives_at_or_above_positive_p10_pct": round(100 * sum(n >= pos_d["p10"] for n in negatives) / len(negatives), 1),
                   "max_negative": neg_d["max"], "min_positive": pos_d["min"],
                   "separable": pos_d["min"] > neg_d["max"]}
    return {"seed": seed, "negatives_per_sentence": NEGATIVES_PER_SENTENCE, "turns": len(turns),
            "positives": pos_d, "negatives": neg_d, "negatives_best_of_3": dist(neg_max), "sanity": dist(sanity),
            "overlap": overlap, "threshold_table": threshold_table(positives, negatives),
            "unresolved_keys": unresolved, "sentences": per_sentence, "sanity_rows": sanity_rows}


def format_calibration(cal: dict) -> str:
    def row(name: str, d: dict) -> str:
        if not d.get("n"):
            return f"{name:22} n=0"
        return f"{name:22} n={d['n']:<4} min {d['min']:.4f}  p10 {d['p10']:.4f}  median {d['median']:.4f}  p90 {d['p90']:.4f}  max {d['max']:.4f}"
    lines = [row("positives (own chunk)", cal["positives"]), row("negatives (3 random)", cal["negatives"]),
             row("negatives best-of-3", cal["negatives_best_of_3"]), row("sanity (verbatim)", cal["sanity"])]
    if cal["overlap"]:
        lines.append(f"overlap: {cal['overlap']}")
    lines += ["", "threshold  positives supported  negatives wrongly supported"]
    lines += [f"{r['threshold']:>9.2f}  {r['positives_supported_pct']!s:>19}%  {r['negatives_supported_pct']!s:>27}%" for r in cal["threshold_table"]]
    if cal["unresolved_keys"]:
        lines.append(f"\nunresolved cited keys (no chunk text): {cal['unresolved_keys']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ the real scorer and the command line
def reranker_scorer() -> Scorer:
    """The local cross-encoder as a Scorer (same model, revision and 0-1 sigmoid scale as retrieval)."""
    from retrieval.rerank import load_reranker, predict
    model = load_reranker()
    return lambda sentence, texts: predict(model, sentence, texts)


def main(argv: list[str] | None = None) -> int:
    from eval import e2e, e2e_checks as ck
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-run", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    try:
        run = e2e.load_run(args.from_run)
    except e2e.HarnessError as exc:
        print(f"e2e_ground: {exc}", file=sys.stderr)
        return 2
    chunks = {c["chunk_id"]: c for c in (json.loads(line) for line in
              (ck.cases_mod.ROOT / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    turns = []
    for r in run["records"]:
        v = ck.view(r)
        if v.ok and v.outcome == "answered" and v.sources and v.key_map:
            turns.append({"id": r["id"], "answer": v.answer, "key_map": v.key_map})
    cal = calibrate(turns, chunks, reranker_scorer(), args.seed)
    print(f"calibration over {len(turns)} answered turns, {cal['positives'].get('n', 0)} cited sentences\n")
    print(format_calibration(cal))
    (args.from_run / "calibration.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsaved {args.from_run / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
