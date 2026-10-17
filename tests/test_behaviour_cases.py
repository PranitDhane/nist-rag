"""The behaviour-case file must be well-formed; the generation phase will run the cases against the real model."""
from pathlib import Path

import yaml

from augmentation import config
from augmentation.context_builder import build_context
from retrieval.models import RetrievalResult, RetrievedChunk

PATH = Path(__file__).resolve().parent.parent / "eval" / "answer_behaviour_cases.yaml"
CHECK_TYPES = {"parser_ok", "exact_d", "contains_all", "contains_any", "not_contains", "quote_exact", "no_pages", "max_words",
               "bullets", "ambiguous_format", "has_not_covered", "names_docs", "cites_key", "only_english", "no_chunk_ids"}
KINDS = {"A", "B", "C_to_D", "D", "conflict", "injection", "language", "length"}


def cases():
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))["cases"]


def test_file_meets_the_plan():
    cs = cases()
    assert len(cs) >= 15 and len({c["id"] for c in cs}) == len(cs)
    assert {c["kind"] for c in cs} == KINDS
    kinds = [c["kind"] for c in cs]
    assert kinds.count("A") >= 5 and kinds.count("B") == 2 and kinds.count("C_to_D") >= 1 and kinds.count("D") >= 1


def test_every_case_is_well_formed():
    for c in cases():
        assert c["expect_case"] in ("A", "B", "D") and c["query"].strip() and c["checks"], c["id"]
        assert c["context"]["mode"] in ("retrieve", "fixture")
        for chk in c["checks"]:
            assert chk["type"] in CHECK_TYPES, (c["id"], chk)
        if c["context"]["mode"] == "fixture":
            assert c["context"]["sources"], c["id"]
            for s in c["context"]["sources"]:
                assert set(s) == {"doc", "section", "type", "part", "text"}, c["id"]
                assert s["type"] in ("prose", "table", "figure-ocr", "footnote")
        else:
            assert c["context"]["note"], c["id"]
        if c["expect_case"] == "D":
            assert any(chk["type"] == "exact_d" for chk in c["checks"]), c["id"]
        if c["expect_case"] == "B":
            assert any(chk["type"] == "ambiguous_format" for chk in c["checks"]), c["id"]


def test_required_behaviours_are_covered():
    ids = {c["id"] for c in cases()}
    for needle in ("four-functions", "cross-cutting", "definition", "table", "figure", "both-readings", "one-reading",
                   "two-part", "near-scope", "two-documents", "injection", "hindi", "broad"):
        assert any(needle in i for i in ids), needle
    by_id = {c["id"]: c for c in cases()}
    assert "Ignore" in by_id["i01-injection-in-source"]["context"]["sources"][0]["text"].title() or \
        "IGNORE" in by_id["i01-injection-in-source"]["context"]["sources"][0]["text"]
    assert any(ord(ch) > 0x900 for ch in by_id["l01-hindi-question"]["query"])               # really Hindi


def test_fixture_contexts_can_be_turned_into_prompt_sources():
    for c in cases():
        if c["context"]["mode"] != "fixture":
            continue
        chunks = []
        for i, s in enumerate(c["context"]["sources"], 1):
            ctype = {"prose": "prose", "table": "genai_action_table", "figure-ocr": "figure", "footnote": "footnote"}[s["type"]]
            chunks.append(RetrievedChunk(chunk_id=f"{c['id']}:{i}", display_text=s["text"], metadata={
                "display_name": s["doc"], "chunk_type": ctype, "section_path": s["section"].split(" > "),
                "part": int(s["part"].split(" of ")[0]), "parts_total": int(s["part"].split(" of ")[1])}))
        block, km = build_context(RetrievalResult(query=c["query"], chunks=chunks, relevant=True, best_score=1, threshold=0, candidate_count=1))
        assert list(km) == [f"C{i}" for i in range(1, len(chunks) + 1)] and block.count("<source ") == len(chunks)


def test_exact_quotes_in_checks_really_appear_in_the_corpus_or_fixture_text():
    import json
    corpus = " ".join(json.loads(l)["display_text"] for l in
                      (config.PROJECT_ROOT / "data" / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines())
    for c in cases():
        for chk in c["checks"]:
            if chk["type"] == "quote_exact":
                text = corpus if c["context"]["mode"] == "retrieve" else " ".join(s["text"] for s in c["context"]["sources"])
                assert chk["value"].lower() in " ".join(text.split()).lower(), (c["id"], chk["value"])
