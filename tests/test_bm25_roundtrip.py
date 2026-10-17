import json

import pytest

from indexing import bm25_index, config
from indexing.bm25_index import build_corpus, make_bm25, save_corpus, search
from indexing.loaders import load_bm25
from indexing.manifest import IndexIncompatible, write_manifest


def chunk(i, text):
    return {"chunk_id": f"d:prose:{i:04d}:1", "embed_text": text, "display_text": text}


CHUNKS = [chunk(1, "GOVERN 1.7 GV-1.7-001 Protocols are put in place to ensure GAI systems can be deactivated."),
          chunk(2, "Confabulation is the production of confidently stated but erroneous content."),
          chunk(3, "MEASURE 2.11 Fairness and bias are evaluated; GV-2.1-004 national security."),
          chunk(4, "Systems must be safely decommissioned and decommissioning plans documented."),
          chunk(5, "Unrelated filler text about weather and gardens.")]
QUERIES = ["GV-1.7-001", "confabulation", "decommissioned systems", "GOVERN 1.7"]


def manifest_for(tmp_path, **over):
    corpus = build_corpus(CHUNKS)
    m = {"bm25": {"tokenizer_version": corpus["tokenizer_version"], **config.BM25_SETTINGS}, **over}
    return write_manifest(m, tmp_path / "manifest.json")


def test_corpus_json_contents(tmp_path):
    corpus = build_corpus(CHUNKS)
    assert corpus["chunk_ids"] == [c["chunk_id"] for c in CHUNKS] and len(corpus["tokens"]) == 5
    assert corpus["tokenizer_version"] == bm25_index.TOKENIZER_VERSION
    assert corpus["k1"] == 1.5 and corpus["b"] == 0.75 and corpus["library"] == "rank_bm25" and corpus["version"]
    path = save_corpus(corpus, tmp_path / "bm25_corpus.json")
    assert json.loads(path.read_text()) == corpus                 # plain JSON, no pickle


def test_save_load_round_trip_gives_identical_scores(tmp_path, monkeypatch):
    corpus = build_corpus(CHUNKS)
    original = make_bm25(corpus)
    path = save_corpus(corpus, tmp_path / "bm25_corpus.json")
    manifest_path = manifest_for(tmp_path)
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest_path)
    loaded, ids = load_bm25(path)
    assert ids == corpus["chunk_ids"]
    for q in QUERIES:
        a = list(original.get_scores(bm25_index.tokenize(q)))
        b = list(loaded.get_scores(bm25_index.tokenize(q)))
        assert a == b, q


def test_exact_identifier_ranks_first():
    corpus = build_corpus(CHUNKS)
    bm25 = make_bm25(corpus)
    assert search(bm25, corpus["chunk_ids"], "GV-1.7-001")[0][0] == "d:prose:0001:1"
    assert search(bm25, corpus["chunk_ids"], "confabulation")[0][0] == "d:prose:0002:1"
    assert search(bm25, corpus["chunk_ids"], "decommissioned")[0][0] == "d:prose:0004:1"      # stemming
    assert search(bm25, corpus["chunk_ids"], "zzzzqqq") == []


def test_mismatched_tokenizer_version_in_corpus_raises(tmp_path, monkeypatch):
    corpus = build_corpus(CHUNKS)
    corpus["tokenizer_version"] = "0-old"
    path = save_corpus(corpus, tmp_path / "bm25_corpus.json")
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest_for(tmp_path))
    with pytest.raises(IndexIncompatible, match="tokenized with '0-old'.*rebuild with `python -m indexing --rebuild`"):
        load_bm25(path)


def test_mismatched_tokenizer_version_in_manifest_raises(tmp_path, monkeypatch):
    corpus = build_corpus(CHUNKS)
    path = save_corpus(corpus, tmp_path / "bm25_corpus.json")
    manifest_path = tmp_path / "manifest.json"
    write_manifest({"bm25": {"tokenizer_version": "0-old", **config.BM25_SETTINGS}}, manifest_path)
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest_path)
    with pytest.raises(IndexIncompatible, match="tokenizer_version='0-old'"):
        load_bm25(path)


def test_changed_bm25_setting_raises(tmp_path, monkeypatch):
    corpus = build_corpus(CHUNKS)
    path = save_corpus(corpus, tmp_path / "bm25_corpus.json")
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest_for(tmp_path))
    monkeypatch.setitem(config.BM25_SETTINGS, "k1", 2.0)
    with pytest.raises(IndexIncompatible, match="k1=1.5, config says 2.0"):
        load_bm25(path)


def test_missing_corpus_or_manifest_section_raises(tmp_path, monkeypatch):
    with pytest.raises(IndexIncompatible, match="missing"):
        load_bm25(tmp_path / "nope.json")
    corpus_path = save_corpus(build_corpus(CHUNKS), tmp_path / "bm25_corpus.json")
    manifest_path = tmp_path / "manifest.json"
    write_manifest({}, manifest_path)
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest_path)
    with pytest.raises(IndexIncompatible, match="no BM25 section"):
        load_bm25(corpus_path)
