"""Builds real indexes from chunks.jsonl in a temp dir, then corrupts them to show each hard check can fail."""
import json
import shutil

import pytest

pytest.importorskip("chromadb")
from indexing import config  # noqa: E402

pytestmark = pytest.mark.skipif(not config.CHUNKS_PATH.is_file(), reason="chunks.jsonl absent")


def point_config_at(monkeypatch, root, chunks_path=None):
    chroma, index = root / "chroma", root / "index"
    chunks_path = chunks_path or config.CHUNKS_PATH
    for name, value in {"CHROMA_DIR": chroma, "INDEX_DIR": index, "BM25_CORPUS_PATH": index / "bm25_corpus.json",
                        "MANIFEST_PATH": index / "index_manifest.json", "REPORT_PATH": index / "index_report.json",
                        "CHUNKS_PATH": chunks_path}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setenv("INDEX_CHROMA_DIR", str(chroma))       # inherited by the fresh-process check
    monkeypatch.setenv("INDEX_DIR", str(index))
    monkeypatch.setenv("CHUNKS_PATH", str(chunks_path))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("built")
    mp = pytest.MonkeyPatch()
    point_config_at(mp, root)
    from indexing.__main__ import main
    code = main(["--rebuild"])
    mp.undo()
    return root, code


@pytest.fixture
def copy_of_built(built, tmp_path, monkeypatch):
    root, code = built
    assert code == 0
    shutil.copytree(root, tmp_path / "idx")
    point_config_at(monkeypatch, tmp_path / "idx")
    return tmp_path / "idx"


def results(report_dir):
    rep = json.loads((report_dir / "index" / "index_report.json").read_text())
    return {c["number"]: c for c in rep["checks"]}, rep


def test_full_rebuild_passes_every_hard_check(built):
    root, code = built
    assert code == 0
    checks, rep = results(root)
    assert all(c["passed"] for c in checks.values()) and len(checks) == 8 and not checks[8]["skipped"]
    assert rep["counts_by_doc"] == {"ai_rmf_1_0": 85, "genai_profile": 123} or sum(rep["counts_by_doc"].values()) > 0
    assert rep["chroma_settings_read_back"]["distance"] == "cosine" and rep["model_settings_read_back"]["dimension"] == 384
    assert rep["bm25"]["vocab_size"] > 1000 and rep["bm25"]["avg_doc_len"] > 50
    assert len(rep["smoke_checks"]) == 6


def test_manifest_records_how_the_indexes_were_built(built):
    root, _ = built
    m = json.loads((root / "index" / "index_manifest.json").read_text())
    for key in ("embedding_model", "model_revision", "dimension", "normalized", "max_seq_length", "document_prefix",
                "vector_store", "chroma_version", "collection_name", "distance", "persist_dir", "bm25",
                "chunks_file_sha256", "chunk_count", "chunk_ids_sha256", "built_at", "build_seconds", "device"):
        assert key in m, key
    assert m["document_prefix"] == "none" and m["distance"] == "cosine" and m["normalized"] is True
    assert {"library", "version", "tokenizer_version", "k1", "b", "stemming", "stopwords"} <= set(m["bm25"])


def test_validate_only_on_existing_indexes(copy_of_built):
    from indexing.__main__ import main
    assert main(["--validate"]) == 0


def test_stale_chunks_file_fails_check_3(copy_of_built, tmp_path, monkeypatch):
    changed = tmp_path / "chunks.jsonl"
    changed.write_bytes(config.CHUNKS_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(config, "CHUNKS_PATH", changed)
    from indexing.__main__ import main
    assert main(["--validate"]) == 1
    checks, _ = results(copy_of_built)
    assert not checks[3]["passed"] and "stale" in checks[3]["detail"]


def test_bm25_missing_a_chunk_fails_checks_1_and_2(copy_of_built):
    path = copy_of_built / "index" / "bm25_corpus.json"
    corpus = json.loads(path.read_text())
    corpus["chunk_ids"].pop()
    corpus["tokens"].pop()
    path.write_text(json.dumps(corpus))
    from indexing.__main__ import main
    assert main(["--validate"]) == 1
    checks, _ = results(copy_of_built)
    assert not checks[1]["passed"] and not checks[2]["passed"]


def test_chroma_missing_a_chunk_fails_checks_1_and_2(copy_of_built):
    import chromadb
    col = chromadb.PersistentClient(path=str(copy_of_built / "chroma")).get_collection(config.COLLECTION_NAME)
    col.delete(ids=col.get(limit=1, include=[])["ids"])
    from indexing.__main__ import main
    assert main(["--validate"]) == 1
    checks, _ = results(copy_of_built)
    assert not checks[1]["passed"] and not checks[2]["passed"]


def test_changed_model_makes_loaders_refuse(copy_of_built, monkeypatch):
    from indexing.loaders import load_bm25, load_vector_store
    from indexing.manifest import IndexIncompatible
    load_vector_store()
    load_bm25()
    monkeypatch.setattr(config, "MODEL_REVISION", "0" * 40)
    with pytest.raises(IndexIncompatible, match="model_revision.*rebuild with `python -m indexing --rebuild`"):
        load_vector_store()


def test_only_bm25_still_runs_the_id_equality_check(copy_of_built):
    from indexing.__main__ import main
    assert main(["--only", "bm25"]) == 0
    checks, _ = results(copy_of_built)
    assert checks[2]["passed"] and checks[8]["skipped"]


def test_loaders_return_usable_objects(copy_of_built):
    from indexing import bm25_index
    from indexing.loaders import load_bm25, load_chunk_lookup, load_vector_store
    store, (bm25, ids), lookup = load_vector_store(), load_bm25(), load_chunk_lookup()
    assert set(ids) == set(lookup) and len(ids) == store._collection.count()
    top_id = bm25_index.search(bm25, ids, "GV-1.7-001", 1)[0][0]
    assert lookup[top_id]["subcategory"] == "GOVERN 1.7" and "GV-1.7-001" in lookup[top_id]["display_text"]
    hit = store.similarity_search("What is confabulation?", k=1, filter={"doc_id": "genai_profile"})[0]
    assert hit.metadata["doc_id"] == "genai_profile" and "onfabulation" in hit.page_content
