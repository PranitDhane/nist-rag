import json

import pytest

from indexing import config
from indexing.manifest import (IndexIncompatible, check_compatibility, config_with, merge_manifest, read_manifest,
                               settings_from_config, sha256_file, sha256_ids, write_manifest)


def good_manifest():
    return {**settings_from_config(config), "vector_store": "chroma", "chunk_count": 3}


def test_match_passes():
    check_compatibility(config, good_manifest())


@pytest.mark.parametrize("override,key", [
    ({"MODEL_NAME": "BAAI/bge-base-en-v1.5"}, "embedding_model"),
    ({"MODEL_REVISION": "0" * 40}, "model_revision"),
    ({"NORMALIZE": False}, "normalized"),
    ({"DISTANCE": "l2"}, "distance"),
    ({"DOCUMENT_PREFIX": "query: "}, "document_prefix"),
])
def test_mismatch_raises_clear_error(override, key):
    with pytest.raises(IndexIncompatible) as e:
        check_compatibility(config_with(**override), good_manifest())
    msg = str(e.value)
    assert key in msg and "python -m indexing --rebuild" in msg and "index was built with" in msg


def test_missing_manifest_field_and_file(tmp_path):
    m = good_manifest()
    del m["distance"]
    with pytest.raises(IndexIncompatible, match="distance"):
        check_compatibility(config, m)
    with pytest.raises(IndexIncompatible, match="no index has been built"):
        check_compatibility(config, manifest_path=tmp_path / "nope.json")


def test_write_read_merge_roundtrip(tmp_path):
    p = tmp_path / "m.json"
    write_manifest(good_manifest(), p)
    m = read_manifest(p)
    assert m["built_at"] and m["chunk_count"] == 3
    merge_manifest({"chunk_count": 5, "bm25": {"k1": 1.5}}, p)
    m = read_manifest(p)
    assert m["chunk_count"] == 5 and m["bm25"] == {"k1": 1.5} and m["embedding_model"] == config.MODEL_NAME
    check_compatibility(config, manifest_path=p)


def test_hashes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert sha256_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256_ids(["b", "a"]) == sha256_ids(["a", "b"])


def test_config_values_are_the_user_decisions():
    assert config.MODEL_NAME == "BAAI/bge-small-en-v1.5" and config.NORMALIZE is True and config.DISTANCE == "cosine"
    assert len(config.MODEL_REVISION) == 40 and config.BATCH_SIZE == 32 and config.COLLECTION_NAME == "nist_ai_rmf"
    assert config.REBUILD_STRATEGY == "full" and config.DEVICE_SETTING == "auto"
    assert config.resolve_device("cpu") == "cpu"
