import math

import pytest

pytest.importorskip("sentence_transformers")
from indexing import config  # noqa: E402
from indexing.embedder import cache_location, count_tokens, get_embeddings, model_info  # noqa: E402

SAMPLES = ["The AI RMF Core provides outcomes and actions to manage AI risks.",
           "GV-1.7-001 Protocols are put in place to ensure GAI systems can be deactivated.",
           "Confabulation is the production of confidently stated but erroneous content."]


@pytest.fixture(scope="module")
def emb():
    return get_embeddings("cpu")           # CPU: deterministic across machines


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))


def test_model_facts(emb):
    info = model_info(emb)
    assert info["max_seq_length"] == config.MAX_SEQ_LENGTH == 512
    assert info["dimension"] == config.DIMENSION == 384


def test_vectors_have_384_dims_and_unit_norm(emb):
    for v in emb.embed_documents(SAMPLES):
        assert len(v) == 384 and abs(norm(v) - 1.0) < 1e-3
    assert abs(norm(emb.embed_query("what is confabulation?")) - 1.0) < 1e-3


def test_similar_beats_unrelated(emb):
    a, b, c = emb.embed_documents(["The system must be safely decommissioned.",
                                   "The system has to be decommissioned safely.",
                                   "Bananas are rich in potassium and grow in warm climates."])
    assert cos(a, b) > cos(a, c) + 0.2


def test_deterministic(emb):
    v1, v2 = emb.embed_documents(SAMPLES[:1])[0], emb.embed_documents(SAMPLES[:1])[0]
    assert max(abs(x - y) for x, y in zip(v1, v2)) < 1e-5


def test_batch_size_does_not_change_vectors(emb):
    one_by_one = [emb.embed_documents([t])[0] for t in SAMPLES]
    batched = emb.embed_documents(SAMPLES)
    for a, b in zip(one_by_one, batched):
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4


def test_model_tokenizer_matches_chunking_tokenizer(emb):
    from chunking.tokens import count_tokens as chunk_count
    for t in SAMPLES + ["x " * 700]:
        assert count_tokens(t, emb) == chunk_count(t)


def test_cache_location_has_the_pinned_revision():
    assert config.MODEL_REVISION in cache_location()
