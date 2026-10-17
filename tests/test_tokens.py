import pytest

pytest.importorskip("transformers")
from chunking.tokens import count_tokens  # noqa: E402


def test_count_includes_special_tokens():
    # BERT wordpiece: "hello world" -> [CLS] hello world [SEP]
    assert count_tokens("hello world") == 4
    assert count_tokens("") == 2


def test_count_is_not_truncated_at_512():
    assert count_tokens("word " * 600) == 602
