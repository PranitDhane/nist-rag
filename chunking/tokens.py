"""Token counting exactly as the embedding model sees text (special tokens included)."""
from __future__ import annotations

from functools import lru_cache

from chunking import config


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    try:                                   # cache first: no network call once the files are downloaded
        return AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(config.MODEL_NAME)


def count_tokens(text: str) -> int:
    """Number of model tokens for `text`, with [CLS]/[SEP]; no truncation."""
    return len(_tokenizer()(text, add_special_tokens=True, truncation=False)["input_ids"])


def cache_location() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(config.MODEL_NAME, allow_patterns=["tokenizer*", "vocab.txt", "special_tokens_map.json"])
