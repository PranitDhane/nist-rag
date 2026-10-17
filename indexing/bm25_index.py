"""Stage E5: BM25 keyword index. The tokenizer here is shared by the index and by retrieval-time queries."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from indexing import config

# Bump whenever tokenize() changes: the corpus JSON and the manifest record it, and loaders refuse a mismatch.
TOKENIZER_VERSION = "1-ids-whole+snowball-english"

_FUNCS = r"(?:govern|map|measure|manage)"
# Alternatives are tried left to right at each position, so identifiers win over plain words.
_TOKEN_RE = re.compile(
    r"(?P<action>\b(?:gv|mp|ms|mg)-\d+\.\d+-\d{3}\b)"                    # action ID: gv-1.7-001
    rf"|(?P<ref>\b(?P<func>{_FUNCS})\s+(?P<num>\d+(?:\.\d+)*)(?![\d.]*\d)\b)"   # subcategory/category ref: govern 1.7
    r"|(?P<dotted>\b\d+(?:\.\d+)+\b)"                                    # dotted section numbers: 2.12, 1.2.3
    r"|(?P<word>[a-z0-9]+)"
)


def base_tokens(text: str) -> list[str]:
    """Lowercase; keep identifiers whole; split everything else on non-alphanumerics.

    'GV-1.7-001' -> ['gv-1.7-001'];  '2.12' -> ['2.12'];  'GOVERN 1.7' -> ['govern', '1.7', 'govern_1.7'].
    """
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        if m.group("action"):
            out.append(m.group("action"))
        elif m.group("ref"):
            func, num = m.group("func"), m.group("num")
            out += [func, num, f"{func}_{num}"]
        elif m.group("dotted"):
            out.append(m.group("dotted"))
        else:
            out.append(m.group("word"))
    return out


@lru_cache(maxsize=1)
def _stemmer():
    import snowballstemmer
    return snowballstemmer.stemmer("english")


def _is_identifier(token: str) -> bool:
    return any(c.isdigit() for c in token) or "-" in token or "_" in token


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return _stemmer().stemWord(word)


def tokenize(text: str) -> list[str]:
    """THE tokenizer: used to build the index and, unchanged, for queries at retrieval time."""
    return [t if _is_identifier(t) else _stem(t) for t in base_tokens(text)]


# ----------------------------------------------------------------------------- build / save / load
def build_corpus(chunks: list[dict]) -> dict:
    """Tokenized corpus (ordered like `chunks`) plus everything needed to rebuild the BM25 object."""
    field = config.BM25_SETTINGS["indexed_text"]
    return {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "tokens": [tokenize(c[field]) for c in chunks],
        "tokenizer_version": TOKENIZER_VERSION,
        **{k: config.BM25_SETTINGS[k] for k in ("k1", "b", "epsilon", "indexed_text", "stopwords", "stemming")},
        "library": config.BM25_SETTINGS["library"], "version": version("rank_bm25"),
    }


def save_corpus(corpus: dict, path: Path | None = None) -> Path:
    path = path or config.BM25_CORPUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corpus, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")   # JSON, never pickle
    return path


def make_bm25(corpus: dict):
    from rank_bm25 import BM25Okapi
    return BM25Okapi(corpus["tokens"], k1=corpus["k1"], b=corpus["b"], epsilon=corpus["epsilon"])


def corpus_stats(corpus: dict) -> dict:
    lengths = [len(t) for t in corpus["tokens"]]
    return {"vocab_size": len({tok for doc in corpus["tokens"] for tok in doc}),
            "avg_doc_len": round(sum(lengths) / max(len(lengths), 1), 2), "chunk_count": len(lengths)}


def search(bm25, chunk_ids: list[str], query: str, k: int = 5) -> list[tuple[str, float]]:
    """Top-k (chunk_id, score) for a query, using the shared tokenizer. Zero-score chunks are not hits."""
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return [(chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]
