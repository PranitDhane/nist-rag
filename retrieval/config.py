"""Every retrieval setting in one place (RETRIEVAL_PLAN Stage T1)."""
from __future__ import annotations

import hashlib
import json
import os

from indexing import config as index_config
from indexing.manifest import check_compatibility

# --- candidates and fusion (T-D1 .. T-D3) -------------------------------------------------------
VECTOR_K = 10
BM25_K = 10
WEIGHTS = [0.6, 0.4]            # order matters: [vector, bm25]
RRF_C = 60                      # EnsembleRetriever's default (`c` field in langchain-classic 1.0.8)
ID_KEY = "chunk_id"             # EnsembleRetriever dedupes on metadata[ID_KEY] instead of page_content

# --- reranking (T-D4 .. T-D6) -------------------------------------------------------------------
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"   # pinned Hugging Face commit
RERANKER_MAX_LENGTH = 512       # query + chunk are cut at this many tokens
RERANKER_DEVICE_SETTING = os.environ.get("RETRIEVAL_DEVICE", "auto")   # Q2: auto = cuda, mps, cpu
TOP_N = 5
RERANK_TEXT = "embed_text"      # which chunk text the reranker reads (Q4: embed_text, with the breadcrumb)
RERANK_SCORE_SCALE = "sigmoid"  # "sigmoid" (0-1) or "logit" (raw). CrossEncoder default for this model is Sigmoid; used everywhere

# --- query prefix (T-D7) ------------------------------------------------------------------------
# Source: the BAAI/bge-small-en-v1.5 model card, "Model List" table, column "query instruction for retrieval".
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- siblings and ordering (T-D8, T-D9; Q5/Q6 decided in Stage T5) ------------------------------
SIBLING_CHUNK_TYPES = ("genai_action_table", "rmf_core_table")   # only split tables are expanded
SIBLING_PLACEMENT = "group"     # "group" = part order, group at its best part's position; "append" = at the end
SIBLINGS_COUNT_TOWARD_TOP_N = False

# --- relevance gate (T-D10; value and mode chosen by the user in Stage T6, Q7) ------------------
THRESHOLD: float | None = 0.15  # chosen by the user from the Stage T6 table (sigmoid scale); plateau 0.073-0.240
GATE_REMOVES_CHUNKS = False     # Q7b: gate only; chunks below the threshold are NOT removed individually


def resolve_device(setting: str | None = None) -> str:
    return index_config.resolve_device(setting or RERANKER_DEVICE_SETTING)


def as_dict() -> dict:
    return {
        "vector_k": VECTOR_K, "bm25_k": BM25_K, "weights": WEIGHTS, "rrf_c": RRF_C, "id_key": ID_KEY,
        "reranker_model": RERANKER_MODEL, "reranker_revision": RERANKER_REVISION, "reranker_max_length": RERANKER_MAX_LENGTH, "top_n": TOP_N,
        "rerank_text": RERANK_TEXT, "rerank_score_scale": RERANK_SCORE_SCALE, "query_prefix": QUERY_PREFIX,
        "sibling_chunk_types": list(SIBLING_CHUNK_TYPES), "sibling_placement": SIBLING_PLACEMENT,
        "siblings_count_toward_top_n": SIBLINGS_COUNT_TOWARD_TOP_N,
        "threshold": THRESHOLD, "gate_removes_chunks": GATE_REMOVES_CHUNKS,
        "embedding_model": index_config.MODEL_NAME, "embedding_revision": index_config.MODEL_REVISION,
        "tokenizer_version": _tokenizer_version(),
    }


def _tokenizer_version() -> str:
    from indexing import bm25_index
    return bm25_index.TOKENIZER_VERSION


def config_version() -> str:
    """Short hash of every value above, so a RetrievalResult says which settings produced it."""
    blob = json.dumps(as_dict(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def startup_check(cfg=index_config) -> None:
    """Embedding model, revision, normalization and distance must equal what the index manifest recorded."""
    check_compatibility(cfg)
