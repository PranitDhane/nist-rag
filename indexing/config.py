"""Every embedding/indexing setting in one place (EMBEDDING_INDEXING_PLAN Stage E1).

Paths and a few settings are overridable through .env (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --- embedding model (E-D1, E-D2) ----------------------------------------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Pinned Hugging Face commit (Q3). Queries must use exactly this model, revision and normalization.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIMENSION = 384
MAX_SEQ_LENGTH = 512                      # expected; the loaded model's value is checked in Stage E2
NORMALIZE = True                          # bge: normalized embeddings + cosine similarity
DOCUMENT_PREFIX = "none"                  # documents are embedded without an instruction prefix
BATCH_SIZE = int(os.environ.get("INDEX_BATCH_SIZE", "32"))
DEVICE_SETTING = os.environ.get("INDEX_DEVICE", "auto")   # Q2: "auto" = cuda, then mps, then cpu; or "cpu"/"cuda"/"mps"


def resolve_device(setting: str | None = None) -> str:
    setting = setting or DEVICE_SETTING
    if setting != "auto":
        return setting
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- vector store (E-D3 .. E-D5) -----------------------------------------------------------
CHROMA_DIR = _path("INDEX_CHROMA_DIR", PROJECT_ROOT / "data" / "chroma")
COLLECTION_NAME = os.environ.get("INDEX_COLLECTION", "nist_ai_rmf")
DISTANCE = "cosine"                       # Chroma's default is L2; fixed at creation time
REBUILD_STRATEGY = "full"                 # Q1: delete and recreate every run

# --- files ---------------------------------------------------------------------------------
CHUNKS_PATH = _path("CHUNKS_PATH", PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
INDEX_DIR = _path("INDEX_DIR", PROJECT_ROOT / "data" / "index")
BM25_CORPUS_PATH = INDEX_DIR / "bm25_corpus.json"
MANIFEST_PATH = INDEX_DIR / "index_manifest.json"
REPORT_PATH = INDEX_DIR / "index_report.json"

# --- BM25 (E-D6, E-D7) ---------------------------------------------------------------------
# Q5: rank_bm25 used directly. Q7: index embed_text (breadcrumb + content). Q8: stemming only, no stopword
# removal; identifiers (tokens containing a digit, '-' or '_') are never stemmed.
BM25_SETTINGS = {
    "library": "rank_bm25", "k1": 1.5, "b": 0.75, "epsilon": 0.25,       # BM25Okapi library defaults
    "indexed_text": "embed_text", "stopwords": False, "stemming": "snowball-english",
}

# --- smoke checks (Stage E6): expected chunks live here, never in pipeline code ------------------
# `expect` is a list of alternatives; a result list "hits" if any chunk in its top 5 matches any alternative.
# Matcher keys: doc_id, chunk_type, subcategory, category, section_leaf_startswith, text_contains.
SMOKE_CHECKS: list[dict] = [
    {"query": "What are the four core functions of the AI RMF?",
     "expect": [{"doc_id": "ai_rmf_1_0", "chunk_type": "prose", "section_leaf_startswith": "5 AI RMF Core"}],
     "note": "AI RMF section 5 AI RMF Core (prose)"},
    {"query": "Which function is cross-cutting?",
     "expect": [{"doc_id": "ai_rmf_1_0", "chunk_type": "prose", "section_leaf_startswith": "5.1 Govern"},
                {"doc_id": "ai_rmf_1_0", "text_contains": "Fig. 5"}],
     "note": "AI RMF 5.1 Govern, or section 5 with the Fig. 5 caption"},
    {"query": "What is confabulation?",
     "expect": [{"doc_id": "genai_profile", "chunk_type": "prose", "section_leaf_startswith": "2.2 Confabulation"}],
     "note": "GenAI section 2.2 Confabulation"},
    {"query": "decommissioning GAI systems",
     "expect": [{"doc_id": "genai_profile", "chunk_type": "genai_action_table", "subcategory": "GOVERN 1.7"}],
     "note": "GenAI GOVERN 1.7 table chunk"},
    {"query": "GV-1.7-001",
     "expect": [{"doc_id": "genai_profile", "chunk_type": "genai_action_table", "subcategory": "GOVERN 1.7"}],
     "bm25_rank": 1, "note": "GenAI GOVERN 1.7 table chunk (BM25 rank 1)"},
    {"query": "How do AI risks differ from traditional software risks?",
     "expect": [{"doc_id": "ai_rmf_1_0", "chunk_type": "prose", "section_leaf_startswith": "Appendix B"}],
     "note": "AI RMF Appendix B"},
]
