"""All chunking limits and paths in one place (CHUNKING_PLAN Stage C1)."""
from __future__ import annotations

from ingest import config as ingest_config

MODEL_NAME = "BAAI/bge-small-en-v1.5"
HARD_MAX_EMBED_TOKENS = 512          # including special tokens
PROSE_TARGET_MIN = 250
PROSE_TARGET_MAX = 350
PROSE_OVERLAP_FRACTION = 0.10        # of the piece size

# Decided by the user after profiling (Stage C2).
TINY_SECTION_MAX_TOKENS = 60         # Q3: a section whose own prose is <= this is merged forward
TABLE_PIECE_MAX_TOKENS = 450         # Q4: max display-text tokens of one table piece (incl. repeated header lines)

# Room kept for the breadcrumb (+ "part i of n") that is added to embed_text only; used as the
# size limit when merging a tiny section into a neighbour (the final 512 check is in Stage C8/C9).
BREADCRUMB_RESERVE_TOKENS = 60
MERGED_MAX_TOKENS = HARD_MAX_EMBED_TOKENS - BREADCRUMB_RESERVE_TOKENS

PAGES_PATH = ingest_config.PROCESSED_DIR / "pages.jsonl"
CHUNKS_PATH = ingest_config.PROCESSED_DIR / "chunks.jsonl"
REPORT_PATH = ingest_config.PROCESSED_DIR / "chunk_report.json"

# NIST AI 600-1 "AI Actor Tasks" names. Used only to repair icon-font garble that ingestion could not
# remove (e.g. "Operation anf i d Monitoring"); a garbled value that matches no name is kept as cleaned.
AI_ACTOR_TASKS = [
    "AI Deployment", "AI Design", "AI Development", "AI Impact Assessment", "Affected Individuals and Communities",
    "Domain Experts", "End-Users", "Governance and Oversight", "Human Factors", "Operation and Monitoring",
    "Procurement", "TEVV", "Third-party entities",
]
