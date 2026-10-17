"""Paths, settings and manifest loading.

Why one module: no other code may hard-code a path or filename, so moving the
data directory or adding a document is a config change only.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ingest.models import DocumentSpec, Manifest, PageTypeRange, Validation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "sources.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def _path(env_name: str, default: Path) -> Path:
    """Env override (relative paths resolve against the project root)."""
    value = os.environ.get(env_name)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


DATA_DIR = _path("DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = _path("RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR = _path("PROCESSED_DIR", DATA_DIR / "processed")
STAGES_DIR = _path("STAGES_DIR", PROCESSED_DIR / "stages")
DEBUG_FIGURES_DIR = PROCESSED_DIR / "debug" / "figures"
HYPHEN_PROTECT_PATH = Path(__file__).resolve().parent / "hyphen_protect.txt"


def _parse_validation(raw: dict) -> Validation:
    return Validation(
        page_types=[PageTypeRange(**r) for r in raw["page_types"]],
        section_ids=[str(s) for s in raw["section_ids"]],
        printed_page_spot_checks={int(k): (None if v is None else str(v))
                                  for k, v in raw["printed_page_spot_checks"].items()},
        expected_figures=raw.get("expected_figures"),
        expected_footnotes=raw.get("expected_footnotes"),
        subcategory_tables=raw.get("subcategory_tables"),
        unique_action_ids=raw.get("unique_action_ids"),
        chunk_action_ids=raw.get("chunk_action_ids"),
        forbidden_lines=[str(x) for x in raw.get("forbidden_lines") or []],
        forbidden_line_patterns=[str(x) for x in raw.get("forbidden_line_patterns") or []],
    )


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    """Load sources.yaml into dataclasses; fails loudly on missing keys."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    docs = []
    for entry in data["documents"]:
        try:
            docs.append(DocumentSpec(
                doc_id=entry["doc_id"],
                display_name=entry["display_name"],
                url=entry["url"],
                filename=entry["filename"],
                sha256=entry.get("sha256") or "",
                expected_pages=int(entry["expected_pages"]),
                expected_title=entry["expected_title"],
                page_number_strategy=entry["page_number_strategy"],
                offset=entry.get("offset"),
                validation=_parse_validation(entry["validation"]),
            ))
        except KeyError as e:
            raise ValueError(f"{path}: document {entry.get('doc_id')!r} is missing key {e}") from e
    return Manifest(documents=docs)
