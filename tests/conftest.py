"""Shared fixtures. Real-PDF tests are skipped when the PDFs are absent."""
from __future__ import annotations

import pytest

from ingest import config
from ingest.config import load_manifest


@pytest.fixture(scope="session")
def manifest():
    return load_manifest()


@pytest.fixture(scope="session")
def pdfs_present(manifest) -> bool:
    return all((config.RAW_DIR / d.filename).is_file() for d in manifest.documents)


@pytest.fixture(scope="session")
def need_pdfs(pdfs_present):
    if not pdfs_present:
        pytest.skip("official PDFs not in data/raw (run python -m ingest)")
