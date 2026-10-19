"""Stage 2: acquire the official PDFs and prove they are the right files.

Why verify: the page numbers and links in every later stage are only valid for
the exact published document, so a wrong/revised/corrupt file must stop the run
here rather than produce plausible but wrong citations.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
import requests

from ingest import config
from ingest.models import DocumentSpec, Manifest

log = logging.getLogger(__name__)

USER_AGENT = "nist-rag-ingest/1.0 (research prototype)"
TIMEOUT_SECONDS = 60
DOWNLOAD_ATTEMPTS = 2  # first try + one retry


class AcquireError(RuntimeError):
    """A document could not be acquired or failed verification."""


@dataclass
class AcquireResult:
    doc_id: str
    path: str
    status: str            # "downloaded" | "copied" | "already_present"
    pages: int
    title: str
    sha256: str
    sha256_pinned: bool    # True if the manifest had a value and it matched


@dataclass
class _Verified:
    pages: int
    title: str
    sha256: str
    sha256_pinned: bool


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_pdf(path: Path, spec: DocumentSpec) -> _Verified:
    """Run checks a-d in order; raise AcquireError naming the first that fails.

    Order matters: the cheap magic-byte check runs first so an HTML error page
    never reaches the PDF parser.
    """
    with open(path, "rb") as f:
        head = f.read(4)
    if head != b"%PDF":
        raise AcquireError(
            f"{spec.doc_id}: check (a) failed for {path.name}: first bytes are {head!r}, "
            f"expected b'%PDF'. The file is not a PDF (an HTML error page?). Check the URL or --from-dir.")
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # corrupt PDF; pymupdf raises various types
        raise AcquireError(f"{spec.doc_id}: {path.name} starts with %PDF but cannot be opened: {e}") from e
    with doc:
        pages = doc.page_count
        title = (doc.metadata or {}).get("title") or ""
    if pages != spec.expected_pages:
        raise AcquireError(f"{spec.doc_id}: check (b) failed: page count is {pages}, expected {spec.expected_pages}. "
                           "The document may have been revised; compare with docs/process/plans/INGESTION_PLAN.md Section 5.")
    if title != spec.expected_title:
        raise AcquireError(f"{spec.doc_id}: check (c) failed: Title is {title!r}, expected {spec.expected_title!r}.")
    digest = sha256_of(path)
    if spec.sha256:
        if digest.lower() != spec.sha256.lower():
            raise AcquireError(f"{spec.doc_id}: check (d) failed: SHA-256 is {digest}, manifest pins {spec.sha256}. "
                               "Either the file changed or the pin is wrong.")
        return _Verified(pages, title, digest, True)
    log.warning("acquire: %s has no pinned sha256; computed %s. Pin it in ingest/sources.yaml.", spec.doc_id, digest)
    return _Verified(pages, title, digest, False)


def _download_to_temp(spec: DocumentSpec, dest_dir: Path) -> Path:
    """Stream the URL into a temp file in dest_dir (same filesystem => atomic move later)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{spec.doc_id}.", suffix=".part", dir=dest_dir)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, requests.get(
                spec.url, stream=True, timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            for block in r.iter_content(1 << 16):
                out.write(block)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _find_local(spec: DocumentSpec, src_dir: Path) -> Path | None:
    """Find a local copy by manifest filename, else by the document number in the name.

    The number (e.g. "100-1") is derived from the manifest filename, so no
    document-specific string lives in code. Users' copies are often renamed.
    """
    exact = src_dir / spec.filename
    if exact.is_file():
        return exact
    m = re.search(r"\d+-\d+", spec.filename)
    key = m.group(0) if m else None
    candidates = sorted(p for p in src_dir.iterdir()
                        if p.is_file() and p.suffix.lower() == ".pdf"
                        and (p.name.lower() == spec.filename.lower() or (key and key in p.name)))
    if len(candidates) > 1:
        raise AcquireError(f"{spec.doc_id}: several files in {src_dir} match: {[p.name for p in candidates]}. "
                           "Leave only one.")
    return candidates[0] if candidates else None


def _result(spec: DocumentSpec, dest: Path, status: str, v: _Verified) -> AcquireResult:
    return AcquireResult(spec.doc_id, str(dest), status, v.pages, v.title, v.sha256, v.sha256_pinned)


def acquire_document(spec: DocumentSpec, raw_dir: Path, from_dir: Path | None = None) -> AcquireResult:
    dest = raw_dir / spec.filename

    if dest.is_file():  # idempotent: keep a good file
        try:
            v = verify_pdf(dest, spec)
            log.info("acquire: %s already present (%s)", spec.doc_id, dest.name)
            return _result(spec, dest, "already_present", v)
        except AcquireError as e:
            log.warning("acquire: existing %s failed verification, re-acquiring: %s", dest.name, e)

    raw_dir.mkdir(parents=True, exist_ok=True)

    if from_dir is not None:
        src = _find_local(spec, from_dir)
        if src is None:
            raise AcquireError(f"{spec.doc_id}: no file for {spec.filename} found in {from_dir}.")
        v = verify_pdf(src, spec)  # verify the source before touching data/raw
        if not (dest.exists() and os.path.samefile(src, dest)):
            shutil.copyfile(src, dest)
        log.info("acquire: %s copied from %s", spec.doc_id, src)
        return _result(spec, dest, "copied", v)

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        log.info("acquire: %s downloading %s (attempt %d/%d)", spec.doc_id, spec.url, attempt, DOWNLOAD_ATTEMPTS)
        tmp: Path | None = None
        try:
            tmp = _download_to_temp(spec, raw_dir)
            v = verify_pdf(tmp, spec)
            os.replace(tmp, dest)  # only after all checks pass
            return _result(spec, dest, "downloaded", v)
        except (requests.RequestException, AcquireError) as e:
            last_error = e
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            log.warning("acquire: %s attempt %d failed: %s", spec.doc_id, attempt, e)
    raise AcquireError(f"{spec.doc_id}: download failed after {DOWNLOAD_ATTEMPTS} attempts. Last error: {last_error}")


def acquire_all(manifest: Manifest, raw_dir: Path | None = None, from_dir: Path | None = None) -> list[AcquireResult]:
    """Acquire every document and save the outcome for the final report."""
    raw_dir = raw_dir or config.RAW_DIR
    results = [acquire_document(spec, raw_dir, from_dir) for spec in manifest.documents]
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    (config.STAGES_DIR / "02_acquire.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return results
