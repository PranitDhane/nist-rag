"""Stage 3: record each document's properties and the machine's OCR capability.

Why a profile stage: later stages (figure OCR, page numbering) decide from
recorded data rather than assumptions, and the user can inspect the facts in
stages/03_profile.json before anything is extracted.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pymupdf

from ingest import config
from ingest.models import DocumentSpec, Manifest

log = logging.getLogger(__name__)

TINY_IMAGE_PT = 50          # images with BOTH sides below this are logos/bullets
CAPTION_MAX_GAP_PT = 40     # a caption must touch or sit this close to its image
CAPTION_RE = re.compile(r"^Fig\.\s*(\d+)\b")


def detect_tesseract() -> dict:
    """Return {"available": bool, "version": str|None}; never raises.

    Tesseract is optional, so a missing binary is a warning, not an error.
    """
    try:
        import pytesseract
        version = str(pytesseract.get_tesseract_version())
        return {"available": True, "version": version}
    except Exception as e:  # TesseractNotFoundError, ImportError, broken install
        log.warning("profile: Tesseract unavailable (%s); figure OCR will be skipped.", type(e).__name__)
        return {"available": False, "version": None}


def is_rotated(transform: tuple[float, ...]) -> bool:
    """90-degree rotation shows as zero a and d with non-zero b and c in the image matrix."""
    a, b, c, d = transform[:4]
    return a == 0 and d == 0 and (b != 0 or c != 0)


def _rect_gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Distance between two rectangles; 0 if they touch or overlap."""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return (dx * dx + dy * dy) ** 0.5


def find_caption(image_bbox: pymupdf.Rect, blocks: list[tuple]) -> tuple[int, str] | None:
    """Nearest text block starting with 'Fig. N' that touches or is within CAPTION_MAX_GAP_PT.

    Captions are not always below the image (Fig. 3's runs vertically beside
    it), so we use rectangle distance rather than 'first block underneath'.
    Returns (figure_number, caption_text) or None.
    """
    best: tuple[float, int, str] | None = None
    for x0, y0, x1, y1, text, *_ in blocks:
        m = CAPTION_RE.match(text.strip())
        if not m:
            continue
        gap = _rect_gap(image_bbox, pymupdf.Rect(x0, y0, x1, y1))
        if gap <= CAPTION_MAX_GAP_PT and (best is None or gap < best[0]):
            best = (gap, int(m.group(1)), " ".join(text.split()))
    return (best[1], best[2]) if best else None


def profile_images(page: pymupdf.Page) -> tuple[list[dict], int]:
    """Images on one page (tiny ones excluded) and the number skipped."""
    blocks = page.get_text("blocks")
    images: list[dict] = []
    skipped = 0
    for info in page.get_image_info(xrefs=True):
        bbox = pymupdf.Rect(info["bbox"])
        if bbox.width < TINY_IMAGE_PT and bbox.height < TINY_IMAGE_PT:
            skipped += 1
            continue
        cap = find_caption(bbox, blocks)
        images.append({
            "pdf_page": page.number + 1,
            "xref": info["xref"],
            "bbox": [round(v, 1) for v in bbox],
            "transform": [round(v, 2) for v in info["transform"]],
            "rotated": is_rotated(info["transform"]),
            "figure_number": cap[0] if cap else None,
            "caption": cap[1] if cap else None,
        })
    return images, skipped


def profile_document(spec: DocumentSpec, pdf_path: Path) -> dict:
    with pymupdf.open(pdf_path) as doc:
        labels = [page.get_label() for page in doc]
        pages_with_text = sum(1 for page in doc if page.get_text("words"))
        images: list[dict] = []
        skipped = 0
        for page in doc:
            imgs, n_skipped = profile_images(page)
            images.extend(imgs)
            skipped += n_skipped
        result = {
            "doc_id": spec.doc_id,
            "pages": doc.page_count,
            "title": (doc.metadata or {}).get("title") or "",
            "pages_with_text": pages_with_text,
            # "present" = most pages carry extractable words (not a scanned PDF)
            "text_layer_present": pages_with_text > doc.page_count / 2,
            "page_labels_present": any(labels),
            "page_number_strategy": spec.page_number_strategy,
            "images": images,
            "tiny_images_skipped": skipped,
        }
    log.info("profile: %s: %d pages, %d images kept, %d tiny skipped, %d with figure caption",
             spec.doc_id, result["pages"], len(images), skipped,
             sum(1 for i in images if i["figure_number"] is not None))
    return result


def profile_all(manifest: Manifest, raw_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or config.RAW_DIR
    docs = []
    for spec in manifest.documents:
        path = raw_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"profile: {spec.doc_id}: {path} is missing. Run Stage 2 (python -m ingest) first.")
        docs.append(profile_document(spec, path))
    profile = {"tesseract": detect_tesseract(), "documents": docs}
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    (config.STAGES_DIR / "03_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def load_profile() -> dict:
    path = config.STAGES_DIR / "03_profile.json"
    if not path.is_file():
        raise FileNotFoundError(f"profile: {path} is missing. Run Stage 3 first (python -m ingest --only-stage 3).")
    return json.loads(path.read_text(encoding="utf-8"))
