"""Stage 6: recover the text that exists only inside figure images, labelled as OCR.

Why separate from page Markdown: OCR text is noisy and tool-dependent, so it is
stored in its own field with method/confidence and never mixed into the page
text (plan Section 6, Stage 6, step 7). Tesseract is optional: without it we
still emit one record per figure (caption only) so downstream code never has to
guess whether a figure existed.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pymupdf
from PIL import Image

from ingest import config
from ingest.models import Manifest

log = logging.getLogger(__name__)

OCR_DPI = 300
DEFAULT_CONFIDENCE_THRESHOLD = 60.0  # chosen by the user after the first run (Figs 1,3,4,5 scored 88-94)


def confidence_threshold() -> float | None:
    """OCR confidence threshold: OCR_CONFIDENCE_THRESHOLD (.env) or the default; "none" disables filtering."""
    value = os.environ.get("OCR_CONFIDENCE_THRESHOLD", "").strip()
    if value.lower() == "none":
        return None
    return float(value) if value else DEFAULT_CONFIDENCE_THRESHOLD


def upright_rotation(transform: list[float]) -> int:
    """PIL rotate() angle that makes a 90-degree-rotated image's text horizontal.

    The matrix maps the image's x axis to (a, b) on the page. b < 0 means the
    image's left-to-right runs bottom-to-top, i.e. text reads upwards, so the
    rendered clip must turn 90 degrees clockwise (PIL angle -90); b > 0 is the opposite.
    """
    b = transform[1]
    return -90 if b < 0 else 90


def render_figure(page: pymupdf.Page, bbox: list[float], rotated: bool, transform: list[float]) -> Image.Image:
    """Render only the image area (not the whole page) so page text is not OCR'd as figure text."""
    pix = page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=OCR_DPI)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if rotated:
        img = img.rotate(upright_rotation(transform), expand=True)
    return img


def ocr_image(img: Image.Image) -> tuple[str, float | None]:
    """Run Tesseract; return (text, mean word confidence). Confidence -1 entries are not words."""
    import pytesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(data["conf"], data["text"]) if float(c) >= 0 and t.strip()]
    text = pytesseract.image_to_string(img).strip()
    return text, (round(sum(confs) / len(confs), 1) if confs else None)


def apply_threshold(rec: dict, threshold: float | None) -> dict:
    """Below the threshold the text is dropped (kept as null) but the confidence stays visible."""
    if threshold is not None and rec["ocr_status"] == "ok" and (rec["confidence"] or 0.0) < threshold:
        return {**rec, "text": None, "ocr_status": "below_threshold"}
    return rec


def figure_record(doc_id: str, page: pymupdf.Page, image: dict, tesseract: dict,
                  out_dir: Path, threshold: float | None) -> dict:
    base = {"doc_id": doc_id, "figure_number": image["figure_number"], "pdf_page": image["pdf_page"],
            "caption": image["caption"]}
    if not tesseract["available"]:
        return {**base, "text": None, "method": "caption_only", "confidence": None,
                "tesseract_version": None, "ocr_status": "skipped_unavailable", "png": None}
    img = render_figure(page, image["bbox"], image["rotated"], image["transform"])
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{doc_id}_fig{image['figure_number']}.png"
    img.save(png)
    text, conf = ocr_image(img)
    rec = {**base, "text": text, "method": "ocr", "confidence": conf,
           "tesseract_version": tesseract["version"], "ocr_status": "ok", "png": str(png)}
    return apply_threshold(rec, threshold)


def figures_all(manifest: Manifest, profile: dict, classified: list[dict], raw_dir: Path | None = None,
                out_dir: Path | None = None, threshold: float | None = None) -> list[dict]:
    """One record per captioned figure that sits on an index-eligible page."""
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.DEBUG_FIGURES_DIR
    eligible = {(r["doc_id"], r["pdf_page"]) for r in classified if r["index_eligible"]}
    if not profile["tesseract"]["available"]:
        log.warning("figures: Tesseract unavailable; writing caption-only figure records (no OCR).")
    records: list[dict] = []
    for spec in manifest.documents:
        prof = next((d for d in profile["documents"] if d["doc_id"] == spec.doc_id), None)
        if prof is None:
            raise ValueError(f"figures: {spec.doc_id} not in the profile. Re-run Stage 3.")
        with pymupdf.open(raw_dir / spec.filename) as doc:
            for image in prof["images"]:
                if image["figure_number"] is None or (spec.doc_id, image["pdf_page"]) not in eligible:
                    continue
                records.append(figure_record(spec.doc_id, doc[image["pdf_page"] - 1], image,
                                             profile["tesseract"], out_dir, threshold))
        log.info("figures: %s: %d figure records", spec.doc_id, sum(r["doc_id"] == spec.doc_id for r in records))
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STAGES_DIR / "06_figures.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_figures() -> list[dict]:
    path = config.STAGES_DIR / "06_figures.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"figures: {path} is missing. Run Stage 6 first (python -m ingest --only-stage 6).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
