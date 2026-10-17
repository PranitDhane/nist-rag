"""Stage 6 tests: rotation direction, threshold, Tesseract-unavailable path, real OCR run."""
from __future__ import annotations

import json

import pytest
from PIL import Image

from ingest import config
from ingest.classify import classify_all
from ingest.extract import load_extract
from ingest.figures import apply_threshold, figures_all, upright_rotation
from ingest.profile import detect_tesseract, profile_all


def test_upright_rotation_direction():
    assert upright_rotation([0, -632.41, 488.87, 0, 0, 0]) == -90
    assert upright_rotation([0, 632.41, -488.87, 0, 0, 0]) == 90


def test_threshold_only_drops_low_confidence_ok_records():
    ok = {"ocr_status": "ok", "confidence": 40.0, "text": "abc"}
    assert apply_threshold(ok, None) == ok
    assert apply_threshold(ok, 50) == {"ocr_status": "below_threshold", "confidence": 40.0, "text": None}
    assert apply_threshold({**ok, "confidence": 80.0}, 50)["text"] == "abc"
    skipped = {"ocr_status": "skipped_unavailable", "confidence": None, "text": None}
    assert apply_threshold(skipped, 50) == skipped


@pytest.fixture(scope="module")
def inputs(manifest, need_pdfs):
    profile = json.loads((config.STAGES_DIR / "03_profile.json").read_text())
    return profile, classify_all(load_extract())


def test_caption_only_when_tesseract_hidden(manifest, inputs, tmp_path, monkeypatch):
    profile, classified = inputs
    monkeypatch.setattr(config, "STAGES_DIR", tmp_path)
    hidden = {**profile, "tesseract": {"available": False, "version": None}}
    recs = figures_all(manifest, hidden, classified, out_dir=tmp_path / "png")
    ai = [r for r in recs if r["doc_id"] == "ai_rmf_1_0"]
    assert len(ai) == 5
    assert all(r["method"] == "caption_only" and r["text"] is None and r["caption"]
               and r["ocr_status"] == "skipped_unavailable" for r in ai)
    assert not (tmp_path / "png").exists()


def test_real_ocr_five_figures_upright(manifest, inputs, tmp_path, monkeypatch):
    if not detect_tesseract()["available"]:
        pytest.skip("Tesseract not installed")
    profile, classified = inputs
    monkeypatch.setattr(config, "STAGES_DIR", tmp_path)
    recs = figures_all(manifest, profile, classified, out_dir=tmp_path / "png")
    assert [r["figure_number"] for r in recs] == [1, 2, 3, 4, 5]
    assert all(r["method"] == "ocr" and r["ocr_status"] == "ok" for r in recs)
    for r in recs:
        with Image.open(r["png"]) as im:
            # every AI RMF figure is wider than tall once upright (Fig. 3 starts rotated)
            assert im.width > im.height, r["figure_number"]
    assert next(r for r in recs if r["figure_number"] == 3)["confidence"] > 60
