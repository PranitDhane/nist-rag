"""Stage C2: measure real sizes (token counts, tiny sections, repeated table headers). Produces no chunks."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter

from chunking import config
from chunking.join import join_all, load_pages
from chunking.segment import Segmentation, segment
from chunking.tokens import count_tokens

THRESHOLDS = (350, 400, 450, 480)
HEADER_ROW_RE = re.compile(r"^\|\s*\*\*(?:Action ID|Categories)\*\*\s*\|", re.M)


def stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    hist = Counter((v // 50) * 50 for v in values)
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values),
            "histogram_50": {f"{k}-{k + 49}": hist[k] for k in sorted(hist)},
            "over": {str(t): sum(v > t for v in values) for t in THRESHOLDS}}


def repeated_header_rows(seg: Segmentation) -> list[dict]:
    """Table header rows that are NOT directly after a table's first line (i.e. repeated on a continued page)."""
    out = []
    text = seg.doc.text
    for u in seg.units:
        if u.kind == "prose":
            continue
        body = text[u.start:u.end]
        first_line_end = body.find("\n")
        for m in HEADER_ROW_RE.finditer(body):
            # the genuine header is the first header row of the unit; later ones are repeats
            pass
        hits = list(HEADER_ROW_RE.finditer(body))
        for m in hits[1:] if u.kind == "genai_table" else hits[1:]:
            out.append({"unit": u.marker_id, "pdf_page": seg.doc.page_at(u.start + m.start()).pdf_page,
                        "row": body[m.start():body.find("\n", m.start())]})
    return out


def profile() -> dict:
    pages = load_pages()
    result = {}
    for doc in join_all(pages):
        seg = segment(doc)
        text = doc.text
        by_section: dict[int, list] = {}
        for u in seg.units:
            if u.kind == "prose":
                by_section.setdefault(u.section_index, []).append(u)
        section_tokens = {}
        for idx, us in by_section.items():
            section_tokens[idx] = sum(count_tokens(text[u.start:u.end]) for u in us)
        tiny = [{"section_path": seg.sections[i].path, "tokens": t,
                 "text": " ".join(text[u.start:u.end] for u in by_section[i])[:300]}
                for i, t in sorted(section_tokens.items()) if t < 60]
        prose_units = [count_tokens(text[u.start:u.end]) for u in seg.units if u.kind == "prose"]
        genai = [{"marker": u.marker_id, "tokens": count_tokens(text[u.start:u.end])}
                 for u in seg.units if u.kind == "genai_table"]
        rmf = []
        for u in seg.units:
            if u.kind == "rmf_table":
                for c in seg.categories[u.sequence]:
                    rmf.append({"table": u.marker_id, "category": c.label, "tokens": count_tokens(text[c.start:c.end])})
        figs = [f for p in doc.page_records.values() for f in p["figures"]]
        fig_rows = [{"figure": f["figure_number"], "has_ocr": bool(f["text"]),
                     "tokens": count_tokens(f["text"]) if f["text"] else 0, "status": f["ocr_status"]} for f in figs]
        notes = [count_tokens(f"Footnote {n['number']}: {n['text']}") for p in doc.page_records.values() for n in p["footnotes"]]
        result[doc.doc_id] = {
            "pages_joined": len(doc.pages), "characters": len(text), "sections": len(seg.sections),
            "units": dict(Counter(u.kind for u in seg.units)),
            "prose_per_section_tokens": stats(list(section_tokens.values())),
            "prose_unit_tokens": stats(prose_units),
            "sections_over_prose_max": sum(t > config.PROSE_TARGET_MAX for t in section_tokens.values()),
            "tiny_sections_under_60": tiny,
            "genai_tables": {**stats([g["tokens"] for g in genai]), "items": genai},
            "rmf_categories": {**stats([r["tokens"] for r in rmf]), "items": rmf},
            "figures": fig_rows,
            "footnotes": stats(notes),
            "repeated_header_rows": repeated_header_rows(seg),
        }
    return result


def write_profile() -> dict:
    prof = profile()
    report = {}
    if config.REPORT_PATH.is_file():
        report = json.loads(config.REPORT_PATH.read_text(encoding="utf-8"))
    report["profile"] = prof
    config.REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return prof
