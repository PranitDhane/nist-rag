"""Runs stages C3-C8 for every document and returns the records plus a log of everything repaired."""
from __future__ import annotations

from chunking.assemble import assemble_document
from chunking.extras import build_figure_chunks, build_footnote_chunks
from chunking.join import JoinedDoc, join_all, load_pages
from chunking.prose import build_prose_chunks
from chunking.segment import Segmentation, segment
from chunking.tables import build_genai_chunks, build_rmf_chunks


def run_document(doc: JoinedDoc) -> tuple[list[dict], dict, Segmentation]:
    seg = segment(doc)
    prose, merge_log = build_prose_chunks(seg)
    genai, genai_log = build_genai_chunks(seg)
    rmf, rmf_log = build_rmf_chunks(seg)
    figures, fig_log = build_figure_chunks(doc)
    footnotes = build_footnote_chunks(doc)
    records = assemble_document(seg, prose, genai, rmf, figures, footnotes)
    log = {"merges": merge_log, "table_repairs": genai_log, "rmf_notes": rmf_log, "figure_notes": fig_log}
    return records, log, seg


def run_all(pages: list[dict] | None = None):
    pages = pages if pages is not None else load_pages()
    out = []
    for doc in join_all(pages):
        records, log, seg = run_document(doc)
        out.append((doc, records, log, seg))
    return out
