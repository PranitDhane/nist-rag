"""Stage 9: clean, safe external links with readable anchor text, tagged by where they appear.

Why from PDF annotations (not Markdown): the Markdown shows link text but not the URL, and
long URLs are often broken or garbled by line wraps. The annotation holds the exact URI, and
the words under its rectangle give the anchor text. Only http, https and mailto links are kept:
document metadata is untrusted input and a 'javascript:' or 'file:' link must never reach a UI.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import pymupdf

from ingest import config
from ingest.clean import _is_footnote_block, join_hyphenated, load_protect_list
from ingest.models import Manifest
from ingest.structure import path_at

log = logging.getLogger(__name__)

ALLOWED_SCHEMES = {"http", "https", "mailto"}
MIN_WORD_OVERLAP = 0.6       # a word belongs to the link if >=60% of its area is inside the rect
PARTIAL_WORD_OVERLAP = 0.05  # a word touched by the rect but under 60% inside means a cut-off anchor
MAX_LINE_GAP_FACTOR = 1.5    # split-link pieces sit on consecutive lines (gap <= 1.5 line heights)
TRAILING_PUNCT = ".,;:)]}"
BARE_URL_RE = re.compile(r"https?://[^\s|<>*]+")
URL_CONTINUATION_RE = re.compile(r"\s+(?!https?://)([A-Za-z0-9][A-Za-z0-9._~/%?=&#:+@!$'()-]*)")


def scheme_of(uri: str) -> str:
    m = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", uri.strip())
    return m.group(1).lower() if m else ""


def external_links(raw: list[dict]) -> tuple[list[dict], list[str], int]:
    """(kept external links, rejected URIs, internal count).

    Internal links (goto/named, no URI) are navigation only and dropped. A URI with any other
    scheme is rejected and returned so it can be logged.
    """
    kept, rejected, internal = [], [], 0
    for link in raw:
        uri = (link.get("uri") or "").strip()
        if not uri:
            internal += 1
        elif scheme_of(uri) in ALLOWED_SCHEMES:
            kept.append({"uri": uri, "rect": link["rect"]})
        else:
            rejected.append(uri)
    return kept, rejected, internal


def _join_pieces(a: str, b: str) -> str:
    """Join two anchor pieces that sit on consecutive lines.

    A trailing hyphen is a line-wrap hyphen and is handled like Stage 7 (soft hyphen dropped
    unless the word is protected); a trailing '/' means a URL broke there, so no space.
    """
    pre, post = re.search(r"([A-Za-z]+)-$", a), re.match(r"[A-Za-z0-9][A-Za-z]*", b)
    if pre and post:
        joined = join_hyphenated(pre.group(1), post.group(0), load_protect_list())
        return a[:pre.start()] + joined + b[post.end():]
    return a + b if a.endswith("/") else f"{a} {b}".strip()


def anchor_text(words: list[tuple], rect: list[float]) -> tuple[str, bool]:
    """(text, cut_off) of the words >=60% inside `rect`, in reading order.

    `words` are PyMuPDF word tuples (x0, y0, x1, y1, text, block, line, word_no). Text is
    NFKC-normalized like the Markdown ('proﬁle' -> 'profile') and words broken over a line
    are joined like Stage 7. cut_off is True when a word on the same line is only partly under
    the rect, i.e. the anchor starts or ends mid-word.
    """
    r = pymupdf.Rect(rect)
    inside, partial = [], False
    for w in words:
        wr = pymupdf.Rect(w[:4])
        if wr.width <= 0 or wr.height <= 0 or not wr.intersects(r):
            continue
        h = max(0.0, min(wr.x1, r.x1) - max(wr.x0, r.x0)) / wr.width
        v = max(0.0, min(wr.y1, r.y1) - max(wr.y0, r.y0)) / wr.height
        if h * v >= MIN_WORD_OVERLAP:
            inside.append(w)
        elif v >= 0.5 and PARTIAL_WORD_OVERLAP < h < MIN_WORD_OVERLAP:
            partial = True
    inside.sort(key=lambda w: (w[5], w[6], w[7]))
    text = ""
    for i, w in enumerate(inside):
        word = unicodedata.normalize("NFKC", w[4])
        if i and (inside[i - 1][5], inside[i - 1][6]) != (w[5], w[6]):
            text = _join_pieces(text, word)       # new line: may be a wrapped word
        else:
            text = f"{text} {word}".strip()
    return text, partial


def merge_split_links(items: list[dict]) -> list[dict]:
    """Merge consecutive annotations with the same URI that sit on consecutive lines.

    A link wrapped over two lines is stored as two annotations. Requiring vertical adjacency
    keeps two separate links to the same URL on one line apart (they are deduplicated later).
    """
    out: list[dict] = []
    for item in items:
        prev = out[-1] if out else None
        if prev and prev["uri"] == item["uri"]:
            pr, cr = pymupdf.Rect(prev["rect"]), pymupdf.Rect(item["rect"])
            if cr.y0 >= pr.y1 - 2 and cr.y0 - pr.y1 <= MAX_LINE_GAP_FACTOR * pr.height:
                out[-1] = {**prev, "rect": list(pr | cr), "text": _join_pieces(prev["text"], item["text"]),
                           "cut_off": prev["cut_off"] or item["cut_off"]}
                continue
        out.append(item)
    return out


def strip_trailing_punct(url: str) -> str:
    return url.rstrip(TRAILING_PUNCT)


def bare_urls(markdown: str) -> list[tuple[str, int]]:
    """(url, offset) for URLs written out in the text.

    A URL broken after a hyphen is rejoined with its next fragment (the Markdown shows the break
    as a space or paragraph gap). Trailing '.,;:)' is stripped: it belongs to the sentence.
    """
    out = []
    for m in BARE_URL_RE.finditer(markdown):
        url, end = m.group(0), m.end()
        while url.endswith("-") and (c := URL_CONTINUATION_RE.match(markdown, end)):
            url, end = url + c.group(1), c.end()
        out.append((strip_trailing_punct(url), m.start()))
    return out


def same_url(bare: str, annotation_uri: str) -> bool:
    """Bare text matches an annotation if equal or one is a prefix of the other (after trailing punctuation).

    The prefix case covers a bare URL cut short by a line break.
    """
    a, b = strip_trailing_punct(bare), strip_trailing_punct(annotation_uri)
    return a == b or (min(len(a), len(b)) >= 15 and (a.startswith(b) or b.startswith(a)))


def _footnote_for(rect: list[float], page: dict) -> str | None:
    """Number of the footnote whose block contains the rect, else None."""
    r = pymupdf.Rect(rect)
    for b in page["bottom_blocks"]:
        if _is_footnote_block(b, page.get("body_font_size")) and pymupdf.Rect(b["bbox"]).intersects(r):
            return b["first_span_text"].strip()
    return None


def _offset_of(markdown: str, text: str) -> int | None:
    for probe in (text, text[:20]):
        if probe and (i := markdown.find(probe)) >= 0:
            return i
    return None


def build_page_links(page: dict, words: list[tuple]) -> dict:
    """Replace a page's raw links with clean ones; add rejected/cut-off bookkeeping."""
    kept, rejected, internal = external_links(page["links"])
    for uri in rejected:
        log.warning("links: %s p.%s: rejected link with scheme %r (%s)", page["doc_id"], page["pdf_page"],
                    scheme_of(uri) or "?", uri[:80])
    items = []
    for k in kept:
        text, cut = anchor_text(words, k["rect"])
        items.append({**k, "text": text, "cut_off": cut})
    merged = merge_split_links(items)

    md, links, seen, cut_off = page["markdown"], [], set(), []
    def add(text: str, url: str, rect: list[float] | None, offset: int | None) -> None:
        if (url, text) in seen:
            return
        seen.add((url, text))
        fn = _footnote_for(rect, page) if rect else None
        if page["page_type"] == "references":
            location = "references"
        elif fn is not None:
            location = "footnote"
        else:
            location = "body"
        if location == "footnote" or offset is None:
            path = path_at(page, len(md)) if location == "footnote" else page["section_path_start"]
        else:
            path = path_at(page, offset)
        link = {"text": text, "url": url, "location": location, "section_path": path, "pdf_page": page["pdf_page"]}
        if fn is not None:
            link["footnote_number"] = fn
        links.append(link)

    for m in merged:
        add(m["text"], m["uri"], m["rect"], _offset_of(md, m["text"]))
        if m["cut_off"]:
            cut_off.append({"text": m["text"], "url": m["uri"]})
    annotation_uris = [m["uri"] for m in merged]
    for url, offset in bare_urls(md):
        if scheme_of(url) in ALLOWED_SCHEMES and not any(same_url(url, a) for a in annotation_uris):
            add(url, url, None, offset)
    return {**page, "links": links, "rejected_links": rejected, "internal_links_dropped": internal,
            "cut_off_links": cut_off}


def links_all(manifest: Manifest, structured: list[dict], raw_dir: Path | None = None) -> list[dict]:
    raw_dir = raw_dir or config.RAW_DIR
    result: list[dict] = []
    for spec in manifest.documents:
        path = raw_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"links: {spec.doc_id}: {path} is missing. Run Stage 2 (python -m ingest) first.")
        pages = sorted((r for r in structured if r["doc_id"] == spec.doc_id), key=lambda r: r["pdf_page"])
        with pymupdf.open(path) as doc:
            done = [build_page_links(p, doc[p["pdf_page"] - 1].get_text("words")) for p in pages]
        log.info("links: %s: %d links kept, %d internal dropped, %d rejected", spec.doc_id,
                 sum(len(p["links"]) for p in done), sum(p["internal_links_dropped"] for p in done),
                 sum(len(p["rejected_links"]) for p in done))
        result.extend(done)
    config.STAGES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.STAGES_DIR / "09_links.jsonl", "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return result


def link_counts(records: list[dict]) -> dict[str, dict[str, int]]:
    """Links per document per location."""
    out: dict[str, dict[str, int]] = {}
    for r in records:
        for l in r["links"]:
            doc = out.setdefault(r["doc_id"], {})
            doc[l["location"]] = doc.get(l["location"], 0) + 1
    return out


def load_links() -> list[dict]:
    path = config.STAGES_DIR / "09_links.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"links: {path} is missing. Run Stage 9 first (python -m ingest --only-stage 9).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
