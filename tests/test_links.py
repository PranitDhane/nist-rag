"""Stage 9 tests: scheme filtering, anchor text, split-link merge, bare URLs, locations (inline
fixtures, no PDFs), plus counts on the real output (skipped without PDFs)."""
from __future__ import annotations

import pytest

from ingest.links import (anchor_text, bare_urls, build_page_links, external_links, merge_split_links,
                          same_url, strip_trailing_punct)


def word(x0, y0, x1, y1, text, block=0, line=0, n=0):
    return (x0, y0, x1, y1, text, block, line, n)


# ---- scheme filtering
def test_only_http_https_mailto_kept_and_others_rejected():
    raw = [{"uri": u, "rect": [0, 0, 1, 1]} for u in
           ["http://a.org", "https://b.org", "mailto:x@y.org", "javascript:alert(1)", "file:///etc/passwd",
            "JaVaScRiPt:alert(1)", "data:text/html,x", "ftp://c.org"]]
    raw += [{"uri": None, "rect": [0, 0, 1, 1]}, {"uri": "", "rect": [0, 0, 1, 1]}]
    kept, rejected, internal = external_links(raw)
    assert [k["uri"] for k in kept] == ["http://a.org", "https://b.org", "mailto:x@y.org"]
    assert len(rejected) == 5 and internal == 2


# ---- anchor text
def test_anchor_text_overlap_order_ligature():
    words = [word(10, 10, 50, 20, "proﬁle", 0, 0, 0), word(52, 10, 80, 20, "link", 0, 0, 1),
             word(10, 40, 50, 50, "outside", 0, 1, 0)]
    assert anchor_text(words, [5, 8, 85, 22]) == ("profile link", False)


def test_anchor_text_requires_60_percent_and_flags_cut_off_word():
    words = [word(10, 10, 50, 20, "whole", 0, 0, 0), word(52, 10, 92, 20, "cutword", 0, 0, 1)]
    text, cut = anchor_text(words, [5, 8, 72, 22])      # covers 50% of the second word
    assert text == "whole" and cut is True


def test_anchor_text_words_on_other_lines_do_not_count_as_cut_off():
    words = [word(10, 10, 50, 20, "link", 0, 0, 0), word(10, 21.5, 50, 31.5, "below", 0, 1, 0)]
    assert anchor_text(words, [5, 8, 55, 22]) == ("link", False)


def test_anchor_text_joins_hyphenated_line_wrap():
    words = [word(10, 10, 50, 20, "recom-", 0, 0, 0), word(10, 22, 50, 32, "mendations,", 0, 1, 0)]
    assert anchor_text(words, [5, 8, 60, 34])[0] == "recommendations,"


def test_anchor_text_keeps_real_hyphen_before_uppercase_and_protected_words():
    assert anchor_text([word(0, 0, 9, 9, "human-", 0, 0, 0), word(0, 10, 9, 19, "AI", 0, 1, 0)], [0, 0, 20, 20])[0] == "human-AI"
    assert anchor_text([word(0, 0, 9, 9, "third-", 0, 0, 0), word(0, 10, 9, 19, "party", 0, 1, 0)], [0, 0, 20, 20])[0] == "third-party"


def test_anchor_text_empty_when_no_words():
    assert anchor_text([], [0, 0, 10, 10]) == ("", False)


# ---- merge of a link split over two lines
def piece(uri, rect, text):
    return {"uri": uri, "rect": rect, "text": text, "cut_off": False}


def test_two_line_link_merged_with_joined_text():
    merged = merge_split_links([piece("https://x.org/a", [300, 100, 540, 112], "see https://x.org/"),
                                piece("https://x.org/a", [72, 113, 150, 125], "a")])
    assert len(merged) == 1 and merged[0]["text"] == "see https://x.org/a"
    assert merged[0]["rect"][:2] == [72, 100]


def test_no_merge_for_different_uri_same_line_or_far_apart():
    a = piece("https://x.org", [72, 100, 150, 112], "one")
    assert len(merge_split_links([a, piece("https://y.org", [72, 113, 150, 125], "two")])) == 2
    assert len(merge_split_links([a, piece("https://x.org", [200, 100, 260, 112], "again")])) == 2
    assert len(merge_split_links([a, piece("https://x.org", [72, 300, 150, 312], "far")])) == 2


# ---- bare URLs
@pytest.mark.parametrize("src,expected", [
    ("see https://x.org/a.", ["https://x.org/a"]),
    ("(https://x.org/a), and https://y.org/b;", ["https://x.org/a", "https://y.org/b"]),
    ("**https://doi.org/10.6028/NIST.AI.100-1**", ["https://doi.org/10.6028/NIST.AI.100-1"]),
    ("https://x.org/long-\n\nname-here and text", ["https://x.org/long-name-here"]),
    ("no urls here, just ftp://x.org", []),
])
def test_bare_urls(src, expected):
    assert [u for u, _ in bare_urls(src)] == expected


def test_strip_trailing_punct():
    assert strip_trailing_punct("https://x.org/a).;,:") == "https://x.org/a"


def test_annotation_wins_over_bare_url():
    assert same_url("https://x.org/a.", "https://x.org/a")
    assert same_url("https://x.org/long-pa", "https://x.org/long-path/file.pdf")   # cut short by a line break
    assert not same_url("https://x.org/a", "https://y.org/a")
    assert not same_url("https://x.org/a", "https://x.org/b")


# ---- whole page
def make_page(raw, markdown="", page_type="body", bottom=None):
    return {"doc_id": "d", "pdf_page": 3, "page_type": page_type, "markdown": markdown, "links": raw,
            "section_path_start": ["2 Overview"], "heading_paths": [], "bottom_blocks": bottom or [],
            "body_font_size": 11.0}


def test_build_page_links_dedup_annotation_wins_and_rejects(caplog):
    words = [word(10, 10, 50, 20, "NIST", 0, 0, 0)]
    page = make_page([{"kind": 2, "uri": "https://x.org/a", "rect": [5, 8, 55, 22]},
                      {"kind": 2, "uri": "https://x.org/a", "rect": [5, 8, 55, 22]},
                      {"kind": 2, "uri": "javascript:alert(1)", "rect": [5, 8, 55, 22]},
                      {"kind": 1, "uri": None, "rect": [5, 8, 55, 22]}],
                     markdown="NIST is at https://x.org/a. Also https://z.org/only.")
    out = build_page_links(page, words)
    assert [(l["text"], l["url"], l["location"]) for l in out["links"]] == [
        ("NIST", "https://x.org/a", "body"), ("https://z.org/only", "https://z.org/only", "body")]
    assert out["rejected_links"] == ["javascript:alert(1)"] and out["internal_links_dropped"] == 1
    assert any("javascript" in r.message for r in caplog.records)
    assert out["links"][0]["section_path"] == ["2 Overview"]


def test_location_references_and_footnote():
    raw = [{"kind": 2, "uri": "https://x.org/a", "rect": [72, 600, 200, 612]}]
    words = [word(72, 600, 200, 612, "src", 0, 0, 0)]
    assert build_page_links(make_page(raw, page_type="references"), words)["links"][0]["location"] == "references"
    fn = [{"bbox": [72, 595, 540, 620], "first_span_text": "9", "first_span_size": 6.6, "text": "9 src"}]
    link = build_page_links(make_page(raw, bottom=fn), words)["links"][0]
    assert link["location"] == "footnote" and link["footnote_number"] == "9"
    body = [{"bbox": [72, 700, 540, 720], "first_span_text": "9", "first_span_size": 6.6, "text": "9 other"}]
    assert build_page_links(make_page(raw, bottom=body), words)["links"][0]["location"] == "body"


def test_empty_anchor_text_is_kept_but_empty():
    out = build_page_links(make_page([{"kind": 2, "uri": "https://x.org", "rect": [0, 0, 5, 5]}]), [])
    assert out["links"][0]["text"] == ""


# ---- real data
@pytest.fixture(scope="module")
def linked(need_pdfs):
    from ingest.config import load_manifest
    from ingest.links import links_all
    from ingest.structure import load_structure
    return links_all(load_manifest(), load_structure())


def test_real_every_url_is_safe_and_counts_reported(linked):
    urls = [l["url"] for r in linked for l in r["links"]]
    assert urls and all(u.startswith(("http://", "https://", "mailto:")) for u in urls)
    from ingest.structure import load_structure
    ai = {l["url"] for r in linked if r["doc_id"] == "ai_rmf_1_0" for l in r["links"]}
    ai_annotated = {l["uri"] for r in load_structure() if r["doc_id"] == "ai_rmf_1_0" for l in r["links"] if l["uri"]}
    assert len(ai_annotated) == 12             # plan: 21 external annotations, 12 unique
    assert ai_annotated <= ai and ai - ai_annotated == {"https://doi.org/10.6028/NIST.AI.100-1"}   # DOI is bare text only
    gen = {l["url"] for r in linked if r["doc_id"] == "genai_profile" for l in r["links"]}
    assert len(gen) >= 140                     # plan: 142 unique external


def test_real_locations(linked):
    for r in linked:
        for l in r["links"]:
            if r["page_type"] == "references":
                assert l["location"] == "references"
            assert l["location"] in {"body", "footnote", "references"}
    assert any(l["location"] == "footnote" for r in linked for l in r["links"])


def test_real_no_ligatures_or_internal_links_leak(linked):
    assert not any(c in l["text"] for r in linked for l in r["links"] for c in "ﬁﬂﬀﬃ")
    assert all(not l["url"].startswith("#") for r in linked for l in r["links"])
