import pytest

pytest.importorskip("transformers")
from chunking import config  # noqa: E402
from chunking.join import join_document  # noqa: E402
from chunking.prose import ProseChunk, build_prose_chunks, merge_tiny, split_segment  # noqa: E402
from chunking.segment import segment  # noqa: E402
from chunking.tokens import count_tokens  # noqa: E402


def sentences(n, tag="s"):
    return " ".join(f"This is sentence number {i} of the {tag} test passage about managing risk." for i in range(n))


def one_doc(md, heads):
    rec = {"doc_id": "d", "display_name": "D", "source_url": "u", "pdf_page": 1, "printed_page": "1",
           "index_eligible": True, "markdown": md, "table_markers": [], "section_path_start": [],
           "headings": [{"section_id": sid, "title": title, "level": 1, "char_offset": md.index(f"# {sid} "),
                         "section_path": [f"{sid} {title}"]} for sid, title in heads]}
    return join_document([rec])


def test_short_segment_is_one_chunk():
    t = sentences(5)
    assert split_segment(t, 100) == [(100, 100 + len(t), False)]


def test_long_segment_pieces_respect_bounds_and_overlap():
    t = sentences(120)
    pieces = split_segment(t, 0)
    assert len(pieces) > 2
    for a, b, _ in pieces:
        assert count_tokens(t[a:b]) <= config.PROSE_TARGET_MAX
    for a, b, _ in pieces[:-1]:
        assert count_tokens(t[a:b]) >= config.PROSE_TARGET_MIN
    assert pieces[0][2] is False and all(p[2] for p in pieces[1:])
    for (a1, b1, _), (a2, b2, _) in zip(pieces, pieces[1:]):
        assert a2 < b1 and a2 > a1                # overlaps, and moves forward
        assert 0.03 < count_tokens(t[a2:b1]) / count_tokens(t[a1:b1]) < 0.2   # roughly 10%


def test_pieces_cover_every_character_of_the_segment():
    t = sentences(90)
    pieces = split_segment(t, 0)
    covered = set()
    for a, b, _ in pieces:
        covered.update(range(a, b))
    assert all(i in covered or t[i].isspace() for i in range(len(t)))


def test_overlap_never_crosses_a_section():
    md = f"# 1 A\n\n{sentences(80, 'a')}\n\n# 2 B\n\n{sentences(80, 'b')}"
    d = one_doc(md, [("1", "A"), ("2", "B")])
    chunks, _ = build_prose_chunks(segment(d))
    for c in chunks:
        body = c.text
        assert not ("the a test" in body and "the b test" in body)
    b_start = d.headings[1].start
    assert all(c.end <= b_start or c.start >= b_start for c in chunks)


def test_piece_never_ends_on_a_bare_heading():
    md = f"# 1 A\n\n{sentences(40)}\n\n## **Sub heading**\n\n{sentences(40)}"
    d = one_doc(md, [("1", "A")])
    chunks, _ = build_prose_chunks(segment(d))
    assert all(not c.text.rstrip().splitlines()[-1].startswith("#") for c in chunks)


def mk(text, sec, seq):
    return ProseChunk("d", seq, [sec], [(seq * 1000, seq * 1000 + len(text))], text, count_tokens(text), seq)


def test_tiny_merges_forward_and_keeps_next_section_path():
    a, b = mk("# Part 1", "Part 1", 0), mk(sentences(10), "1 Framing", 1)
    out, log = merge_tiny([a, b], [True, False], "")
    assert len(out) == 1 and out[0].section_path == ["1 Framing"] and out[0].merged_sections == ["Part 1"]
    assert out[0].text.startswith("# Part 1\n\n") and len(out[0].spans) == 2
    assert log[0]["direction"] == "forward"


def test_tiny_without_next_chunk_merges_backward():
    a, b = mk(sentences(10), "1 Framing", 0), mk("# A.1 Gov", "A.1", 1)
    out, log = merge_tiny([a, b], [False, True], "")
    assert len(out) == 1 and out[0].section_path == ["1 Framing"] and out[0].text.endswith("# A.1 Gov")
    assert log[0]["direction"] == "backward"


def test_merge_that_would_exceed_the_limit_is_not_done_and_is_logged():
    big = sentences(40)
    assert count_tokens(big) > config.MERGED_MAX_TOKENS - 5
    a, b = mk("# Part 1", "Part 1", 0), mk(big, "Big", 1)
    out, log = merge_tiny([a, b], [True, False], "")
    assert len(out) == 2 and log[0]["direction"] == "not_merged_too_large"


def test_consecutive_tiny_sections_merge_into_the_same_next_chunk():
    a, b, c = mk("# P", "P", 0), mk("# Q", "Q", 1), mk(sentences(5), "N", 2)
    out, _ = merge_tiny([a, b, c], [True, True, False], "")
    assert len(out) == 1 and out[0].merged_sections == ["P", "Q"]
