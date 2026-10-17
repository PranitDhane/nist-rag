import pytest

pytest.importorskip("transformers")
from chunking import config  # noqa: E402
from chunking.join import join_document  # noqa: E402
from chunking.segment import segment  # noqa: E402
from chunking.tables import (TableTooBig, build_genai_chunks, build_rmf_chunks, fit_groups,  # noqa: E402
                             normalize_action_id, parse_genai_table, partition_min_max, render_genai_piece)
from chunking.tokens import count_tokens  # noqa: E402

HDR = "|**Action ID**|**Suggested Action**|**GAI Risks**|\n|---|---|---|\n"


def genai_doc(rows, actors="|**AI Actor Tasks**|**: AI Deployment, TEVV**||", sub="**GOVERN 1.7:** Decommission safely."):
    md = "# 3 Actions\n\n" + sub + "\n\n" + HDR + "\n".join(rows) + "\n" + actors
    rec = {"doc_id": "d", "display_name": "D", "source_url": "u", "pdf_page": 1, "printed_page": "1", "index_eligible": True,
           "markdown": md, "section_path_start": [],
           "headings": [{"section_id": "3", "title": "Actions", "level": 1, "char_offset": 0, "section_path": ["3 Actions"]}],
           "table_markers": [{"id": "GOVERN 1.7", "kind": "genai_subcategory", "char_offset": md.index("**GOVERN")}]}
    return join_document([rec])


def row(i, words=5):
    return f"|GV-1.7-{i:03d}|{'word ' * words}<br>end|Confabulation; Value<br>Chain|"


def test_small_table_is_one_chunk_with_everything():
    d = genai_doc([row(1), row(2)])
    chunks, _ = build_genai_chunks(segment(d))
    assert len(chunks) == 1 and chunks[0].part == chunks[0].parts_total == 1
    c = chunks[0]
    assert c.action_ids == ["GV-1.7-001", "GV-1.7-002"] and c.ai_actor_tasks == ["AI Deployment", "TEVV"]
    assert c.gai_risks == ["Confabulation", "Value Chain"] and c.subcategory == "GOVERN 1.7" and c.function == "GOVERN"
    assert "AI Actor Tasks: AI Deployment, TEVV" in c.text


def test_big_table_splits_between_rows_and_repeats_header_and_actor_line(monkeypatch):
    monkeypatch.setattr(config, "TABLE_PIECE_MAX_TOKENS", 120)
    d = genai_doc([row(i, words=40) for i in range(1, 7)])
    seg = segment(d)
    chunks, _ = build_genai_chunks(seg)
    assert len(chunks) > 1
    unit = next(u for u in seg.units if u.kind == "genai_table")
    for i, c in enumerate(chunks, start=1):
        assert c.part == i and c.parts_total == len(chunks) and c.tokens <= 120
        assert c.text.startswith("GOVERN 1.7: Decommission safely.")
        assert c.text.endswith("AI Actor Tasks: AI Deployment, TEVV")
        for aid in c.action_ids:                        # no row split: each row text is complete in its piece
            assert aid in c.text and c.text.count(aid) == 1
    assert [a for c in chunks for a in c.action_ids] == [f"GV-1.7-{i:03d}" for i in range(1, 7)]
    assert chunks[0].start == unit.start and chunks[-1].end == unit.end
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))       # spans tile the unit


def test_row_too_big_stops_instead_of_splitting_a_row(monkeypatch):
    monkeypatch.setattr(config, "TABLE_PIECE_MAX_TOKENS", 60)
    d = genai_doc([row(1, words=200)])
    with pytest.raises(TableTooBig):
        build_genai_chunks(segment(d))


def test_typo_action_id_is_repaired_and_wrapped_row_joined():
    t = parse_genai_table("**GOVERN 4.3:** X.\n\n" + HDR + "|GV4.3--001|Do a thing|Confabulation|\n||and continue it||\n"
                          "|**AI Actors: AI D**|**esign**||", "GOVERN 4.3")
    assert [r.action_id for r in t.rows] == ["GV-4.3-001"]
    assert t.rows[0].action == "Do a thing and continue it"
    assert t.actors == ["AI Design"] and any("repaired" in r for r in t.repairs)
    assert normalize_action_id("GV-1.1-002") == ("GV-1.1-002", False)


def test_icon_font_garble_is_removed_from_actor_line():
    t = parse_genai_table("**MAP 1.1:** X.\n\n" + HDR + "|MP-1.1-001|a|b|\n|**AI Actor Task    f   i**|**s: AI Deployment, TEVV**||", "MAP 1.1")
    assert t.actors == ["AI Deployment", "TEVV"]


def test_ai_actor_paragraph_variant():
    t = parse_genai_table("**MAP 1.1:** X.\n\n" + HDR + "|MP-1.1-001|a|b|\n\n**AI Actor Tasks: Governance and Oversight**", "MAP 1.1")
    assert t.actors == ["Governance and Oversight"]


def test_partition_balances_and_fit_groups_counts_exactly():
    assert partition_min_max([5, 5, 5, 5], 2) == [2, 2]
    assert sum(partition_min_max([1, 9, 1, 1, 9, 1], 3)) == 6
    d = genai_doc([row(1), row(2), row(3)])
    t = parse_genai_table(d.text[d.markers[0].start:], "GOVERN 1.7")
    spans = fit_groups(3, lambda i, j: render_genai_piece(t, t.rows[i:j], "lines"), 10_000)
    assert spans == [(0, 3)]


RMF_MD = ("# 5 Core\n\nTable 1: Cats.\n\n|**Categories**|**Subcategories**|\n|---|---|\n"
          "|**GOVERN 1:**|**GOVERN 1.1:** Legal and regulatory|\n|Policies and|requirements are understood.|\n"
          "|procedures exist.|**GOVERN 1.2:** Trust is integrated.|\n"
          "|**GOVERN 2:**<br>Accountability<br>is in place.|**GOVERN 2.1:** Roles are documented.|\n")


def rmf_doc():
    rec = {"doc_id": "d", "display_name": "D", "source_url": "u", "pdf_page": 1, "printed_page": "1", "index_eligible": True,
           "markdown": RMF_MD, "section_path_start": [],
           "headings": [{"section_id": "5", "title": "Core", "level": 1, "char_offset": 0, "section_path": ["5 Core"]}],
           "table_markers": [{"id": "Table 1", "kind": "rmf_table", "char_offset": RMF_MD.index("Table 1")}]}
    return join_document([rec])


def test_rmf_category_text_is_rebuilt_from_fragments():
    chunks, _ = build_rmf_chunks(segment(rmf_doc()))
    assert [c.category for c in chunks] == ["GOVERN 1", "GOVERN 2"]
    g1 = chunks[0]
    assert g1.text.splitlines()[0] == "GOVERN 1: Policies and procedures exist."
    assert g1.text.splitlines()[1] == "GOVERN 1.1: Legal and regulatory requirements are understood."
    assert g1.subcategories == ["GOVERN 1.1", "GOVERN 1.2"] and g1.function == "GOVERN" and g1.caption == "Table 1"
    assert chunks[1].text.startswith("GOVERN 2: Accountability is in place.")


def test_rmf_category_splits_between_subcategories_and_repeats_category_text(monkeypatch):
    monkeypatch.setattr(config, "TABLE_PIECE_MAX_TOKENS", 30)
    d = rmf_doc()
    seg = segment(d)
    chunks, _ = build_rmf_chunks(seg)
    g1 = [c for c in chunks if c.category == "GOVERN 1"]
    assert len(g1) == 2 and [c.subcategories for c in g1] == [["GOVERN 1.1"], ["GOVERN 1.2"]]
    assert all(c.text.startswith("GOVERN 1: Policies and procedures exist.") for c in g1)
    assert g1[0].end == g1[1].start


def test_real_tables_if_available():
    if not config.PAGES_PATH.is_file():
        pytest.skip("pages.jsonl absent")
    from chunking.join import join_all, load_pages
    for d in join_all(load_pages()):
        seg = segment(d)
        if d.doc_id == "genai_profile":
            chunks, _ = build_genai_chunks(seg)
            ids = [a for c in chunks for a in c.action_ids]
            assert len(ids) == len(set(ids)) == 212
            assert {c.subcategory for c in chunks} == {m.id for m in d.markers}
            assert all(c.tokens <= config.TABLE_PIECE_MAX_TOKENS for c in chunks)
        else:
            chunks, _ = build_rmf_chunks(seg)
            assert len(chunks) >= 19 and all(c.tokens <= config.TABLE_PIECE_MAX_TOKENS for c in chunks)
