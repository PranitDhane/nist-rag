import json

import pytest

from attribution import format as fmt
from augmentation import config as aug_config
from indexing import config as icfg
from insufficient.related import MAX_RELATED, RelatedEntry, build_related_sections, entry_for
from retrieval.models import RetrievedChunk


def meta(cid, chunk_type="prose", doc_id="ai_rmf_1_0", display="NIST AI 100-1 (AI RMF 1.0)", section=("Part 2", "5 AI RMF Core"),
         printed=("22", "22"), pdf=(27, 27), **extra):
    m = {"chunk_id": cid, "doc_id": doc_id, "display_name": display, "chunk_type": chunk_type, "section_path": list(section),
         "printed_page_start": printed[0], "printed_page_end": printed[1], "pdf_page_start": pdf[0], "pdf_page_end": pdf[1],
         "source_url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf", "subcategory": None, "category": None,
         "figure_number": None, "footnote_number": None, "part": 1, "parts_total": 1}
    m.update(extra)
    return m


def chunk(cid, rank=None, sibling=False, text="SECRET ANSWER TEXT that must never be listed", **kw):
    return RetrievedChunk(chunk_id=cid, display_text=text, metadata=meta(cid, **kw), rerank_score=0.9 if rank else None,
                          rerank_rank=rank, added_as_sibling=sibling, sibling_of="x" if sibling else None)


def distinct(n, start=1):
    return [chunk(f"c{i}", rank=i, section=("Part 2", f"5.{i} Sec"), printed=(str(10 + i), str(10 + i)), pdf=(15 + i, 15 + i))
            for i in range(start, start + n)]


def test_order_is_preserved_best_first_in_both_modes():
    chunks = distinct(5)
    assert [e.section for e in build_related_sections(chunks, "gate")] == [f"Part 2 > 5.{i} Sec" for i in range(1, 6)]
    assert [e.section for e in build_related_sections(chunks, "prompt")] == [f"Part 2 > 5.{i} Sec" for i in range(1, 6)]


def test_gate_mode_uses_reranker_order_even_if_the_list_is_shuffled_prompt_mode_keeps_prompt_order():
    chunks = distinct(3)
    shuffled = [chunks[2], chunks[0], chunks[1]]
    assert [e.section[-5:] for e in build_related_sections(shuffled, "gate")] == ["1 Sec", "2 Sec", "3 Sec"]
    assert [e.section[-5:] for e in build_related_sections(shuffled, "prompt")] == ["3 Sec", "1 Sec", "2 Sec"]


def test_at_most_five_and_fewer_only_when_fewer_chunks_exist():
    assert len(build_related_sections(distinct(8), "gate")) == MAX_RELATED == 5
    assert len(build_related_sections(distinct(3), "gate")) == 3
    assert build_related_sections([], "gate") == []


def test_dedup_by_citation_label_keeps_the_higher_ranked_chunk_and_still_fills_five():
    dup = [chunk("a1", rank=1, section=("5 Core",)), chunk("a2", rank=2, section=("5 Core",))]      # same label
    rest = distinct(5, start=3)
    out = build_related_sections(dup + rest, "gate")
    assert len(out) == 5 and len({e.citation_label for e in out}) == 5
    assert out[0].section == "5 Core" and sum(e.section == "5 Core" for e in out) == 1


def test_different_pages_of_one_section_are_different_labels():
    a = chunk("a", rank=1, printed=("22", "22"))
    b = chunk("b", rank=2, printed=("23", "23"), pdf=(28, 28))
    assert len(build_related_sections([a, b], "gate")) == 2


def test_siblings_are_never_listed_in_either_mode():
    sib = chunk("s", sibling=True, chunk_type="genai_action_table", subcategory="GOVERN 1.7", section=("3 Actions",))
    chunks = distinct(2) + [sib]
    assert all("GOVERN 1.7" not in e.section for e in build_related_sections(chunks, "prompt"))
    assert all("GOVERN 1.7" not in e.section for e in build_related_sections(chunks, "gate"))


def test_every_field_is_present_and_there_is_no_score_field():
    e = build_related_sections([chunk("a", rank=1)], "gate")[0]
    assert set(e.to_dict()) == {"doc", "section", "pages", "citation_label", "citation_url", "kind"}
    assert all(isinstance(v, str) and v for v in e.to_dict().values())
    assert e.doc == "NIST AI 100-1 (AI RMF 1.0)" and e.pages == "p. 22" and e.kind == "prose"
    assert "score" not in RelatedEntry.__dataclass_fields__                              # Q3: no scores


@pytest.mark.parametrize("chunk_type,expected", [("prose", "prose"), ("genai_action_table", "table"), ("rmf_core_table", "table"),
                                                 ("figure", "figure-ocr"), ("footnote", "footnote")])
def test_kind_covers_all_five_chunk_types(chunk_type, expected):
    c = chunk("a", rank=1, chunk_type=chunk_type, subcategory="GOVERN 1.7", category="GOVERN 1", figure_number=5, footnote_number=1)
    assert entry_for(c).kind == expected


def test_table_section_includes_subcategory_or_category():
    g = chunk("g", rank=1, chunk_type="genai_action_table", subcategory="GOVERN 1.7", section=("3 Suggested Actions",))
    r = chunk("r", rank=2, chunk_type="rmf_core_table", category="GOVERN 4", section=("Part 2", "5.1 Govern"))
    assert entry_for(g).section == "3 Suggested Actions > GOVERN 1.7" and entry_for(r).section == "Part 2 > 5.1 Govern > GOVERN 4"


def test_no_chunk_text_appears_anywhere_in_the_output():
    out = build_related_sections(distinct(5), "gate")
    blob = json.dumps([e.to_dict() for e in out])
    assert "SECRET" not in blob and "ANSWER TEXT" not in blob


def test_citation_url_uses_the_pdf_page_not_the_printed_page():
    e = entry_for(chunk("a", rank=1, printed=("21", "21"), pdf=(26, 26)))
    assert e.pages == "p. 21" and e.citation_label.endswith("p. 21") and e.citation_url.endswith("#page=26")
    assert e.citation_url == "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf#page=26"


def test_page_range_uses_an_en_dash_and_start_page_for_the_url():
    e = entry_for(chunk("a", rank=1, printed=("22", "23"), pdf=(27, 28)))
    assert e.pages == "pp. 22–23" and e.citation_url.endswith("#page=27")


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError):
        build_related_sections([], "bogus")


# ---- the shared formatter against the REAL chunks ------------------------------------------------------
def real_chunks():
    return [json.loads(l) for l in icfg.CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.skipif(not icfg.CHUNKS_PATH.is_file(), reason="chunks.jsonl absent")
def test_formatter_reproduces_every_stored_label_and_url():
    chunks = real_chunks()
    assert len(chunks) > 100
    for c in chunks:
        assert fmt.citation_label(c) == c["citation_label"], c["chunk_id"]
        assert fmt.citation_url(c) == c["citation_url"], c["chunk_id"]


@pytest.mark.skipif(not icfg.CHUNKS_PATH.is_file(), reason="chunks.jsonl absent")
def test_formatter_label_examples_and_kinds_on_real_chunks():
    by = {}
    for c in real_chunks():
        by.setdefault(c["chunk_type"], c)
    assert fmt.citation_label(by["genai_action_table"]).startswith("NIST AI 600-1, §3 Suggested Actions to Manage GAI Risks, GOVERN ")
    assert fmt.citation_label(by["footnote"]).endswith("footnote 1, p. 1")
    assert ", Fig. " in fmt.citation_label(by["figure"])
    assert {fmt.kind(c) for c in real_chunks()} == {"prose", "table", "figure-ocr", "footnote"}


def test_formatter_unit_rules():
    assert fmt.section_label({"section_path": ["Executive Summary"]}) == "Executive Summary"
    assert fmt.section_label({"section_path": ["Part 2", "5.1 Govern"]}) == "§5.1 Govern"
    assert fmt.section_label({"section_path": []}) == ""
    assert fmt.pages_text({"printed_page_start": None, "printed_page_end": None, "pdf_page_start": 26, "pdf_page_end": 26}) == "PDF p. 26"
    with pytest.raises(KeyError, match="no citation name"):
        fmt.doc_name({"doc_id": "other"})
