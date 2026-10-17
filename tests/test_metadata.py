import json

import pytest

from indexing import config
from indexing.metadata import DERIVED_FIELDS, LIST_FIELDS, SCALAR_FIELDS, from_chroma_metadata, to_chroma_metadata


def full_chunk(**over):
    c = {"chunk_id": "d:prose:0001:1", "doc_id": "d", "display_name": "D (x)", "chunk_type": "prose",
         "section_path": ["Part 1", "1 A", "1.1 B"], "section_path_is_page_level": False, "merged_sections": [],
         "sequence": 1, "part": 1, "parts_total": 1, "char_start": 10, "char_end": 50, "char_spans": [[10, 50]],
         "pdf_page_start": 5, "pdf_page_end": 6, "printed_page_start": "1", "printed_page_end": "2",
         "source_url": "https://x/y.pdf", "citation_url": "https://x/y.pdf#page=5", "citation_label": "X, p. 1",
         "display_text": "text “quoted” — dash", "embed_text": "crumb\n\ntext “quoted” — dash", "token_count_embed": 12,
         "function": None, "category": None, "subcategory": None, "subcategories": [], "action_ids": [],
         "gai_risks": [], "ai_actor_tasks": [], "trustworthy_characteristics": [], "figure_number": None,
         "ocr_confidence": None, "footnote_number": None, "links": [{"text": "NIST", "url": "https://nist.gov"}]}
    c.update(over)
    return c


TYPES = {
    "prose": {},
    "genai_action_table": dict(chunk_type="genai_action_table", function="GOVERN", subcategory="GOVERN 1.7",
                               action_ids=["GV-1.7-001", "GV-1.7-002"], gai_risks=["Confabulation", "Data Privacy"],
                               ai_actor_tasks=["AI Deployment"], part=2, parts_total=2),
    "rmf_core_table": dict(chunk_type="rmf_core_table", function="GOVERN", category="GOVERN 1",
                           subcategories=["GOVERN 1.1", "GOVERN 1.2"]),
    "figure": dict(chunk_type="figure", figure_number=3, ocr_confidence=91.9, char_start=None, char_end=None,
                   char_spans=None, links=[]),
    "footnote": dict(chunk_type="footnote", footnote_number=4, section_path_is_page_level=True, char_start=None,
                     char_end=None, char_spans=None, printed_page_start=None, printed_page_end=None),
}


@pytest.mark.parametrize("kind", TYPES)
def test_round_trip_every_chunk_type(kind):
    chunk = full_chunk(**TYPES[kind])
    meta = to_chroma_metadata(chunk)
    back = from_chroma_metadata(meta, chunk["embed_text"])
    assert back == {k: chunk[k] for k in chunk}, [k for k in chunk if back.get(k) != chunk[k]]


@pytest.mark.parametrize("kind", TYPES)
def test_no_none_and_only_chroma_types(kind):
    meta = to_chroma_metadata(full_chunk(**TYPES[kind]))
    assert all(v is not None for v in meta.values())
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
    assert not any(isinstance(v, list) for v in meta.values())


def test_null_scalars_are_left_out_and_restored_as_null():
    meta = to_chroma_metadata(full_chunk())
    for k in ("function", "category", "subcategory", "figure_number", "ocr_confidence", "footnote_number"):
        assert k not in meta
        assert from_chroma_metadata(meta)[k] is None


def test_section_path_helpers():
    meta = to_chroma_metadata(full_chunk())
    assert meta["section_path_str"] == "Part 1 > 1 A > 1.1 B"
    assert meta["section_top"] == "Part 1" and meta["section_leaf"] == "1.1 B"
    assert json.loads(meta["section_path_json"]) == ["Part 1", "1 A", "1.1 B"]
    empty = to_chroma_metadata(full_chunk(section_path=[]))
    assert empty["section_path_str"] == "" and "section_top" not in empty


def test_lists_are_json_strings_and_keys_match_the_plan():
    meta = to_chroma_metadata(full_chunk(**TYPES["genai_action_table"]))
    assert meta["action_ids_json"] == '["GV-1.7-001","GV-1.7-002"]'
    for key in ("section_path_json", "merged_sections_json", "subcategories_json", "action_ids_json", "gai_risks_json",
                "ai_actor_tasks_json", "trustworthy_characteristics_json", "links_json"):
        assert key in meta
    assert {"section_path_str", "section_top", "section_leaf"} == set(DERIVED_FIELDS)


def test_unsupported_value_is_rejected():
    with pytest.raises(TypeError):
        to_chroma_metadata(full_chunk(sequence=[1, 2]))


def test_real_chunks_round_trip_exactly():
    if not config.CHUNKS_PATH.is_file():
        pytest.skip("chunks.jsonl absent")
    chunks = [json.loads(l) for l in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    seen = set()
    for c in chunks:
        meta = to_chroma_metadata(c)
        assert from_chroma_metadata(meta, c["embed_text"]) == c, c["chunk_id"]
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
        seen.add(c["chunk_type"])
    assert seen == {"prose", "genai_action_table", "rmf_core_table", "figure", "footnote"}
    assert set(SCALAR_FIELDS) | set(LIST_FIELDS) | {"embed_text"} <= set(chunks[0])
