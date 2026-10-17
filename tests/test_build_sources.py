"""Stage S2: build_sources() and its consistency checks (ATTRIBUTION_PLAN Section 4)."""
import json
import logging

import pytest

from attribution.build import AttributionError, build_sources, check_consistency
from attribution.models import SourceEntry

RMF = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
GAI = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"


def meta(cid, doc="ai_rmf_1_0", section=("Part 2", "5.1 Govern"), printed=("22", "22"), pdf=(27, 27), seq=0, chunk_type="prose", **extra):
    m = {"chunk_id": cid, "doc_id": doc, "chunk_type": chunk_type, "section_path": list(section), "sequence": seq,
         "printed_page_start": printed[0], "printed_page_end": printed[1], "pdf_page_start": pdf[0], "pdf_page_end": pdf[1],
         "source_url": RMF if doc == "ai_rmf_1_0" else GAI, "subcategory": None, "category": None,
         "figure_number": None, "footnote_number": None}
    m.update(extra)
    return m


def setup(*metas):
    """key_map C1..Cn -> chunk ids in the given order, plus the chunk lookup."""
    return {f"C{i}": m["chunk_id"] for i, m in enumerate(metas, 1)}, {m["chunk_id"]: m for m in metas}


def run(text, cited, metas, **kw):
    key_map, lookup = setup(*metas)
    return build_sources(text, cited, key_map, lookup, strict=True, **kw)


def test_single_citation():
    a = meta("a", printed=("22", "23"), pdf=(27, 28))
    [e] = run("Govern is about culture [C1].", ["C1"], [a])
    assert e == SourceEntry(keys=["C1"], label="NIST AI 100-1, §5.1 Govern, pp. 22–23", url=f"{RMF}#page=27", kind="prose")
    assert e.to_dict()["keys"] == ["C1"]


def test_documents_ordered_ai_rmf_before_genai_then_by_page():
    g_early = meta("g1", doc="genai_profile", section=("1 Intro",), printed=("2", "2"), pdf=(8, 8))
    r_late = meta("r2", section=("5.2 Map",), printed=("30", "30"), pdf=(35, 35))
    r_early = meta("r1", section=("5.1 Govern",), printed=("22", "22"), pdf=(27, 27))
    entries = run("A [C1]. B [C2]. C [C3].", ["C1", "C2", "C3"], [g_early, r_late, r_early])
    assert [e.keys for e in entries] == [["C3"], ["C2"], ["C1"]]
    assert [e.label.split(",")[0] for e in entries] == ["NIST AI 100-1", "NIST AI 100-1", "NIST AI 600-1"]


def test_sequence_breaks_ties_deterministically():
    a = meta("a", section=("5.1 Govern",), seq=9)
    b = meta("b", section=("5.1 Govern A",), seq=3)      # same pages, different label
    entries = run("x [C1]. y [C2].", ["C1", "C2"], [a, b])
    assert [e.keys for e in entries] == [["C2"], ["C1"]]


def test_two_keys_with_the_same_label_merge_with_sorted_keys():
    a, b, c = meta("a", seq=1), meta("b", seq=2), meta("c", section=("5.2 Map",), pdf=(30, 30), printed=("25", "25"))
    entries = run("x [C3]. y [C1]. z [C2].", ["C3", "C1", "C2"], [a, c, b])        # C1=a, C2=c, C3=b
    assert [e.keys for e in entries] == [["C1", "C3"], ["C2"]]
    assert len({e.label for e in entries}) == 2


def test_merge_sorts_numerically_not_lexically():
    ms = [meta(f"m{i}", seq=i) for i in range(1, 12)]
    [e] = run(" ".join(f"s [C{i}]." for i in (10, 2, 1, 11)), ["C10", "C2", "C1", "C11"], ms)
    assert e.keys == ["C1", "C2", "C10", "C11"]


def test_prompt_chunk_that_was_not_cited_is_not_listed():
    a, b = meta("a"), meta("b", section=("5.2 Map",))
    key_map, lookup = setup(a, b)                       # both were in the prompt
    entries = build_sources("Only one [C1].", ["C1"], key_map, lookup, strict=True)
    assert [e.keys for e in entries] == [["C1"]] and len(entries) == 1


def test_sibling_parts_on_different_pages_are_separate_entries_and_on_the_same_page_merge():
    t1 = meta("t1", doc="genai_profile", chunk_type="genai_action_table", section=("3 Actions",), subcategory="GOVERN 1.3",
              printed=("14", "14"), pdf=(20, 20), seq=1)
    t2 = meta("t2", doc="genai_profile", chunk_type="genai_action_table", section=("3 Actions",), subcategory="GOVERN 1.3",
              printed=("14", "15"), pdf=(20, 21), seq=2)
    t3 = meta("t3", doc="genai_profile", chunk_type="genai_action_table", section=("3 Actions",), subcategory="GOVERN 1.3",
              printed=("14", "14"), pdf=(20, 20), seq=3)
    entries = run("a [C1]. b [C2]. c [C3].", ["C1", "C2", "C3"], [t1, t2, t3])
    assert [e.keys for e in entries] == [["C1", "C3"], ["C2"]]
    assert entries[0].label.endswith("GOVERN 1.3, p. 14") and entries[1].label.endswith("GOVERN 1.3, pp. 14–15")
    assert {e.kind for e in entries} == {"table"}


def test_kind_is_carried_per_entry_and_labels_never_say_ocr():
    fig = meta("f", chunk_type="figure", figure_number=3, section=("2 Audience",), printed=("11", "11"), pdf=(16, 16))
    foot = meta("n", doc="genai_profile", chunk_type="footnote", footnote_number=2, section=("1 Intro",), printed=("1", "1"), pdf=(7, 7))
    entries = run("a [C1]. b [C2].", ["C1", "C2"], [fig, foot])
    assert [(e.kind, e.label) for e in entries] == [("figure-ocr", "NIST AI 100-1, §2 Audience, Fig. 3, p. 11"),
                                                    ("footnote", "NIST AI 600-1, §1 Intro, footnote 2, p. 1")]


def test_case_b_ambiguous_answer_lists_keys_from_all_readings_in_one_list():
    a, b = meta("a", section=("5.1 Govern",)), meta("b", doc="genai_profile", section=("1 Intro",), printed=("1", "1"), pdf=(7, 7))
    text = "Reading one: x [C1]. Reading two: y [C2][C1]."
    entries = run(text, ["C1", "C2"], [a, b])
    assert [e.keys for e in entries] == [["C1"], ["C2"]]


def test_output_is_deterministic_and_does_not_depend_on_cited_key_order():
    ms = [meta(f"m{i}", section=(f"5.{i} S",), pdf=(20 + i, 20 + i), printed=(str(15 + i),) * 2) for i in range(1, 6)]
    text = " ".join(f"s [C{i}]." for i in range(1, 6))
    one = run(text, ["C1", "C2", "C3", "C4", "C5"], ms)
    two = run(text, ["C5", "C3", "C1", "C4", "C2"], ms)
    assert one == two and one == run(text, ["C1", "C2", "C3", "C4", "C5"], ms)


def test_no_citations_gives_an_empty_list():
    assert run("No markers here.", [], [meta("a")]) == []


# --- consistency checks ------------------------------------------------------------------------------------------------

def entry(keys, label="L"):
    return SourceEntry(keys=keys, label=label, url="u", kind="prose")


def test_check_passes_for_a_consistent_list():
    assert check_consistency("a [C1]. b [C2][C1].", [entry(["C1", "C2"])]) == []


def test_check_catches_a_key_in_the_text_missing_from_the_list():
    assert check_consistency("a [C1]. b [C2].", [entry(["C1"])]) == ["C2 is in the text but in no entry"]


def test_check_catches_a_key_in_the_list_missing_from_the_text():
    assert check_consistency("a [C1].", [entry(["C1", "C2"])]) == ["C2 is in the list but not in the text"]


def test_check_catches_empty_keys_duplicate_keys_and_duplicate_labels():
    problems = check_consistency("a [C1].", [entry([], "X"), entry(["C1"], "Y"), entry(["C1"], "Y")])
    assert any("no keys" in p for p in problems)
    assert any("C1 appears in 2 entries" in p for p in problems)
    assert any("listed twice" in p for p in problems)


def test_build_raises_in_strict_mode_when_a_key_has_no_mapping():
    a = meta("a")
    with pytest.raises(AttributionError, match="C2 has no chunk"):
        build_sources("a [C1]. b [C2].", ["C1", "C2"], {"C1": "a"}, {"a": a}, strict=True)
    with pytest.raises(AttributionError, match="not in chunk_lookup"):
        build_sources("a [C1].", ["C1"], {"C1": "gone"}, {"a": a}, strict=True)


def test_build_logs_error_but_still_returns_in_production_mode(caplog):
    a = meta("a")
    with caplog.at_level(logging.ERROR, logger="attribution.build"):
        entries = build_sources("a [C1]. b [C2].", ["C1", "C2"], {"C1": "a"}, {"a": a})
    assert [e.keys for e in entries] == [["C1"]]
    assert any("C2" in r.message for r in caplog.records)


def test_build_flags_a_key_cited_in_the_text_but_not_in_cited_keys():
    a = meta("a")
    with pytest.raises(AttributionError, match="C2 is in the text"):
        build_sources("a [C1]. b [C2].", ["C1"], {"C1": "a", "C2": "a"}, {"a": a}, strict=True)


def test_build_never_edits_the_text():
    text = "Keep [C2] and [C1] exactly."
    a, b = meta("a"), meta("b", section=("5.2 Map",))
    key_map, lookup = setup(a, b)
    before = text
    build_sources(text, ["C2", "C1"], key_map, lookup, strict=True)
    assert text == before


# --- real chunks -------------------------------------------------------------------------------------------------------

def test_real_chunks_merge_and_order_correctly():
    from ingest.config import PROCESSED_DIR
    path = PROCESSED_DIR / "chunks.jsonl"
    if not path.exists():
        pytest.skip("chunks.jsonl not built")
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lookup = {c["chunk_id"]: c for c in chunks}
    from collections import defaultdict
    from attribution import format as fmt
    by_label = defaultdict(list)
    for c in chunks:
        by_label[fmt.citation_label(c)].append(c)
    dup = next(v for v in by_label.values() if len(v) > 1)
    other = next(c for c in reversed(chunks) if c not in dup)
    picked = [other, dup[0], dup[1]]
    key_map = {f"C{i}": c["chunk_id"] for i, c in enumerate(picked, 1)}
    entries = build_sources("a [C1]. b [C2]. c [C3].", ["C1", "C2", "C3"], key_map, lookup, strict=True)
    assert len(entries) == 2 and any(e.keys == ["C2", "C3"] for e in entries)
    assert all(e.url.startswith("https://") and "#page=" in e.url for e in entries)
