import pytest

from augmentation import config
from augmentation.parser import parse_answer, split_units

KM = {f"C{i}": f"chunk:{i}" for i in range(1, 6)}          # keys C1..C5 exist
D = config.INSUFFICIENT_CONTEXT
WORDS_180 = " ".join(["word"] * 180)

# name -> (text, expected case, expected violations (as a set), extra expectations)
CASES = {
    # ---- valid ----
    "A paragraph": ('The AI RMF Core is composed of four functions: "GOVERN," "MAP," "MEASURE," and "MANAGE" [C1]. '
                    'GOVERN is a cross-cutting function [C2].', "A_or_B", set()),
    "A bullets": ("The Core has four functions:\n- GOVERN cultivates a culture of risk management [C2].\n"
                  "- MAP establishes context [C1].\n- MEASURE assesses risks [C1][C3].\n- MANAGE prioritizes risks [C4].",
                  "A_or_B", set()),
    "B one covered one uncovered": (f"{config.B_FIRST_LINE}\n- GOVERN as a function: it is described as cross-cutting [C2].\n"
                                    f"- Governance in profiles: {config.NOT_COVERED}.", "A_or_B", set()),
    "D exact": (D, "D", set()),
    "D no final period": (D.rstrip("."), "D", set()),
    "D with surrounding whitespace": (f"\n  {D}  \n", "D", set()),
    # ---- invalid ----
    "one uncited sentence": ("GOVERN is cross-cutting [C2]. MAP maps context.", "A_or_B", {"UNCITED_SENTENCE"}),
    "invented C9": ("GOVERN is cross-cutting [C2]. MAP maps context [C9].", "A_or_B", {"UNKNOWN_CITATION"}),
    "malformed [C1, C2]": ("GOVERN is cross-cutting [C1, C2].", "A_or_B", {"MALFORMED_CITATION", "NO_CITATIONS", "UNCITED_SENTENCE"}),
    "malformed (C1)": ("GOVERN is cross-cutting (C1).", "A_or_B", {"MALFORMED_CITATION", "NO_CITATIONS", "UNCITED_SENTENCE"}),
    "malformed [1]": ("GOVERN is cross-cutting [1].", "A_or_B", {"MALFORMED_CITATION", "NO_CITATIONS", "UNCITED_SENTENCE"}),
    "Sources list": ("GOVERN is cross-cutting [C2].\n\nSources:\n- NIST AI 100-1 [C2]", "A_or_B", {"HAS_SOURCES_LIST"}),
    "References list": ("GOVERN is cross-cutting [C2].\nReferences: [C2]", "A_or_B", {"HAS_SOURCES_LIST"}),
    "URL": ("GOVERN is cross-cutting [C2]. See https://www.nist.gov/ai [C2].", "A_or_B", {"HAS_URL"}),
    "markdown heading": ("## Answer\nGOVERN is cross-cutting [C2].", "A_or_B", {"HAS_HEADING_OR_CODE", "UNCITED_SENTENCE"}),
    "code fence": ("GOVERN is cross-cutting [C2].\n```\ncode\n```", "A_or_B", {"HAS_HEADING_OR_CODE", "UNCITED_SENTENCE"}),
    "D plus extra text": (f"{D} However, GOVERN is cross-cutting [C2].", "D", {"D_WITH_EXTRA_TEXT"}),
    "D plus extra paragraph": (f"GOVERN is cross-cutting [C2].\n\n{D}", "D", {"D_WITH_EXTRA_TEXT"}),
    "180 words": (WORDS_180 + " end [C1].", "A_or_B", {"OVER_LENGTH"}),
    "empty": ("", "invalid", {"EMPTY"}),
    "whitespace only": (" \n\t ", "invalid", {"EMPTY"}),
    "zero citations": ("GOVERN is a cross-cutting function.", "A_or_B", {"NO_CITATIONS", "UNCITED_SENTENCE"}),
    # ---- edge cases ----
    "e.g. stays one sentence": ("Risks arise in many places, e.g., data, models and people [C1].", "A_or_B", set()),
    "i.e. and etc.": ("Outcomes (i.e., results, etc.) are documented [C1]. Actions follow [C2].", "A_or_B", set()),
    "adjacent markers": ("GOVERN spans functions [C1][C3].", "A_or_B", set()),
    "quoted term with a period": ('NIST defines the goal as "trustworthy AI." Organizations pursue it [C1].', "A_or_B", set()),
    "section number 5.1": ("Section 5.1 describes GOVERN as cross-cutting [C2].", "A_or_B", set()),
    "Fig. abbreviation": ("Fig. 5 shows the Core functions [C4]. It marks GOVERN as cross-cutting [C4].", "A_or_B", set()),
    "lead-in and bullets exempt": ("The functions are:\n- GOVERN [C1].\n- MAP [C1].", "A_or_B", set()),
}


@pytest.mark.parametrize("name", CASES)
def test_case_and_violations(name):
    text, case, violations = CASES[name]
    p = parse_answer(text, KM)
    assert p.case == case, (name, p)
    assert set(p.violations) == violations, (name, p.violations, p.uncited_sentences)
    assert p.ok == (not violations)


def test_ambiguous_format_flag_and_cited_keys_order_and_dedup():
    p = parse_answer(CASES["B one covered one uncovered"][0], KM)
    assert p.is_ambiguous_format and p.cited_keys == ["C2"]
    p = parse_answer("A [C2]. B [C1]. C [C2][C3].", KM)
    assert p.cited_keys == ["C2", "C1", "C3"] and not p.is_ambiguous_format


def test_first_line_must_be_exactly_the_approved_phrase_for_ambiguous_format():
    assert not parse_answer("This question can be read in more than one way\n- a: b [C1].", KM).is_ambiguous_format
    assert not parse_answer("Intro. " + config.B_FIRST_LINE + "\n- a: b [C1].", KM).is_ambiguous_format


def test_unknown_keys_listed_in_order_and_removed_only_from_clean_text():
    text = "A is true [C1]. B is true [C9]. C is true [C1][C8]."
    p = parse_answer(text, KM)
    assert p.unknown_keys == ["C9", "C8"] and p.cited_keys == ["C1", "C9", "C8"]
    assert p.clean_text == "A is true [C1]. B is true. C is true [C1]."
    assert "[C9]" in text                                                       # input untouched
    assert parse_answer("fine [C1].", KM).clean_text == "fine [C1]."             # nothing else rewritten


def test_word_count_excludes_citation_markers_and_bullet_dashes():
    assert parse_answer("One two three [C1][C2].", KM).word_count == 3          # markers are not words
    assert parse_answer("- one two\n- three four [C1].", KM).word_count == 4


def test_word_limit_boundary():
    ok = " ".join(["word"] * 149) + " [C1]."
    assert "OVER_LENGTH" not in parse_answer(ok, KM).violations
    assert parse_answer(" ".join(["word"] * 151) + " [C1].", KM).violations == ["OVER_LENGTH"]
    ambiguous = config.B_FIRST_LINE + "\n- " + " ".join(["word"] * 160) + " [C1]."
    assert "OVER_LENGTH" in parse_answer(ambiguous, KM).violations              # Q1: same 150 limit for Case B


def test_case_b_answer_exempts_first_line_and_not_covered_bullets_but_not_other_bullets():
    text = f"{config.B_FIRST_LINE}\n- reading one: GOVERN is cross-cutting.\n- reading two: {config.NOT_COVERED}."
    p = parse_answer(text, KM)
    assert p.uncited_sentences == ["reading one: GOVERN is cross-cutting."]


def test_not_covered_bullet_with_a_citation_is_still_fine():
    p = parse_answer(f"{config.B_FIRST_LINE}\n- a: x [C1].\n- b: {config.NOT_COVERED}.", KM)
    assert p.ok


def test_split_units_details():
    assert split_units("First fact [C1]. Second fact [C2].") == ["First fact [C1].", "Second fact [C2]."]
    assert split_units("Risks, e.g., data and models [C1].") == ["Risks, e.g., data and models [C1]."]
    assert split_units('The term is "valid. reliable" [C1].') == ['The term is "valid. reliable" [C1].']
    assert split_units("Intro:\n- a [C1].\n- b [C2].") == ["a [C1].", "b [C2]."]
    assert split_units(f"{config.B_FIRST_LINE}\n- x: {config.NOT_COVERED}.") == []


def test_d_detection_does_not_fire_on_similar_but_different_sentences():
    p = parse_answer("The NIST documents do not contain this information [C1].", KM)
    assert p.case == "A_or_B"


def test_none_text_is_treated_as_empty():
    assert parse_answer(None, KM).violations == ["EMPTY"]                       # type: ignore[arg-type]


def test_empty_key_map_makes_every_citation_unknown():
    p = parse_answer("GOVERN is cross-cutting [C1].", {})
    assert p.unknown_keys == ["C1"] and "UNKNOWN_CITATION" in p.violations and p.clean_text == "GOVERN is cross-cutting."


def test_multiple_violations_are_all_reported():
    p = parse_answer("## Heading\nUncited sentence. See https://x.org [C9].\nSources: none", KM)
    assert {"HAS_HEADING_OR_CODE", "UNCITED_SENTENCE", "HAS_URL", "UNKNOWN_CITATION", "HAS_SOURCES_LIST"} <= set(p.violations)
