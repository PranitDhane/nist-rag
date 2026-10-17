import pytest

from indexing.bm25_index import base_tokens, tokenize


def test_action_id_kept_whole_and_lowercased():
    assert tokenize("GV-1.7-001") == ["gv-1.7-001"]
    assert tokenize("See MS-2.11-003, then MG-4.3-001.") == ["see", "ms-2.11-003", "then", "mg-4.3-001"]


def test_dotted_section_numbers_kept_whole():
    assert tokenize("2.12") == ["2.12"] and tokenize("1.2.3") == ["1.2.3"]
    assert tokenize("see section 2.12.") == ["see", "section", "2.12"]


def test_subcategory_reference_gives_three_tokens():
    assert tokenize("GOVERN 1.7") == ["govern", "1.7", "govern_1.7"]
    assert tokenize("MEASURE 2.11:") == ["measur", "2.11", "measure_2.11"]   # plain word stemmed, combined token whole
    assert tokenize("Map 2.1") == ["map", "2.1", "map_2.1"]


def test_category_reference_also_gets_combined_token():
    assert tokenize("GOVERN 1:") == ["govern", "1", "govern_1"]


def test_plain_words_lowercased_split_and_stemmed():
    assert tokenize("Human-AI Configuration, organization’s") == ["human", "ai", "configur", "organiz", "s"]
    assert tokenize("decommissioning decommissioned decommission") == ["decommiss"] * 3
    assert tokenize("risks risk") == ["risk", "risk"]


def test_identifiers_are_never_stemmed():
    assert tokenize("GV-1.7-001 decommissioning") == ["gv-1.7-001", "decommiss"]
    assert "manage_4.3" in tokenize("MANAGE 4.3") and "manag" in tokenize("MANAGE 4.3")


def test_base_tokens_do_not_stem():
    assert base_tokens("Confabulation refers to") == ["confabulation", "refers", "to"]


def test_empty_and_punctuation_only():
    assert tokenize("") == [] and tokenize("— ; : ( )") == []


def test_query_and_document_use_the_same_function():
    assert tokenize("GOVERN 1.7 decommissioning") == tokenize("govern 1.7   Decommissioning")
