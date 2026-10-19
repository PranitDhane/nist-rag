"""Stage D4: DESIGN.md describes the system as built (CLAUDE.md section 5 and deliverables rule 5).

* every number that is a MEASUREMENT lives in a generated EVIDENCE block and must be current;
* every number written in the prose must be on the allow-list below (design parameters and identifiers), and each design
  parameter is compared with the code's own configuration, so the document cannot drift from the system;
* fixed texts, links and anchors must exist."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DESIGN = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
spec = importlib.util.spec_from_file_location("build_results", ROOT / "docs" / "build_results.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def prose(text: str) -> str:
    """DESIGN.md without generated blocks, code fences, headings and link targets."""
    text = re.sub(r"<!-- EVIDENCE:(\w+):START -->.*?<!-- EVIDENCE:\1:END -->", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^#+ .*$", "", text, flags=re.M)
    return re.sub(r"\]\([^)]*\)", "]", text)


# ---------------------------------------------------------------- generated evidence
def test_every_evidence_block_exists_exactly_once():
    for name in br.EVIDENCE_NAMES:
        start, end = br.evidence_markers(name)
        assert DESIGN.count(start) == 1 and DESIGN.count(end) == 1 and DESIGN.index(start) < DESIGN.index(end), name


needs_build = pytest.mark.skipif(not all((ROOT / f).is_file() for f in br.BUILD_FILES.values()) or not (ROOT / "eval/retrieval_report.json").is_file(),
                                 reason="needs the reports and the output of `make build`")


@needs_build
def test_evidence_blocks_are_current():
    new, missing = br.update_design(DESIGN, br.build_evidence(ROOT))
    assert not missing and new == DESIGN, "DESIGN.md evidence is out of date: run `make results`"
    assert "Evidence not available" not in DESIGN


# ---------------------------------------------------------------- numbers in the prose
ALLOWED = {
    # structure of the document and of the system
    "1", "2", "3", "4", "5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    # identifiers: documents, models, action IDs, sections, HTTP statuses
    "1.7", "001", "100", "600", "1.0", "20", "120", "404", "409",
    # design parameters, each compared with the code below
    "10", "0.6", "0.4", "250", "350", "60", "450", "512", "384", "0.15", "150", "0.2", "8,000",
}
NUMBER = re.compile(r"(?<![\w.])\d[\d,.]*\d|\b\d\b")


def test_prose_contains_no_unexplained_numbers():
    found = {m.group(0) for m in NUMBER.finditer(prose(DESIGN))}
    assert found <= ALLOWED, f"numbers in DESIGN.md prose that are not design parameters or identifiers: {sorted(found - ALLOWED)}. " \
                             f"Measurements belong in a generated EVIDENCE block (docs/build_results.py)."


def test_the_allow_list_catches_a_typed_metric():
    assert "0.879" not in ALLOWED and "97.1" not in ALLOWED
    assert {m.group(0) for m in NUMBER.finditer("Hits are 0.879 and routing 97.1% over 33 cases")} == {"0.879", "97.1", "33"}


# ---------------------------------------------------------------- design parameters equal the code's configuration
def test_retrieval_parameters_match_the_code():
    from retrieval import config as rc
    assert (rc.VECTOR_K, rc.BM25_K, list(rc.WEIGHTS), rc.TOP_N, rc.THRESHOLD) == (10, 10, [0.6, 0.4], 5, 0.15)
    for text in ("(k=10)", "(k=10)", "weights 0.6 vector / 0.4 BM25", "best 5", "(0.15)"):
        assert text in DESIGN, text


def test_chunking_parameters_match_the_code():
    from chunking import config as cc
    assert (cc.PROSE_TARGET_MIN, cc.PROSE_TARGET_MAX, cc.PROSE_OVERLAP_FRACTION) == (250, 350, 0.10)
    assert (cc.TINY_SECTION_MAX_TOKENS, cc.TABLE_PIECE_MAX_TOKENS, cc.HARD_MAX_EMBED_TOKENS) == (60, 450, 512)
    for text in ("250–350 tokens", "about 10 % overlap", "(60 tokens or fewer)", "at most 450 tokens", "512 tokens"):
        assert text in DESIGN, text


def test_model_and_llm_parameters_match_the_code():
    from app.config import Settings
    from augmentation import config as ac
    from generation import config as gc
    from indexing import config as ic
    assert Settings().router_model.endswith("gpt-oss-120b") and gc.GEN_MODEL.endswith("gpt-oss-20b")
    assert Settings().history_turns == 2 and ac.WORD_LIMIT == 150 and gc.TPM_LIMIT == 8000 and gc.TEMPERATURE == 0.2
    assert ic.MODEL_NAME == "BAAI/bge-small-en-v1.5" and ic.DIMENSION == 384 and ic.NORMALIZE is True
    for text in ("gpt-oss-20b", "gpt-oss-120b", "last 2 turns", "about 150 words", "(8,000)", "384 dimensions", "bge-small-en-v1.5"):
        assert text in DESIGN, text
    assert "cosine" in DESIGN


def test_fixed_texts_in_the_document_are_the_ones_in_the_code():
    from augmentation import config as ac
    from graph.replies import OUT_OF_SCOPE_REPLY
    assert OUT_OF_SCOPE_REPLY in DESIGN and ac.INSUFFICIENT_CONTEXT in DESIGN


def test_graph_nodes_named_in_the_diagram_exist_in_the_documented_graphs():
    graph_md = (ROOT / "docs" / "graph.md").read_text(encoding="utf-8")
    diagram = re.search(r"```\n(reset_turn.*?)```", DESIGN, re.S).group(1)
    for node in ("reset_turn", "rule_greeting_check", "classify_and_rewrite", "greeting_reply", "out_of_scope_reply", "finalize", "error_finalize",
                 "hybrid_search", "rerank", "gate", "expand_context", "generate", "attribute", "gate_failure_reply", "finalize_model_refusal"):
        assert node in diagram, node
        assert node in graph_md, node


# ---------------------------------------------------------------- links
def test_relative_links_point_at_files_and_anchors_that_exist():
    results = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).strip().replace(" ", "-") for h in re.findall(r"^## (.+)$", results, re.M)}
    for target in re.findall(r"\]\(([^)]+)\)", DESIGN):
        path, _, anchor = target.partition("#")
        assert (ROOT / path).exists(), f"broken link (DESIGN.md is at the repository root): {target}"
        if path.endswith("RESULTS.md") and anchor:
            assert anchor in anchors, f"unknown RESULTS.md anchor: {anchor}"
    for name in br.EVIDENCE_NAMES:                                              # the generated links too
        for target in re.findall(r"\]\(([^)]+)\)", br.build_evidence(ROOT)[name]):
            path, _, anchor = target.partition("#")
            assert (ROOT / path).exists() and (not anchor or anchor in anchors), target
