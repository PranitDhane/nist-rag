"""The chunking validators must actually fail on bad data (each check is exercised with a corrupted record)."""
import copy

import pytest

pytest.importorskip("transformers")
from chunking import config  # noqa: E402
from chunking import validate as v  # noqa: E402
from chunking.join import join_all, load_pages  # noqa: E402

pytestmark = pytest.mark.skipif(not config.PAGES_PATH.is_file(), reason="pages.jsonl absent")


@pytest.fixture(scope="module")
def good():
    from chunking.pipeline import run_all
    res = run_all()
    from ingest.config import load_manifest
    return [r for _, rs, _, _ in res for r in rs], {d.doc_id: d for d, _, _, _ in res}, load_manifest(), load_pages()


def bad(good_records, pick, mutate):
    recs = copy.deepcopy(good_records)
    mutate(next(r for r in recs if pick(r)))
    return recs


def test_good_data_passes_every_check(good):
    recs, docs, manifest, pages = good
    checks = [v.check_embed_limit(recs), v.check_coverage(recs, docs), v.check_sections(recs, docs),
              v.check_tables(recs, manifest), v.check_table_integrity(recs), v.check_fields(recs),
              v.check_no_breadcrumb(recs), v.check_extras(recs, pages)]
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_embed_limit_fails_on_long_text(good):
    recs = bad(good[0], lambda r: r["chunk_type"] == "prose", lambda r: r.update(embed_text=("word " * 600), token_count_embed=602))
    assert not v.check_embed_limit(recs).ok


def test_coverage_fails_when_a_span_is_dropped(good):
    recs, docs, *_ = good
    recs = copy.deepcopy(recs)
    victim = next(r for r in recs if r["chunk_type"] == "prose" and r["doc_id"] == "ai_rmf_1_0")
    recs.remove(victim)
    assert not v.check_coverage(recs, docs).ok


def test_section_check_fails_when_a_chunk_spans_a_heading(good):
    recs, docs, *_ = good
    doc = docs["ai_rmf_1_0"]
    h = doc.headings[3]
    recs = bad(recs, lambda r: r["doc_id"] == "ai_rmf_1_0" and r["chunk_type"] == "prose",
               lambda r: r.update(char_spans=[[h.start - 50, h.start + 50]]))
    assert not v.check_sections(recs, docs).ok


def test_table_check_fails_on_duplicate_or_missing_action_id(good):
    recs, _, manifest, _ = good
    dup = bad(recs, lambda r: r["chunk_type"] == "genai_action_table", lambda r: r["action_ids"].append(r["action_ids"][0]))
    assert not v.check_tables(dup, manifest).ok
    gone = bad(recs, lambda r: r["chunk_type"] == "genai_action_table", lambda r: r["action_ids"].pop())
    assert not v.check_tables(gone, manifest).ok


def test_table_integrity_fails_without_actor_line_or_header(good):
    recs, *_ = good
    no_actor = bad(recs, lambda r: r["chunk_type"] == "genai_action_table",
                   lambda r: r.update(display_text="\n".join(r["display_text"].splitlines()[:-1])))
    assert not v.check_table_integrity(no_actor).ok
    no_head = bad(recs, lambda r: r["chunk_type"] == "rmf_core_table", lambda r: r.update(display_text="x\n" + r["display_text"]))
    assert not v.check_table_integrity(no_head).ok


def test_field_check_fails_on_empty_fields_and_duplicate_ids(good):
    recs, *_ = good
    assert not v.check_fields(bad(recs, lambda r: True, lambda r: r.update(citation_url=""))).ok
    dup = copy.deepcopy(recs)
    dup[1]["chunk_id"] = dup[0]["chunk_id"]
    assert not v.check_fields(dup).ok


def test_breadcrumb_in_display_text_fails(good):
    recs, *_ = good
    def leak(r):
        r["display_text"] = r["embed_text"].split("\n\n", 1)[0] + "\n\n" + r["display_text"]
    assert not v.check_no_breadcrumb(bad(recs, lambda r: r["chunk_type"] == "prose", leak)).ok


def test_extras_check_fails_when_a_footnote_chunk_is_missing(good):
    recs, _, _, pages = good
    fewer = copy.deepcopy(recs)
    fewer.remove(next(r for r in fewer if r["chunk_type"] == "footnote"))
    assert not v.check_extras(fewer, pages).ok


def test_determinism_check_detects_difference(good):
    recs = good[0]
    other = copy.deepcopy(recs)
    other[0]["display_text"] += " "
    assert not v.check_determinism(recs, other).ok and v.check_determinism(recs, recs).ok
