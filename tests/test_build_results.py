"""Stage D3: docs/build_results.py. Uses copies of the real report files in a temp directory, so no number is hand-typed here."""
import ast
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_results", ROOT / "docs" / "build_results.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)

REPORT_FILES = [rel for rel, _ in br.REPORTS.values()]
have_reports = all((ROOT / f).is_file() for f in REPORT_FILES) and bool(list(ROOT.glob(br.E2E_GLOB)))
needs_reports = pytest.mark.skipif(not have_reports, reason="needs the committed report files")


@pytest.fixture()
def repo(tmp_path):
    """A temp 'repository' holding copies of every report and a README with the markers."""
    for rel in REPORT_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    run = sorted(ROOT.glob(br.E2E_GLOB))[-1]
    dest = tmp_path / "eval" / "e2e_runs" / run.parent.name
    dest.mkdir(parents=True)
    shutil.copy(run, dest / "report.json")
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "README.md").write_text(f"# Title\n\nintro\n\n{br.START}\nold block\n{br.END}\n\nthe rest\n", encoding="utf-8")
    return tmp_path


def edit(root: Path, rel: str, fn) -> None:
    path = root / rel
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------- determinism and independence from the machine
@needs_reports
def test_output_is_deterministic_and_does_not_depend_on_file_times(repo):
    first = br.build(repo)
    import os, time
    for rel in REPORT_FILES:
        os.utime(repo / rel, (time.time() + 5000, time.time() + 5000))           # a fresh clone has new modification times
    assert br.build(repo) == first
    assert br.build(ROOT) == first                                                # same content, different location


@needs_reports
def test_main_twice_gives_identical_files(repo):
    assert br.main(repo) == 0
    one = ((repo / "docs" / "RESULTS.md").read_text(), (repo / "README.md").read_text())
    assert br.main(repo) == 0
    assert ((repo / "docs" / "RESULTS.md").read_text(), (repo / "README.md").read_text()) == one


# ---------------------------------------------------------------- every number comes from a report
@needs_reports
def test_numbers_change_when_the_report_changes(repo):
    def change(d):
        for k in ("vector_only", "bm25_only", "ensemble", "ensemble_rerank"):
            d["summary"][k]["hit_at_5"] = 0.123
            d["summary"][k]["mrr"] = 0.456
    edit(repo, "eval/retrieval_report.json", change)
    results, block = br.build(repo)
    assert "0.123" in results and "0.456" in results and "Hit@5 vector 0.123" in block
    assert "0.879" not in results.split("## Routing")[0]                          # the old value is gone from the retrieval section


@needs_reports
def test_each_section_shows_values_read_from_its_report(repo):
    results, block = br.build(repo)
    load = lambda rel: json.loads((repo / rel).read_text())                       # noqa: E731
    routing, links, ins = load("eval/routing_report.json"), load("eval/link_check_report.json"), load("eval/insufficient_report.json")
    assert f"{routing['passed']}/{routing['cases']} cases passed" in results
    assert f"{links['chunks_checked']} chunks checked: {links['passed_strict']} pass strictly" in results and f"**{links['failed']} failed**" in results
    assert f"{ins['unanswerable']['insufficient']}/{ins['unanswerable']['cases']} correctly returned" in results
    assert f"{links['chunks_checked'] - links['failed']}/{links['chunks_checked']} chunks open a PDF page" in block
    e2e = json.loads(sorted((repo / "eval" / "e2e_runs").glob("*/report.json"))[-1].read_text())
    assert f"run `{e2e['run']}`" in results and f"{len(e2e['failures'])} failed checks" in results


@needs_reports
def test_every_section_names_its_source_file_and_hash(repo):
    results, _ = br.build(repo)
    for rel in REPORT_FILES:
        assert f"`{rel}`" in results
    assert results.count("(sha256 `") == len(REPORT_FILES) + 1                  # + the end-to-end report
    import hashlib
    sha = hashlib.sha256((repo / "eval/link_check_report.json").read_bytes()).hexdigest()[:12]
    assert sha in results


# ---------------------------------------------------------------- missing / broken reports
@needs_reports
def test_a_missing_report_becomes_a_clear_note_not_a_crash(repo):
    (repo / "eval/routing_report.json").unlink()
    results, block = br.build(repo)
    assert "_Not run: `eval/routing_report.json` was not found. Run `make eval-routing`" in results
    assert "## Retrieval" in results and "## Citation links" in results and "- **Routing:** not run." in block


@needs_reports
def test_a_corrupt_report_is_treated_as_missing(repo):
    (repo / "eval/link_check_report.json").write_text("{not json")
    assert "_Not run: `eval/link_check_report.json`" in br.build(repo)[0]


def test_with_no_reports_at_all_every_section_says_not_run(tmp_path):
    results, block = br.build(tmp_path)
    assert results.count("_Not run:") == 8 and "End-to-end evaluation" in results and "not run" in block
    assert br.main(tmp_path) == 0 and (tmp_path / "docs" / "RESULTS.md").is_file()


@needs_reports
def test_the_usage_report_is_optional_inside_the_generation_section(repo):
    (repo / "eval/generation_usage_report.json").unlink()
    results, _ = br.build(repo)
    assert "## Generation" in results and "Token-usage report not found" in results


@needs_reports
def test_the_newest_end_to_end_run_is_used(repo):
    run = sorted((repo / "eval" / "e2e_runs").iterdir())[-1]
    older = repo / "eval" / "e2e_runs" / "20200101-000000"
    older.mkdir()
    data = json.loads((run / "report.json").read_text())
    data["run"] = "20200101-000000"
    (older / "report.json").write_text(json.dumps(data))
    assert "run `20200101-000000`" not in br.build(repo)[0]
    newer = repo / "eval" / "e2e_runs" / "20991231-000000"
    newer.mkdir()
    data["run"] = "20991231-000000"
    (newer / "report.json").write_text(json.dumps(data))
    assert "run `20991231-000000`" in br.build(repo)[0]


# ---------------------------------------------------------------- the README block
def test_update_readme_replaces_only_the_marked_block():
    text = f"top\n{br.START}\nold\n{br.END}\nbottom\n"
    new = br.update_readme(text, "NEW\n")
    assert new == f"top\n{br.START}\nNEW\n{br.END}\nbottom\n"
    assert br.update_readme(new, "NEW\n") == new                                  # idempotent


@pytest.mark.parametrize("text", ["no markers", f"{br.START} only start", f"{br.END} then {br.START}",
                                  f"{br.START}a{br.END}{br.START}b{br.END}"])
def test_update_readme_refuses_missing_duplicate_or_reversed_markers(text):
    assert br.update_readme(text, "x\n") is None


@needs_reports
def test_main_updates_the_block_and_leaves_the_rest_of_the_readme_alone(repo):
    br.main(repo)
    text = (repo / "README.md").read_text()
    assert text.startswith("# Title\n\nintro\n\n") and text.endswith("\n\nthe rest\n") and "old block" not in text
    assert br.build(repo)[1] in text


@needs_reports
def test_a_readme_without_markers_is_left_untouched_and_reported(repo, capsys):
    (repo / "README.md").write_text("# Plain README\n")
    br.main(repo)
    assert (repo / "README.md").read_text() == "# Plain README\n" and "has no <!-- RESULTS:START -->" in capsys.readouterr().out


# ---------------------------------------------------------------- the real repository
def test_the_script_uses_only_the_standard_library():
    tree = ast.parse((ROOT / "docs" / "build_results.py").read_text())
    import sys
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


@needs_reports
def test_the_committed_results_file_is_not_stale():
    """Fails when a report changed (or the script did) but `make results` was not re-run."""
    committed = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert committed == br.build(ROOT)[0], "docs/RESULTS.md is out of date: run `make results`"
    assert "_Not run:" not in committed


# ---------------------------------------------------------------- sentinel values: a hard-coded number cannot survive these
SENTINELS = [
    ("eval/link_check_report.json", lambda d: d.update(chunks_checked=9991, passed_strict=9992, explained_mismatch=9993), ["9991 chunks checked", "9992 pass strictly", "9993 have a text-sample"]),
    ("eval/routing_report.json", lambda d: d.update(cases=9981, passed=9982, tokens=9983, llm_calls=9984), ["9982/9981", "9,983", "9984"]),
    ("eval/insufficient_report.json", lambda d: d["unanswerable"].update(cases=9971, insufficient=9972, via_gate=9973), ["9972/9971", "9973 through"]),
    ("eval/prompt_size_report.json", lambda d: (d.update(queries=9961, prompt_budget_tokens=9962), d["over_budget"].update(observed_est=9963, observed_est_margin=9964)),
     ["9961 real prompts", "9,962 tokens", "9963 by the plain estimate, 9964 with the 20% margin"]),
    ("eval/generation_usage_report.json", lambda d: d.update(requests=9951), ["9951 live requests"]),
    ("eval/generation_behaviour_report.json", lambda d: d.update(cases=9941, errors=9942), ["9941 behaviour cases, 9942 ended"]),
    ("eval/integration_report.json", lambda d: d["live"].update(startup_seconds=99.9), ["99.9 s"]),
    ("eval/retrieval_report.json", lambda d: d.update(cases=9931, answerable=9932), ["9931 cases (9932 answerable"]),
]


@needs_reports
@pytest.mark.parametrize("rel,mutate,expected", SENTINELS, ids=[s[0].split("/")[1] for s in SENTINELS])
def test_sentinel_values_appear_in_the_output(repo, rel, mutate, expected):
    edit(repo, rel, mutate)
    results, _ = br.build(repo)
    for text in expected:
        assert text in results, text


@needs_reports
def test_end_to_end_sentinels_and_readme_block(repo):
    run = sorted((repo / "eval" / "e2e_runs").glob("*/report.json"))[-1]
    data = json.loads(run.read_text())
    data["aggregates"]["route_accuracy"] = {"passed": 9921, "applicable": 9922, "pct": 99.9}
    data["requests"] = 9923
    run.write_text(json.dumps(data))
    results, block = br.build(repo)
    assert "99.9% (9921/9922)" in results and "9923 `/chat` requests" in results and "route 9921/9922" in block and "9923 requests" in block
