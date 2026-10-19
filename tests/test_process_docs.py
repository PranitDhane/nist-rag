"""Stage D6: the process documents moved to docs/process/ and nothing points at the old places."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESS = ROOT / "docs" / "process"
PLANS = ("INGESTION", "CHUNKING", "EMBEDDING_INDEXING", "API_ROUTING", "RETRIEVAL", "AUGMENTATION", "GENERATION",
         "INSUFFICIENT_CONTEXT", "ATTRIBUTION", "GRAPH_INTEGRATION", "E2E_EVALUATION", "DELIVERABLES")

OLD_PATHS = re.compile(r"docs/(?!process/)(?:\w+_PLAN\.md|PROGRESS\.md|PROBLEMS_AND_SOLUTIONS\.md|(?:RETRIEVAL|GENERATION)_RESULTS\.md)")
SKIP_DIRS = {".venv", ".git", "data", "__pycache__", "e2e_runs", ".pytest_cache"}
# Not rewritten on purpose: the log keeps the paths that were valid when each entry was written, and eval/e2e_cases.yaml is pinned by the
# SHA-256 recorded in the saved end-to-end run (a comment edit would break that verification).
HISTORICAL = {PROCESS / "PROGRESS.md", ROOT / "eval" / "e2e_cases.yaml", ROOT / "tests" / "test_process_docs.py"}


def text_files(*suffixes: str):
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in suffixes and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
            yield path


def slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


# ---------------------------------------------------------------- the move
def test_everything_is_where_the_plan_says():
    for name in PLANS:
        assert (PROCESS / "plans" / f"{name}_PLAN.md").is_file(), name
    for name in ("PROGRESS.md", "PROBLEMS_AND_SOLUTIONS.md", "README.md", "phase_notes.md"):
        assert (PROCESS / name).is_file(), name
    assert (ROOT / "CLAUDE.md").is_file()                                           # Q6: stays at the root
    leftovers = [p.name for p in (ROOT / "docs").glob("*.md") if p.name.endswith("_PLAN.md") or p.name in ("PROGRESS.md", "PROBLEMS_AND_SOLUTIONS.md")]
    assert not leftovers, leftovers
    assert {p.name for p in (ROOT / "docs").iterdir() if p.is_file()} >= {"RESULTS.md", "E2E_RESULTS.md", "graph.md", "build_results.py", "architecture.html"}


def test_no_file_still_points_at_the_old_locations():
    offenders = []
    for path in text_files(".md", ".py", ".html", ".yaml", ".yml", ".sh", ".txt", ""):
        if path in HISTORICAL or path.name == "Makefile" and False:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in OLD_PATHS.finditer(text):
            offenders.append(f"{path.relative_to(ROOT)}: {m.group(0)}")
    assert not offenders, offenders


def test_the_old_path_detector_works():
    assert OLD_PATHS.search("see docs/RETRIEVAL_PLAN.md") and OLD_PATHS.search("docs/PROGRESS.md")
    assert not OLD_PATHS.search("docs/process/plans/RETRIEVAL_PLAN.md") and not OLD_PATHS.search("docs/RESULTS.md")


def test_every_docs_path_named_in_claude_md_exists():
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    paths = set(re.findall(r"`((?:docs|eval|prompts)/[\w./-]+)`", claude))
    assert paths
    for rel in paths:
        if rel.endswith("/") or "*" in rel or "<" in rel or "e2e_runs" in rel:
            continue
        assert (ROOT / rel).exists(), f"CLAUDE.md names a path that does not exist: {rel}"


# ---------------------------------------------------------------- links
def test_relative_markdown_links_resolve_including_anchors():
    broken = []
    for path in text_files(".md"):
        if path == PROCESS / "phase_notes.md":                                     # historical text, links written for another layout
            continue
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http", "mailto", "<")):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (path.parent / file_part).resolve() if file_part else path
            if not dest.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
            elif anchor and dest.suffix == ".md":
                heads = {slug(h) for h in re.findall(r"^#{1,6} (.+)$", dest.read_text(encoding="utf-8"), re.M)}
                if anchor not in heads:
                    broken.append(f"{path.relative_to(ROOT)} -> {target} (no such heading)")
    assert not broken, broken


# ---------------------------------------------------------------- PROBLEMS_AND_SOLUTIONS.md cites real log entries
def log_fields() -> dict[str, list[str]]:
    """date (MM-DD) -> the 'phase | stage' field of each progress-log line of that date."""
    out: dict[str, list[str]] = {}
    for line in (PROCESS / "PROGRESS.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"-?\s*2026-(\d\d-\d\d) \| ([^|]+)\|", line)
        if m:
            out.setdefault(m.group(1), []).append(m.group(2).strip())
    return out


def test_every_problem_row_cites_an_existing_progress_entry():
    fields, text = log_fields(), (PROCESS / "PROBLEMS_AND_SOLUTIONS.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| ") and not line.startswith(("| Problem", "|---"))]
    assert len(rows) > 40
    for row in rows:
        cites = row.rstrip("|").rsplit("|", 1)[1].strip()
        status = row.rstrip("|").rsplit("|", 2)[1].strip()
        assert status in {"fixed", "handled", "accepted", "open", "decided", "resolved"}, row
        for group in cites.split(";"):
            date, _, rest = group.strip().partition(" ")
            assert re.fullmatch(r"\d\d-\d\d", date), row
            candidates = fields.get(date, [])
            assert candidates, f"no progress entry on {date}: {row}"
            for item in rest.split(","):
                words = item.strip().split(" ")
                phase_word, stage = (words[0], " ".join(words[1:])) if words[0] in ("Embedding", "Retrieval", "Chunking", "Graph", "API", "Insufficient", "Attribution",
                                                                                   "Generation", "Augmentation", "E2E", "Deliverables") else ("", item.strip())
                assert any(phase_word in c and (not stage or stage in c) for c in candidates), f"'{item.strip()}' not found among {candidates}: {row}"


def test_the_log_cross_check_catches_a_made_up_citation():
    fields = log_fields()
    assert any("Stage 4" in c for c in fields["10-17"]) and not any("Stage 99" in c for c in fields["10-17"])
