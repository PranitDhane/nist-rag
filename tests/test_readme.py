"""Stage D5: README.md describes the system as built (deliverables rules 1, 2 and 5)."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DESIGN = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
spec = importlib.util.spec_from_file_location("build_results", ROOT / "docs" / "build_results.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation removed, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def prose(text: str) -> str:
    text = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", "", text, flags=re.S)
    text = re.sub(r"<!-- TIMINGS:START -->.*?<!-- TIMINGS:END -->", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^#+ .*$", "", text, flags=re.M)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------- generated parts
def test_markers_exist_exactly_once():
    for marker in (br.START, br.END, br.T_START, br.T_END):
        assert README.count(marker) == 1


def test_timings_block_is_current():
    import json
    acc = json.loads((ROOT / br.ACCEPTANCE[0]).read_text()) if (ROOT / br.ACCEPTANCE[0]).is_file() else None
    assert br.replace_block(README, br.T_START, br.T_END, br.timings_block(acc)) == README, "README timings block is out of date: run `make results`"


@pytest.mark.skipif(not (ROOT / "eval/retrieval_report.json").is_file(), reason="needs the committed reports")
def test_results_block_is_current():
    _, block = br.build(ROOT)
    assert br.update_readme(README, block) == README, "README results block is out of date: run `make results`"


# ---------------------------------------------------------------- numbers
ALLOWED = {
    "1", "2", "3", "4", "5", "6", "7", "8", "10", "21", "22", "5.1", "1.7", "001", "100", "600", "1.0", "20", "120", "256",   # structure, identifiers, example labels
    "3.12", "127.0.0.1", "8000", "2000", "20.0", "30.0", "32", "60", "512",                                                       # parameters, compared with the code below
    "404", "409", "422", "500", "503",                                                                                           # HTTP statuses, compared below
}
NUMBER = re.compile(r"(?<![\w.])\d[\d,.]*\d|\b\d\b")


def test_prose_contains_no_typed_metrics():
    found = {m.group(0) for m in NUMBER.finditer(prose(README))}
    assert found <= ALLOWED, f"numbers in README prose that are not parameters or identifiers: {sorted(found - ALLOWED)}; generated results belong in the RESULTS block"


def test_config_defaults_in_the_readme_match_the_code():
    from app.config import Settings
    from generation import config as gc
    from indexing import config as ic
    s = Settings()
    assert (s.host, s.port, s.max_message_chars, s.history_turns, s.llm_max_attempts, s.llm_base_delay, s.llm_max_delay, s.llm_timeout) == \
           ("127.0.0.1", 8000, 2000, 2, 4, 1.0, 20.0, 30.0)
    assert s.router_model == "openai/gpt-oss-120b" and s.router_reasoning_effort == "low" and gc.GEN_MODEL == "openai/gpt-oss-20b"
    assert (ic.BATCH_SIZE, ic.COLLECTION_NAME) == (32, "nist_ai_rmf")
    for text in ("`openai/gpt-oss-120b`", "`openai/gpt-oss-20b`", "`HISTORY_TURNS` | `2`", "`127.0.0.1`, `8000`", "`2000`", "`4`, `1.0`, `20.0`, `30.0`", "`32`, `nist_ai_rmf`"):
        assert text in README, text
    assert "**Python 3.12**" in README


def test_every_variable_in_the_config_table_is_in_env_example():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    table = README.split("## Configuration")[1].split("## Downloading")[0]
    names = set(re.findall(r"`([A-Z][A-Z0-9_]{3,})`", table))
    assert len(names) > 15
    for name in names:
        assert re.search(rf"^#?\s*{name}=", env, re.M), f"{name} is in the README table but not in .env.example"


# ---------------------------------------------------------------- commands
def make_targets() -> set[str]:
    return set(re.findall(r"^([a-z][a-z0-9-]*):", MAKEFILE, re.M))


def test_every_make_target_the_readme_mentions_exists():
    mentioned = set(re.findall(r"`make ([a-z][a-z0-9-]*)", README)) | set(re.findall(r"^make ([a-z][a-z0-9-]*)", README, re.M))
    assert mentioned and mentioned <= make_targets(), mentioned - make_targets()


def test_the_python_command_table_covers_every_documented_target():
    table = README.split("## Make targets and their Python commands")[1].split("## Running the application")[0]
    documented = set(re.findall(r"`make ([a-z][a-z0-9-]*)", table))
    for target in ("install", "download", "ingest", "chunk", "index", "build", "rebuild", "serve", "test", "eval-routing", "eval-retrieval", "e2e", "results", "clean"):
        assert target in documented, target
    assert documented <= make_targets()


def test_every_python_module_in_a_command_exists():
    for module in set(re.findall(r"python3? -m ([a-z0-9_.]+)", README)):
        if module in ("pip", "venv", "pytest", "json.tool"):
            continue
        assert (ROOT / (module.replace(".", "/") + ".py")).exists() or (ROOT / module.replace(".", "/") / "__main__.py").exists(), module
    assert (ROOT / "docs" / "build_results.py").exists() and (ROOT / "testing_curl.sh").exists()


def test_python_command_table_matches_the_makefile():
    for fragment in ("-m ingest --only-stage 2", "-m chunking", "-m indexing", "-m app.main", "-m eval.routing", "-m eval.retrieval_eval", "-m eval.e2e_report", "docs/build_results.py"):
        assert fragment in MAKEFILE, fragment                      # the Makefile runs it ...
        assert fragment.replace("-m ", "python -m ") in README or fragment in README, fragment   # ... and the README documents the same command


# ---------------------------------------------------------------- API facts
def test_documented_endpoints_errors_and_events_exist_in_the_code():
    api = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
    for route in ('"/chat"', '"/chat/stream"', '"/health"', '"/threads/{thread_id}"'):
        assert route in api, route
    for code in ("thread_not_found", "thread_busy", "llm_unavailable"):
        assert code in api, code
        assert code in README
    for event in ("meta", "token", "correction", "sources", "related", "done", "error"):
        assert f'format_event("{event}"' in api, event
    assert "`POST /chat/stream`" in README and "`DELETE /threads/{thread_id}`" in README


def test_response_fields_in_the_readme_exist_in_the_schema():
    schemas = (ROOT / "app" / "schemas.py").read_text(encoding="utf-8")
    for field in ("thread_id", "route", "answer", "sources", "standalone_query", "related_sections", "meta", "outcome", "insufficient_reason", "llm_calls", "latency_ms", "usage"):
        assert field in schemas, field
    for kind in ("prose", "table", "figure-ocr", "footnote"):
        assert kind in schemas and f"`{kind}`" in README


def test_fixed_texts_in_the_readme_are_the_ones_in_the_code():
    from augmentation import config as ac
    from graph.replies import OUT_OF_SCOPE_REPLY
    assert OUT_OF_SCOPE_REPLY in README and ac.INSUFFICIENT_CONTEXT in README


# ---------------------------------------------------------------- examples
def test_example_questions_exist_in_the_case_file():
    import yaml
    data = yaml.safe_load((ROOT / "eval" / "e2e_cases.yaml").read_text(encoding="utf-8"))
    messages = {t["message"] for t in data["single_turn"]} | {t["message"] for c in data["conversations"] for t in c["turns"]}
    for question in ("What are the four core functions of the NIST AI RMF?", "What are the four functions of the AI RMF?", "Which one is cross-cutting?",
                     "What categories does it include?", "What does the Generative AI Profile suggest for decommissioning GAI systems?",
                     "What are the four functions of the AI RMF, and how much does an AI RMF certification cost?",
                     "What is the capital of France?", "Hello!", "What is suggested action GV-1.7-001?"):
        assert question in messages, question


def test_the_example_response_matches_a_real_saved_answer():
    import json
    runs = sorted((ROOT / "eval" / "e2e_runs").glob("*/raw/S1.json"))
    if not runs:
        pytest.skip("no saved end-to-end run")
    body = json.loads(runs[-1].read_text())["body"]
    assert body["answer"] in README
    for s in body["sources"]:
        assert s["label"] in README and s["url"] in README


# ---------------------------------------------------------------- diagrams and links
def test_diagrams_are_the_ones_in_docs_graph_md():
    graph = (ROOT / "docs" / "graph.md").read_text(encoding="utf-8")
    for block in re.findall(r"```mermaid\n(.*?)```", graph, re.S):
        assert block in README, "README diagram differs from docs/graph.md: regenerate docs/graph.md and copy the blocks"


def test_design_links_point_at_existing_sections():
    anchors = {slug(h) for h in re.findall(r"^#{2,3} (.+)$", DESIGN, re.M)}
    links = re.findall(r"\]\(DESIGN\.md#([^)]+)\)", README)
    assert links
    for anchor in links:
        assert anchor in anchors, f"unknown DESIGN.md anchor: {anchor}"


def test_relative_file_links_exist():
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", README):
        if target.startswith(("http", "mailto")):
            continue
        assert (ROOT / target).exists(), f"broken link: {target}"
    results = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    heads = {slug(h) for h in re.findall(r"^## (.+)$", results, re.M)}
    for anchor in re.findall(r"RESULTS\.md#([^)]+)\)", README):
        assert anchor in heads, anchor


def test_required_sections_are_present_in_the_required_order():
    titles = re.findall(r"^## (.+)$", README, re.M)
    required = ["Quickstart", "Requirements", "Installation", "Configuration", "Downloading and processing the documents", "Running the application",
                "Example questions", "How it works", "Important design decisions", "Results", "Known limitations", "What I'd do with more time",
                "Project structure", "Testing and evaluation", "Troubleshooting", "Development process"]
    positions = [titles.index(t) for t in required]
    assert positions == sorted(positions)
    assert README.startswith("# NIST AI RMF Assistant")
