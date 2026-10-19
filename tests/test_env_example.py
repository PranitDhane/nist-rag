"""Every environment variable the code reads must be documented in .env.example (CLAUDE.md: keep it current), and the
example must never hold a real secret."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("app", "graph", "llm", "retrieval", "generation", "augmentation", "insufficient", "attribution", "ingest", "chunking", "indexing")
READ = re.compile(r"""(?:environ(?:\.get)?\(|environ\[|getenv\(|\bget\(|_path\()\s*["']([A-Z][A-Z0-9_]{3,})["']""")
SECRET_NAMES = ("GROQ_API_KEY", "LANGSMITH_API_KEY")


def variables_read() -> set[str]:
    names = set()
    for pkg in PACKAGES:
        for path in (ROOT / pkg).rglob("*.py"):
            names |= set(READ.findall(path.read_text(encoding="utf-8")))
    return names


def documented() -> set[str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]{3,})=", text, re.M))


def test_every_variable_the_code_reads_is_in_env_example():
    missing = sorted(variables_read() - documented())
    assert not missing, f"add to .env.example: {missing}"


def test_the_scanner_actually_finds_variables():
    assert {"GROQ_API_KEY", "APP_PORT", "INDEX_DEVICE", "RAW_DIR", "GEN_MODEL"} <= variables_read()


def test_env_example_has_no_secret_values_and_env_is_ignored():
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        for name in SECRET_NAMES:
            if line.startswith(name + "="):
                assert line.strip() == name + "=", f"{name} must be empty in .env.example"
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore and "!.env.example" in ignore and "data/" in ignore
