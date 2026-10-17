"""docs/graph.md must contain the Mermaid exported from the REAL graphs (so the diagrams cannot drift)."""
from __future__ import annotations

from pathlib import Path

from app.config import Settings
from graph.build import build_app_graph
from graph.rag_subgraph import build_rag_subgraph
from tests.graph_fakes import make_components
from tests.test_main_graph import RouterClient

DOC = Path(__file__).resolve().parent.parent / "docs" / "graph.md"


def mermaid_blocks() -> tuple[str, str]:
    comps = make_components()
    main = build_app_graph(Settings(groq_api_key="x"), RouterClient(), components=comps).get_graph().draw_mermaid()
    expanded = build_app_graph(Settings(groq_api_key="x"), RouterClient(), components=comps).get_graph(xray=1).draw_mermaid()
    return main.strip(), expanded.strip()


def test_graph_doc_contains_the_current_diagrams():
    text = DOC.read_text(encoding="utf-8")
    main, expanded = mermaid_blocks()
    assert main in text, "docs/graph.md main-graph diagram is stale; regenerate it (python -m tests.test_graph_docs)"
    assert expanded in text, "docs/graph.md expanded diagram is stale; regenerate it (python -m tests.test_graph_docs)"
    assert "rag_stub" not in text


if __name__ == "__main__":        # regenerate the two blocks and print them
    main, expanded = mermaid_blocks()
    print("MAIN\n" + main + "\n\nEXPANDED\n" + expanded)
