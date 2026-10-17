"""Stage S3: sources in /chat and /chat/stream, and the history rule (ATTRIBUTION_PLAN Sections 5 and 6, Q3, Q5)."""
from fastapi.testclient import TestClient

from attribution.build import build_sources, strip_markers
from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation.fixer import apply_fixes
from tests.insufficient_harness import answered_node, build_harness_graph, gate_failure_node
from tests.test_api import S, FakeClient, d
from tests.test_build_sources import meta, setup
from tests.test_insufficient_api import QUESTION, _classify, chat, names, stream
from tests.test_insufficient_reply import retrieval
from app.main import create_app

SOURCE_KEYS = {"keys", "label", "url", "kind"}


def make(rag_node, *decisions):
    client = FakeClient(*(decisions or (d("nist_question", QUESTION),)))
    graph = build_harness_graph(_classify(client), rag_node)
    return TestClient(create_app(S, graph), raise_server_exceptions=False), client


def real_node(text, metas):
    """A rag node that does what item 5 will do: fix the text, then build sources ONCE from the final text."""
    key_map, lookup = setup(*metas)

    async def node(state):
        fx = apply_fixes(text, key_map)
        sources = [e.to_dict() for e in build_sources(fx.final_text, fx.parsed.cited_keys, key_map, lookup, strict=True)]
        return {"answer": fx.final_text, "sources": sources, "outcome": "answered", "insufficient": None}
    return node


TEXT = "GOVERN is cross-cutting [C1]. It also sets policy [C3]. Culture matters [C2]."
METAS = [meta("a", seq=1), meta("b", section=("5.2 Map",), pdf=(30, 30), printed=("25", "25")), meta("c", seq=2)]   # C1 and C3 share a label


# ---- JSON and SSE carry identical sources ----------------------------------------------------------------------------
def test_chat_and_stream_carry_identical_sources_for_the_same_answer():
    c, _ = make(real_node(TEXT, METAS))
    with c:
        j = chat(c).json()
    c2, _ = make(real_node(TEXT, METAS))
    with c2:
        ev = stream(c2)
    assert names(ev) == ["meta", "token", "sources", "done"]
    assert dict(ev)["sources"] == {"sources": j["sources"]}
    assert [s["keys"] for s in j["sources"]] == [["C1", "C3"], ["C2"]]                # merged, then document/page order
    assert all(set(s) == SOURCE_KEYS for s in j["sources"])
    assert j["sources"][0]["label"] == "NIST AI 100-1, §5.1 Govern, p. 22" and j["sources"][0]["url"].endswith("#page=27")


def test_markers_stay_in_the_answer_shown_to_the_user():
    c, _ = make(real_node(TEXT, METAS))
    with c:
        j = chat(c).json()
    assert j["answer"] == TEXT


# ---- sources only when there is something to cite --------------------------------------------------------------------
def test_insufficient_response_has_no_sources_in_chat_and_stream():
    c, _ = make(gate_failure_node(retrieval()))
    with c:
        assert chat(c).json()["sources"] == []
    c2, _ = make(gate_failure_node(retrieval()))
    with c2:
        assert dict(stream(c2))["sources"] == {"sources": []}


def test_greeting_and_out_of_scope_have_empty_sources():
    c, _ = make(answered_node(), d("out_of_scope"))
    with c:
        assert chat(c, "What is the capital of France?").json()["sources"] == []


# ---- stream order and the corrected text -----------------------------------------------------------------------------
def test_sources_follow_the_correction_and_match_the_corrected_text(monkeypatch):
    from app import api
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    streamed = "GOVERN is cross-cutting [C1]. An invented claim [C9]. Culture matters [C2]."
    inner = real_node(streamed, METAS)                          # the fixer removes the invented [C9]

    async def node(state):
        await GenericFakeChatModel(messages=iter([AIMessage(content=streamed)]), tags=["answer_generation"]).ainvoke("q")
        return await inner(state)
    c, _ = make(node)
    with c:
        ev = stream(c)
    order = names(ev)
    assert order.index("correction") < order.index("sources") < order.index("done")
    corrected = dict(ev)["correction"]["text"]
    assert "[C9]" not in corrected
    keys = [k for s in dict(ev)["sources"]["sources"] for k in s["keys"]]
    assert sorted(keys) == ["C1", "C2"]


# ---- history (Q3) ----------------------------------------------------------------------------------------------------
def test_history_stores_the_answer_without_markers_while_the_response_keeps_them():
    c, client = make(real_node(TEXT, METAS), d("nist_question", QUESTION), d("nist_question", "again"))
    with c:
        j = chat(c).json()
        msgs = c.get(f"/threads/{j['thread_id']}").json()["messages"]
        assert j["answer"] == TEXT
        assert msgs[1]["content"] == "GOVERN is cross-cutting. It also sets policy. Culture matters."
        chat(c, "And what about MAP?", j["thread_id"])
    assert "[C" not in client.prompts[1] and "Assistant: GOVERN is cross-cutting." in client.prompts[1]


def test_strip_markers_rules():
    assert strip_markers("x [C1][C2].") == "x."
    assert strip_markers("a [C1] and b [C10].") == "a and b."
    assert strip_markers("- one [C1]\n- two [C2]") == "- one\n- two"
    assert strip_markers("[C1] starts a line") == "starts a line"
    assert strip_markers("No markers (C1) or [C] or [1] here.") == "No markers (C1) or [C] or [1] here."
    assert strip_markers(D) == D
