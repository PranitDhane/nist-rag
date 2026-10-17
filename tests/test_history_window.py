from langchain_core.messages import AIMessage, HumanMessage

from graph.history import build_history_window, format_history


def convo(n):
    out = []
    for i in range(1, n + 1):
        out += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}", additional_kwargs={"route": "nist_question"})]
    return out


def test_window_is_last_two_turns_oldest_first():
    assert build_history_window(convo(5), 2) == [("User", "q4"), ("Assistant", "a4"), ("User", "q5"), ("Assistant", "a5")]


def test_fewer_turns_than_window():
    assert build_history_window(convo(1), 2) == [("User", "q1"), ("Assistant", "a1")]
    assert build_history_window([], 2) == []


def test_greeting_and_out_of_scope_turns_count_toward_the_window():          # Q9
    msgs = convo(1) + [HumanMessage(content="hi"), AIMessage(content="Hello!", additional_kwargs={"route": "greeting"}),
                       HumanMessage(content="capital?"), AIMessage(content="Sorry", additional_kwargs={"route": "out_of_scope"})]
    assert [t for _, t in build_history_window(msgs, 2)] == ["hi", "Hello!", "capital?", "Sorry"]


def test_only_answer_text_is_used_not_sources_or_metadata():
    msgs = [HumanMessage(content="q"), AIMessage(content="the answer", additional_kwargs={"sources": [{"x": 1}], "route": "x"})]
    assert build_history_window(msgs, 2)[1] == ("Assistant", "the answer")


def test_unanswered_trailing_user_message_is_ignored_and_turns_zero():
    assert build_history_window(convo(1) + [HumanMessage(content="dangling")], 2) == [("User", "q1"), ("Assistant", "a1")]
    assert build_history_window(convo(2), 0) == []


def test_format():
    assert format_history([]) == "(no previous messages)"
    assert format_history([("User", "q"), ("Assistant", "a")]) == "User: q\nAssistant: a"
