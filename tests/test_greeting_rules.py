import pytest

from graph.greeting_rules import is_pure_greeting, normalize

POSITIVES = ["hi", "Hello!", "hey there", "Good morning", "thanks!", "Thank you so much", "bye",
             "HELLO", "  hi   ", "Hi Claude!", "hey bot", "heyyy", "thx", "Thanks 🙏", "hi 👋", "hello, thanks", "thanks, bye!",
             "good evening", "See you later.", "hi again!", "thank you very much!!!", "Cheers", "good night", "hello!!!"]
NEGATIVES = ["Hi, what is GOVERN?", "Hi again! Which one is cross-cutting?", "hello, explain confabulation",
             "thanks, and what about MAP?", "hey can you list the 12 risks", "hello world program in python", "good",
             "thanks for nothing, what is NIST?", "", "   ", "?", "hi hi hi hi", "hello what", "what is GOVERN 1.7",
             "thanks for the help", "bye the way what is MAP", "hi, I have a question", "How are you?", "hey claude what is the AI RMF"]


@pytest.mark.parametrize("text", POSITIVES)
def test_pure_greetings_are_caught(text):
    assert is_pure_greeting(text) is True, text


@pytest.mark.parametrize("text", NEGATIVES)
def test_mixed_or_unlisted_messages_are_never_caught(text):
    assert is_pure_greeting(text) is False, text


def test_normalization():
    assert normalize("  Hello!!  ") == "hello" and normalize("Heyyyy") == "hey" and normalize("Thanks 🙏") == "thanks"
    assert normalize("hello") == "hello"                               # two repeated letters are kept


def test_list_comes_from_the_data_file_not_the_code():
    from graph.greeting_rules import PHRASES_PATH
    assert PHRASES_PATH.name == "greeting_phrases.yaml" and PHRASES_PATH.is_file()
