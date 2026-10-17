"""Stage R3: pure-greeting rule. No LLM. Fires only when the ENTIRE message is a greeting / thanks / goodbye."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

PHRASES_PATH = Path(__file__).with_name("greeting_phrases.yaml")


def normalize(text: str) -> str:
    """Lowercase; punctuation and emoji become spaces; 3+ repeated letters collapse ('heyyy' -> 'hey'); trim."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)       # punctuation / emoji -> space
    text = text.replace("_", " ")
    text = re.sub(r"(\w)\1{2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _pattern() -> re.Pattern:
    data = yaml.safe_load(PHRASES_PATH.read_text(encoding="utf-8"))
    phrases = sorted({p for key in ("greetings", "thanks", "goodbyes") for p in data[key]}, key=len, reverse=True)
    names = sorted(data["name_words"], key=len, reverse=True)
    alt = lambda items: "(?:" + "|".join(re.escape(i) for i in items) + ")"
    unit = rf"{alt(phrases)}(?: {alt(names)})?"
    return re.compile(rf"^{unit}(?: and)?(?: {unit})?$")


def is_pure_greeting(text: str) -> bool:
    """True only if the whole normalized message matches the approved list. Mixed messages are never caught."""
    norm = normalize(text)
    return bool(norm) and _pattern().match(norm) is not None
