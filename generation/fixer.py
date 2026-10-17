"""Stage N2: apply_fixes() - the violation policy of GENERATION_PLAN Section 5, as a pure function.

The fixer never adds content and never rewords sentences: it removes or normalizes markup, or replaces the whole
text with the exact Case D sentence. Fixes are applied in the plan's order, then the text is parsed again."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from augmentation import config as aug_config
from augmentation.models import KeyMap, ParsedAnswer
from augmentation.parser import CODE_FENCE, HEADING, SOURCES_LIST, parse_answer
from generation.models import ANSWERED, INSUFFICIENT

CASE_D = aug_config.INSUFFICIENT_CONTEXT

_BRACKET_LIST = re.compile(r"\[\s*(C\d+(?:\s*[,;&]\s*C\d+)+)\s*\]")           # [C1, C3] [C1,C3]
_PAREN_LIST = re.compile(r"\(\s*(C\d+(?:\s*[,;&]\s*C\d+)*)\s*\)")              # (C1)  (C1, C2)
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\(\s*(?:https?://|www\.)[^)\s]*\s*\)")      # [text](http://...) -> text
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\]\)]+", re.IGNORECASE)
_FENCE_LINE = re.compile(r"^\s*(?:```|~~~)[^\n]*$")


@dataclass
class FixResult:
    final_text: str
    outcome: str                                 # ANSWERED | INSUFFICIENT
    fixes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parsed: ParsedAnswer | None = None


def _keys(group: str) -> str:
    return "".join(f"[{k}]" for k in re.findall(r"C\d+", group))


def _normalize_citations(text: str) -> str:
    text = _BRACKET_LIST.sub(lambda m: _keys(m.group(1)), text)
    return _PAREN_LIST.sub(lambda m: _keys(m.group(1)), text)


def _remove_sources_list(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if SOURCES_LIST.match(line):
            return "\n".join(lines[:i]).rstrip()
    return text


def _remove_urls(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = re.sub(r"\(\s*\)|(?<!\[C\d)\[\s*\]", "", text)                       # empty (), [] left behind
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", text)


def _remove_heading_and_code(text: str) -> str:
    out = []
    for line in text.splitlines():
        if _FENCE_LINE.match(line):
            continue
        out.append(re.sub(r"^ {0,3}#{1,6}\s+", "", line) if HEADING.match(line) else line)
    return "\n".join(out)


def apply_fixes(raw_text: str, key_map: KeyMap) -> FixResult:
    parsed = parse_answer(raw_text, key_map)
    fixes: list[str] = []

    def insufficient(code: str) -> FixResult:
        fixes.append(code)
        return FixResult(CASE_D, INSUFFICIENT, fixes, [], parse_answer(CASE_D, key_map))

    if "D_WITH_EXTRA_TEXT" in parsed.violations:
        return insufficient("D_WITH_EXTRA_TEXT")
    if "EMPTY" in parsed.violations:
        return insufficient("EMPTY")
    if parsed.case == "D":                                    # exactly the Case D sentence: nothing to fix
        return FixResult(raw_text.strip(), INSUFFICIENT, [], [], parsed)

    text = raw_text.strip()
    if "MALFORMED_CITATION" in parsed.violations:
        new = _normalize_citations(text)
        if new != text:
            fixes.append("MALFORMED_CITATION")
            text = new
    if "UNKNOWN_CITATION" in parse_answer(text, key_map).violations:
        text = parse_answer(text, key_map).clean_text         # invented markers removed, nothing else changes
        fixes.append("UNKNOWN_CITATION")
    now = parse_answer(text, key_map)
    if "HAS_SOURCES_LIST" in now.violations:
        text = _remove_sources_list(text)
        fixes.append("HAS_SOURCES_LIST")
    if "HAS_URL" in parse_answer(text, key_map).violations:
        text = _remove_urls(text)
        fixes.append("HAS_URL")
    if "HAS_HEADING_OR_CODE" in parse_answer(text, key_map).violations:
        text = _remove_heading_and_code(text)
        fixes.append("HAS_HEADING_OR_CODE")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()

    final = parse_answer(text, key_map)
    if "EMPTY" in final.violations:
        return insufficient("EMPTY")
    if final.case == "D":                                     # a fix exposed a bare Case D sentence
        return FixResult(text, INSUFFICIENT, fixes, [], final)
    if "NO_CITATIONS" in final.violations:                    # re-checked AFTER the fixes (e.g. all citations invented)
        return insufficient("NO_CITATIONS")
    warnings = [v for v in final.violations if v in ("UNCITED_SENTENCE", "OVER_LENGTH")]
    if "MALFORMED_CITATION" in final.violations:
        warnings.append("MALFORMED_CITATION_UNFIXED")         # unrecognised form: left untouched
    return FixResult(text, ANSWERED, fixes, warnings, final)
