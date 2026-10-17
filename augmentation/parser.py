"""Stage A4: parse_answer() - classify and check raw model output against the output contract (Section 5).

Pure and deterministic: never calls an LLM and never rewrites content, except that `clean_text` has invented
(unknown) citation keys removed.
"""
from __future__ import annotations

import re

from augmentation import config
from augmentation.models import KeyMap, ParsedAnswer

VALID_MARKER = re.compile(r"\[C(\d+)\]")
# Bracket / paren groups that LOOK like citations but are not the exact form: [C1, C3], [1], [C1 and C2], (C1) ...
BRACKET_GROUP = re.compile(r"\[([^\[\]\n]*)\]")
PAREN_CITATION = re.compile(r"\(\s*C\d+(?:\s*[,;&]\s*C\d+)*\s*\)")
CITATIONISH = re.compile(r"^\s*(?:C?\d+(?:\s*[,;&/-]\s*C?\d+)*|.*\bC\d+\b.*)\s*$")
SOURCES_LIST = re.compile(r"^[\s>*_#\-]*(?:sources?|references?)\b[\s*_]*[:\-—]", re.IGNORECASE)
URL = re.compile(r"https?://|\bwww\.", re.IGNORECASE)
HEADING = re.compile(r"^ {0,3}#{1,6}\s")
CODE_FENCE = re.compile(r"```|~~~")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"”’')\]]*\s+(?=[A-Z0-9\"“(\[*_])")
ABBREVIATIONS = re.compile(r"\b(?:e\.g|i\.e|etc|vs|cf|Figs?|U\.S|approx|Dr|Mr|Mrs|Ms|St)\.|\bno\.(?=\s*\d)", re.IGNORECASE)
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”')
PLACEHOLDER = "\x01"                      # stands in for a period that must not end a sentence


def _protect(text: str) -> str:
    """Hide periods that do not end a sentence: quoted spans and common abbreviations."""
    text = QUOTED.sub(lambda m: m.group(0).replace(".", PLACEHOLDER).replace("!", "\x02").replace("?", "\x03"), text)
    return ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", PLACEHOLDER), text)


def _restore(text: str) -> str:
    return text.replace(PLACEHOLDER, ".").replace("\x02", "!").replace("\x03", "?")


def split_units(text: str) -> list[str]:
    """Candidate fact-stating units: one per bullet line, one per sentence of a paragraph line.
    Exempt (not returned): the Case B first line, lead-in lines ending with ':', and "not covered" bullets."""
    units: list[str] = []
    lines = text.strip().splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if i == 0 and line == config.B_FIRST_LINE:
            continue
        body = BULLET.sub("", line, count=1).strip()
        if line.endswith(":"):                                  # lead-in line
            continue
        if config.NOT_COVERED in body and not VALID_MARKER.search(body):
            continue                                            # "<label>: not covered by the provided documents."
        for sentence in SENTENCE_END.split(_protect(body)):
            sentence = _restore(sentence).strip()
            if sentence and re.search(r"\w", sentence):
                units.append(sentence)
    return units


def _normalized_end(text: str) -> str:
    return text.strip()


def _is_case_d_only(stripped: str) -> bool:
    return stripped in (config.INSUFFICIENT_CONTEXT, config.INSUFFICIENT_CONTEXT.rstrip("."))


def _word_count(text: str) -> int:
    no_markers = VALID_MARKER.sub(" ", text)
    return len([w for w in no_markers.split() if re.search(r"\w", w)])


def parse_answer(text: str, key_map: KeyMap) -> ParsedAnswer:
    """Classify and check raw model output. `key_map` is the key -> chunk_id map from build_context()."""
    text = text or ""
    stripped = _normalized_end(text)
    markers = VALID_MARKER.findall(text)
    cited = list(dict.fromkeys(f"C{n}" for n in markers))
    unknown = [k for k in cited if k not in key_map]
    clean = text
    for key in unknown:
        clean = re.sub(r"[ \t]?\[" + re.escape(key) + r"\]", "", clean)
    parsed = ParsedAnswer(case="invalid", is_ambiguous_format=False, cited_keys=cited, unknown_keys=unknown,
                          word_count=_word_count(text), clean_text=clean)
    v = parsed.violations

    if not stripped:
        v.append("EMPTY")
        return parsed

    # ---- Case D ----------------------------------------------------------------------------------------------
    if _is_case_d_only(stripped):
        parsed.case = "D"
        return parsed
    if config.INSUFFICIENT_CONTEXT.rstrip(".") in stripped:      # the sentence plus other text: partial answers are refused (G-D7)
        parsed.case = "D"
        v.append("D_WITH_EXTRA_TEXT")
        return parsed

    # ---- Case A / B ------------------------------------------------------------------------------------------
    parsed.case = "A_or_B"
    lines = stripped.splitlines()
    parsed.is_ambiguous_format = lines[0].strip() == config.B_FIRST_LINE
    if unknown:
        v.append("UNKNOWN_CITATION")
    if not markers:
        v.append("NO_CITATIONS")
    malformed = [m.group(0) for m in BRACKET_GROUP.finditer(text)
                 if not VALID_MARKER.fullmatch(m.group(0)) and CITATIONISH.match(m.group(1))] + PAREN_CITATION.findall(text)
    if malformed:
        v.append("MALFORMED_CITATION")
    parsed.uncited_sentences = [u for u in split_units(text) if not VALID_MARKER.search(u)]
    if parsed.uncited_sentences:
        v.append("UNCITED_SENTENCE")
    if parsed.word_count > config.WORD_LIMIT:                   # Q1: same limit for Case B
        v.append("OVER_LENGTH")
    if any(SOURCES_LIST.match(line) for line in lines):
        v.append("HAS_SOURCES_LIST")
    if URL.search(text):
        v.append("HAS_URL")
    if any(HEADING.match(line) for line in lines) or CODE_FENCE.search(text):
        v.append("HAS_HEADING_OR_CODE")
    return parsed
