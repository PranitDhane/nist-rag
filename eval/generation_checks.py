"""Pure check evaluators for eval/answer_behaviour_cases.yaml (types listed in that file's header). No LLM, no network."""
from __future__ import annotations

import re
import unicodedata

from augmentation import config as aug_config
from augmentation.parser import BULLET, VALID_MARKER, _word_count, parse_answer

PAGES = re.compile(r"\bpages?\b|\bpp?\.\s*\d", re.IGNORECASE)
CHUNK_ID = re.compile(r"\b[a-z0-9_]+:[a-z_]+:\d{4}:\d+\b|https?://|<\s*/?\s*source", re.IGNORECASE)
CASE_D = aug_config.INSUFFICIENT_CONTEXT


_QUOTED_SPAN = re.compile(r'"[^"]*"|“[^”]*”')


def _quoted(text: str, value: str) -> bool:
    """`value` appears, exactly as given, INSIDE a double-quoted span (the quote may continue beyond it)."""
    return any(value in m.group(0) for m in _QUOTED_SPAN.finditer(text))


_HYPHENS = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2212"})     # NFKC leaves these as (non-ASCII) hyphens


def normalize(text: str) -> str:
    """NFKC plus hyphen mapping: typographic variants (U+2011 non-breaking hyphen, U+202F/U+00A0 spaces) -> ASCII."""
    return unicodedata.normalize("NFKC", text).translate(_HYPHENS)


def _only_english(text: str) -> bool:
    return not any(ch.isalpha() and ord(ch) > 0x24F for ch in text)


def run_check(check: dict, text: str, key_map: dict) -> tuple[bool, str]:
    """(passed, detail) for one check against the FINAL (post-fix) text."""
    t = check["type"]
    low = text.lower()
    values = check.get("values") or []
    parsed = parse_answer(text, key_map)
    if t == "parser_ok":
        return not parsed.violations, ",".join(parsed.violations)
    if t == "exact_d":
        return text.strip() in (CASE_D, CASE_D.rstrip(".")), ""
    if t == "contains_all":
        missing = [v for v in values if v.lower() not in low]
        return not missing, f"missing {missing}" if missing else ""
    if t == "contains_any":
        return any(v.lower() in low for v in values), "" if any(v.lower() in low for v in values) else f"none of {values}"
    if t == "not_contains":
        found = [v for v in values if v.lower() in low]
        return not found, f"found {found}" if found else ""
    if t == "quote_exact":
        return _quoted(text, check["value"]), "" if _quoted(text, check["value"]) else f"not quoted exactly: {check['value'][:50]}"
    if t == "no_pages":
        return not PAGES.search(text), ""
    if t == "max_words":
        return _word_count(text) <= check["value"], f"{_word_count(text)} words"
    if t == "bullets":
        n = sum(1 for line in text.splitlines() if BULLET.match(line))
        return n >= check.get("min", 1), f"{n} bullets"
    if t == "ambiguous_format":
        return parsed.is_ambiguous_format, ""
    if t == "has_not_covered":
        return aug_config.NOT_COVERED in text, ""
    if t == "names_docs":
        missing = [v for v in values if v.lower() not in low]
        return not missing, f"missing {missing}" if missing else ""
    if t == "cites_key":
        return check["value"] in parsed.cited_keys, f"cited {parsed.cited_keys}"
    if t == "only_english":
        return _only_english(text), ""
    if t == "no_chunk_ids":
        return not CHUNK_ID.search(text), ""
    raise ValueError(f"unknown check type {t!r}")


def run_checks(checks: list[dict], text: str, key_map: dict) -> list[dict]:
    """Strict result (`passed`) plus `passed_normalized`: the same check on the NFKC-normalized text, so a failure
    caused only by typographic characters (non-breaking hyphens/spaces) is visible as such. Never loosens `passed`."""
    out = []
    for c in checks:
        ok, detail = run_check(c, text or "", key_map)
        ok_norm = ok or run_check(c, normalize(text or ""), key_map)[0]
        out.append({"check": c["type"], "args": {k: v for k, v in c.items() if k != "type"}, "passed": ok,
                    "passed_normalized": ok_norm, "detail": detail})
    return out


def actual_case(outcome: str, final_text: str | None, key_map: dict) -> str:
    """A / B / D / ERROR from a GenerationResult's outcome and final text."""
    if outcome == "ERROR":
        return "ERROR"
    if outcome == "INSUFFICIENT":
        return "D"
    return "B" if parse_answer(final_text or "", key_map).is_ambiguous_format else "A"
