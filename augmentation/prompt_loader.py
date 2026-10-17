"""Stage A1: load the versioned prompt files once and verify them. Startup fails with a clear message if a check fails."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from augmentation import config


class PromptError(Exception):
    """A prompt file is missing or does not satisfy the contract."""


@dataclass(frozen=True)
class Prompts:
    version: str
    system: str
    user_template: str


def _read(path: Path) -> str:
    if not path.is_file():
        raise PromptError(f"prompt file missing: {path}")
    return path.read_text(encoding="utf-8")


def check_system_prompt(text: str, name: str = "system prompt") -> None:
    headings = [line.strip() for line in text.splitlines() if line.startswith("# ")]
    found = []
    for want in config.SECTION_HEADINGS:
        hit = next((h for h in headings if h.startswith(want)), None)
        if hit is None:
            raise PromptError(f"{name}: missing section heading '{want}'")
        found.append(hit)
    order = [headings.index(h) for h in found]
    if order != sorted(order):
        raise PromptError(f"{name}: the five sections are out of order; expected {', '.join(config.SECTION_HEADINGS)}")
    extra = [h for h in headings if h not in found]
    if extra:
        raise PromptError(f"{name}: unexpected extra top-level heading(s): {extra}")
    if config.INSUFFICIENT_CONTEXT not in [line.strip() for line in text.splitlines()]:
        raise PromptError(f"{name}: the exact insufficient-context sentence is not on its own line: "
                          f"{config.INSUFFICIENT_CONTEXT!r}")
    for phrase in (config.B_FIRST_LINE, config.NOT_COVERED):
        if phrase not in text:
            raise PromptError(f"{name}: approved phrase missing: {phrase!r}")


def check_user_template(text: str, name: str = "user template") -> None:
    found = set(re.findall(r"\{(\w+)\}", text))
    if found != config.USER_PLACEHOLDERS:
        missing, extra = config.USER_PLACEHOLDERS - found, found - config.USER_PLACEHOLDERS
        raise PromptError(f"{name}: placeholders must be exactly {sorted(config.USER_PLACEHOLDERS)}"
                          f" (missing {sorted(missing)}, unexpected {sorted(extra)})")
    for ph in config.USER_PLACEHOLDERS:
        if text.count("{" + ph + "}") != 1:
            raise PromptError(f"{name}: placeholder {{{ph}}} must appear exactly once")
    if text.index("{sources_block}") > text.index("{standalone_query}"):
        raise PromptError(f"{name}: the sources must come before the question")


def load_prompts(version: str | None = None, directory: Path | None = None) -> Prompts:
    version = version or config.PROMPT_VERSION
    directory = directory or config.PROMPTS_DIR
    system = _read(directory / config.SYSTEM_PROMPT_TEMPLATE.format(version=version))
    user = _read(directory / config.USER_PROMPT_TEMPLATE.format(version=version))
    check_system_prompt(system, f"answer_system.{version}.md")
    check_user_template(user, f"answer_user.{version}.md")
    return Prompts(version=version, system=system, user_template=user)


@lru_cache(maxsize=1)
def get_prompts() -> Prompts:
    """Loaded once (at startup in the generation phase) and cached."""
    return load_prompts()
