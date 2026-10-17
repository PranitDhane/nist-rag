"""Constants of the augmentation phase (AUGMENTATION_PLAN Sections 2, 3, 5)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROMPT_VERSION = "v1"                                  # selects prompts/answer_{system,user}.<version>.md
SYSTEM_PROMPT_TEMPLATE = "answer_system.{version}.md"
USER_PROMPT_TEMPLATE = "answer_user.{version}.md"

# The five system-prompt sections, in order (G-D1). A heading line must START with these.
SECTION_HEADINGS = ("# PURPOSE", "# CONSTRAINTS", "# INTERPRETATION", "# DECISION", "# OUTPUT FORMAT")
USER_PLACEHOLDERS = frozenset({"sources_block", "standalone_query"})

# Contract strings (G-D8 and Q6, approved). The Case D sentence and the "not covered" phrase are deliberately different.
INSUFFICIENT_CONTEXT = "The NIST documents provided do not contain this information."
B_FIRST_LINE = "This question can be read in more than one way:"
NOT_COVERED = "not covered by the provided documents"

WORD_LIMIT = 150                                       # G-D11; the limit for ambiguous answers is Q1 (Stage A4)
