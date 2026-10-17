import shutil

import pytest

from augmentation import config
from augmentation.prompt_loader import PromptError, get_prompts, load_prompts


def copy_prompts(tmp_path, system=None, user=None):
    for kind in ("system", "user"):
        shutil.copy(config.PROMPTS_DIR / f"answer_{kind}.v1.md", tmp_path / f"answer_{kind}.v1.md")
    if system is not None:
        (tmp_path / "answer_system.v1.md").write_text(system(config_system()), encoding="utf-8")
    if user is not None:
        (tmp_path / "answer_user.v1.md").write_text(user(config_user()), encoding="utf-8")
    return tmp_path


def config_system():
    return (config.PROMPTS_DIR / "answer_system.v1.md").read_text(encoding="utf-8")


def config_user():
    return (config.PROMPTS_DIR / "answer_user.v1.md").read_text(encoding="utf-8")


def test_real_files_load_and_version_comes_from_the_file_names():
    p = load_prompts()
    assert p.version == "v1" == config.PROMPT_VERSION
    assert p.system == config_system() and p.user_template == config_user()
    assert get_prompts() is get_prompts()                       # loaded once, cached


def test_real_system_prompt_has_the_five_sections_in_order_and_the_exact_sentence():
    text = config_system()
    heads = [l for l in text.splitlines() if l.startswith("# ")]
    assert [h.split(" —")[0] for h in heads] == list(config.SECTION_HEADINGS)
    assert config.INSUFFICIENT_CONTEXT in text.splitlines()
    assert config.B_FIRST_LINE in text and config.NOT_COVERED in text


def test_missing_section_fails(tmp_path):
    d = copy_prompts(tmp_path, system=lambda t: t.replace("# DECISION", "# CHOICES"))
    with pytest.raises(PromptError, match="missing section heading '# DECISION'"):
        load_prompts(directory=d)


def test_wrong_order_fails(tmp_path):
    def swap(t):
        return t.replace("# CONSTRAINTS", "# TMP").replace("# PURPOSE", "# CONSTRAINTS").replace("# TMP", "# PURPOSE")
    with pytest.raises(PromptError, match="out of order"):
        load_prompts(directory=copy_prompts(tmp_path, system=swap))


def test_extra_top_level_heading_fails(tmp_path):
    with pytest.raises(PromptError, match="unexpected extra"):
        load_prompts(directory=copy_prompts(tmp_path, system=lambda t: t + "\n# BONUS\n"))


def test_altered_insufficient_sentence_fails(tmp_path):
    d = copy_prompts(tmp_path, system=lambda t: t.replace("do not contain this information.", "do not contain that information."))
    with pytest.raises(PromptError, match="insufficient-context sentence"):
        load_prompts(directory=d)


def test_sentence_not_on_its_own_line_fails(tmp_path):
    d = copy_prompts(tmp_path, system=lambda t: t.replace(config.INSUFFICIENT_CONTEXT + "\n", "Reply: " + config.INSUFFICIENT_CONTEXT + "\n"))
    with pytest.raises(PromptError, match="own line"):
        load_prompts(directory=d)


@pytest.mark.parametrize("phrase", [config.B_FIRST_LINE, config.NOT_COVERED])
def test_approved_phrases_must_be_present(tmp_path, phrase):
    d = copy_prompts(tmp_path, system=lambda t: t.replace(phrase, "changed"))
    with pytest.raises(PromptError, match="approved phrase missing"):
        load_prompts(directory=d)


def test_missing_placeholder_fails(tmp_path):
    d = copy_prompts(tmp_path, user=lambda t: t.replace("{standalone_query}", "QUERY"))
    with pytest.raises(PromptError, match="missing \\['standalone_query'\\]"):
        load_prompts(directory=d)


def test_extra_or_duplicate_placeholder_fails(tmp_path):
    with pytest.raises(PromptError, match="unexpected \\['history'\\]"):
        load_prompts(directory=copy_prompts(tmp_path, user=lambda t: t + "\n{history}"))
    with pytest.raises(PromptError, match="exactly once"):
        load_prompts(directory=copy_prompts(tmp_path, user=lambda t: t + "\n{sources_block}"))


def test_question_before_sources_fails(tmp_path):
    d = copy_prompts(tmp_path, user=lambda t: "<question>{standalone_query}</question>\n<sources>\n{sources_block}\n</sources>")
    with pytest.raises(PromptError, match="sources must come before the question"):
        load_prompts(directory=d)


def test_missing_file_and_unknown_version_fail(tmp_path):
    with pytest.raises(PromptError, match="prompt file missing"):
        load_prompts(directory=tmp_path)
    with pytest.raises(PromptError, match="answer_system.v9.md"):
        load_prompts(version="v9")
