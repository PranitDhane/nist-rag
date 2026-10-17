"""Stage N2: generate() with a mocked LLM (no network)."""
from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from augmentation.config import INSUFFICIENT_CONTEXT as D, PROMPT_VERSION
from generation import config
from generation.client import GenerationClient
from generation.generate import estimate_tokens, generate, precheck_budget
from generation.models import ANSWERED, ERROR, INSUFFICIENT, GenerationError
from tests.gen_fakes import FakeChat, result, status_error

S = Settings(groq_api_key="x", llm_max_attempts=2, llm_base_delay=0.0, llm_max_delay=1.0)
GOOD = "GOVERN is the cross-cutting function [C2]. The Core has four functions [C1]."


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(config, "LENGTH_POLICY", "insufficient")


async def _nosleep(_):
    pass


def run(fake, res=None, **kw):
    client = GenerationClient(S, chat_model=fake, sleep=_nosleep, max_tokens=2000)
    return asyncio.run(generate("Which function is cross-cutting?", res or result(), client=client, **kw))


def test_answered_result_fields():
    r = run(FakeChat(text=GOOD))
    assert r.outcome == ANSWERED and r.final_text == GOOD and r.cited_keys == ["C2", "C1"]
    assert r.key_map == {"C1": "ai:1", "C2": "ai:2", "C3": "gen:1"}
    assert r.usage == {"prompt": 1200, "completion": 300, "reasoning": 220, "total": 1500}
    assert r.finish_reason == "stop" and r.corrected is False and r.streamed_text is None
    assert (r.model, r.prompt_version, r.reasoning_effort, r.temperature) == (
        "openai/gpt-oss-20b", PROMPT_VERSION, "medium", 0.2)
    assert r.error is None and r.timings_ms["total"] >= 0


def test_exact_case_d_from_the_model():
    r = run(FakeChat(text=D))
    assert r.outcome == INSUFFICIENT and r.fixes == [] and r.final_text == D


def test_fixes_are_applied_and_recorded():
    r = run(FakeChat(text="GOVERN is cross-cutting [C2, C1] [C9]."))
    assert r.outcome == ANSWERED and r.final_text == "GOVERN is cross-cutting [C2][C1]." and \
        r.fixes == ["MALFORMED_CITATION", "UNKNOWN_CITATION"]


def test_all_citations_invented_becomes_insufficient():
    r = run(FakeChat(text="GOVERN is cross-cutting [C7]."))
    assert r.outcome == INSUFFICIENT and r.final_text == D


def test_empty_context_raises():
    with pytest.raises(GenerationError) as e:
        run(FakeChat(text=GOOD), result(n=0))
    assert e.value.code == "empty_context"


def test_prompt_has_no_history_and_two_messages():
    fake = FakeChat(text=GOOD)
    run(fake)
    assert len(fake.last_messages) == 2


def test_budget_precheck_fails_fast_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(config, "TPM_LIMIT", 500)
    fake = FakeChat(text=GOOD)
    r = run(fake)
    assert r.outcome == ERROR and r.error["code"] == "budget" and fake.calls == 0 and r.final_text is None


def test_precheck_arithmetic():
    from tests.gen_fakes import chunk  # noqa: F401
    msgs = [type("M", (), {"content": "x" * 4000})()]
    expected = round(4000 / config.CHARS_PER_TOKEN * config.ESTIMATE_MARGIN)
    assert estimate_tokens(msgs) == expected == 1000                 # 4,000 chars at 4 chars/token, margin 1.0
    assert precheck_budget(msgs, 6000, tpm_limit=8000) == expected
    with pytest.raises(GenerationError):
        precheck_budget(msgs, 7100, tpm_limit=8000)                  # 1,000 + 7,100 > 8,000


def test_missing_output_limit_refuses_live_run_but_measurement_mode_works(monkeypatch):
    monkeypatch.setattr(config, "MAX_COMPLETION_TOKENS", None)
    client = GenerationClient(S, chat_model=FakeChat(text=GOOD), sleep=_nosleep)
    with pytest.raises(GenerationError) as e:
        asyncio.run(generate("q", result(), client=client))
    assert e.value.code == "config"
    r = asyncio.run(generate("q", result(), client=client, measurement_mode=True))
    assert r.outcome == ANSWERED


@pytest.mark.parametrize("policy,outcome,code", [("insufficient", INSUFFICIENT, None), ("keep", ANSWERED, None),
                                                 ("error", ERROR, "length")])
def test_finish_reason_length_policy(monkeypatch, policy, outcome, code):
    monkeypatch.setattr(config, "LENGTH_POLICY", policy)
    r = run(FakeChat(text="GOVERN is cross-cutting [C2].", finish="length"))
    assert r.outcome == outcome and (r.error or {}).get("code") == code
    if policy == "keep":
        assert "TRUNCATED" in r.warnings
    if policy == "insufficient":
        assert r.final_text == D and "LENGTH" in r.fixes


def test_rate_limit_retried_then_succeeds_and_exhaustion_is_an_error_result():
    fake = FakeChat(text=GOOD, errors=[status_error(429)])
    assert run(fake).outcome == ANSWERED and fake.calls == 2
    r = run(FakeChat(text=GOOD, errors=[status_error(429)] * 5))
    assert r.outcome == ERROR and r.error["code"] == "rate_limit"
