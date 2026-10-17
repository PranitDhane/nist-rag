"""Stage N3: generate_stream() with a fake streaming LLM (no network)."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from app.config import Settings
from augmentation.config import INSUFFICIENT_CONTEXT as D
from generation import config
from generation.client import GenerationClient
from generation.generate import generate_stream
from generation.models import ANSWERED, INSUFFICIENT, GenerationError
from tests.gen_fakes import FakeChat, result, status_error

S = Settings(groq_api_key="x", llm_max_attempts=3, llm_base_delay=0.0, llm_max_delay=1.0)
GOOD_PARTS = ["GOVERN is the ", "cross-cutting function ", "[C2]. The Core ", "has four functions [C1]."]


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(config, "MAX_COMPLETION_TOKENS", 2000)
    monkeypatch.setattr(config, "LENGTH_POLICY", "insufficient")


async def _nosleep(_):
    pass


def events(fake, res=None, **client_kw):
    client = GenerationClient(S, chat_model=fake, sleep=_nosleep, max_tokens=2000, **client_kw)

    async def collect():
        return [e async for e in generate_stream("Which function is cross-cutting?", res or result(), client=client)]
    return asyncio.run(collect())


def kinds(evs):
    return [e["type"] for e in evs]


def test_normal_stream_has_no_correction():
    evs = events(FakeChat(chunks=GOOD_PARTS))
    assert kinds(evs) == ["token"] * 4 + ["result"]
    r = evs[-1]["result"]
    assert r.outcome == ANSWERED and r.corrected is False and r.streamed_text == r.final_text == "".join(GOOD_PARTS)
    assert r.usage["reasoning"] == 220 and r.timings_ms["ttft"] is not None and r.finish_reason == "stop"


def test_invented_citation_is_removed_by_a_correction_event():
    evs = events(FakeChat(chunks=["GOVERN is cross-cutting [C2]", "[C9]. Four functions [C1]."]))
    assert kinds(evs) == ["token", "token", "correction", "result"]
    corr, r = evs[2], evs[3]["result"]
    assert corr["text"] == "GOVERN is cross-cutting [C2]. Four functions [C1]." and corr["reasons"] == ["UNKNOWN_CITATION"]
    assert r.corrected and r.streamed_text == "GOVERN is cross-cutting [C2][C9]. Four functions [C1]."
    assert r.final_text == corr["text"]


def test_d_with_extra_text_corrected_to_the_case_d_sentence():
    evs = events(FakeChat(chunks=["GOVERN is cross-cutting [C2]. ", D]))
    corr = next(e for e in evs if e["type"] == "correction")
    assert corr["text"] == D and corr["reasons"] == ["D_WITH_EXTRA_TEXT"]
    assert evs[-1]["result"].outcome == INSUFFICIENT


def test_reasoning_chunks_never_reach_tokens_or_streamed_text():
    chunks = [("reasoning", "SECRET THINKING"), "GOVERN is ", ("reasoning", "more SECRET"), "cross-cutting [C2]. ",
              "Four functions [C1]."]
    evs = events(FakeChat(chunks=chunks))
    text = "".join(e["text"] for e in evs if e["type"] == "token")
    assert "SECRET" not in text and text == "GOVERN is cross-cutting [C2]. Four functions [C1]."
    r = evs[-1]["result"]
    assert "SECRET" not in r.streamed_text and "SECRET" not in repr(r.to_dict())


def test_chunk_with_content_and_reasoning_field_is_not_streamed():
    mixed = AIMessageChunk(content="LEAK", additional_kwargs={"reasoning_content": "x"})
    evs = events(FakeChat(chunks=[mixed, "GOVERN [C2]."]))
    assert "".join(e["text"] for e in evs if e["type"] == "token") == "GOVERN [C2]."


def test_429_before_first_token_is_retried_then_succeeds():
    fake = FakeChat(chunks=GOOD_PARTS, errors=[status_error(429)])
    evs = events(fake)
    assert fake.calls == 2 and kinds(evs)[-1] == "result" and evs[-1]["result"].outcome == ANSWERED


def test_retries_exhausted_before_first_token_is_an_error_event():
    evs = events(FakeChat(chunks=GOOD_PARTS, errors=[status_error(429)] * 5))
    assert kinds(evs) == ["error"] and evs[0]["code"] == "rate_limit"


def test_failure_after_three_tokens_is_an_error_event_without_correction_or_result():
    fake = FakeChat(chunks=GOOD_PARTS + ["more"], fail_after=3)
    evs = events(fake)
    assert kinds(evs) == ["token"] * 3 + ["error"] and evs[-1]["code"] == "midstream"
    assert fake.calls == 1                                    # never retried once a token was sent
    assert "correction" not in kinds(evs) and "result" not in kinds(evs)


def test_ttft_timeout_is_an_error_event():
    evs = events(FakeChat(chunks=GOOD_PARTS, hang=True), ttft_timeout=0.05)
    assert kinds(evs) == ["error"] and evs[0]["code"] == "ttft_timeout"


def test_total_timeout_mid_stream_is_an_error_event():
    class Slow(FakeChat):
        async def astream(self, messages):
            yield AIMessageChunk(content="GOVERN ")
            await asyncio.sleep(1)
            yield AIMessageChunk(content="[C2].")
    evs = events(Slow(), total_timeout=0.1)
    assert kinds(evs) == ["token", "error"] and evs[-1]["code"] == "midstream"


def test_empty_context_raises_and_length_policy_error_becomes_error_event(monkeypatch):
    with pytest.raises(GenerationError):
        events(FakeChat(chunks=GOOD_PARTS), result(n=0))
    monkeypatch.setattr(config, "LENGTH_POLICY", "error")
    evs = events(FakeChat(chunks=["GOVERN [C2]."], finish="length"))
    assert kinds(evs) == ["token", "error"] and evs[-1]["code"] == "length"


def test_only_the_final_text_is_what_history_should_store():
    evs = events(FakeChat(chunks=["GOVERN [C2][C9]."]))
    r = evs[-1]["result"]
    assert r.final_text == "GOVERN [C2]." and r.streamed_text != r.final_text
