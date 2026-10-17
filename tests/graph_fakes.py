"""Fake components for the graph tests (no network, no models): chunk records, retrievers, reranker, store, Components."""
from __future__ import annotations

import time

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.config import Settings
from generation.client import GenerationClient
from graph.components import Components
from indexing.metadata import to_chroma_metadata
from retrieval import config as retrieval_config
from tests.gen_fakes import FakeChat

AI = ("ai_rmf_1_0", "NIST AI 100-1 (AI RMF 1.0)", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf")
GEN = ("genai_profile", "NIST AI 600-1 (Generative AI Profile)", "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf")


def record(cid, doc=AI, *, text=None, section=("Part 2", "5 AI RMF Core"), chunk_type="prose", part=1, total=1, seq=0,
           pdf=26, printed="21", subcategory=None, **extra):
    doc_id, name, url = doc
    rec = {"chunk_id": cid, "doc_id": doc_id, "display_name": name, "chunk_type": chunk_type, "sequence": seq, "part": part,
           "parts_total": total, "section_path": list(section), "pdf_page_start": pdf, "pdf_page_end": pdf,
           "printed_page_start": printed, "printed_page_end": printed, "source_url": url, "citation_url": f"{url}#page={pdf}",
           "subcategory": subcategory, "category": None, "function": None, "figure_number": None, "footnote_number": None,
           "display_text": text or f"display {cid}", "embed_text": f"emb {cid}", "citation_label": "stored", "action_ids": [],
           "subcategories": [], "gai_risks": [], "ai_actor_tasks": [], "trustworthy_characteristics": [], "links": [],
           "merged_sections": [], "char_spans": None}
    rec.update(extra)
    return rec


def make_lookup() -> dict:
    recs = [
        record("ai:core", text="The AI RMF Core has four functions: GOVERN, MAP, MEASURE and MANAGE.", seq=1),
        record("ai:govern", text="GOVERN is a cross-cutting function.", section=("Part 2", "5 AI RMF Core", "5.1 Govern"), pdf=27, printed="22", seq=2),
        record("gen:conf", GEN, text="Confabulation is the production of false content.", section=("2 Overview", "2.2 Confabulation"), pdf=11, printed="6", seq=3),
        record("gen:t1", GEN, text="GOVERN 1.7 part one.", section=("3 Suggested Actions",), chunk_type="genai_action_table", subcategory="GOVERN 1.7", part=1, total=2, pdf=22, printed="17", seq=4),
        record("gen:t2", GEN, text="GOVERN 1.7 part two.", section=("3 Suggested Actions",), chunk_type="genai_action_table", subcategory="GOVERN 1.7", part=2, total=2, pdf=23, printed="18", seq=5),
        record("ai:other", text="Other prose.", seq=6),
    ]
    return {r["chunk_id"]: r for r in recs}


class Fixed(BaseRetriever):
    docs: list[Document]
    boom: bool = False
    delay: float = 0.0
    starts: list = Field(default_factory=list)          # time.monotonic() of every call (concurrency tests)

    def _get_relevant_documents(self, query, *, run_manager=None):
        self.starts.append(time.monotonic())
        if self.delay:
            time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("SECRET vector store failure")
        return list(self.docs)


def doc_for(rec):
    return Document(page_content=rec["embed_text"], metadata=to_chroma_metadata(rec), id=rec["chunk_id"])


class FakeStore:
    def __init__(self, lookup, boom=False):
        self.lookup, self.boom = lookup, boom

    def get(self, where, include=None):
        if self.boom:
            raise RuntimeError("SECRET chroma failure")
        conds = where["$and"]
        hit = [r for r in self.lookup.values() if all(to_chroma_metadata(r).get(k) == v for c in conds for k, v in c.items())]
        return {"ids": [r["chunk_id"] for r in hit], "metadatas": [to_chroma_metadata(r) for r in hit],
                "documents": [r["embed_text"] for r in hit]}


class FakeReranker:
    """predict() returns scores looked up by the chunk text `emb <chunk_id>`; counts calls; can fail or sleep."""

    def __init__(self, scores, boom=False, delay=0.0):
        self.scores, self.boom, self.delay, self.calls = scores, boom, delay, 0

    def predict(self, pairs, activation_fn=None, show_progress_bar=False):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("SECRET reranker failure")
        return [self.scores.get(t.replace("emb ", ""), 0.01) for _, t in pairs]


class FakeRetrieval:
    def __init__(self, lookup, vector_ids, bm25_ids, *, boom_vector=False, store_boom=False, delay=0.0):
        self.chunk_lookup = lookup
        self.vector = Fixed(docs=[doc_for(lookup[i]) for i in vector_ids], boom=boom_vector, delay=delay)
        self.bm25 = Fixed(docs=[doc_for(lookup[i]) for i in bm25_ids])
        self.ensemble = EnsembleRetriever(retrievers=[self.vector, self.bm25], weights=list(retrieval_config.WEIGHTS),
                                          c=retrieval_config.RRF_C, id_key=retrieval_config.ID_KEY)
        self.store = FakeStore(lookup, boom=store_boom)


SETTINGS = Settings(groq_api_key="x", llm_max_attempts=2, llm_base_delay=0.0, llm_max_delay=1.0)
GOOD = "GOVERN is the cross-cutting function [C1]. The Core has four functions [C2]."


async def _nosleep(_):
    pass


def make_components(*, scores=None, chat=None, vector_ids=("ai:govern", "ai:core", "gen:conf", "ai:other"),
                    bm25_ids=("ai:core", "ai:govern"), boom_vector=False, store_boom=False, reranker_boom=False, delay=0.0,
                    rerank_delay=0.0) -> Components:
    lookup = make_lookup()
    scores = scores if scores is not None else {"ai:govern": 0.95, "ai:core": 0.90, "gen:conf": 0.3, "ai:other": 0.2}
    client = GenerationClient(SETTINGS, chat_model=chat or FakeChat(chunks=[GOOD]), sleep=_nosleep, max_tokens=2000)
    return Components(retrieval=FakeRetrieval(lookup, list(vector_ids), list(bm25_ids), boom_vector=boom_vector,
                                              store_boom=store_boom, delay=delay),
                      reranker=FakeReranker(scores, boom=reranker_boom, delay=rerank_delay), generation_client=client,
                      settings=SETTINGS)


# ------------------------------------------------------------------ a real LangChain chat model for streaming tests
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

from tests.gen_fakes import USAGE  # noqa: E402


class ScriptedChat(BaseChatModel):
    """Emits `script` through the LangChain callback machinery (so LangGraph's messages stream sees the tokens).
    Items: a string (content), ("reasoning", text) (reasoning chunk), or ("fail",) (raise after what came before)."""
    script: list
    fail_with: Exception | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = "".join(i for i in self.script if isinstance(i, str))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=USAGE,
                                                                         response_metadata={"finish_reason": "stop"}))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for item in self.script:
            if isinstance(item, tuple) and item[0] == "fail":
                raise self.fail_with
            if isinstance(item, tuple):
                chunk = AIMessageChunk(content="", additional_kwargs={"reasoning_content": item[1]})
            else:
                chunk = AIMessageChunk(content=item)
            cg = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, chunk=cg)
            yield cg
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=USAGE,
                                                         response_metadata={"finish_reason": "stop"}))
