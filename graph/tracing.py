"""Optional LangSmith tracing (GRAPH_INTEGRATION_PLAN Stage G7, GI-D5).

OFF by default and enabled ONLY by environment variables (checked by the installed `langsmith`, 0.14.11):
    LANGSMITH_TRACING=true          (older names LANGSMITH_TRACING_V2 / LANGCHAIN_TRACING_V2 also work)
    LANGSMITH_API_KEY=<key>         LANGSMITH_PROJECT=nist-rag        LANGSMITH_ENDPOINT (optional)
When on, prompts and retrieved NIST text are sent to LangSmith's servers. When off, nothing here creates a LangSmith
client or makes a network call, and run configs are exactly what they were before tracing existed."""
from __future__ import annotations

from functools import lru_cache

TRACE_TAG = "nist-rag"


def tracing_enabled() -> bool:
    """Same check LangChain itself uses (reads the env vars at call time; no client is created)."""
    from langsmith.utils import get_env_var, tracing_is_enabled
    get_env_var.cache_clear()                      # the library memoizes env lookups; honour changes made after import
    return bool(tracing_is_enabled())


@lru_cache(maxsize=1)
def _versions() -> dict:
    from augmentation.config import PROMPT_VERSION
    from graph.debug import _indexing_version
    from retrieval import config as retrieval_config
    return {"prompt_version": PROMPT_VERSION, "retrieval_config_version": retrieval_config.config_version(),
            "chunks_sha256": _indexing_version().get("chunks_file_sha256")}


def run_config(thread_id: str) -> dict:
    """The config for one graph run. With tracing off: just the thread id. With tracing on: also the agreed tags and
    metadata (thread_id, prompt_version, retrieval config hash, index version) so a trace can be tied to the exact setup."""
    config: dict = {"configurable": {"thread_id": thread_id}}
    if tracing_enabled():
        v = _versions()
        config["tags"] = [TRACE_TAG, f"prompt:{v['prompt_version']}"]
        config["metadata"] = {"thread_id": thread_id, **v}
    return config
