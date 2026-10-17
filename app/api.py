"""HTTP endpoints. /chat (JSON) here; /chat/stream is added in Stage R7."""
from __future__ import annotations

import logging
import math
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.schemas import ChatRequest, ChatResponse, Meta
from generation.config import ANSWER_TAG
from graph import errors as graph_errors, usage as graph_usage
from graph.debug import build_debug
from graph.tracing import run_config
from insufficient.reply import RELATED_NOTE
from app.sse import SSE_HEADERS, format_event

log = logging.getLogger("app.api")
router = APIRouter()

# One checkpoint per request instead of one per graph step (~7x less memory per conversation; history is intact).
# The in-memory checkpointer cannot prune old checkpoints in this LangGraph version (aprune raises NotImplementedError).
DURABILITY = "exit"


class ApiError(HTTPException):
    """HTTPException that carries our {"error": {"code", "message"}} body."""

    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        super().__init__(status_code=status, detail={"error": {"code": code, "message": message}}, headers=headers)


def check_length(req: ChatRequest, max_chars: int) -> None:
    if len(req.message) > max_chars:                     # same 422 shape FastAPI uses for other validation errors
        raise RequestValidationError([{"type": "string_too_long", "loc": ("body", "message"),
                                       "msg": f"message must be at most {max_chars} characters", "input": None,
                                       "ctx": {"max_length": max_chars}}])


def llm_unavailable(retry_after: float | None) -> ApiError:
    wait = max(1, math.ceil(retry_after)) if retry_after else 15
    return ApiError(503, "llm_unavailable", "The language model is temporarily unavailable. Please retry shortly.",
                    headers={"Retry-After": str(wait)})


def api_error_code(err: dict) -> str:
    """Graph error code (LLM_UNAVAILABLE, RETRIEVAL_FAILED ...) -> the API's lower-case error code."""
    return "internal_error" if err["code"] == graph_errors.INTERNAL else err["code"].lower()


def error_to_api(err: dict, retry_after: float | None = None) -> ApiError:
    """HTTP status + GENERIC body for a graph error. Never carries a stack trace or an internal detail: the node already
    logged those (GI-D4). Status mapping: graph.errors.HTTP_STATUS."""
    if err["code"] == graph_errors.LLM_UNAVAILABLE:
        return llm_unavailable(retry_after)
    return ApiError(graph_errors.HTTP_STATUS.get(err["code"], 500), api_error_code(err),
                    graph_errors.GENERIC_MESSAGES.get(err["code"], graph_errors.GENERIC_MESSAGES[graph_errors.INTERNAL]))


async def thread_exists(graph, thread_id: str) -> bool:
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return bool(snap.values)


class ThreadLock:
    """Per-thread guard (Q13): a second simultaneous request on the same thread is rejected with 409."""

    def __init__(self, locks: dict, thread_id: str):
        self.lock = locks.setdefault(thread_id, __import__("asyncio").Lock())
        self.thread_id = thread_id

    async def __aenter__(self):
        if self.lock.locked():
            raise ApiError(409, "thread_busy", "Another request on this conversation is still being processed.")
        await self.lock.acquire()
        return self

    async def __aexit__(self, *exc):
        self.lock.release()


async def prepare(request: Request, body: ChatRequest) -> str:
    """Validate, resolve (or create) the thread_id. Unknown ids are 404 (Q12)."""
    check_length(body, request.app.state.settings.max_message_chars)
    if body.thread_id is None:
        return str(uuid.uuid4())
    if not await thread_exists(request.app.state.graph, body.thread_id):
        raise ApiError(404, "thread_not_found",
                       "Unknown or expired thread_id. Omit thread_id to start a new conversation.")
    return body.thread_id


def nist_fields(out: dict) -> dict:
    """INSUFFICIENT_CONTEXT_PLAN Section 5. For the NIST route only: (outcome, insufficient_reason, related_sections).
    The graph puts an InsufficientReply dict under state["insufficient"] when the answer was insufficient; related
    sections are NEVER mixed into `sources`. Other routes get an empty dict, so their responses are unchanged."""
    if out.get("route") != "nist_question":
        return {}
    ins = out.get("insufficient")
    if ins:
        return {"outcome": "insufficient", "insufficient_reason": ins["reason"], "related_sections": list(ins["related_sections"])}
    return {"outcome": out.get("outcome") or "answered", "insufficient_reason": None, "related_sections": []}


def meta_fields(out: dict, latency_ms: int, debug: bool = False) -> dict:
    """The meta fields set explicitly on a response (GRAPH_INTEGRATION_PLAN Section 7). Same source for /chat `meta` and the
    stream's `done` event. Generation fields appear only when generation ran; `debug` only when requested."""
    extra = nist_fields(out)
    fields: dict = {"llm_calls": out.get("llm_calls", 0), "route_source": out.get("route_source"), "latency_ms": latency_ms,
                    "outcome": extra.get("outcome") or out.get("outcome"),
                    "insufficient_reason": extra.get("insufficient_reason"),
                    "usage": out.get("usage") or graph_usage.total()}
    gen = out.get("generation")
    if gen:
        fields.update(model=gen.get("model"), prompt_version=gen.get("prompt_version"), fixes=list(gen.get("fixes") or []),
                      warnings=list(gen.get("warnings") or []))
    if debug:
        fields["debug"] = build_debug(out)
    return fields


def log_request(thread_id: str, out: dict, latency_ms: int, stream: bool = False) -> None:
    log.info("chat%s thread_id=%s route=%s route_source=%s llm_calls=%s latency_ms=%d", "_stream" if stream else "",
             thread_id, out.get("route"), out.get("route_source"), out.get("llm_calls"), latency_ms)


@router.post("/chat", response_model=ChatResponse, response_model_exclude_unset=True)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    thread_id = await prepare(request, body)
    started = time.perf_counter()
    async with ThreadLock(request.app.state.locks, thread_id):
        out = await request.app.state.graph.ainvoke({"user_message": body.message}, run_config(thread_id), durability=DURABILITY)
    latency_ms = round((time.perf_counter() - started) * 1000)
    if out.get("error"):
        log.warning("chat thread_id=%s error=%s node=%s latency_ms=%d", thread_id, out["error"]["code"], out["error"].get("node"), latency_ms)
        raise error_to_api(out["error"], out.get("retry_after"))
    log_request(thread_id, out, latency_ms)
    log.debug("chat message=%r", body.message)
    extra = nist_fields(out)
    insufficient = extra.get("outcome") == "insufficient"
    meta = Meta(**meta_fields(out, latency_ms, body.debug))
    response = ChatResponse(thread_id=thread_id, route=out["route"], answer=out["answer"],
                            sources=[] if insufficient else (out.get("sources") or []),      # nothing supports an insufficient reply
                            standalone_query=out.get("standalone_query"), meta=meta,
                            **({"related_sections": extra["related_sections"]} if extra else {}))
    return response


@router.get("/health")
async def health(request: Request) -> dict:
    s = request.app.state.settings
    return {"status": "ok", "model": s.router_model, "key_configured": bool(s.groq_api_key)}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, request: Request) -> dict:
    snap = await request.app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        raise ApiError(404, "thread_not_found", "Unknown or expired thread_id.")
    msgs = []
    for m in snap.values["messages"]:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content, "route": m.additional_kwargs.get("route")})
    return {"thread_id": thread_id, "messages": msgs}


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: str, request: Request) -> None:
    graph = request.app.state.graph
    if not await thread_exists(graph, thread_id):
        raise ApiError(404, "thread_not_found", "Unknown or expired thread_id.")
    await graph.checkpointer.adelete_thread(thread_id)
    request.app.state.locks.pop(thread_id, None)


# ----------------------------------------------------------------------------- /chat/stream (Stage R7)
# Top-level nodes whose update carries the final answer. The RAG subgraph is ONE node of the main graph ("rag"): its update
# arrives (after the subgraph finishes) with the answer, sources, related sections, generation data and any error.
ANSWER_NODES = {"greeting_reply", "out_of_scope_reply", "rag"}
# Only LLM calls carrying this tag may stream tokens to the client (GRAPH_INTEGRATION_PLAN Section 7). The router's call, the
# reasoning chunks, the state-update messages LangGraph also emits (the user message, finalize's stored text) and anything
# else are NEVER forwarded.
STREAM_TAG = ANSWER_TAG


def is_answer_token(token, metadata: dict) -> bool:
    kwargs = getattr(token, "additional_kwargs", None) or {}
    return (STREAM_TAG in (metadata.get("tags") or []) and bool(getattr(token, "content", ""))
            and not kwargs.get("reasoning_content"))


async def drop_dangling_user_message(graph, config) -> None:
    """After an interrupted turn the history must not end with an unanswered user message."""
    snap = await graph.aget_state(config)
    msgs = snap.values.get("messages") or [] if snap.values else []
    if msgs and isinstance(msgs[-1], HumanMessage):
        await graph.aupdate_state(config, {"messages": [RemoveMessage(id=msgs[-1].id)]})


async def chat_events(graph, thread_id: str, message: str, release=None, debug: bool = False):
    """Async generator of SSE frames for one message. `release` is called when the stream ends for any reason.

    Order: meta -> token* -> [correction] -> sources -> [related] -> done, or ... -> error (nothing after it).
    The graph is streamed with subgraphs=True: events are (namespace, mode, data). Only top-level node updates are read
    (the `rag` subgraph's own update carries its whole output once it finishes), and only tagged generation tokens stream."""
    config = run_config(thread_id)                         # + LangSmith tags/metadata only when tracing is enabled (GI-D5)
    started = time.perf_counter()
    state: dict = {}
    streamed_from_model = False
    streamed_text = ""
    meta_sent = completed = False
    answer, sources, error = None, [], None
    try:
        async for namespace, mode, chunk in graph.astream({"user_message": message}, config, durability=DURABILITY,
                                                           stream_mode=["updates", "messages"], subgraphs=True):
            if mode == "updates":
                if namespace:                             # a node INSIDE the rag subgraph: its result arrives with `rag`
                    continue
                for node, update in chunk.items():
                    update = update or {}
                    state.update({k: v for k, v in update.items() if k != "messages"})
                    if update.get("error"):
                        error = update["error"]           # keep consuming: error_finalize must still roll the history back
                    if update.get("route") and not meta_sent and not error:
                        meta_sent = True
                        yield format_event("meta", {"thread_id": thread_id, "route": update["route"]})
                    if node in ANSWER_NODES:
                        answer, sources = update.get("answer"), update.get("sources") or []
            elif mode == "messages":
                token, metadata = chunk
                if is_answer_token(token, metadata):
                    streamed_from_model = True
                    streamed_text += token.content
                    yield format_event("token", {"text": token.content})
        completed = True
        latency_ms = round((time.perf_counter() - started) * 1000)
        if error:
            log.warning("chat_stream thread_id=%s error=%s node=%s latency_ms=%d", thread_id, error["code"], error.get("node"), latency_ms)
            yield format_event("error", {"code": api_error_code(error), "message": graph_errors.GENERIC_MESSAGES.get(
                error["code"], graph_errors.GENERIC_MESSAGES[graph_errors.INTERNAL])})
            return
        if not streamed_from_model:                       # template replies and gate-failure replies: ONE token event
            yield format_event("token", {"text": answer})
        elif answer is not None and answer != streamed_text:
            # the validated final text differs from what was streamed (e.g. the fixer forced Case D): replace it
            correction = {"text": answer}
            reasons = list(((state.get("generation") or {}).get("fixes")) or [])
            if reasons:                                    # generation phase Q5: {text, reasons}
                correction["reasons"] = reasons
            yield format_event("correction", correction)
        extra = nist_fields({**state, "route": state.get("route")})
        insufficient = extra.get("outcome") == "insufficient"
        yield format_event("sources", {"sources": [] if insufficient else sources})
        if insufficient:                                   # separate event; never inside `sources` (IC-D5)
            yield format_event("related", {"note": RELATED_NOTE, "sections": extra["related_sections"]})
        done = {k: v for k, v in meta_fields(state, latency_ms, debug).items() if k != "route_source"}
        yield format_event("done", done)
        log_request(thread_id, state, latency_ms, stream=True)
    except Exception:                                      # noqa: BLE001
        log.exception("chat_stream thread_id=%s failed", thread_id)
        yield format_event("error", {"code": "internal_error", "message": "Internal server error."})
    finally:
        if not completed:                                  # client disconnected (generator closed/cancelled) or crashed
            log.info("chat_stream thread_id=%s interrupted before completion; cleaning up", thread_id)
            try:
                await drop_dangling_user_message(graph, config)
            except Exception:                              # noqa: BLE001
                log.exception("cleanup after interrupted stream failed")
        if release:
            release()


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Same body and validation as /chat. 422/404/409 are plain HTTP errors sent before the stream starts."""
    thread_id = await prepare(request, body)
    lock = ThreadLock(request.app.state.locks, thread_id)
    await lock.__aenter__()                               # 409 if busy; released when the stream ends
    released = False

    def release() -> None:                                # idempotent: called by the generator AND as a background task
        nonlocal released
        if not released:
            released = True
            lock.lock.release()

    return StreamingResponse(chat_events(request.app.state.graph, thread_id, body.message, release=release, debug=body.debug),
                             media_type="text/event-stream", headers=SSE_HEADERS, background=BackgroundTask(release))
