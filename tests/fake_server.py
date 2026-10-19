"""A tiny stand-in for the API (stdlib only, real sockets) so the E2E harness is tested without Groq (plan V2)."""
from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

Reply = tuple[int, dict, dict]          # status, JSON body, extra headers


def answered(thread_id: str, message: str, *, llm_calls: int = 2, generation: int = 2800) -> dict:
    return {"thread_id": thread_id, "route": "nist_question", "answer": "GOVERN is cross-cutting [C1].",
            "sources": [{"keys": ["C1"], "label": "NIST AI 100-1, §5.1 Govern, pp. 21–22", "url": "x#page=26", "kind": "prose"}],
            "standalone_query": message, "related_sections": [],
            "meta": {"llm_calls": llm_calls, "outcome": "answered", "usage": {"router": 1200, "generation": generation, "total": 1200 + generation},
                     "debug": {"route": "nist_question", "key_map": {"C1": "ai_rmf_1_0:prose:0022:1"}}}}


def greeting(thread_id: str) -> dict:
    return {"thread_id": thread_id, "route": "greeting", "answer": "Hello!", "sources": [], "standalone_query": None,
            "meta": {"llm_calls": 0, "outcome": "greeting", "usage": {"router": 0, "generation": 0, "total": 0}}}


class FakeServer:
    """`handler(path, payload, n)` returns a Reply; `n` counts POST /chat calls (1-based). Every request is logged."""

    def __init__(self, handler: Callable[[str, dict, int], Reply] | None = None, health: Reply | None = None):
        self.requests: list[dict] = []
        self.chat_calls = 0
        self.handler = handler or self.default_handler
        self.health = health or (200, {"status": "ok", "key_configured": True}, {})
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):          # keep test output quiet
                pass

            def _send(self, status: int, body: dict, headers: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer.requests.append({"method": "GET", "path": self.path})
                self._send(*outer.health)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.chat_calls += 1
                outer.requests.append({"method": "POST", "path": self.path, "payload": payload})
                self._send(*outer.handler(self.path, payload, outer.chat_calls))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @staticmethod
    def default_handler(path: str, payload: dict, n: int) -> Reply:
        tid = payload.get("thread_id") or str(uuid.uuid4())
        return 200, answered(tid, payload["message"]), {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def posts(self) -> list[dict]:
        return [r["payload"] for r in self.requests if r["method"] == "POST"]
