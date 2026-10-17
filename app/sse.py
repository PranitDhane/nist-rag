"""Server-Sent Events formatting (hand-formatted, Q3) and a small parser used by tests."""
from __future__ import annotations

import json

SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def format_event(event: str, data: dict) -> str:
    """One SSE frame: `event:` line, a single-line JSON `data:` line, blank line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def parse_events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for frame in raw.strip().split("\n\n"):
        if not frame.strip():
            continue
        fields = dict(line.split(": ", 1) for line in frame.split("\n"))
        out.append((fields["event"], json.loads(fields["data"])))
    return out
