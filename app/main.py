"""FastAPI app. The graph is built ONCE at startup (lifespan), never per request.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000      (or: python -m app.main)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api import router
from app.config import ConfigError, Settings, get_settings

log = logging.getLogger("app.main")


def create_app(settings: Settings | None = None, graph=None) -> FastAPI:
    """`graph` can be injected (tests); otherwise the production graph is built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or get_settings()
        app.state.locks = {}
        if graph is not None:
            app.state.graph = graph
        else:
            from graph.build import build_app_graph
            from llm.groq_client import GroqClient
            s = app.state.settings.require_key()          # fail at startup with a clear message
            app.state.graph = build_app_graph(s, GroqClient(s))
            log.info("graph built; router model %s", s.router_model)
        yield

    app = FastAPI(title="NIST AI RMF assistant", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        body = exc.detail if isinstance(exc.detail, dict) and "error" in exc.detail else \
            {"error": {"code": "http_error", "message": str(exc.detail)}}
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)     # details logged, never returned
        return JSONResponse({"error": {"code": "internal_error", "message": "Internal server error."}}, status_code=500)

    return app


app = create_app()


def main() -> None:
    import uvicorn
    s = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        s.require_key()
    except ConfigError as e:
        raise SystemExit(f"Cannot start: {e}")
    uvicorn.run("app.main:app", host=s.host, port=s.port)


if __name__ == "__main__":
    main()
