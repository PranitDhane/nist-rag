# Usage: make dev | make test | make test-fast | make health | make ask Q="your question"
PY := .venv/bin/python

.PHONY: dev test test-fast health ask help

help:
	@echo "make dev        start the API (loads models, ~8 s); host/port from APP_HOST/APP_PORT in .env (default 127.0.0.1:8000)"
	@echo "make health     check that the running server is up"
	@echo "make ask Q='..' ask a question through /chat (debug on)"
	@echo "make test       full test suite"
	@echo "make test-fast  test suite without the slow real-model tests"

# Fails early with a clear message if the venv, the key or the indexes are missing.
dev:
	@test -x $(PY) || { echo "Missing .venv: run 'python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt'"; exit 1; }
	@grep -Eq '^GROQ_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]' .env 2>/dev/null || [ -n "$$GROQ_API_KEY" ] || { echo "GROQ_API_KEY is not set (add it to .env)"; exit 1; }
	@test -f data/index/index_manifest.json || { echo "Indexes missing: run '$(PY) -m indexing --rebuild'"; exit 1; }
	$(PY) -m app.main

health:
	@curl -s http://127.0.0.1:$${APP_PORT:-8000}/health; echo

Q ?= Which of the four AI RMF functions is cross-cutting?
ask:
	@curl -s -X POST http://127.0.0.1:$${APP_PORT:-8000}/chat -H 'Content-Type: application/json' \
	  -d "{\"message\": \"$(Q)\", \"debug\": true}" | $(PY) -m json.tool --no-ensure-ascii

test:
	$(PY) -m pytest -q -W ignore

test-fast:
	$(PY) -m pytest -q -W ignore --deselect tests/test_retrievers.py --deselect tests/test_retrieval_integration.py
