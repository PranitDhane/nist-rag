# NIST AI RMF Agentic RAG: one-command build and run.
# Works with an activated venv (`python` on the PATH) or with a local .venv (used automatically when present).
# Override with:  make PYTHON=python3.12 <target>
# No `make` (e.g. Windows)? Every target prints the Python command it runs: copy that line.

PYTHON   ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,python))
BASE_URL ?= http://127.0.0.1:8000
FROM_DIR ?=
RUN      ?=
Q        ?= Which of the four AI RMF functions is cross-cutting?

# Print the command, then run it (so the README can pair each target with its Python command).
run = echo "+ $(1)" && $(1)

.DEFAULT_GOAL := help
.PHONY: help install download ingest chunk index build rebuild serve dev health ask test test-fast \
        eval-routing eval-retrieval e2e e2e-report results clean clean-all check-env check-index

help:
	@echo "NIST AI RMF Agentic RAG - targets (activated venv, or a local .venv, is used automatically)"
	@echo ""
	@echo "  setup"
	@echo "    install         pip install -r requirements.txt"
	@echo "  data pipeline"
	@echo "    download        acquire + verify the two NIST PDFs into data/raw (ingest stage 2)"
	@echo "    ingest          PDFs -> pages (FROM_DIR=<dir> uses local PDFs instead of downloading)"
	@echo "    chunk           pages -> chunks"
	@echo "    index           chunks -> Chroma + BM25 indexes"
	@echo "    build           ingest -> chunk -> index (stops at the first failure)"
	@echo "    rebuild         the same with --rebuild on each step (PDFs are kept)"
	@echo "  run"
	@echo "    serve           start the API (host/port from APP_HOST/APP_PORT in .env; default 127.0.0.1:8000)"
	@echo "    health          check that the running server is up"
	@echo "    ask Q='...'     ask one question through /chat (debug on)"
	@echo "  tests and evaluation"
	@echo "    test            full test suite (offline; live tests skip without a key)"
	@echo "    test-fast       test suite without the slow real-model tests"
	@echo "    eval-routing    routing evaluation (live Groq calls, paced)"
	@echo "    eval-retrieval  retrieval evaluation (local models, no LLM)"
	@echo "    e2e             end-to-end run against a RUNNING server (BASE_URL=$(BASE_URL))"
	@echo "    e2e-report RUN=eval/e2e_runs/<timestamp>   checks + report from a saved run"
	@echo "    results         regenerate docs/RESULTS.md and the README results block"
	@echo "  cleanup"
	@echo "    clean           remove data/processed, data/chroma, data/index (never data/raw)"
	@echo "    clean-all       also remove data/raw (asks for confirmation)"

# ------------------------------------------------------------------ setup
install:
	@$(call run,$(PYTHON) -m pip install -r requirements.txt)

# ------------------------------------------------------------------ data pipeline
download:
	@$(call run,$(PYTHON) -m ingest --only-stage 2)

ingest:
	@$(call run,$(PYTHON) -m ingest $(if $(FROM_DIR),--from-dir $(FROM_DIR)))

chunk:
	@$(call run,$(PYTHON) -m chunking)

index:
	@$(call run,$(PYTHON) -m indexing)

# Sub-makes in sequence (not prerequisites) so `make -j` can never reorder the steps; `&&` stops at the first failure.
build:
	@$(MAKE) --no-print-directory ingest && $(MAKE) --no-print-directory chunk && $(MAKE) --no-print-directory index

rebuild:
	@$(call run,$(PYTHON) -m ingest --rebuild $(if $(FROM_DIR),--from-dir $(FROM_DIR))) \
	  && $(call run,$(PYTHON) -m chunking --rebuild) && $(call run,$(PYTHON) -m indexing --rebuild)

# ------------------------------------------------------------------ run
# Fails early, with a clear message, if the key or the indexes are missing.
check-env:
	@$(PYTHON) --version >/dev/null 2>&1 || { echo "No Python found: create the venv first (python3.12 -m venv .venv) and run 'make install'"; exit 1; }
	@grep -Eq '^GROQ_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]' .env 2>/dev/null || [ -n "$$GROQ_API_KEY" ] || { echo "GROQ_API_KEY is not set: copy .env.example to .env and add your key"; exit 1; }

check-index:
	@test -f data/index/index_manifest.json -a -d data/chroma || { echo "Indexes missing: run 'make build' first"; exit 1; }

serve: check-env check-index
	@$(call run,$(PYTHON) -m app.main)

dev: serve

health:
	@curl -s http://127.0.0.1:$${APP_PORT:-8000}/health; echo

ask:
	@curl -s -X POST http://127.0.0.1:$${APP_PORT:-8000}/chat -H 'Content-Type: application/json' \
	  -d "{\"message\": \"$(Q)\", \"debug\": true}" | $(PYTHON) -m json.tool --no-ensure-ascii

# ------------------------------------------------------------------ tests and evaluation
test:
	@$(call run,$(PYTHON) -m pytest -q -W ignore)

test-fast:
	@$(call run,$(PYTHON) -m pytest -q -W ignore --deselect tests/test_retrievers.py --deselect tests/test_retrieval_integration.py)

eval-routing: check-env
	@$(call run,$(PYTHON) -m eval.routing)

eval-retrieval: check-index
	@$(call run,$(PYTHON) -m eval.retrieval_eval)

e2e: check-env
	@$(call run,$(PYTHON) -m eval.e2e --base-url $(BASE_URL))

e2e-report:
	@test -n "$(RUN)" || { echo "Usage: make e2e-report RUN=eval/e2e_runs/<timestamp>"; exit 1; }
	@$(call run,$(PYTHON) -m eval.e2e_report --from-run $(RUN))

results:
	@test -f docs/build_results.py || { echo "docs/build_results.py does not exist yet (deliverables stage D3)"; exit 1; }
	@$(call run,$(PYTHON) docs/build_results.py)

# ------------------------------------------------------------------ cleanup
clean:
	@echo "+ rm -rf data/processed data/chroma data/index   (data/raw is kept)"
	@rm -rf data/processed data/chroma data/index

clean-all:
	@printf "This also deletes the downloaded PDFs in data/raw. Type 'yes' to continue: " && read ans && [ "$$ans" = "yes" ] \
	  && echo "+ rm -rf data" && rm -rf data || echo "cancelled"
