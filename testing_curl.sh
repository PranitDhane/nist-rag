#!/usr/bin/env bash
# All curl tests for the NIST AI RMF assistant in one place.
#
#   ./testing_curl.sh               list the available tests
#   ./testing_curl.sh greeting      run one test
#   ./testing_curl.sh all           run everything (about 6 LLM-backed calls; it pauses between them for the free tier)
#
# Start the server first (in another terminal):  make dev
# Override the address with:  BASE=http://127.0.0.1:9000 ./testing_curl.sh health
BASE="${BASE:-http://127.0.0.1:8000}"
PAUSE="${PAUSE:-20}"          # seconds between NIST questions (free tier: ~8,000 tokens/minute)
JSON='Content-Type: application/json'

pretty() { python3 -m json.tool --no-ensure-ascii 2>/dev/null || cat; }   # keeps curly quotes/hyphens readable
say()    { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
chat()   { curl -s -X POST "$BASE/chat" -H "$JSON" -d "$1" | pretty; }           # $1 = JSON body
tid()    { python3 -c 'import sys,json; print(json.load(sys.stdin)["thread_id"])' 2>/dev/null; }

# Every test starts with this: fail with a clear message instead of printing empty output.
need_server() {
  if ! curl -s -f -m 5 "$BASE/health" >/dev/null 2>&1; then
    echo "Cannot reach the server at $BASE. Start it first (in another terminal):  make dev" >&2
    exit 1
  fi
}

t_health() {
  say "health: is the server up?"
  curl -s "$BASE/health" | pretty
}

t_greeting() {
  say "greeting by rule: 0 LLM calls, instant"
  chat '{"message": "Hello!"}'
}

t_out_of_scope() {
  say "out of scope: fixed reply 'Sorry, we don't have this answer.' (1 router call)"
  chat '{"message": "What is the capital of France?"}'
}

t_answered() {
  say "NIST question, answered: answer + sources (label + page link) + meta.debug"
  chat '{"message": "Which of the four AI RMF functions is cross-cutting?", "debug": true}'
}

t_definition() {
  say "NIST question about the GenAI profile (confabulation)"
  chat '{"message": "How does the Generative AI Profile define confabulation?"}'
}

t_table() {
  say "exact ID lookup (table chunk)"
  chat '{"message": "What does suggested action GV-1.7-001 say?"}'
}

t_insufficient() {
  say "insufficient: the fixed sentence + related_sections (closest sections, not answers)"
  chat '{"message": "What does the AI RMF Playbook recommend for auditing third-party vendors?"}'
}

t_followup() {
  say "follow-up: turn 1 creates a thread_id, turn 2 depends on it"
  local id
  id=$(curl -s -X POST "$BASE/chat" -H "$JSON" -d '{"message": "What are the four core functions of the AI RMF?"}' | tid)
  if [ -z "$id" ]; then
    echo "Turn 1 did not return a thread_id (rate limit or server error?). Wait a minute and try again." >&2
    return 1
  fi
  echo "thread_id = $id"
  sleep "$PAUSE"
  chat "{\"message\": \"Which one is cross-cutting?\", \"thread_id\": \"$id\", \"debug\": true}"
  say "stored conversation for that thread"
  curl -s "$BASE/threads/$id" | pretty
  say "delete the thread (expect HTTP 204)"
  curl -s -o /dev/null -w "HTTP %{http_code}\n" -X DELETE "$BASE/threads/$id"
}

t_stream() {
  say "streaming (SSE): meta -> token... -> sources -> done"
  curl -N -s -X POST "$BASE/chat/stream" -H "$JSON" -d '{"message": "Which of the four AI RMF functions is cross-cutting?"}'
  echo
}

t_stream_greeting() {
  say "streaming a greeting: one token event"
  curl -N -s -X POST "$BASE/chat/stream" -H "$JSON" -d '{"message": "thanks!"}'
  echo
}

# ---- error cases (no LLM call) ----------------------------------------------------------------------
t_errors() {
  say "422: empty message"
  curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/chat" -H "$JSON" -d '{"message": "   "}'
  say "422: thread_id is not a UUID"
  curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/chat" -H "$JSON" -d '{"message": "hi", "thread_id": "nope"}'
  say "404: unknown thread_id"
  curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/chat" -H "$JSON" \
       -d '{"message": "hi", "thread_id": "11111111-1111-4111-8111-111111111111"}'
  say "404: unknown thread when reading it"
  curl -s -w "\nHTTP %{http_code}\n" "$BASE/threads/11111111-1111-4111-8111-111111111111"
}

TESTS="health greeting out_of_scope answered definition table insufficient followup stream stream_greeting errors"

usage() {
  echo "Usage: $0 <test>|all"
  echo "Tests: $TESTS"
  echo "(NIST questions cost ~4,000 tokens each; the free tier allows about 2 per minute.)"
}

run() {
  local name="$1"
  case "$name" in
    ""|-h|--help|help) usage; return 0 ;;
    all) ;;
    *) if ! declare -f "t_$name" >/dev/null; then echo "Unknown test: $name"; usage; exit 1; fi ;;
  esac
  need_server
  if [ "$name" = "all" ]; then
    for t in $TESTS; do
      "t_$t"
      case "$t" in answered|definition|table|insufficient|followup) sleep "$PAUSE" ;; esac
    done
  else
    "t_$name"
  fi
}
run "$1"
