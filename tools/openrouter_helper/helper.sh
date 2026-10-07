#!/usr/bin/env bash
# OpenRouter planning helper for the person simulation.
#
#   bash tools/openrouter_helper/helper.sh <command>
#
#   plan        read the current state of the project (run files, complexity, machine) and ask the
#               planner model what the next step should be; writes state/next_task.md
#   serve       keep the rate-limiting gateway running forever (restarts it if it dies; survives the
#               day rolling over; stops when the file state/STOP exists)
#   probe       send "Reply with OK" once to each of the five models (5 requests) and print the result
#   resources   how many simulations the machine can run in parallel right now
#   context     (re)build state/context.md only
#   snapshot    save the editable files (before a coder starts)
#   verify      compile what changed, import at base and rich, run a smoke simulation: PASS / FAIL
#   rollback    restore the snapshot (undo a failed step)
#   status      usage today, last plan, last verification
#   stop        stop the gateway and ask `serve` to exit
#
# The models are used strictly in the order of models.json, 200 requests each per UTC day, at most
# 20 requests a minute and 1000 a day -- enforced by proxy.py, whoever asks.
# The API key is read from tools/openrouter_helper/.env (or $OPENROUTER_API_KEY) and never printed.
#
# This script plans, measures, snapshots and verifies.  It does not launch a coding agent: run your
# coder (for example Pi) yourself on state/next_task.md.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HELPER_ROOT:-$(cd "$HERE/../.." && pwd)}"
STATE="${HELPER_STATE:-$HERE/state}"
PORT="${OR_PORT:-8765}"
PY="${PYTHON:-python}"
mkdir -p "$STATE"
export HELPER_ROOT="$ROOT" HELPER_STATE="$STATE" OR_STATE="$STATE" OR_PORT="$PORT"
cd "$ROOT" || exit 1

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$HERE/.env" ]; then
  set -a; . "$HERE/.env"; set +a
fi

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

proxy_up() { curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

proxy_start() {
  if proxy_up; then return 0; fi
  if [ -z "${OR_UPSTREAM:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    log "no OPENROUTER_API_KEY (put it in $HERE/.env)"; return 1
  fi
  nohup "$PY" "$HERE/proxy.py" --port "$PORT" >>"$STATE/proxy.log" 2>&1 &
  echo $! >"$STATE/proxy.pid"
  for _ in $(seq 1 40); do proxy_up && return 0; sleep 0.25; done
  log "gateway did not start; see $STATE/proxy.log"; return 1
}

proxy_stop() {
  if [ -f "$STATE/proxy.pid" ]; then kill "$(cat "$STATE/proxy.pid")" 2>/dev/null; rm -f "$STATE/proxy.pid"; fi
}

cmd_plan() {
  proxy_start || return 1
  log "reading the project state"
  "$PY" "$HERE/hlib.py" context || return 1
  "$PY" "$HERE/hlib.py" plan-request || { log "no quota left right now"; return 3; }
  log "asking the planner ($(sed -n 's/.*"model": *"\([^"]*\)".*/\1/p' "$STATE/current_model.json" | head -1))"
  code=$(curl -sS --max-time "${PLAN_TIMEOUT:-3600}" -H 'Content-Type: application/json' \
         --data-binary @"$STATE/plan_req.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
         -o "$STATE/plan_resp.json" -w '%{http_code}')
  if [ "$code" != "200" ]; then
    log "planner request failed (HTTP $code): $(head -c 300 "$STATE/plan_resp.json")"; return 1
  fi
  "$PY" "$HERE/hlib.py" plan-parse "$STATE/plan_resp.json" || { log "the reply was not in the expected format (kept in $STATE/bad_plan.txt)"; return 2; }
  "$PY" "$HERE/hlib.py" snapshot >/dev/null
  log "snapshot taken. Give $STATE/next_task.md to your coder, then run:  bash tools/openrouter_helper/helper.sh verify"
}

cmd_serve() {
  rm -f "$STATE/STOP"
  log "serving the gateway (stop with: bash tools/openrouter_helper/helper.sh stop)"
  fails=0
  while [ ! -f "$STATE/STOP" ]; do
    if ! proxy_up; then
      proxy_start && fails=0 || fails=$((fails + 1))
      [ "$fails" -gt 0 ] && sleep $((fails < 6 ? fails * 5 : 30))
    fi
    sleep 5
  done
  proxy_stop; log "stopped"
}

cmd_status() {
  if proxy_up; then
    curl -s "http://127.0.0.1:$PORT/current"; echo
  else
    echo "gateway not running"; [ -f "$STATE/usage.json" ] && cat "$STATE/usage.json"; echo
  fi
  [ -f "$STATE/next_plan.json" ] && { echo "last plan:"; cat "$STATE/next_plan.json"; echo; }
  [ -f "$STATE/last_verify.json" ] && { echo "last verify:"; head -c 600 "$STATE/last_verify.json"; echo; }
}

case "${1:-}" in
  plan)       cmd_plan ;;
  serve)      cmd_serve ;;
  probe)      "$PY" "$HERE/proxy.py" --probe ;;
  resources)  "$PY" "$HERE/hlib.py" resources ;;
  context)    "$PY" "$HERE/hlib.py" context ;;
  snapshot)   "$PY" "$HERE/hlib.py" snapshot ;;
  verify)     "$PY" "$HERE/hlib.py" verify ;;
  rollback)   "$PY" "$HERE/hlib.py" rollback ;;
  record)     shift; "$PY" "$HERE/hlib.py" record "$@" ;;
  status)     cmd_status ;;
  stop)       touch "$STATE/STOP"; proxy_stop; log "stop requested" ;;
  *)          sed -n '2,25p' "${BASH_SOURCE[0]}" ;;
esac
