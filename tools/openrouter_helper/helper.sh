#!/usr/bin/env bash
# OpenRouter planning helper for the person simulation.        bash tools/openrouter_helper/helper.sh run
#
#   run         THE ONE COMMAND.  Runs until you stop it (day after day):
#                 1. looks at the project: .npz / JSON run files, complexity, git, what earlier steps did,
#                    and how many parallel simulations this machine can spare right now
#                 2. asks the planner model (models in order, 200 requests each per UTC day, 20/min,
#                    1000/day) for the single most valuable next step, as a full prompt for a coder; the
#                    planner may also name web pages, which are fetched read-only and attached
#                 3. snapshots the editable files and writes state/next_task.md
#                 4. waits for your coder (any tool: it reads next_task.md, works in the project directory
#                    with its own shell, and finishes with `helper.sh done`)
#                 5. verifies the result (compiles, imports at base and rich, smoke run, helper/key
#                    untouched); on FAIL it rolls the files back; records the outcome; goes to 1
#   plan        just step 1-3 once
#   done        tell `run` the coder has finished (or `touch state/CODER_DONE`)
#   verify      PASS / FAIL for the current tree        rollback   undo back to the snapshot
#   snapshot    snapshot now                            resources  how many simulations fit right now
#   fetch URL   read a public web page as text           probe      "Reply with OK" to each model once
#   serve       only keep the rate-limiting gateway running (restarts it if it dies)
#   status      usage today, last plan, last verification   stop   ask a running `run`/`serve` to exit
#
# Knobs (environment): RUN_WAIT_TIMEOUT (seconds to wait for a coder per step, default 21600),
#   AUTO_ROLLBACK=1 (default) roll back a failed step, PLAN_TIMEOUT, OR_PORT, HELPER_ROOT.
# The API key is read from tools/openrouter_helper/.env (or $OPENROUTER_API_KEY) and is never printed.
# This script never launches a coding agent itself: that is the one step it leaves to you.

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

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
hl() { "$PY" "$HERE/hlib.py" "$@"; }
proxy_up() { curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

proxy_start() {
  proxy_up && return 0
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

json_get() { sed -n "s/.*\"$1\": *\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\1/p" | head -1; }

cmd_plan() {
  proxy_start || return 1
  log "reading the project state"
  hl context || return 1
  hl plan-request || { log "no quota left right now"; return 3; }
  log "asking the planner ($(json_get model <"$STATE/current_model.json"))"
  rm -f "$STATE/plan_resp.json"
  code=$(curl -sS --max-time "${PLAN_TIMEOUT:-3600}" -H 'Content-Type: application/json' \
         --data-binary @"$STATE/plan_req.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
         -o "$STATE/plan_resp.json" -w '%{http_code}') || code=000
  if [ "$code" != "200" ]; then
    log "planner request failed (HTTP $code): $(head -c 300 "$STATE/plan_resp.json" 2>/dev/null)"; return 1
  fi
  hl plan-parse "$STATE/plan_resp.json" || { log "reply not in the expected format (kept in $STATE/bad_plan.txt)"; return 2; }
  hl attach-web || true
  hl snapshot >/dev/null
  log "task ready: $STATE/next_task.md   (snapshot taken)"
}

seconds_to_wait_for_quota() {
  curl -s "http://127.0.0.1:$PORT/current" | sed -n 's/.*"retry_after": *\([0-9]*\).*/\1/p' | head -1
}

wait_for_coder() {
  local timeout="${RUN_WAIT_TIMEOUT:-21600}" waited=0
  rm -f "$STATE/CODER_DONE"
  log "waiting for the coder. Give it $STATE/next_task.md; it should end with:  bash tools/openrouter_helper/helper.sh done"
  while [ ! -f "$STATE/CODER_DONE" ] && [ ! -f "$STATE/STOP" ] && [ "$waited" -lt "$timeout" ]; do
    sleep "${RUN_POLL:-5}"; waited=$((waited + ${RUN_POLL:-5}))
  done
  [ -f "$STATE/CODER_DONE" ] && log "coder reported done" || log "no coder report (waited ${waited}s): verifying anyway"
  rm -f "$STATE/CODER_DONE"
}

cmd_run() {
  rm -f "$STATE/STOP" "$STATE/CODER_DONE"
  trap 'proxy_stop; log "helper stopped"; exit 0' INT TERM
  log "helper started: models in order, 200/model/day, 20/min, 1000/day. Stop with: bash tools/openrouter_helper/helper.sh stop"
  cycle=0; fails=0
  while [ ! -f "$STATE/STOP" ]; do
    proxy_start || { sleep 30; continue; }
    left=$(seconds_to_wait_for_quota)
    if [ -n "$left" ]; then
      log "all models are at today's quota; resuming in ${left}s (UTC midnight)"
      while [ "$left" -gt 0 ] && [ ! -f "$STATE/STOP" ]; do sleep 30; left=$((left - 30)); done
      continue
    fi
    cycle=$((cycle + 1))
    log "=== step $cycle ==="; hl resources
    if ! cmd_plan; then
      fails=$((fails + 1)); b=$((fails < 8 ? fails * 20 : 180)); log "planning failed ($fails in a row); retry in ${b}s"
      for _ in $(seq 1 "$b"); do [ -f "$STATE/STOP" ] && break; sleep 1; done
      continue
    fi
    fails=0
    title=$("$PY" -c "import json;print(json.load(open('$STATE/next_plan.json'))['title'])" 2>/dev/null || echo "?")
    wait_for_coder
    [ -f "$STATE/STOP" ] && break
    out=$(hl verify); echo "$out"
    if echo "$out" | head -1 | grep -q '^PASS'; then
      hl record "$title" "accepted" "$(echo "$out" | sed -n 2p)"
    else
      hl record "$title" "rejected" "$(echo "$out" | head -1 | cut -c1-280)"
      if [ "${AUTO_ROLLBACK:-1}" = "1" ]; then log "verification failed: rolling back"; hl rollback; fi
    fi
  done
  proxy_stop; log "helper stopped"
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
  if proxy_up; then curl -s "http://127.0.0.1:$PORT/current"; echo
  else echo "gateway not running"; [ -f "$STATE/usage.json" ] && cat "$STATE/usage.json"; echo; fi
  [ -f "$STATE/next_plan.json" ] && { echo "last plan:"; cat "$STATE/next_plan.json"; echo; }
  [ -f "$STATE/last_verify.json" ] && { echo "last verify:"; head -c 600 "$STATE/last_verify.json"; echo; }
  [ -f "$STATE/history.jsonl" ] && { echo "history:"; tail -n 5 "$STATE/history.jsonl"; }
}

case "${1:-}" in
  run)        cmd_run ;;
  plan)       cmd_plan ;;
  serve)      cmd_serve ;;
  done)       touch "$STATE/CODER_DONE"; log "marked done" ;;
  probe)      "$PY" "$HERE/proxy.py" --probe ;;
  resources)  hl resources ;;
  context)    hl context ;;
  snapshot)   hl snapshot ;;
  verify)     hl verify ;;
  rollback)   hl rollback ;;
  fetch)      shift; hl fetch "$@" ;;
  record)     shift; hl record "$@" ;;
  status)     cmd_status ;;
  stop)       touch "$STATE/STOP"; proxy_stop; log "stop requested" ;;
  *)          sed -n '2,28p' "${BASH_SOURCE[0]}" ;;
esac
