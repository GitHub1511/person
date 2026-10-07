#!/usr/bin/env bash
# Hourly README agent driver.   bash tools/openrouter_helper/readme_helper/run.sh once [--apply]
#
#   once [--apply]   rebuild the prompt and ask the model once.
#                    Default is a dry-run (writes readme_helper/state/preview.md,
#                    README.md untouched); --apply backs up README.md and writes it.
#   loop [--apply]   sleep until the next wall-clock hour boundary (:00), then
#                    run `once`, forever. Re-aligns to :00 after every run, so it
#                    never drifts (it does NOT just sleep 3600).
#   preview          offline dry-run: build the prompt, print its size + section
#                    list, save readme_helper/state/prompt.md. No network, no quota.
#
# Honors HELPER_ROOT like the main helper (project root override).
# Quota: exactly 1 request per `once` run, so 24/day in `loop` mode.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HELPER_ROOT:-$(cd "$HERE/../../.." && pwd)}"
PY="${PYTHON:-python}"
export HELPER_ROOT="$ROOT"
cd "$ROOT" || exit 1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

usage() { sed -n '2,14p' "${BASH_SOURCE[0]}"; }

cmd_once() {
  # $1 (optional): --apply
  if [ "${1:-}" = "--apply" ]; then
    log "readme agent: single run, APPLY mode (README.md will be rewritten)"
    "$PY" "$HERE/agent.py" --apply
  elif [ -n "${1:-}" ]; then
    log "usage: run.sh once [--apply]"; return 2
  else
    log "readme agent: single run, dry-run mode (README.md untouched)"
    "$PY" "$HERE/agent.py" --dry-run
  fi
}

secs_to_next_hour() {
  # Integer seconds until the next wall-clock :00 (3600 when exactly on it).
  "$PY" -c "import time; print(int(3600 - time.time() % 3600) or 3600)"
}

cmd_loop() {
  # $1 (optional): --apply
  mode="${1:-}"; [ -z "$mode" ] || [ "$mode" = "--apply" ] || { log "usage: run.sh loop [--apply]"; return 2; }
  trap 'log "readme loop stopped"; exit 0' INT TERM
  log "readme loop started ($([ -n "$mode" ] && echo "APPLY" || echo "dry-run") mode; aligns to :00 each hour). Stop with Ctrl-C."
  while true; do
    secs="$(secs_to_next_hour)"
    log "sleeping ${secs}s until the next hour boundary"
    sleep "$secs"
    log "=== hourly run $(date '+%Y-%m-%d %H:00') ==="
    cmd_once "$mode" || log "run failed (exit $?); next try at the following :00"
  done
}

cmd_preview() {
  "$PY" "$HERE/agent.py" --offline --print-prompt
}

case "${1:-}" in
  once)    shift; cmd_once "$@" ;;
  loop)    shift; cmd_loop "$@" ;;
  preview) cmd_preview ;;
  *)       usage ;;
esac
