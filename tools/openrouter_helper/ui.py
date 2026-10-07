#!/usr/bin/env python3
"""Local web console for the OpenRouter planning helper.

A localhost-only dashboard where you type a goal prompt and watch Pi build
the simulated-human project towards it, step by step:

    bash tools/openrouter_helper/helper.sh ui     # proxy + this console
    # or directly:
    python tools/openrouter_helper/ui.py [--port 8770]

Open http://127.0.0.1:8770 in a browser.  The page lets you:

  * type a goal prompt (e.g. "make the person learn to stand up from lying")
    plus a max-cycle budget, and save it as the loop's standing goal;
  * start the recursive loop (Pi coder via the local gateway, snapshot,
    verify, rollback on failure) and stop it again;
  * read live feedback: loop state, quota usage, current step, last
    verification (PASS/FAIL + reasons), step history, Pi's latest report.

The loop itself is still ``helper.sh run`` (AUTO_CODER=1, i.e. Pi).  The
console only writes goal files / STOP / spawns that loop and reports state.
It listens on 127.0.0.1 only and uses only the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HELPER_ROOT") or HERE.parents[1]).resolve()
STATE = Path(os.environ.get("HELPER_STATE") or HERE / "state")
STATE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import hlib  # noqa: E402  (goal helpers + context)

UI_PORT = int(os.environ.get("OR_UI_PORT", "8770"))
OR_PORT = os.environ.get("OR_PORT", "8765")
MAX_POST = 64 * 1024
LOOP_PID = STATE / "loop.pid"
RUN_LOG = STATE / "run.log"

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>person helper console</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222;background:#fafafa}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px;color:#444}
.sub{color:#666;font-size:13px;margin-bottom:12px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:10px 0}
textarea{width:100%;min-height:90px;font:13px/1.45 inherit;padding:8px;box-sizing:border-box}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:8px}
button{padding:7px 14px;font-size:13px;border-radius:6px;border:1px solid #999;background:#f0f0f0;cursor:pointer}
button.primary{background:#1a73e8;color:#fff;border-color:#1a73e8}
button.danger{background:#fff;color:#b00;border-color:#b00}
button:disabled{opacity:.5;cursor:default}
input[type=number]{width:70px;padding:6px}
pre{background:#111;color:#ddd;padding:10px;border-radius:6px;overflow:auto;max-height:260px;font-size:12px;white-space:pre-wrap}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}
.ok{background:#d9f2d9;color:#0a5}.bad{background:#fdd;color:#a00}.idle{background:#eee;color:#555}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid #eee;padding:5px 6px;text-align:left;vertical-align:top}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
#err{color:#a00;font-size:13px;min-height:18px}
.bar{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-top:6px}
.bar>div{height:100%;background:#1a73e8;width:0%}
</style></head><body>
<h1>person helper console</h1>
<div class="sub">Goal-directed recursive improvement: the planner aims every step at your goal,
Pi (via the local gateway, quotas enforced) edits the project, each step is snapshotted,
verified, and rolled back on failure.</div>
<div id="err"></div>

<h2>1. Goal prompt</h2>
<div class="card">
<textarea id="goal" placeholder="e.g. make the person stand up from lying, then take three steps without falling"></textarea>
<div class="row">
<label>max cycles <input id="cycles" type="number" value="10" min="1" max="200"></label>
<button class="primary" id="bGo" onclick="saveStart()">Save goal &amp; start loop</button>
<button onclick="saveOnly()">Save goal only</button>
</div>
<div id="goalState" class="mono" style="margin-top:8px"></div>
<div class="bar"><div id="goalBar"></div></div>
</div>

<h2>2. Loop control</h2>
<div class="card"><div class="row">
<button class="primary" onclick="act('start')">Start / resume</button>
<button class="danger" onclick="act('stop')">Stop</button>
<button onclick="refresh()">Refresh now</button>
<span id="loopState"></span>
</div></div>

<h2>3. Feedback</h2>
<div class="card"><div id="status"></div></div>
<div class="card"><h2 style="margin-top:0">Step history</h2><div id="hist"></div></div>
<div class="card"><h2 style="margin-top:0">Pi coder — latest report tail</h2><pre id="pi">(none yet)</pre></div>
<div class="card"><h2 style="margin-top:0">Run log tail</h2><pre id="runlog">(none yet)</pre></div>

<script>
let S = {};
async function api(path, body) {
  const o = {method: body ? 'POST' : 'GET'};
  if (body) { o.headers = {'Content-Type': 'application/json'}; o.body = JSON.stringify(body); }
  const r = await fetch(path, o);
  return r.json();
}
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
async function refresh() {
  try {
    S = await api('/api/status');
    document.getElementById('err').textContent = '';
  } catch(e){ document.getElementById('err').textContent = 'console unreachable: ' + e; return; }
  const g = S.goal_meta || {};
  document.getElementById('goalState').textContent =
    (S.goal ? 'active goal (' + (g.cycles_done||0) + '/' + (g.max_cycles||'?') + ', ' + (g.status||'active') + '): ' + S.goal.slice(0,180) : '(no goal set)');
  const pct = g.max_cycles ? Math.min(100, 100*(g.cycles_done||0)/g.max_cycles) : 0;
  document.getElementById('goalBar').style.width = pct + '%';
  if (!document.getElementById('goal').value && S.goal)
    document.getElementById('goal').value = S.goal;
  const L = S.loop || {};
  document.getElementById('loopState').innerHTML =
    (L.running ? '<span class="pill ok">RUNNING' + (L.pid ? ' pid ' + L.pid : '') + '</span>'
               : '<span class="pill idle">STOPPED</span>') +
    (L.stop_requested ? ' <span class="pill idle">stop requested</span>' : '');
  const v = S.last_verify || {};
  const pill = !S.last_verify ? '<span class="pill idle">no verify yet</span>'
    : v.pass ? '<span class="pill ok">PASS</span>' : '<span class="pill bad">FAIL</span>';
  const probs = (v.problems || []).map(esc).join('<br>');
  document.getElementById('status').innerHTML =
    '<table><tr><th>gateway</th><td>' + esc(S.proxy && S.proxy.up ? 'up, model ' + (S.proxy.current||'?') : 'DOWN') +
    ' &nbsp; quota ' + esc(String((S.usage||{}).total||0)) + '/' + esc(String((S.usage||{}).daily||1000)) + '</td></tr>' +
    '<tr><th>current step</th><td>' + esc((S.next_plan||{}).title || '(none)') +
    ' <span class="mono">' + esc((S.next_plan||{}).model || '') + '</span></td></tr>' +
    '<tr><th>last verify</th><td>' + pill + (probs ? '<br>' + probs : '') +
    ' <span class="mono">' + esc((v.changed||[]).length + ' files changed') + '</span></td></tr></table>';
  document.getElementById('hist').innerHTML = '<table>' +
    (S.history||[]).map(h => '<tr><td class="mono">' + esc(h.time||'') + '</td><td>' + esc(h.title||'') +
      '</td><td><b>' + esc(h.result||'') + '</b></td><td>' + esc(h.note||'') + '</td></tr>').join('') +
    '</table>' || '(empty)';
  document.getElementById('pi').textContent = S.pi_output || '(none yet)';
  document.getElementById('runlog').textContent = S.run_log || '(none yet)';
}
async function saveOnly() {
  const goal = document.getElementById('goal').value.trim();
  if (!goal) { document.getElementById('err').textContent = 'type a goal first'; return; }
  const r = await api('/api/goal', {goal, max_cycles: +document.getElementById('cycles').value || 10});
  if (r.error) document.getElementById('err').textContent = r.error; else refresh();
}
async function saveStart() { await saveOnly(); await act('start'); }
async function act(a) {
  const r = await api('/api/' + a, {});
  if (r.error) document.getElementById('err').textContent = r.error;
  setTimeout(refresh, 800);
}
refresh(); setInterval(refresh, 3000);
</script></body></html>
"""


def _read_json(path: Path, default):
    try:
        d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return d
    except (OSError, ValueError):
        return default


def _tail(path: Path, max_bytes: int = 20000, max_lines: int = 60) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = data.splitlines()
    return "\n".join(lines[-max_lines:])


def _proxy_get(route: str, timeout: float = 4.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{OR_PORT}{route}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _find_bash() -> str | None:
    cands = [os.environ.get("HELPER_BASH"),
             r"C:\Program Files\Git\bin\bash.exe",
             r"C:\Program Files\Git\usr\bin\bash.exe",
             shutil.which("bash")]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def _loop_status() -> dict:
    pid = None
    if LOOP_PID.exists():
        try:
            pid = int(LOOP_PID.read_text(encoding="utf-8", errors="replace").strip().split()[0])
        except (ValueError, IndexError, OSError):
            pid = None
    running = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            running = True
        except OSError:
            running = False
    # Fallback activity hint: run.log touched in the last 3 minutes.
    active_recently = False
    try:
        active_recently = time.time() - RUN_LOG.stat().st_mtime < 180
    except OSError:
        pass
    return {"running": running, "pid": pid, "active_recently": active_recently,
            "stop_requested": (STATE / "STOP").exists()}


def _spawn_loop() -> dict:
    st = _loop_status()
    if st["running"]:
        return {"ok": True, "pid": st["pid"], "already": True}
    bash = _find_bash()
    if not bash:
        return {"error": "bash not found (install Git for Windows)"}
    helper = (HERE / "helper.sh").as_posix()
    env = dict(os.environ, AUTO_CODER="1", HELPER_ROOT=str(ROOT),
               HELPER_STATE=str(STATE), OR_STATE=str(STATE), OR_PORT=str(OR_PORT))
    try:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        logf = open(RUN_LOG, "a", encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": f"cannot open run log: {e}"}
    try:
        kw: dict = {"cwd": str(ROOT), "env": env, "stdout": logf, "stderr": subprocess.STDOUT}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kw["start_new_session"] = True
        p = subprocess.Popen([bash, helper, "run"], **kw)
    except Exception as e:
        return {"error": f"cannot start loop: {e}"}
    try:
        LOOP_PID.write_text(str(p.pid), encoding="utf-8")
        (STATE / "STOP").unlink(missing_ok=True)
    except OSError:
        pass
    return {"ok": True, "pid": p.pid}


def _stop_loop() -> dict:
    try:
        (STATE / "STOP").write_text("stop requested from console\n", encoding="utf-8")
    except OSError as e:
        return {"error": f"cannot write STOP: {e}"}
    st = _loop_status()
    if st["running"] and st["pid"]:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(st["pid"])], capture_output=True, timeout=15)
            else:
                os.kill(st["pid"], signal.SIGTERM)
        except Exception:
            pass
    return {"ok": True}


def status_payload() -> dict:
    proxy_info = _proxy_get("/current") or {}
    usage = _read_json(STATE / "usage.json", {})
    if not isinstance(usage, dict):
        usage = {}
    hist = []
    try:
        lines = (STATE / "history.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-8:]:
            if line.strip():
                try:
                    hist.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    plan = _read_json(STATE / "next_plan.json", {})
    verify = _read_json(STATE / "last_verify.json", {})
    proxy_up = _proxy_get("/health") is not None
    return {
        "goal": hlib.goal_text()[:4000],
        "goal_meta": hlib.goal_meta(),
        "loop": _loop_status(),
        "proxy": {"up": proxy_up,
                  "current": (proxy_info.get("model") if isinstance(proxy_info, dict) else None)},
        "usage": {"total": usage.get("total", 0), "daily": usage.get("daily", 1000),
                  "counts": usage.get("counts", {})},
        "history": hist,
        "next_plan": {"title": plan.get("title", ""), "model": plan.get("model", ""),
                      "time": plan.get("time", 0),
                      "files": (plan.get("files", []) or [])[:10]} if isinstance(plan, dict) else {},
        "last_verify": {"pass": verify.get("pass"), "problems": (verify.get("problems", []) or [])[:6],
                        "changed": (verify.get("changed", []) or [])[:12]} if isinstance(verify, dict) else {},
        "pi_output": _tail(STATE / "pi_last_output.txt", 12000, 40),
        "run_log": _tail(RUN_LOG, 20000, 50),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self) -> None:
        data = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send_html()
        if self.path.startswith("/api/status"):
            return self._send(200, status_payload())
        if self.path.startswith("/api/log"):
            name = self.path.split("log", 1)[1].strip("?/ =")
            allow = {"run": RUN_LOG, "pi": STATE / "pi_last_output.txt",
                     "proxy": STATE / "proxy.log"}
            p = allow.get(name, RUN_LOG)
            return self._send(200, {"log": _tail(p, 30000, 80)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            return self._send(400, {"error": "bad Content-Length"})
        if n < 0 or n > MAX_POST:
            return self._send(413 if n > MAX_POST else 400, {"error": "bad body size"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if not isinstance(body, dict):
            return self._send(400, {"error": "bad json"})
        if self.path.startswith("/api/goal"):
            goal = str(body.get("goal", "")).strip()
            if not goal:
                return self._send(400, {"error": "empty goal"})
            try:
                mc = int(body.get("max_cycles", 10))
            except (ValueError, TypeError):
                return self._send(400, {"error": "bad max_cycles"})
            try:
                meta = hlib.set_goal(goal, mc)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, {"ok": True, "meta": meta})
        if self.path.startswith("/api/start"):
            return self._send(200, _spawn_loop())
        if self.path.startswith("/api/stop"):
            return self._send(200, _stop_loop())
        if self.path.startswith("/api/clear-goal"):
            try:
                hlib.clear_goal()
            except Exception as e:
                return self._send(500, {"error": str(e)})
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=UI_PORT)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"helper console on http://127.0.0.1:{a.port}  (localhost only)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
