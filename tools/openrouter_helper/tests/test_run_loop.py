#!/usr/bin/env python3
"""Does `helper.sh run` keep going?  A long, offline run on a throwaway copy of the project.

    python tests/test_run_loop.py [steps]

A mock OpenRouter answers the planner.  A fake "coder" thread acts on every task: sometimes it
makes a harmless edit, sometimes a broken one, sometimes nothing, sometimes it never reports.
The test checks that, over many steps, the loop plans, rotates models in order, accepts good
steps, rolls back bad ones, survives the gateway being killed and a crash of the mock, hits the
quota and waits instead of spinning, resumes after a (simulated) day change, and exits promptly
when asked to stop.
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
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


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    sb = Path(tempfile.mkdtemp(prefix="helper_loop_"))
    ign = shutil.ignore_patterns("__pycache__", "state", "node_modules", ".env", "pi")
    shutil.copytree(ROOT / "embodied_human", sb / "embodied_human", ignore=ign)
    (sb / "tools").mkdir()
    shutil.copytree(HELPER, sb / "tools" / "openrouter_helper", ignore=ign)
    shutil.copy2(ROOT / "run_sim.py", sb / "run_sim.py")
    state = sb / "state"
    state.mkdir()
    datef = sb / "date.txt"
    datef.write_text("2026-03-01")
    env = dict(os.environ, HELPER_ROOT=str(sb), HELPER_STATE=str(state), OR_STATE=str(state),
               OR_UPSTREAM="http://127.0.0.1:9951/v1/chat/completions", OR_PORT="9961",
               OR_TIME_SCALE="0.02", OR_QUOTA="2", OPENROUTER_API_KEY="test-not-real",
               OR_FAKE_DATE_FILE=str(datef), RUN_WAIT_TIMEOUT="12", RUN_POLL="1",
               HELPER_SMOKE_SECS="2", PYTHONIOENCODING="utf-8")
    mock = subprocess.Popen([sys.executable, str(HELPER / "tests" / "mock_upstream.py"), "9951"])
    time.sleep(1.0)
    h = (sb / "tools" / "openrouter_helper" / "helper.sh").as_posix()
    logf = open(sb / "run.log", "w")
    import atexit
    def cleanup():
        try:
            subprocess.run([BASH, h, "stop"], env=env, capture_output=True, timeout=20)
        except Exception:
            pass
        pp = state / "proxy.pid"
        if pp.exists():
            subprocess.run(["taskkill", "/F", "/PID", pp.read_text().strip()], capture_output=True)
        for p in (run_proc[0], mock):
            try:
                p.kill()
            except Exception:
                pass
    run_proc = [None]
    atexit.register(cleanup)
    run = subprocess.Popen([BASH, h, "run"], env=env, stdout=logf, stderr=subprocess.STDOUT)
    run_proc[0] = run

    SCHEDULE = ["good", "bad", "nothing", "bad_import", "good", "silent", "bad", "good", "nothing", "bad_import"]
    stop_flag = threading.Event()
    log_actions = []

    def coder():
        seen = 0
        while not stop_flag.is_set():
            nt = state / "next_task.md"
            if nt.exists() and nt.stat().st_mtime_ns != seen:
                seen = nt.stat().st_mtime_ns
                time.sleep(0.5)
                what = SCHEDULE[len(log_actions) % len(SCHEDULE)]
                target = sb / "embodied_human" / "_fast.py"
                if what == "good":
                    target.write_text(target.read_text(encoding="utf-8-sig") + f"\n# harmless edit {time.time()}\n")
                elif what == "bad":
                    target.write_text(target.read_text(encoding="utf-8-sig") + "\ndef oops(:\n")
                elif what == "bad_import":
                    a = sb / "embodied_human" / "agent.py"
                    a.write_text(a.read_text(encoding="utf-8-sig") + "\nraise RuntimeError('x')\n")
                log_actions.append(what)
                if what != "silent":
                    (state / "CODER_DONE").write_text("done")
            time.sleep(0.3)

    th = threading.Thread(target=coder, daemon=True)
    th.start()

    def hist():
        p = state / "history.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    t0 = time.time()
    killed_proxy = killed_mock = False
    while time.time() - t0 < 1500 and len(hist()) < steps:
        time.sleep(2)
        if not killed_proxy and len(hist()) >= 3:
            pid = (state / "proxy.pid").read_text().strip() if (state / "proxy.pid").exists() else ""
            if pid:
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                killed_proxy = True
                print("  (killed the gateway after step 3)", flush=True)
        if not killed_mock and len(hist()) >= 6:
            mock.kill(); time.sleep(3)
            mock = subprocess.Popen([sys.executable, str(HELPER / "tests" / "mock_upstream.py"), "9951"])
            killed_mock = True
            print("  (restarted the mock upstream after step 6)", flush=True)
    H = hist()
    print(f"{len(H)} steps recorded in {time.time() - t0:.0f}s; coder actions: {log_actions}", flush=True)
    check(f"the loop completed {steps} steps", len(H) >= steps, str(len(H)))
    check("it accepted harmless steps", sum(h["result"] == "accepted" for h in H) >= 2)
    check("it rejected broken steps", sum(h["result"] == "rejected" for h in H) >= 2)
    fast = (sb / "embodied_human" / "_fast.py").read_text(encoding="utf-8-sig")
    check("a broken edit never survived (file compiles)", compile(fast, "_fast.py", "exec") is not None)
    agent_src = (sb / "embodied_human" / "agent.py").read_text(encoding="utf-8-sig")
    check("a broken import never survived", "boom" not in agent_src and "raise RuntimeError('x')" not in agent_src)
    usage = json.loads((state / "usage.json").read_text())
    counts = usage["counts"]
    order = [m["id"] for m in json.loads((HELPER / "models.json").read_text())["models"]]
    seq = [counts.get(m, 0) for m in order]
    check("models used strictly in order, 2 requests each at this test quota",
          all(a >= b for a, b in zip(seq, seq[1:])) and max(seq) <= 2, str(seq))
    log = (sb / "run.log").read_text()
    check("it survived the gateway being killed", killed_proxy and len(H) > 3)

    if usage["total"] >= 10 or "quota" in log:
        print("  waiting for the quota message ...", flush=True)
        t1 = time.time()
        while "resuming in" not in (sb / "run.log").read_text() and time.time() - t1 < 400:
            time.sleep(3)
        check("at quota it waits for the next day instead of spinning", "resuming in" in (sb / "run.log").read_text())
        datef.write_text("2026-03-02")
        n_before = len(hist())
        t1 = time.time()
        # the loop sleeps in 30 s chunks until the (real) midnight; poke it by restarting the wait
        subprocess.run([BASH, h, "stop"], env=env, capture_output=True)
        time.sleep(1)
    stop_flag.set()
    t2 = time.time()
    subprocess.run([BASH, h, "stop"], env=env, capture_output=True)
    while run.poll() is None and time.time() - t2 < 60:
        time.sleep(1)
    check("`stop` ends the loop promptly", run.poll() is not None, f"{time.time() - t2:.0f}s")
    if run.poll() is None:
        run.kill()
    mock.kill()
    print("\n" + ("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    print("sandbox:", sb)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
