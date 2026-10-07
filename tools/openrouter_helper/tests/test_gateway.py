#!/usr/bin/env python3
"""Offline tests for the gateway (proxy.py) against a mock OpenRouter.

    python tests/test_gateway.py

Each scenario starts a fresh mock and a fresh gateway with its own state directory.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent
PY = sys.executable
FAILS = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class Rig:
    def __init__(self, port_mock, port_proxy, env=None, mock_env=None, state=None):
        self.state = Path(state or tempfile.mkdtemp(prefix="gw_state_"))
        self.pm, self.pp = port_mock, port_proxy
        base = dict(OR_UPSTREAM=f"http://127.0.0.1:{port_mock}/v1/chat/completions",
                    OR_PORT=str(port_proxy), OR_STATE=str(self.state), OR_TIME_SCALE="0.02",
                    OPENROUTER_API_KEY="test-not-real")
        base.update(env or {})
        self.env = dict(os.environ, **base)
        self.mock = subprocess.Popen([PY, str(HERE / "mock_upstream.py"), str(port_mock)],
                                     env=dict(os.environ, **(mock_env or {})))
        time.sleep(0.8)
        self.start()

    def start(self):
        self.proxy = subprocess.Popen([PY, str(HELPER / "proxy.py"), "--port", str(self.pp)], env=self.env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.pp}/health", timeout=1).read()
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("gateway did not start")

    def kill_proxy(self):
        self.proxy.kill(); self.proxy.wait()

    def ask(self, text="hello", **extra):
        body = {"messages": [{"role": "user", "content": text}], **extra}
        req = urllib.request.Request(f"http://127.0.0.1:{self.pp}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def usage(self):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.pp}/usage", timeout=3).read())

    def close(self):
        for p in (self.proxy, self.mock):
            try:
                p.kill()
            except Exception:
                pass


def main() -> int:
    models = [m["id"] for m in json.loads((HELPER / "models.json").read_text())["models"]]

    print("A. strict model order, a quota per model, then exhaustion")
    r = Rig(9921, 9931, env={"OR_QUOTA": "3"})
    seq = []
    for i in range(17):
        code, out = r.ask(f"q{i}")
        seq.append(out["x_helper"]["model"] if code == 200 else f"HTTP{code}")
    expect = [m for m in models for _ in range(3)]
    check("15 requests walk the five models in order, 3 each", seq[:15] == expect, str(seq[:15]))
    check("the 16th is refused with a retry time", seq[15] == "HTTP429")
    code, out = r.ask("x")
    check("refusal says quota_exhausted", out["error"].get("code") == "quota_exhausted"
          and out["error"]["retry_after"] > 0)
    check("usage counts are exact", sum(r.usage()["counts"].values()) == 15, str(r.usage()["counts"]))
    r.close()

    print("B. a 429 is retried and every attempt is counted")
    r = Rig(9922, 9932, mock_env={"MOCK_429_FIRST": "2"})
    code, out = r.ask("hi")
    check("succeeds after two 429s", code == 200 and out["choices"][0]["message"]["content"] == "OK")
    check("all three attempts counted", r.usage()["total"] == 3, str(r.usage()["total"]))
    r.close()

    print("C. a model that does not exist is skipped for the day")
    r = Rig(9923, 9933, mock_env={"MOCK_FAIL_MODEL": models[0]})
    code, out = r.ask("hi")
    check("falls through to the second model", code == 200 and out["x_helper"]["model"] == models[1],
          str(out.get("x_helper")))
    check("first model recorded as skipped", models[0] in r.usage()["skipped"])
    r.close()

    print("D. an over-large max_tokens steps down the ladder, and it is remembered")
    r = Rig(9924, 9934, mock_env={"MOCK_REJECT_BIG": "120000"})
    code, out = r.ask("hi")
    check("accepted at the lower rung", code == 200 and out["x_helper"]["max_tokens"] == 100000,
          str(out.get("x_helper")))
    code, out = r.ask("again")
    check("next request goes straight to the remembered rung", out["x_helper"]["max_tokens"] == 100000)
    r.close()

    print("E. counts survive a restart of the gateway")
    r = Rig(9925, 9935)
    for _ in range(3):
        r.ask("x")
    r.kill_proxy(); r.start()
    check("three requests still counted after restart", r.usage()["total"] == 3, str(r.usage()["total"]))
    r.ask("x")
    check("and counting continues", r.usage()["total"] == 4)
    r.close()

    print("F. the day rolls over by itself")
    ff = Path(tempfile.mkdtemp(prefix="gw_date_")) / "date.txt"
    ff.write_text("2026-01-01")
    r = Rig(9926, 9936, env={"OR_FAKE_DATE_FILE": str(ff), "OR_QUOTA": "1"})
    seq = [r.ask("x")[0] for _ in range(6)]
    check("one per model, then refused", seq == [200] * 5 + [429], str(seq))
    ff.write_text("2026-01-02")
    code, out = r.ask("new day")
    check("a new UTC date restores the quota", code == 200 and out["x_helper"]["model"] == models[0])
    r.close()

    print("G. never more than the per-minute limit (real time, 20/min => >= ~3 s apart)")
    r = Rig(9927, 9937, env={"OR_TIME_SCALE": "1.0", "OR_RPM": "20"})
    t0 = time.time()
    for _ in range(4):
        r.ask("x")
    el = time.time() - t0
    check("4 requests take at least 9 s", el >= 9.0, f"{el:.1f}s")
    r.close()

    print("H. the planner request is accepted by the full pipeline (probe)")
    r = Rig(9928, 9938)
    pr = subprocess.run([PY, str(HELPER / "proxy.py"), "--probe"], env=r.env, capture_output=True, text=True)
    check("probe reaches all five models", pr.stdout.count(" ok ") == 5, pr.stdout[-300:])
    r.close()

    print("\n" + ("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
