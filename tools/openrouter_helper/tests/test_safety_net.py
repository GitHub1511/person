#!/usr/bin/env python3
"""Tests snapshot / verify / rollback / protection on a throwaway COPY of the project.

    python tests/test_safety_net.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import find_bash as _find_bash, free_port as _free_port, wait_http as _wait_http, kill_proc as _kill_proc, kill_pid_file as _kill_pid_file

HELPER = Path(__file__).resolve().parent.parent
ROOT = HELPER.parents[1]
BASH = _find_bash()
FAILS = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    sb = Path(tempfile.mkdtemp(prefix="helper_sandbox_"))
    ign = shutil.ignore_patterns("__pycache__", "state", "node_modules", ".env", "pi")
    shutil.copytree(ROOT / "embodied_human", sb / "embodied_human", ignore=ign)
    (sb / "tools").mkdir()
    shutil.copytree(HELPER, sb / "tools" / "openrouter_helper", ignore=ign)
    for f in ("run_sim.py", "README.md"):
        shutil.copy2(ROOT / f, sb / f)
    h = (sb / "tools" / "openrouter_helper" / "helper.sh").as_posix()
    env = dict(os.environ, HELPER_ROOT=str(sb), PYTHONIOENCODING="utf-8")
    env.pop("HELPER_STATE", None)
    env.pop("OR_STATE", None)

    def run(cmd, timeout=600):
        p = subprocess.run([BASH, h, cmd], env=env, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()

    print("snapshot, then verify an untouched tree")
    out = run("snapshot")
    check("snapshot taken", "snapshot" in out and "files" in out, out)
    out = run("verify")
    check("verify passes on an unchanged tree", out.splitlines()[0].startswith("PASS"), out[-300:])

    print("a broken edit and a stray new file are caught and undone")
    fast = sb / "embodied_human" / "_fast.py"
    orig = fast.read_text()
    fast.write_text(orig + "\ndef broken(:\n")
    (sb / "embodied_human" / "newmod.py").write_text("x = 1\n")
    out = run("verify")
    check("verify fails on a syntax error", out.startswith("FAIL") and "does not compile" in out, out[-300:])
    out = run("rollback")
    check("rollback restores the file", fast.read_text() == orig, out)
    check("rollback removes the new file", not (sb / "embodied_human" / "newmod.py").exists())
    out = run("verify")
    check("verify passes again after rollback", out.splitlines()[0].startswith("PASS"), out[-300:])

    print("an import-time failure is caught")
    agent = sb / "embodied_human" / "agent.py"
    a0 = agent.read_text()
    agent.write_text(a0 + "\nraise RuntimeError('boom at import')\n")
    out = run("verify")
    check("verify fails when the package no longer imports", out.startswith("FAIL") and "import fails" in out, out[-300:])
    run("rollback")
    check("agent.py restored", agent.read_text() == a0)

    print("the helper itself and the key are protected")
    prox = sb / "tools" / "openrouter_helper" / "proxy.py"
    p0 = prox.read_text()
    prox.write_text(p0 + "\n# tampered\n")
    out = run("verify")
    check("verify fails when a protected file changed", out.startswith("FAIL") and "protected" in out, out[-300:])
    prox.write_text(p0)

    print("\n" + ("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    shutil.rmtree(sb, ignore_errors=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
