#!/usr/bin/env python3
"""Watch the project and push to GitHub whenever a file is saved.

    python tools/autopush.py                  # watch, commit and push (default)
    python tools/autopush.py --no-push        # watch and commit locally only
    python tools/autopush.py --once           # commit+push whatever changed now, then exit

How it works
------------
Every ``--interval`` seconds it looks at the modification times of every file git
would track (respecting ``.gitignore``, so ``out/``, caches and model weights are
never sent).  When something changed it waits until nothing has changed for
``--debounce`` seconds (so a burst of saves becomes one commit), then runs
``git add -A``, ``git commit`` and ``git push``.

It does not handle credentials.  It simply calls ``git push``, which uses whatever
you already have set up (Git Credential Manager sign-in, SSH key, ...).  If the
push fails (offline, not signed in) the commit is kept and the push is retried
on the next change / every ``--retry`` seconds.

Stop it with Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_git() -> str:
    g = shutil.which("git")
    if g:
        return g
    for cand in (r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files (x86)\Git\cmd\git.exe"):
        if os.path.exists(cand):
            return cand
    sys.exit("git was not found: install it from https://git-scm.com or add it to PATH")


GIT = find_git()


def git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([GIT, *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=check)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def snapshot() -> dict[str, tuple[float, int]]:
    """mtime + size of every non-ignored file (tracked or untracked)."""
    r = git("ls-files", "-co", "--exclude-standard", "-z")
    snap = {}
    for rel in filter(None, r.stdout.split("\0")):
        try:
            st = (ROOT / rel).stat()
        except OSError:
            snap[rel] = (0.0, -1)               # deleted
            continue
        snap[rel] = (st.st_mtime, st.st_size)
    return snap


def has_changes() -> bool:
    return bool(git("status", "--porcelain").stdout.strip())


def commit_changes() -> str | None:
    git("add", "-A")
    names = git("diff", "--cached", "--name-only").stdout.split()
    if not names:
        return None
    shown = ", ".join(names[:4]) + (f" (+{len(names) - 4} more)" if len(names) > 4 else "")
    msg = f"autosave: {shown}"
    r = git("commit", "-q", "-m", msg)
    if r.returncode != 0:
        log(f"commit failed: {(r.stderr or r.stdout).strip()[:300]}")
        return None
    log(f"committed {len(names)} file(s): {shown}")
    return msg


def unpushed() -> bool:
    r = git("rev-list", "--count", "@{u}..HEAD")
    if r.returncode != 0:                       # no upstream yet
        return git("rev-parse", "--verify", "-q", "HEAD").returncode == 0
    return r.stdout.strip() not in ("", "0")


def push() -> bool:
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "main"
    has_up = git("rev-parse", "--abbrev-ref", "@{u}").returncode == 0
    args = ["push"] if has_up else ["push", "-u", "origin", branch]
    r = git(*args)
    if r.returncode == 0:
        log("pushed to GitHub")
        return True
    log(f"push failed (will retry): {(r.stderr or r.stdout).strip()[-300:]}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between scans")
    ap.add_argument("--debounce", type=float, default=5.0,
                    help="seconds of quiet after the last save before committing")
    ap.add_argument("--retry", type=float, default=60.0,
                    help="seconds between retries of a failed push")
    ap.add_argument("--no-push", action="store_true", help="commit locally, never push")
    ap.add_argument("--once", action="store_true", help="sync once and exit")
    a = ap.parse_args()

    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        sys.exit(f"{ROOT} is not a git repository")
    if not a.no_push and git("remote", "get-url", "origin").returncode != 0:
        sys.exit("no 'origin' remote set: git remote add origin <url>")

    def sync() -> None:
        if has_changes():
            commit_changes()
        if not a.no_push and unpushed():
            push()

    if a.once:
        sync()
        return 0

    log(f"watching {ROOT}  (debounce {a.debounce}s, {'commit only' if a.no_push else 'push on'})"
        "  - Ctrl+C to stop")
    last = snapshot()
    dirty_since: float | None = None
    last_try = 0.0
    sync()                                       # catch up on anything already pending
    try:
        while True:
            time.sleep(a.interval)
            cur = snapshot()
            if cur != last:
                last = cur
                dirty_since = time.time()
            if dirty_since is not None and time.time() - dirty_since >= a.debounce:
                dirty_since = None
                sync()
                last_try = time.time()
            elif (dirty_since is None and not a.no_push and time.time() - last_try >= a.retry
                  and unpushed()):
                last_try = time.time()
                push()
    except KeyboardInterrupt:
        log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
