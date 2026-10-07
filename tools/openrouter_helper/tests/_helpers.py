#!/usr/bin/env python3
"""Portable test helpers: free ports, bash discovery, process-portable kills.

Shared by test_gateway.py / test_run_loop.py / test_pi_coder.py so the
helper's own tests do not depend on hardcoded ports or Windows-only tools.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def find_bash() -> str:
    for cand in (os.environ.get("HELPER_BASH"), shutil.which("bash"),
                 r"C:\Program Files\Git\bin\bash.exe",
                 r"C:\Program Files\Git\usr\bin\bash.exe"):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("bash not found: install Git for Windows or put bash on PATH")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_http(url: str, timeout: float = 15.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except Exception:
            time.sleep(0.2)
    return False


def kill_proc(p: "subprocess.Popen | None") -> None:
    if p is None:
        return
    try:
        if p.poll() is not None:
            return
        if os.name == "nt":
            p.terminate()
        else:
            p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def kill_pid_file(pid_file: Path) -> None:
    try:
        pid = int(pid_file.read_text(encoding="utf-8", errors="replace").strip().split()[0])
    except (OSError, ValueError, IndexError):
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=15)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


if __name__ == "__main__":
    print(find_bash())
    print(free_port())
    sys.exit(0)
