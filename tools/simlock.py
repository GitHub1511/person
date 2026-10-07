#!/usr/bin/env python
"""Run a heavy command only when the machine can afford it.

    python tools/simlock.py [--mem-mb 1500] [--slots 2] [--timeout 1800] -- python tools/ultra_bench.py ...

Many workers (people, scripts, coding agents) may want to run simulations or benchmarks at
once, and each one can need 1-3 GB.  This waits until fewer than ``--slots`` such commands are
running *and* enough memory is free, then runs the command and releases its slot when it ends.
It does not limit anything else; commands that skip it are simply not counted.

Slots are files in ``.simlock/`` (git-ignored).  A slot whose process has died is reclaimed.
The exit code of the command is returned.  Windows and POSIX.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

LOCKDIR = Path(__file__).resolve().parent.parent / ".simlock"


def free_memory_mb() -> tuple[float, float]:
    """(free physical MB, free commit MB)."""
    if os.name == "nt":
        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                        ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        m = MS(); m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ap / 1e6, m.apf / 1e6
    try:
        info = dict(l.split(":") for l in open("/proc/meminfo"))
        kb = lambda k: float(info[k].split()[0]) / 1024
        return kb("MemAvailable"), kb("MemAvailable") + kb("SwapFree")
    except Exception:
        return 8000.0, 8000.0


def alive(pid: int) -> bool:
    if os.name == "nt":
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def try_take(slots: int) -> Path | None:
    LOCKDIR.mkdir(exist_ok=True)
    for i in range(slots):
        p = LOCKDIR / f"slot{i}.lock"
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return p
        except FileExistsError:
            try:
                pid = int(p.read_text() or 0)
            except Exception:
                pid = 0
            if pid and alive(pid):
                continue
            try:
                p.unlink()
            except OSError:
                pass
            return try_take(slots) if i == 0 else None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mem-mb", type=float, default=1500, help="memory this command needs")
    ap.add_argument("--slots", type=int, default=int(os.environ.get("SIMLOCK_SLOTS", 2)))
    ap.add_argument("--reserve-mb", type=float, default=2500, help="always leave this much free")
    ap.add_argument("--timeout", type=float, default=3600, help="give up waiting after this long (s)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    t0 = time.time()
    slot = None
    while True:
        phys, commit = free_memory_mb()
        if min(phys * 1.5, commit) >= a.mem_mb + a.reserve_mb:
            slot = try_take(a.slots)
            if slot:
                break
        if time.time() - t0 > a.timeout:
            print(f"simlock: gave up after {a.timeout:.0f}s (free {phys:.0f} MB, need {a.mem_mb + a.reserve_mb:.0f})",
                  file=sys.stderr)
            return 75
        time.sleep(2.0 + (os.getpid() % 7) * 0.3)
    try:
        return subprocess.call(cmd)
    finally:
        try:
            slot.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
