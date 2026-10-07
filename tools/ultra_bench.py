#!/usr/bin/env python
"""Benchmark the ultra-tier subsystems one at a time, each in its own process.

    python tools/ultra_bench.py                       # every subsystem at ultra and mega
    python tools/ultra_bench.py --only afferents --levels ultra --secs 5
    python tools/ultra_bench.py --one NAME LEVEL SECS # (internal) run one and print JSON

It needs no simulator: inputs come from ``ultra.SyntheticBus``.  For each (subsystem, level) it
reports state variables, memory (declared and peak process memory), wall milliseconds per
simulated second, whether everything stayed finite, and whether the declared budgets
(``wall_budget_ms_per_sim_s``, ``ram_budget_mb``) hold.  Results go to ``out/ultra_bench.json``.
Run it through ``tools/simlock.py`` when other jobs are running.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def peak_rss_mb() -> float:
    if os.name == "nt":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e6
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def run_one(name: str, level: str, secs: float) -> dict:
    os.environ["PERSON_COMPLEXITY"] = level
    from embodied_human import ultra
    errs = ultra.load_all()
    if name not in ultra.REGISTRY:
        return {"name": name, "level": level, "error": f"not registered ({errs})"}
    try:
        r = ultra.bench(ultra.REGISTRY[name], ultra.Level(level), sim_seconds=secs)
    except Exception as e:
        import traceback
        return {"name": name, "level": level, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:]}
    r["peak_process_mb"] = peak_rss_mb()
    ok = r["finite"]
    if r.get("budget_ms") is not None and r["ms_per_sim_s"] > r["budget_ms"]:
        ok = False
        r["over_time"] = True
    if r.get("budget_mb") is not None and max(r["mem_mb"], 0) > r["budget_mb"]:
        ok = False
        r["over_ram"] = True
    r["ok"] = ok
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", default=["ultra", "mega"])
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--one", nargs=3, metavar=("NAME", "LEVEL", "SECS"))
    ap.add_argument("--out", default=str(ROOT / "out" / "ultra_bench.json"))
    a = ap.parse_args()
    if a.one:
        print("JSON:" + json.dumps(run_one(a.one[0], a.one[1], float(a.one[2]))))
        return 0
    from embodied_human import ultra
    errs = ultra.load_all()
    for e in errs:
        print("IMPORT ERROR", e)
    names = [n for n in sorted(ultra.REGISTRY) if not a.only or n in a.only]
    rows = []
    print(f"{'subsystem':28s} {'level':6s} {'state vars':>12s} {'MB':>8s} {'peak MB':>8s} {'ms/sim-s':>10s} {'budget':>8s}  ok")
    for n in names:
        for lv in a.levels:
            p = subprocess.run([sys.executable, __file__, "--one", n, lv, str(a.secs)],
                               capture_output=True, text=True, timeout=1800)
            line = [l for l in p.stdout.splitlines() if l.startswith("JSON:")]
            if not line:
                r = {"name": n, "level": lv, "error": (p.stderr or p.stdout)[-400:]}
            else:
                r = json.loads(line[0][5:])
            rows.append(r)
            if "error" in r:
                print(f"{n:28s} {lv:6s}  ERROR {r['error'][:150]}")
            else:
                print(f"{n:28s} {lv:6s} {r['n_state']:12,d} {r['mem_mb']:8.1f} {r['peak_process_mb']:8.0f} "
                      f"{r['ms_per_sim_s']:10.1f} {str(r['budget_ms']):>8s}  {'ok' if r['ok'] else 'FAIL'}")
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    bad = [r for r in rows if "error" in r or not r.get("ok")]
    tot = {lv: (sum(r.get("n_state", 0) for r in rows if r["level"] == lv),
                sum(r.get("mem_mb", 0) for r in rows if r["level"] == lv),
                sum(r.get("ms_per_sim_s", 0) for r in rows if r["level"] == lv)) for lv in a.levels}
    for lv, (ns, mb, ms) in tot.items():
        print(f"TOTAL {lv}: {ns:,} state variables, {mb:,.0f} MB, {ms:,.0f} ms per simulated second")
    print(f"{len(rows) - len(bad)}/{len(rows)} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
