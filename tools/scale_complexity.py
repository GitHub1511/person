#!/usr/bin/env python
"""
Set, inspect and measure how complicated the person is.

    python tools/scale_complexity.py --list                    # the presets and their knobs
    python tools/scale_complexity.py --measure                 # build each level, count, time
    python tools/scale_complexity.py --measure rich extreme    # just those
    python tools/scale_complexity.py --set extreme             # make a level the default
    python tools/scale_complexity.py --set rich --override skin_density=3 neural_units=20000
    python tools/scale_complexity.py --reset                   # back to the built-in default

Setting a level writes ``embodied_human/complexity.json``; the package reads it once, at
import, because the skin geometry (and so the MuJoCo model) is generated from it.  The
environment variable ``PERSON_COMPLEXITY`` (a preset name or a JSON file) overrides it for
one run without changing anything.

``--measure`` builds a person at each level in a *separate process* and reports

* skin: taxels, receptor channels per taxel, tactile scalars
* every sensory scalar a single frame carries (skin + proprioception + spindles + ears +
  vestibular hair cells + nose + tongue + retina ...)
* the internal world: dynamic state variables (organs, chemistry, neural mass, clock, eyes...)
  and the episodic store
* the behaviour space (channels, arm configurations, descriptors)
* the cost: milliseconds per physics step and the resulting speed relative to real time
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _child() -> None:
    """Runs inside a subprocess: build, run a little, print one JSON line."""
    import numpy as np
    import tempfile
    from embodied_human.agent import EmbodiedHuman
    from embodied_human.complexity import C
    from embodied_human.config import SimConfig
    from embodied_human import receptors as R

    t0 = time.perf_counter()
    cfg = SimConfig(out_dir=Path(tempfile.gettempdir()) / "scale_complexity")
    vision = len(sys.argv) > 2 and sys.argv[2] == "vision"
    ag = EmbodiedHuman(cfg, vision=vision, touch_sensors=False)
    build = time.perf_counter() - t0
    steps = 1500
    for _ in range(300):
        ag.step()
    t1 = time.perf_counter()
    for _ in range(steps):
        ag.step()
    wall = time.perf_counter() - t1
    f = ag.frame
    out = {
        "level": C.name, "build_s": round(build, 1),
        "taxels": int(ag.receptors.tactile.n_taxels), "channels": R.N_TACTILE_CH,
        "tactile_scalars": int(f.tactile.size),
        "sensory_scalars": int(f.n_scalars()),
        "retina": bool(vision),
        "populations": {k: int(v.size) for k, v in f.ext.items()},
        "latent_dim": int(ag.latent_spec.dim),
        "ms_per_step": round(1000 * wall / steps, 2),
        "speed_x_realtime": round(steps * ag.dt / wall, 2),
        "fallen": bool(ag.state.fallen),
    }
    if ag.inner is not None:
        out["inner"] = ag.inner.counts()
    out["eyes_state_vars"] = int(ag.ocular.n_state)
    out["behaviour_space"] = ag.behavior.space.describe()
    out["behaviour_space"].pop("channels", None)
    print("@@" + json.dumps(out))


def measure(levels: list[str], vision: bool) -> list[dict]:
    rows = []
    for lv in levels:
        env = dict(os.environ)
        env["PERSON_COMPLEXITY"] = lv
        print(f"  measuring {lv} ...", flush=True)
        args = [sys.executable, __file__, "--child"] + (["vision"] if vision else [])
        p = subprocess.run(args, env=env, capture_output=True, text=True, cwd=ROOT)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("@@")]
        if not line:
            print(f"    FAILED: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.stdout[-300:]}")
            continue
        rows.append(json.loads(line[0][2:]))
    return rows


def show(rows: list[dict]) -> None:
    if not rows:
        return
    base = next((r for r in rows if r["level"] == "base"), rows[0])
    print()
    hdr = f"{'':34s}" + "".join(f"{r['level']:>14s}" for r in rows)
    print(hdr)

    def line(label, fn, fmt="{:,}"):
        vals = []
        for r in rows:
            try:
                vals.append(fmt.format(fn(r)))
            except Exception:
                vals.append("-")
        print(f"{label:34s}" + "".join(f"{v:>14s}" for v in vals))
    line("skin taxels", lambda r: r["taxels"])
    line("receptor channels per taxel", lambda r: r["channels"])
    line("sensory scalars per frame", lambda r: r["sensory_scalars"])
    line("  x the base body", lambda r: r["sensory_scalars"] / base["sensory_scalars"], "{:.1f}x")
    line("internal dynamic variables", lambda r: r["inner"]["dynamic_state_variables"] + 136)
    line("  neural-mass units", lambda r: r["inner"]["neural_mass_units"])
    line("  motor units", lambda r: r["inner"]["motor_units"])
    line("  chemistry analytes", lambda r: r["inner"]["chemistry_analytes"])
    line("  episodic store values", lambda r: r["inner"]["episodic_store_values"])
    line("behaviour channels", lambda r: r["behaviour_space"]["n_channels"])
    line("  configurations per arm", lambda r: r["behaviour_space"]["per_arm"])
    line("  descriptors (log10)", lambda r: r["behaviour_space"]["descriptors_log10"], "{:.1f}")
    line("latent dimension", lambda r: r["latent_dim"])
    line("ms per physics step", lambda r: r["ms_per_step"], "{:.2f}")
    line("speed (x real time)", lambda r: r["speed_x_realtime"], "{:.2f}")
    print("\n(internal dynamic variables include the original ~136: 67 interoceptive, 28 emotions,"
          " 25 neuromodulators, 16 drives)")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child()
        return 0
    from embodied_human.complexity import PRESETS, Complexity, _FILE, load, save
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--measure", nargs="*", metavar="LEVEL")
    ap.add_argument("--vision", action="store_true", help="measure with the retina enabled")
    ap.add_argument("--set", metavar="LEVEL")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    if a.reset:
        if _FILE.exists():
            _FILE.unlink()
        print("removed", _FILE)
    if a.list:
        cur = load()
        keys = list(asdict(PRESETS["base"]))
        print(f"current default: {cur.name}   ({_FILE if _FILE.exists() else 'built-in'})")
        print(f"{'':26s}" + "".join(f"{n:>10s}" for n in PRESETS))
        for k in keys:
            if k == "name":
                continue
            print(f"{k:26s}" + "".join(f"{str(asdict(p)[k]):>10s}" for p in PRESETS.values()))
    if a.set:
        if a.set not in PRESETS:
            sys.exit(f"unknown level {a.set!r}; choose from {', '.join(PRESETS)}")
        d = asdict(PRESETS[a.set])
        valid = set(d)
        for kv in a.override:
            k, _, v = kv.partition("=")
            if k not in valid:
                sys.exit(f"unknown knob {k!r}")
            cur = d[k]
            d[k] = type(cur)(json.loads(v)) if not isinstance(cur, bool) else v.lower() in ("1", "true", "yes")
        d["name"] = a.set + ("+custom" if a.override else "")
        c = Complexity(**d)
        print("wrote", save(c))
    if a.measure is not None:
        levels = a.measure or list(PRESETS)
        show(measure(levels, a.vision))
    if not (a.list or a.set or a.measure is not None or a.reset):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
