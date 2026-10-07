#!/usr/bin/env python3
"""Short smoke simulation used by `helper.sh verify`:  python smoke.py <level> <sim-seconds>

Builds the person at the current complexity level, stands it (ambient behaviour on, as in
the mind-driven mode), runs a few simulated seconds and reports one JSON line starting
with '@@'.  Fails (non-zero exit) on any exception.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    level = argv[1] if len(argv) > 1 else "base"
    if level not in ("base", "rich", "extreme"):
        # Unknown levels fall back to base rather than crashing the import path.
        level = "base"
    os.environ["PERSON_COMPLEXITY"] = level
    try:
        secs = float(argv[2]) if len(argv) > 2 else 20.0
    except (ValueError, TypeError):
        print("bad sim-seconds", file=sys.stderr)
        return 2
    # Clamp: negative/huge values would hang or OOM the verifier.
    secs = max(1.0, min(secs, 120.0))
    ROOT = Path(os.environ.get("HELPER_ROOT") or Path(__file__).resolve().parents[2])
    sys.path.insert(0, str(ROOT))

    import numpy as np  # noqa: E402

    from embodied_human.agent import EmbodiedHuman  # noqa: E402
    from embodied_human.config import SimConfig  # noqa: E402

    out_dir = Path(tempfile.mkdtemp(prefix="helper_smoke_"))
    cfg = SimConfig(seed=11, out_dir=out_dir)
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    try:
        dt = float(getattr(ag, "dt", 0.002))
        if not (dt > 0 and dt < 1.0):
            dt = 0.002
    except (TypeError, ValueError):
        dt = 0.002
    n = max(1, min(int(secs / dt), 600000))
    for _ in range(300):
        ag.step()
    t0 = time.perf_counter()
    for _ in range(n):
        ag.step()
    wall = time.perf_counter() - t0
    try:
        nan = bool(np.isnan(ag.latent).any() or np.isnan(ag.preferred).any())
    except Exception:
        nan = True
    print("@@" + json.dumps({"ms_per_step": 1000 * wall / n, "fallen": bool(ag.state.fallen), "nan": nan,
                             "latent_dim": int(ag.latent_spec.dim), "sim_s": secs}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
