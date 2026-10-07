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

os.environ.setdefault("PERSON_COMPLEXITY", sys.argv[1] if len(sys.argv) > 1 else "base")
ROOT = Path(os.environ.get("HELPER_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from embodied_human.agent import EmbodiedHuman  # noqa: E402
from embodied_human.config import SimConfig  # noqa: E402

secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
cfg = SimConfig(seed=11, out_dir=Path(tempfile.gettempdir()) / "helper_smoke")
ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
ag.autonomous = False
n = int(secs / ag.dt)
for _ in range(300):
    ag.step()
t0 = time.perf_counter()
for _ in range(n):
    ag.step()
wall = time.perf_counter() - t0
nan = bool(np.isnan(ag.latent).any() or np.isnan(ag.preferred).any())
print("@@" + json.dumps({"ms_per_step": 1000 * wall / n, "fallen": bool(ag.state.fallen), "nan": nan,
                         "latent_dim": int(ag.latent_spec.dim), "sim_s": secs}))
