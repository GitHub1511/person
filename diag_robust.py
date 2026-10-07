"""Sweep stabilisation parameters for robust standing across seeds."""
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig


def trial(seed, prio_gain, prio_thr, hip_gain, horizon=15.0, vision=False):
    cfg = SimConfig(duration=horizon, log_every=10**9, seed=seed)
    cfg.vision.enabled = vision
    a = EmbodiedHuman(cfg, vision=vision)
    a.motor.prio_gain = prio_gain
    a.motor.prio_thr = prio_thr
    a.motor.M.hip_strategy_gain = hip_gain
    n = int(horizon / a.dt)
    for i in range(n):
        a.step()
        if a.state.root_pos[2] < 0.55:
            return a.t
    return horizon


def score(**kw):
    ts = [trial(s, **kw) for s in (7, 13, 21, 33)]
    return float(np.mean(ts)), float(np.min(ts)), ts


print("baseline: prio_gain=9 thr=0.025 hip=1.0")
for pg, pt, hg in [
    (9, 0.025, 1.0),
    (25, 0.010, 1.0),
    (25, 0.010, 0.3),
    (40, 0.008, 0.3),
    (25, 0.020, 0.6),
    (60, 0.006, 0.2),
    (60, 0.10, 0.2),
    (120, 0.10, 0.2),
    (200, 0.15, 0.1),
    (120, 0.20, 0.1),
]:
    mean, mn, ts = score(prio_gain=pg, prio_thr=pt, hip_gain=hg)
    print(f"  prio_gain={pg:3d} thr={pt:.3f} hip={hg:.1f} -> "
          f"mean={mean:5.2f}s min={mn:5.2f}s  {[round(t,1) for t in ts]}")
