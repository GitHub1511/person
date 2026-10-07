"""Sweep for stability AND expressiveness with segmental postural priority."""
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig


def trial(seed, prio_gain, prio_thr, horizon=20.0, vision=False):
    cfg = SimConfig(duration=horizon, log_every=10**9, seed=seed)
    cfg.vision.enabled = vision
    a = EmbodiedHuman(cfg, vision=vision)
    a.motor.prio_gain = prio_gain
    a.motor.prio_thr = prio_thr
    n = int(horizon / a.dt)
    # expressiveness proxy: how far the voluntary target travels from nominal
    travel = 0.0
    prev = a.motor.q_nom.copy()
    pols = set()
    for i in range(n):
        a.step()
        pols.add(a.inference.current.name)
        if i % 20 == 0:
            travel += float(np.abs(a.voluntary_target - prev).sum())
            prev = a.voluntary_target.copy()
        if a.state.root_pos[2] < 0.55:
            return a.t, travel, len(pols)
    return a.t, travel, len(pols)


print(f"{'gain':>5} {'thr':>6}  {'mean t':>7} {'min t':>7} {'travel':>8} {'policies':>8}")
for pg, pt in [(30, 0.012), (8, 0.30), (6, 0.40), (4, 0.50), (10, 0.25),
               (6, 0.55), (3, 0.60)]:
    ts, tr, np_ = [], [], []
    for s in (7, 13, 21):
        t, tv, p = trial(s, pg, pt)
        ts.append(t); tr.append(tv); np_.append(p)
    print(f"{pg:5d} {pt:6.2f}  {np.mean(ts):7.2f} {min(ts):7.2f} "
          f"{np.mean(tr):8.2f} {np.mean(np_):8.1f}")
