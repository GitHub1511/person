"""Sweep balance-controller variants with forced 'hold' and measure survival."""
import numpy as np
from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig


def trial(knee_mult=1.0, ankle_mult=1.0, hip_mult=1.0, kd_mult=1.0,
          use_balance=True, horizon=10.0, seed=7, verbose=False):
    cfg = SimConfig(duration=horizon, log_every=10**9, seed=seed)
    a = EmbodiedHuman(cfg, vision=False)
    # apply the multipliers
    for prefix, mult in (("knee", knee_mult), ("ankle", ankle_mult),
                         ("hip", hip_mult)):
        for i, jn in enumerate(a.motor.names):
            if jn.startswith(prefix):
                a.motor.kp[i] *= mult * kd_mult
                a.motor.kd[i] *= mult * kd_mult
    n = int(horizon / a.dt)
    minz = 9.0
    for i in range(n):
        a.step()
        a.inference.current = a.inference.policies[0]
        a.inference.hold_timer = 1e9
        a.voluntary_target = a.motor.q_nom.copy()
        if not use_balance:
            a.motor.strategy[:] = 0.0
            # neutralise the balance term by zeroing the CoP gains
            a.motor.mass = 0.0
        z = a.state.root_pos[2]
        minz = min(minz, z)
        if z < 0.55:
            return a.t, minz
    return a.t, minz


print("baseline (balance on, m=0 disables it):")
print(f"  balance OFF      -> survived {trial(use_balance=False)[0]:5.2f}s")
print(f"  balance ON       -> survived {trial()[0]:5.2f}s")

print("\nknee stiffness multiplier:")
for km in (0.6, 0.8, 1.0, 1.3, 1.8):
    t, mz = trial(knee_mult=km)
    print(f"  knee x{km:4.1f} -> survived {t:5.2f}s (min z {mz:.3f})")

print("\nankle stiffness multiplier:")
for am in (0.6, 0.8, 1.0, 1.3):
    t, mz = trial(ankle_mult=am)
    print(f"  ankle x{am:4.1f} -> survived {t:5.2f}s (min z {mz:.3f})")

print("\nglobal kd multiplier:")
for kdm in (0.7, 1.0, 1.5, 2.0):
    t, mz = trial(kd_mult=kdm)
    print(f"  kd x{kdm:4.1f} -> survived {t:5.2f}s (min z {mz:.3f})")

print("\nseeds (default gains):")
for s in (1, 7, 13, 21):
    t, mz = trial(seed=s)
    print(f"  seed {s:3d} -> survived {t:5.2f}s")
