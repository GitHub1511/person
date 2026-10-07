"""Walk in a circle (turning while walking) and turn on the spot.

    python diag_walk_turn.py --speed 0.2 --turn 0.33 --seconds 25
    python diag_walk_turn.py --speed 0.0 --turn 0.4 --seconds 10     # on the spot
"""
import argparse
import math
import time

import mujoco
import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.2)
    ap.add_argument("--turn", type=float, default=0.33)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--y0", type=float, default=1.0)
    a = ap.parse_args()
    cfg = SimConfig()
    cfg.rates.receptor = 100.0
    cfg.rates.afferent = 100.0
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    ag.data.qpos[ag.meta.root_qpos_addr + 1] += a.y0
    mujoco.mj_forward(ag.model, ag.data)
    t0 = time.perf_counter()
    last_print = 0.0
    fell = None
    while ag.t < 1.0 + a.seconds:
        if ag.t >= 1.0 and not ag.gait.cmd_active:
            ag.gait.walk(a.speed, a.turn)
        ag.step()
        if ag.t - last_print >= 2.0:
            last_print = ag.t
            c = ag.state.com
            yaw = ag.skills.world.heading()
            print(f"t={ag.t:5.1f} {ag.gait.diag.mode:6s} steps={ag.gait.steps:3d} com=({c[0]:+.2f},{c[1]:+.2f}) "
                  f"yaw={math.degrees(yaw):+7.1f} deg  speed={ag.gait.diag.speed:.2f}")
        if ag.state.fallen:
            fell = ag.t
            print("FELL at", round(ag.t, 2))
            break
    print(f"end t={ag.t:.1f} steps={ag.gait.steps} fell={fell} wall={time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
