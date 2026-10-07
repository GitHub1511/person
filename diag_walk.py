"""Walking diagnostic: stand, walk forward, optionally turn, stop.

    python diag_walk.py --speed 0.4 --walk 8
"""
import argparse
import time

import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--turn", type=float, default=0.0)
    ap.add_argument("--stand", type=float, default=1.0)
    ap.add_argument("--walk", type=float, default=6.0)
    ap.add_argument("--settle", type=float, default=4.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--y0", type=float, default=1.3, help="start this far back from the origin")
    ap.add_argument("--no-furniture", action="store_true", help="move the tables out of the way")
    ap.add_argument("--params", type=str, default="", help="JSON file of gait parameter overrides")
    a = ap.parse_args()

    ag = EmbodiedHuman(SimConfig(), vision=False, touch_sensors=False)
    ag.autonomous = False
    # start well back from the table, which otherwise stops a forward walk
    import mujoco
    ag.data.qpos[ag.meta.root_qpos_addr + 1] += a.y0
    if a.no_furniture:
        for i in range(ag.model.ngeom):
            nm = mujoco.mj_id2name(ag.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if nm.startswith("table_"):
                ag.model.geom_pos[i][1] += 60.0
    mujoco.mj_forward(ag.model, ag.data)
    if a.params:
        import json
        from pathlib import Path
        for k, v in json.loads(Path(a.params).read_text()).items():
            if k.startswith("wbc."):
                setattr(ag.gait.wbc, k[4:], v)
            else:
                setattr(ag.gait.P, k, v)
    dt = ag.dt
    n = int((a.stand + a.walk + a.settle) / dt)
    t_walk, t_stop = a.stand, a.stand + a.walk
    last = ""
    t0 = time.perf_counter()
    for i in range(n):
        t = ag.t
        if t >= t_walk and not ag.gait.cmd_active and t < t_stop:
            ag.gait.walk(a.speed, a.turn)
        if t >= t_stop and ag.gait.cmd_active:
            ag.gait.stop()
        ag.step()
        g = ag.gait.diag
        if a.verbose and i % 100 == 0 and ag.gait.active:
            print(f"t={ag.t:5.2f} {g.mode:6s} st={g.stance} com=({g.com[0]:+.3f},{g.com[1]:+.3f},{g.com[2]:.3f}) "
                  f"v=({g.com_vel[0]:+.2f},{g.com_vel[1]:+.2f}) dcm=({g.dcm[0]:+.3f},{g.dcm[1]:+.3f}) "
                  f"cop=({g.cop[0]:+.3f},{g.cop[1]:+.3f}) steps={g.steps}")
        mode = g.mode
        if mode != last:
            pl = ag.gait._foot_center(ag.data, 'l'); pr = ag.gait._foot_center(ag.data, 'r')
            print(f"  t={ag.t:5.2f} mode -> {mode:6s} stance={g.stance} steps={g.steps} "
                  f"com=({g.com[0]:+.2f},{g.com[1]:+.2f},{g.com[2]:.2f}) v=({g.com_vel[0]:+.2f},{g.com_vel[1]:+.2f}) "
                  f"dcm=({g.dcm[0]:+.2f},{g.dcm[1]:+.2f}) L=({pl[0]:+.2f},{pl[1]:+.2f}) R=({pr[0]:+.2f},{pr[1]:+.2f}) "
                  f"next=({g.step_target[0]:+.2f},{g.step_target[1]:+.2f}) head={ag.gait.heading:+.2f} "
                  f"yaw={ag.gait._heading_of(ag.data):+.2f}")
            last = mode
        if ag.state.fallen:
            print(f"FELL at t={ag.t:.2f}  com z={ag.state.com[2]:.2f}")
            break
    print("pin log (cop / torsion):", ag.gait.wbc.pin_log)
    com = ag.state.com
    print(f"end t={ag.t:.2f} com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:.3f}) steps={ag.gait.steps} "
          f"distance={ag.gait.diag.distance:.2f} m  wall={time.perf_counter()-t0:.1f}s "
          f"fallen={ag.state.fallen}")


if __name__ == "__main__":
    main()
