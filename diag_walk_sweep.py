"""Random search over gait parameters, run in parallel.

    python diag_walk_sweep.py --n 120 --workers 20 --speed 0.3

Prints the best parameter sets by number of steps survived.  This is how the
defaults in ``GaitParams`` were tuned: the gait is a hybrid of analytic
control laws and a handful of gains that interact, and a search is far cheaper
than reasoning about every interaction by hand.
"""
import os

# one BLAS thread per worker: the WBC solves small dense systems, and without
# this N workers x M threads oversubscribe the machine
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np

RANGES = {
    "step_width": (0.16, 0.30),
    "t_ss": (0.34, 0.60),
    "t_ds": (0.08, 0.20),
    "beta_step": (0.2, 1.0),
    "k_dcm": (0.0, 5.0),
    "com_height": (0.82, 0.85),
    "swing_height": (0.03, 0.07),
    "cop_lat": (0.024, 0.040),
    "wn_tilt": (2.5, 8.0),
    "apa_horizon": (0.2, 0.5),
    "wz": (4.0, 8.0),
    "kp_swing": (120.0, 500.0),
    "kd_swing": (18.0, 50.0),
    "w_com": (150.0, 800.0),
    "w_pel": (15.0, 150.0),
    "k_yaw": (8.0, 50.0),
    "wbc.yaw_ratio": (0.02, 0.8),
    "wbc.w_swing_p": (8.0, 120.0),
    "wbc.w_swing_r": (3.0, 60.0),
    "wbc.margin": (0.6, 0.95),
    "wbc.wn_post": (8.0, 20.0),
    "leg_knee_ref": (0.15, 0.5),
    "posture_weight": (1.0, 10.0),
}


SMALL = {
    # beta must exceed 1 - exp(-omega*Tr) ~ 0.78 for the foot placement to be
    # stable (the error map is e' = (1 - beta) exp(omega Tr) e), so search above it
    "beta_step": (0.9, 1.15),
    "k_speed": (0.0, 1.2),
    "max_extra_step": (0.03, 0.16),
    "k_dcm": (0.0, 8.0),
    "t_ss": (0.30, 0.52),
    "t_ds": (0.08, 0.22),
    "cop_sag": (0.06, 0.10),
    "step_width": (0.18, 0.28),
    "com_height": (0.825, 0.852),
    "apa_horizon": (0.2, 0.45),
    "wbc.mu_slip": (0.5, 0.9),
    "wbc.margin": (0.7, 0.97),
    "wbc.w_swing_p": (15.0, 150.0),
    "kp_swing": (150.0, 450.0),
    "kd_swing": (20.0, 50.0),
    "wn_tilt": (3.0, 7.0),
    "w_pel": (20.0, 120.0),
}


YAW = {
    "wbc.yaw_ratio": (0.05, 2.0),
    "k_yaw": (8.0, 80.0),
    "d_yaw": (3.0, 20.0),
    "wbc.mu_torsion": (0.04, 0.12),
    "wbc.w_swing_r": (10.0, 150.0),
    "kr_swing": (100.0, 500.0),
    "kw_swing": (10.0, 50.0),
    "w_pel": (30.0, 150.0),
    "wn_tilt": (4.0, 9.0),
}


REACH = {
    # the legs straightened out (both knees at their limit) at the sixth step:
    # steps too long for the height of the hips.  Search lower hips, shorter steps.
    "com_height": (0.76, 0.83),
    "max_step_fwd": (0.16, 0.34),
    "max_extra_step": (0.0, 0.10),
    "k_speed": (0.0, 1.0),
    "swing_height": (0.03, 0.07),
    "step_width": (0.17, 0.25),
    "t_ss": (0.30, 0.45),
    "t_ds": (0.07, 0.14),
    "beta_step": (0.92, 1.1),
    "k_dcm": (0.5, 5.0),
    "leg_knee_ref": (0.2, 0.7),
}


ALL = {
    "com_height": (0.80, 0.83), "max_step_fwd": (0.20, 0.32), "max_extra_step": (0.0, 0.08),
    "k_speed": (0.1, 0.8), "swing_height": (0.03, 0.06), "step_width": (0.18, 0.24),
    "t_ss": (0.33, 0.45), "t_ds": (0.08, 0.14), "beta_step": (0.95, 1.1), "k_dcm": (1.0, 5.0),
    "leg_knee_ref": (0.3, 0.65), "wbc.wn_trunk": (3.0, 10.0), "trunk_weight": (0.2, 2.0),
    "wbc.yaw_ratio": (0.05, 1.0), "k_yaw": (20.0, 70.0), "wn_tilt": (5.0, 8.0),
    "w_pel": (40.0, 120.0), "wbc.margin": (0.7, 0.9), "wbc.mu_slip": (0.5, 0.75),
    "kp_swing": (150.0, 260.0), "kd_swing": (18.0, 30.0), "heading_pull": (0.0, 1.0),
    "k_path": (0.0, 0.6),
}


def run_one(args):
    idx, params, speed, walk_t = args
    from embodied_human.agent import EmbodiedHuman
    from embodied_human.config import SimConfig
    import os
    from pathlib import Path
    cfg = SimConfig(out_dir=Path(os.environ.get("TEMP", ".")) / f"walk_sweep_{os.getpid()}")
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    # start well back from the table, which otherwise stops a forward walk dead
    # (it did: every "survivor" of the first sweeps had simply walked into it)
    import mujoco
    ag.data.qpos[ag.meta.root_qpos_addr + 1] += float(os.environ.get("WALK_Y0", "1.3"))
    if os.environ.get("NO_TABLE"):
        for gi in range(ag.model.ngeom):
            nm = mujoco.mj_id2name(ag.model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            if nm.startswith("table_"):
                ag.model.geom_pos[gi][1] += 60.0
    mujoco.mj_forward(ag.model, ag.data)
    y_start = float(ag.data.qpos[ag.meta.root_qpos_addr + 1])
    for k, v in params.items():
        if k.startswith("wbc."):
            setattr(ag.gait.wbc, k[4:], v)
        else:
            setattr(ag.gait.P, k, v)
    dt = ag.dt
    n = int((1.0 + walk_t) / dt)
    t_fall = None
    for i in range(n):
        if ag.t >= 1.0 and not ag.gait.cmd_active:
            ag.gait.walk(speed, 0.0)
        ag.step()
        if ag.state.fallen:
            t_fall = ag.t
            break
    g = ag.gait
    com = ag.state.com
    return {"idx": idx, "params": params, "steps": g.steps, "fell": t_fall,
            "dist": float(y_start - com[1]), "lateral": float(abs(com[0])),
            "final_z": float(com[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--walk", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=str, default="")
    ap.add_argument("--base-file", type=str, default="")
    ap.add_argument("--jitter", type=float, default=1.0,
                    help="1 = sample the full ranges, <1 = perturb the base set")
    ap.add_argument("--out", type=str, default="out/walk_sweep.json")
    ap.add_argument("--small", action="store_true", help="use the reduced parameter set")
    ap.add_argument("--yaw", action="store_true", help="search the yaw/trunk-control parameters")
    ap.add_argument("--reach", action="store_true", help="search hip height / step length")
    ap.add_argument("--all", action="store_true", help="search the union of the gait parameters")
    ap.add_argument("--y0", type=float, default=1.3, help="start this far back from the origin")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    global RANGES
    if a.small:
        RANGES = SMALL
    if a.yaw:
        RANGES = YAW
    if a.reach:
        RANGES = REACH
    if a.all:
        RANGES = ALL
    if a.base_file:
        base = json.loads(Path(a.base_file).read_text())
    else:
        base = json.loads(a.base) if a.base else {}
    jobs = []
    for i in range(a.n):
        p = {}
        for k, (lo, hi) in RANGES.items():
            if k in base and a.jitter < 1.0:
                span = (hi - lo) * a.jitter
                v = float(np.clip(base[k] + rng.uniform(-span, span) * 0.5, lo, hi))
            else:
                v = float(rng.uniform(lo, hi))
            p[k] = v
        jobs.append((i, p, a.speed, a.walk))
    t0 = time.time()
    with mp.Pool(a.workers, maxtasksperchild=2) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    # a good walk survives AND actually travels: surviving while marching on
    # the spot is not walking
    res.sort(key=lambda r: -((10.0 if r["fell"] is None else 0.0)
                             + min(r["dist"], a.speed * a.walk * 0.9) + 0.01 * r["steps"]))
    print(f"{len(res)} runs in {time.time()-t0:.0f}s")
    for r in res[:8]:
        print(f"steps={r['steps']:3d} fell={r['fell']} dist={r['dist']:.2f} lat={r['lateral']:.2f}")
        print("   ", json.dumps({k: round(v, 3) for k, v in r["params"].items()}))
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    n_ok = sum(1 for r in res if r["fell"] is None)
    print(f"{n_ok}/{len(res)} survived the whole walk")


if __name__ == "__main__":
    mp.freeze_support()
    main()
