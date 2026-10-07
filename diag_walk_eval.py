"""Re-test the best gait parameter sets from a sweep under different noise seeds
and speeds, to pick one that is robust rather than lucky.

    python diag_walk_eval.py out/walk_sweep5.json --top 10
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


def run(args):
    params, seed, speed, walk_t, y0 = args
    import mujoco
    from embodied_human.agent import EmbodiedHuman
    from embodied_human.config import SimConfig
    cfg = SimConfig(seed=seed, out_dir=Path(os.environ.get("TEMP", ".")) / f"walk_eval_{os.getpid()}")
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    ag.data.qpos[ag.meta.root_qpos_addr + 1] += y0
    # long walks need the furniture out of the way
    for gi in range(ag.model.ngeom):
        nm = mujoco.mj_id2name(ag.model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if nm.startswith("table_"):
            ag.model.geom_pos[gi][1] += 60.0
    mujoco.mj_forward(ag.model, ag.data)
    y_start = float(ag.data.qpos[ag.meta.root_qpos_addr + 1])
    merged = dict(params)
    merged.update(json.loads(os.environ.get("EVAL_OVERRIDE", "{}")))
    for k, v in merged.items():
        if k.startswith("wbc."):
            setattr(ag.gait.wbc, k[4:], v)
        else:
            setattr(ag.gait.P, k, v)
    fell = None
    for i in range(int((1.0 + walk_t) / ag.dt)):
        if ag.t >= 1.0 and not ag.gait.cmd_active:
            ag.gait.walk(speed, 0.0)
        ag.step()
        if ag.state.fallen:
            fell = ag.t
            break
    return {"fell": fell, "dist": y_start - float(ag.state.com[1]), "steps": ag.gait.steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--walk", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--speed", type=float, default=0.3)
    a = ap.parse_args()
    res = json.loads(Path(a.sweep).read_text())
    surv = [r for r in res if r["fell"] is None][:a.top]
    jobs = []
    for i, r in enumerate(surv):
        for seed in range(1, a.seeds + 1):
            jobs.append((i, (r["params"], seed, a.speed, a.walk, 0.5)))
    t0 = time.time()
    with mp.Pool(a.workers, maxtasksperchild=2) as pool:
        out = pool.map(run, [j[1] for j in jobs], chunksize=1)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s")
    table = {}
    for (i, _), o in zip(jobs, out):
        table.setdefault(i, []).append(o)
    ranked = []
    for i, lst in table.items():
        ok = sum(1 for o in lst if o["fell"] is None)
        dist = float(np.mean([o["dist"] for o in lst]))
        ranked.append((ok, dist, i))
    ranked.sort(reverse=True)
    for ok, dist, i in ranked:
        print(f"config {i}: survived {ok}/{len(table[i])}, mean distance {dist:.2f} m   "
              f"falls at {[None if o['fell'] is None else round(o['fell'], 1) for o in table[i]]}")
    best = ranked[0][2]
    print("best:", json.dumps({k: round(v, 3) for k, v in surv[best]["params"].items()}))
    Path("out").mkdir(exist_ok=True)
    Path("out/walk_best.json").write_text(json.dumps(surv[best]["params"], indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
