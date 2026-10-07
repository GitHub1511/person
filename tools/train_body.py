#!/usr/bin/env python
"""
Learn, from many bodies at once, which behaviours are safe to perform.

    python tools/train_body.py                         # use most cores
    python tools/train_body.py --workers 10 --sim 240 --rounds 2
    python tools/train_body.py --eval-only             # just compare, do not refit

What it does
------------
1. **Babbling.**  ``--workers`` independent copies of the person each execute
   random behaviours from the generative behaviour space for ``--sim`` simulated
   seconds (half in autonomous mode, half in ambient mode).  For every
   behaviour they record whether it unbalanced the body: did the centre of mass
   leave the feet by more than 10 cm, or did the person fall (in which case the
   copy is rebuilt and carries on).
2. **Fitting.**  Everything is pooled and ``BodySafety`` (a logistic model over the
   parts of a behaviour) is fitted; ``--rounds`` > 1 repeats this, with the later
   rounds babbling *under the current model* so that they spend their time near
   the boundary between safe and unsafe.
3. **Evaluation.**  A fresh batch of copies runs (a) unconstrained and (b) with
   the learned model choosing, and the two are compared: unsafe behaviours per
   hundred, falls per simulated hour, and how much of the behaviour space (distinct
   behaviours, channel entropy) is still being used.

The model is written to ``embodied_human/body_safety.json`` and is loaded by every
new person; each person then keeps adapting it online from its own outcomes.

Training runs at ``base`` sensory complexity (the balance physics does not depend on
how many receptors there are, and it is several times faster).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["PERSON_COMPLEXITY"] = "base"          # must precede the package import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

MODEL_PATH = ROOT / "embodied_human" / "body_safety.json"


def build_agent(seed: int):
    import tempfile
    from embodied_human.agent import EmbodiedHuman
    from embodied_human.config import SimConfig
    cfg = SimConfig(seed=seed, out_dir=Path(tempfile.gettempdir()) / f"train_body_{os.getpid()}")
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    return ag


def babble(job: dict) -> dict:
    """One copy of the person: run behaviours, report outcomes."""
    seed, budget = job["seed"], job["sim"]
    mode = job["mode"]                      # 'auto' | 'ambient'
    mode_safe = job["safe"]                 # use the learned model while choosing
    ag = build_agent(seed)
    rng = np.random.default_rng(seed)

    def configure(ag):
        ag.autonomous = (mode == "auto")
        ex = ag.behavior
        ex.learn_online = False
        ex.max_hold = 3.0 if job.get("short", True) else None
        if mode_safe:
            ex.safety.load(MODEL_PATH)
            ex.safety_weight = 5.0
            ex.explore_boost = 1.0
            ex.desire_noise = 0.5
        else:
            ex.safety_weight = 0.0           # unbiased babbling
            ex.explore_boost = 1.7
            ex.desire_noise = 1.0
        if job.get("model_weight"):
            ex.safety.load(MODEL_PATH)
            ex.safety_weight = float(job["model_weight"])
        return ex

    ex = configure(ag)
    records: list[tuple] = []
    err = None
    total = 0.0
    falls = 0
    distinct = set()
    n_dec = 0
    chan_counts: list[dict] = []
    seen = 0
    t_wall = time.perf_counter()\n    try:\n      while total < budget:
        ag.step()
        # harvest finished behaviours
        if len(ex.outcomes) > seen:
            for o in ex.outcomes[seen:]:
                records.append((o[0].tolist(), int(o[1]), float(o[2]), float(o[3]), int(o[4])))
                distinct.add(o[0].tobytes())
                n_dec += 1
            seen = len(ex.outcomes)
        if ag.state.fallen:
            falls += 1
            ex.run_fell = True
            ex._close_outcome(ag.t)
            if len(ex.outcomes) > seen:
                for o in ex.outcomes[seen:]:
                    records.append((o[0].tolist(), 1, float(o[2]), float(o[3]), 1))
                    distinct.add(o[0].tobytes())
                    n_dec += 1
                seen = len(ex.outcomes)
            # a slow loss of balance is often the *previous* behaviour's doing
            if len(records) >= 2 and records[-2][3] < 6.0:
                r = records[-2]
                records[-2] = (r[0], 1, r[2], r[3], r[4])
            total += ag.t
            del ex
            del ag
            import gc
            gc.collect()                      # a MuJoCo model + data per body: do not pile them up
            ag = build_agent(int(rng.integers(0, 2 ** 31)))
            ex = configure(ag)
            seen = 0
        if ag.t + total >= budget:
            break
    total += ag.t if not ag.state.fallen else 0.0
    return {"records": records, "falls": falls, "sim": float(total), "decisions": n_dec,
            "distinct": len(distinct), "wall": time.perf_counter() - t_wall,
            "mode": mode, "safe": mode_safe}


def make_space():
    from embodied_human.behavior_space import BehaviorSpace
    from embodied_human.build_model import load_model
    from embodied_human.config import SimConfig
    from embodied_human.skeleton import nominal_posture
    _, _, meta = load_model(SimConfig(), include_touch_sensors=False)
    names = [n for _, n, _ in meta.joint_order]
    lo = np.array([j.lo for _, _, j in meta.joint_order])
    hi = np.array([j.hi for _, _, j in meta.joint_order])
    nom = nominal_posture()
    return BehaviorSpace(names, lo, hi, np.array([nom.get(n, 0.0) for n in names]))


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_pool(jobs: list[dict], workers: int) -> list[dict]:
    with mp.Pool(workers, maxtasksperchild=1) as pool:
        return pool.map(babble, jobs, chunksize=1)


def pooled(results: list[dict]) -> list[tuple]:
    out = []
    for r in results:
        out += r["records"]
    return out


def fit(space, recs: list[tuple]) -> dict:
    from embodied_human.body_learning import BodySafety
    ms = BodySafety(space, load=False)
    X = np.stack([ms.features(space.compile(np.array(r[0], dtype=np.int64))) for r in recs])
    y = np.array([r[1] for r in recs], float)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    ntr = int(0.8 * len(y))
    tr, te = idx[:ntr], idx[ntr:]
    ms.fit(X[tr], y[tr])
    sc = X[te] @ ms.w + ms.b
    a = auc(sc, y[te])
    base = float(y[te].mean())
    p = 1 / (1 + np.exp(-sc))
    ll = float(-np.mean(y[te] * np.log(p + 1e-9) + (1 - y[te]) * np.log(1 - p + 1e-9)))
    ll0 = float(-np.mean(y[te] * np.log(base + 1e-9) + (1 - y[te]) * np.log(1 - base + 1e-9)))
    info = ms.fit(X, y)          # final model on all data
    ms.save(MODEL_PATH)
    return {"n": len(y), "unsafe_rate": float(y.mean()), "heldout_auc": a,
            "heldout_logloss": ll, "constant_logloss": ll0, "top_risks": ms.top_risks(14)}


def summarize(label: str, results: list[dict]) -> dict:
    sim = sum(r["sim"] for r in results)
    falls = sum(r["falls"] for r in results)
    recs = pooled(results)
    unsafe = np.mean([r[1] for r in recs]) if recs else float("nan")
    dec = sum(r["decisions"] for r in results)
    dist = len(set(tuple(r[0]) for r in recs))
    s = {"label": label, "sim_seconds": sim, "behaviours": len(recs), "unsafe_per_100": 100 * unsafe,
         "falls": falls, "falls_per_sim_hour": falls / max(sim / 3600.0, 1e-9),
         "distinct": dist, "distinct_frac": dist / max(len(recs), 1)}
    print(f"  {label:34s} {len(recs):5d} behaviours | unsafe {100 * unsafe:5.1f}/100 | "
          f"falls {falls:3d} in {sim / 60:5.1f} sim-min ({s['falls_per_sim_hour']:.0f}/h) | "
          f"distinct {dist} ({100 * s['distinct_frac']:.0f}%)")
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=0, help="parallel copies (default: most cores)")
    ap.add_argument("--sim", type=float, default=200.0, help="simulated seconds per copy per round")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--eval-sim", type=float, default=150.0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing model")
    a = ap.parse_args()
    ncpu = os.cpu_count() or 4
    workers = a.workers or max(2, min(ncpu - 4, 12))
    print(f"{workers} parallel copies of the person, {a.sim:.0f} simulated seconds each per round")
    space = make_space()
    print(f"behaviour space: {len(space.layout)} channels, ~10^{sum(np.log10(float(s)) for _, s in space.layout):.1f} descriptors")
    if a.fresh and MODEL_PATH.exists():
        MODEL_PATH.unlink()

    all_results: list[dict] = []
    if not a.eval_only:
        for rnd in range(1, a.rounds + 1):
            t0 = time.time()
            jobs = []
            for w in range(workers):
                jobs.append({"seed": 1000 * rnd + w, "sim": a.sim,
                             "mode": "auto" if w % 2 == 0 else "ambient",
                             "safe": False,
                             # later rounds: half the copies babble under the current model
                             "model_weight": 2.0 if (rnd > 1 and w % 4 >= 2) else 0.0})
            res = run_pool(jobs, workers)
            all_results += res
            print(f"round {rnd}: {time.time() - t0:.0f}s wall")
            summarize(f"round {rnd} babbling", res)
            info = fit(space, pooled(all_results))
            print(f"  fitted on {info['n']} behaviours (unsafe rate {100 * info['unsafe_rate']:.1f}%): "
                  f"held-out AUC {info['heldout_auc']:.3f}, log-loss {info['heldout_logloss']:.3f} "
                  f"(constant predictor {info['constant_logloss']:.3f})")
        print("  riskiest parts:", ", ".join(f"{n} ({w:+.2f})" for n, w in info["top_risks"][:10]))

    print("evaluation (fresh copies, same budget)")
    ev = []
    for safe in (False, True):
        jobs = [{"seed": 9000 + w + (500 if safe else 0), "sim": a.eval_sim,
                 "mode": "auto" if w % 2 == 0 else "ambient", "safe": safe}
                for w in range(workers)]
        res = run_pool(jobs, workers)
        ev.append(summarize("with the learned safety model" if safe else "unconstrained babbling", res))
    out = {"workers": workers, "sim_per_copy": a.sim, "rounds": a.rounds,
           "training_behaviours": len(pooled(all_results)), "evaluation": ev}
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "train_body_report.json").write_text(json.dumps(out, indent=1))
    print("saved model:", MODEL_PATH, "| report: out/train_body_report.json")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
