"""Fall-recovery test: knock the body down, verify it gets back up.

    python diag_getup.py
    python diag_getup.py --trials 2 --tip back

Trial: settle 1 s standing, tip the root over (quat + shove), step until
``state.fallen``, then ``api_stand_up()`` and step up to 30 s. Success =
not fallen and COM > 0.72 m. Prints per-trial verdict plus balance trace.
Exit 0 iff every trial stands again.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig

TIPS = {
    # axis-angle-ish quats: roll onto the side / pitch forward / backward
    "side": np.array([0.7071, 0.7071, 0.0, 0.0]),
    "forward": np.array([0.7071, 0.0, 0.7071, 0.0]),
    "back": np.array([0.7071, 0.0, -0.7071, 0.0]),
}


def knock_down(ag, tip: str) -> None:
    ra = ag.meta.root_qpos_addr
    # Open floor, well clear of the table (0,TABLE_Y) and shelf: pure floor
    # recovery, no furniture interference. Furniture cases are separate.
    ag.data.qpos[ra] = 1.2
    ag.data.qpos[ra + 1] = 0.6
    ag.data.qpos[ra + 2] = 0.45
    ag.data.qpos[ra + 3:ra + 7] = TIPS[tip] / np.linalg.norm(TIPS[tip])
    ag.data.qvel[:] = 0.0
    mujoco.mj_forward(ag.model, ag.data)
    for _ in range(int(1.5 / ag.dt)):
        ag.step()
        if ag.state is not None and ag.state.fallen:
            break


def trial(tip: str, auto: bool) -> dict:
    cfg = SimConfig(out_dir=Path("out"))
    cfg.rates.receptor = 100.0
    cfg.rates.afferent = 100.0
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    ag.gait.hold_stance = True
    for _ in range(int(1.0 / ag.dt)):
        ag.step()
    knock_down(ag, tip)
    fell0 = bool(ag.state.fallen)
    com0 = float(ag.state.com[2])
    if not auto:
        ag.skills.api_stand_up()
    t0 = ag.t
    bals = []
    com_peak = 0.0
    i = 0
    while ag.t - t0 < 45.0:
        ag.step()
        bals.append(ag.motor.balance_error)
        i += 1
        if i % 100 == 0:
            com_peak = max(com_peak, float(ag.state.com[2]))
    st = ag.state
    ok = (not st.fallen) and float(st.com[2]) > 0.72
    return {"tip": tip, "auto": auto, "fell_after_tip": fell0,
            "com_after_tip": round(com0, 3),
            "stood_up": bool(ok), "fallen": bool(st.fallen),
            "com_z": round(float(st.com[2]), 3),
            "com_peak": round(com_peak, 3),
            "mean_bal": round(float(np.mean(bals)), 4),
            "max_bal": round(float(np.max(bals)), 4),
            "events": [e for e in ag.skills.events if "stand_up" in e or "fallen" in e or "got back" in e]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tip", default="side", choices=list(TIPS))
    a = ap.parse_args()
    tips = [a.tip, "forward", "back"][:a.trials]
    rows = []
    for i, tip in enumerate(tips):
        r = trial(tip, auto=(i == len(tips) - 1))
        rows.append(r)
        print(f"[trial {i}] tip={r['tip']} auto={r['auto']} "
              f"fell_after_tip={r['fell_after_tip']} com0={r['com_after_tip']} "
              f"-> stood_up={r['stood_up']} fallen={r['fallen']} "
              f"com_z={r['com_z']} peak={r['com_peak']} mean_bal={r['mean_bal']} max_bal={r['max_bal']}")
        for e in r["events"]:
            print(f"    event: {e}")
    ok = all(r["stood_up"] for r in rows)
    print("RESULT:", "ALL STOOD UP" if ok else "STILL DOWN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
