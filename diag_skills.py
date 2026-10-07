"""Exercise the skills one by one while standing (no walking): head, hands,
reach, grab, hold, put down.  Writes a contact sheet of the stages.

    python diag_skills.py --obj apple --side left
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.viewer import SceneRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="apple")
    ap.add_argument("--hand", default=None)
    ap.add_argument("--advance", type=float, default=0.32,
                    help="metres the person is moved towards the table before starting")
    ap.add_argument("--out", default="out/skills")
    ap.add_argument("--steps", default="grab,put")
    a = ap.parse_args()

    cfg = SimConfig(out_dir=Path("out"))
    cfg.rates.receptor = 100.0
    cfg.rates.afferent = 100.0
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    ag.gait.hold_stance = True
    ra = ag.meta.root_qpos_addr
    ag.data.qpos[ra + 1] -= a.advance
    mujoco.mj_forward(ag.model, ag.data)
    sk = ag.skills
    view = SceneRenderer(ag, (480, 360))
    view.set_view("close")
    frames = []

    def snap(label):
        img = view.render()
        frames.append(img.copy())

    def run(seconds, every=None, label=None):
        n = int(seconds / ag.dt)
        k = int((every or seconds + 1) / ag.dt)
        for i in range(n):
            ag.step()
            if every and i % k == k - 1:
                snap(label)
            if ag.state.fallen:
                print("FELL at", ag.t)
                return False
        return True

    run(1.0)
    obj = a.obj
    p0 = sk.world.obj_pos(obj)
    print(f"{obj} at {np.round(p0, 3)}, body at {np.round(sk.world.body_pos(), 3)}")
    sk.api_look_at(obj)
    run(0.8)
    snap("look")
    for step in a.steps.split(","):
        if step == "grab":
            sk.api_grab(obj, a.hand)
        elif step == "put":
            sk.api_put_down("table", a.hand)
        elif step.startswith("wave"):
            sk.api_gesture("wave")
        elif step == "reach":
            sk.api_reach(obj, a.hand)
        elif step.startswith("pose:"):
            sk.api_hand_pose("right", step.split(":")[1])
            run(1.0)
            snap(step)
            continue
        t0 = ag.t
        while sk.busy and ag.t - t0 < 40.0:
            ok = run(0.5, every=0.5, label=step)
            if not ok:
                break
        print(f"after {step}: t={ag.t:.1f} result='{sk.last_result}' held={sk.held} "
              f"events={sk.events[-3:]}")
        p = sk.world.obj_pos(obj)
        print(f"   {obj} now at {np.round(p, 3)}")
    run(1.0)
    snap("end")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = 5
    n = len(frames)
    # keep it readable: at most 20 frames, evenly sampled
    if n > 20:
        idx = np.linspace(0, n - 1, 20).astype(int)
        frames = [frames[i] for i in idx]
        n = 20
    rows = (n + cols - 1) // cols
    w, h = frames[0].size
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % cols) * w, (i // cols) * h))
    path = out / "sheet.png"
    sheet.save(path)
    print("saved", path)


if __name__ == "__main__":
    main()
