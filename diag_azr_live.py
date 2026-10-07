"""Live AZR mind test — uses the real model in LM Studio, no stubs.

    python diag_azr_live.py
    python diag_azr_live.py --url http://127.0.0.1:1234/v1 --model absolute_zero_reasoner-coder-3b

Builds the body at the table, asks the real AZR backend to think once after
being spoken to, dispatches the parsed calls into the skill system, steps the
sim, and reports balance/fall status. Exit 0 only if a thought came back and
the body is still standing.
"""
import argparse
import sys
from pathlib import Path

import mujoco

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.mind import Mind, OpenAICompatBackend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", default="absolute_zero_reasoner-coder-3b")
    ap.add_argument("--ask", default="could you pick up the apple?")
    a = ap.parse_args()

    cfg = SimConfig(out_dir=Path("out"))
    cfg.rates.receptor = 100.0
    cfg.rates.afferent = 100.0
    ag = EmbodiedHuman(cfg, vision=False, touch_sensors=False)
    ag.autonomous = False
    ag.gait.hold_stance = True
    ag.data.qpos[ag.meta.root_qpos_addr + 1] -= 0.32
    mujoco.mj_forward(ag.model, ag.data)
    for _ in range(1200):
        ag.step()

    backend = OpenAICompatBackend(base_url=a.url, model=a.model, timeout=180.0)
    mind = Mind(ag, backend)
    ag.skills.hear(a.ask)
    try:
        mind._think_once(False)
    except Exception as exc:
        print(f"BACKEND FAILED: {type(exc).__name__}: {exc}")
        return 2
    print("THOUGHT :", (mind.thought or "")[:400])
    print("ANSWER  :", (mind.last_reply.answer if mind.last_reply else "")[:400].replace("\n", " ; "))
    print("CALLS   :", mind.last_reply.calls if mind.last_reply else [])
    print("ERRORS  :", mind.last_reply.errors if mind.last_reply else [])
    print("QUEUE   :", [q[0] for q in ag.skills.queue])
    ok_think = bool(mind.thought)
    # Silence is legal, but for this prompted ask we want at least an attempt
    # to look/grab/speak; warn (not fail) if it stayed silent.
    if not (mind.last_reply and mind.last_reply.calls):
        print("NOTE: AZR stayed silent (legal); prompt hardening may need tuning.")
    for _ in range(int(10 / ag.dt)):
        ag.step()
        if ag.state.fallen or any(ag.skills.held.values()):
            break
    print(f"after acting: held={ag.skills.held} fallen={ag.state.fallen} "
          f"balance={ag.motor.balance_error:.4f} t={ag.t:.1f}")
    if ag.state.fallen:
        print("RESULT: body fell during AZR-directed action")
        return 3
    if not ok_think:
        print("RESULT: no thought returned")
        return 4
    print("RESULT: live AZR thought + dispatch OK, body standing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
