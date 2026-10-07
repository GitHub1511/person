"""AZR balance/move benchmark — real model only, no stubs.

    python diag_azr_balance.py
    python diag_azr_balance.py --cycles 4 --url http://127.0.0.1:1234/v1

Each cycle: speak/prompt AZR, dispatch through the control-loop gate,
step the sim up to 12 s (or until skill idle), then verify_outcome().
Reports per-cycle thought/calls/outcome plus totals: falls, holds acquired,
mean balance. Exit 0 iff zero falls.
"""
import argparse
import sys
from pathlib import Path

import mujoco

from embodied_human.agent import EmbodiedHuman
from embodied_human.config import SimConfig
from embodied_human.mind import Mind, OpenAICompatBackend

ASKS = [
    "look at the apple",
    "could you pick up the apple?",
    "put it down on the table",
    "look around the room",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", default="absolute_zero_reasoner-coder-3b")
    ap.add_argument("--cycles", type=int, default=4)
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
    falls = holds = 0
    bals = []
    for i in range(min(a.cycles, len(ASKS))):
        ag.skills.hear(ASKS[i])
        mind.poke()
        try:
            mind._think_once(False)
        except Exception as exc:
            print(f"[cycle {i}] BACKEND FAILED: {type(exc).__name__}: {exc}")
            return 2
        calls = mind.last_reply.calls if mind.last_reply else []
        print(f"[cycle {i}] ask={ASKS[i]!r}")
        print(f"  think: {(mind.thought or '')[:220]}")
        print(f"  calls: {calls} errors: {mind.last_reply.errors if mind.last_reply else []}")
        t0 = ag.t
        while ag.t - t0 < 12.0:
            ag.step()
            if ag.state.fallen:
                break
            if not ag.skills.busy and not ag.skills.queue:
                # let motion settle one more second
                if ag.t - t0 > 2.0:
                    break
        line = mind.verify_last()
        bals.append(ag.motor.balance_error)
        falls += int(ag.state.fallen)
        holds += int(any(ag.skills.held.values()))
        print(f"  outcome: {line} fallen={ag.state.fallen} "
              f"held={ag.skills.held} t={ag.t:.1f}")
        if ag.state.fallen:
            print("  body fell; stopping benchmark")
            break
    print(f"CYCLES={min(a.cycles, len(ASKS))} FALLS={falls} HOLDS={holds} "
          f"MEAN_BAL={sum(bals)/max(len(bals),1):.4f}")
    return 1 if falls else 0


if __name__ == "__main__":
    sys.exit(main())
