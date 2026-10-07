"""AZR control loop: reasoning proposes, physics disposes.

AZR is a code-reasoning model, not a balance controller. It must never drive
torques directly. Instead every proposed call passes through three stages:

1. **Assess (precondition gate)** — ``assess()`` checks the live body state
   (fallen, balance error, gait busy, distance to target) and either allows,
   rewrites (e.g. a redundant ``walk_to`` for something already in reach
   becomes ``face``), or denies the call with a reason AZR can learn from.
2. **Execute** — the surviving call runs on the existing controllers
   (whole-body stance hold, capture-point gait, analytic arm servo). AZR never
   touches ``data.ctrl`` itself.
3. **Verify (outcome check)** — ``verify_outcome()`` compares body snapshots
   before/after and returns a one-line outcome (fell / held / balance / still
   moving) that the mind appends to episodic memory and the next prompt's
   events. Acting in simulation and checking the result is the domain-level
   training signal: reasoning that repeatedly falls stops being proposed.

This is deliberately *not* fine-tuning the 3B weights (no GPU budget here);
it is the scaffolding fine-tuning would need: constrained decoding via the
whitelist + precondition gate + verified outcomes.
"""

from __future__ import annotations

import numpy as np

# Calls that move the base or commit a hand. Only one per thought.
LOCOMOTION = ("walk_to", "walk", "turn", "face", "grab", "put_down", "reach")
# Calls safe at any time.
ALWAYS_SAFE = ("say", "look_at", "look_forward", "hand_pose", "gesture",
               "blink", "touch_self", "express", "rub_eyes", "scratch",
               "wait", "nothing", "stop", "stand", "crouch", "release",
               "point_at", "pick_up")


def _balance(agent) -> float:
    try:
        return float(getattr(getattr(agent, "motor", None),
                             "balance_error", 0.0) or 0.0)
    except Exception:
        return 0.0


def _fallen(agent) -> bool:
    st = getattr(agent, "state", None)
    return bool(st is not None and st.fallen)


def _shoulder_dist(agent, xyz: np.ndarray) -> float | None:
    try:
        sk = agent.skills
        d = min(float(np.linalg.norm(xyz - sk._shoulder_pos(s)))
                for s in "lr")
        return d
    except Exception:
        return None


def _target_xyz(agent, name: str, args: tuple) -> np.ndarray | None:
    """World position of the call's target, if it resolves to one."""
    try:
        sk = agent.skills
        if not args:
            return None
        spec = sk.resolve(args[0])
        if spec is None:
            return None
        kind, val = spec
        w = sk.world
        if kind == "object":
            return w.obj_pos(val)
        if kind == "surface":
            return w.surface_point(val)
        if kind == "point":
            return np.array(val, float)
        if kind == "ego":
            return w.from_ego(val[0], val[1], val[2] if len(val) > 2 else 0.0)
        return None
    except Exception:
        return None


def assess(agent, name: str, args: tuple, kwargs: dict,
           loco_done: bool) -> dict:
    """Gate one AZR call. Returns {verdict, calls, reason}.

    verdict: "allow" (run as-is), "rewrite" (run ``calls`` instead),
    "deny" (skip; ``reason`` goes back to AZR as an event).
    """
    sk = agent.skills
    bal = _balance(agent)
    fallen = _fallen(agent)
    busy = bool(getattr(sk, "busy", False))
    gait_active = bool(getattr(getattr(agent, "gait", None), "active", False))

    if name in ("nothing",):
        return {"verdict": "allow", "calls": [(name, args, kwargs)], "reason": ""}
    if fallen and name in ("walk_to", "walk", "turn", "grab", "reach",
                            "put_down", "crouch", "face"):
        return {"verdict": "deny", "calls": [],
                "reason": f"{name}: I am on the floor, cannot move"}
    if name in LOCOMOTION and loco_done:
        return {"verdict": "deny", "calls": [],
                "reason": f"{name}: one move per turn; reconsider next thought"}
    if name in ("walk_to", "walk", "grab", "reach") and bal > 0.06:
        return {"verdict": "deny", "calls": [],
                "reason": f"{name}: off balance ({bal:.3f} m), stood still instead"}
    if name in ("walk_to", "walk", "turn") and (busy or gait_active):
        return {"verdict": "deny", "calls": [],
                "reason": f"{name}: body still moving, wait for it to finish"}

    # Proximity rewrite: walking to something already in reach only risks a
    # start/stop fall. Face it instead; grab/reach work from here.
    if name == "walk_to":
        xyz = _target_xyz(agent, name, args)
        if xyz is not None:
            d = _shoulder_dist(agent, xyz)
            try:
                from .skills import ARM_REACH
            except Exception:
                ARM_REACH = 0.54
            if d is not None and d < ARM_REACH + 0.18:
                return {"verdict": "rewrite",
                        "calls": [("face", args, {})],
                        "reason": f"walk_to: already within reach ({d:.2f} m), face instead"}
        # Unknown target: let skills raise the FAILED so AZR learns the name.
        return {"verdict": "allow", "calls": [(name, args, kwargs)], "reason": ""}

    # Grab out of reach already auto-walks inside the skill; nothing to add.
    return {"verdict": "allow", "calls": [(name, args, kwargs)], "reason": ""}


def snapshot(agent) -> dict:
    try:
        sk = agent.skills
        return {"t": float(agent.t),
                "fallen": _fallen(agent),
                "balance": _balance(agent),
                "held": dict(sk.held),
                "busy": bool(sk.busy),
                "doing": sk.current_description()}
    except Exception:
        return {}


def verify_outcome(before: dict, after: dict, name: str) -> str:
    """One-line verified outcome of a dispatched call."""
    if not before or not after:
        return f"{name}: outcome unknown"
    if after.get("fallen") and not before.get("fallen"):
        return f"{name}: FELL during execution (balance {after.get('balance', 0):.3f} m)"
    hb, ha = before.get("held", {}), after.get("held", {})
    got = [k for k in ha if ha[k] and ha[k] != hb.get(k)]
    lost = [k for k in hb if hb[k] and hb[k] != ha.get(k)]
    if got:
        return f"{name}: holding {got[0]} now, standing={not after.get('fallen')}"
    if lost:
        return f"{name}: released, standing={not after.get('fallen')}"
    if after.get("fallen"):
        return f"{name}: still on the floor"
    return (f"{name}: done, standing, "
            f"balance {before.get('balance', 0):.3f}->{after.get('balance', 0):.3f} m")
