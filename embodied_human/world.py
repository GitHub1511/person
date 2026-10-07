"""
The world, as the body and the mind can query it.

Everything here is *ground truth read straight out of MuJoCo* (object poses,
furniture) filtered through what a body could plausibly know: an object is
"seen" only if it is inside the head's field of view and range, and positions
are always reported **egocentrically** (forward / left / up of the person) the
way a person would describe them, never as world coordinates.

This is the symbolic stand-in for a visual system.  The README says plainly
that the retina in :mod:`embodied_human.receptors` is 48x36 with hand-built
features; a language model cannot read that, so the mind is given this.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError("MuJoCo is required: python -m pip install mujoco") from exc

from .build_model import FURNITURE, TABLE_THICKNESS
from .locomotion import fwd, left


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class World:
    ODORANT_EMISSION = 0.02
    TASTANT_EMISSION = 0.01

    def __init__(self, agent, complexity=None):
        self.agent = agent
        self.complexity = complexity
        self.m = agent.model
        self.meta = agent.meta
        self.objects = {o.name: o for o in self.meta.objects}
        self.geom = {n: self.meta.geom_ids[n] for n in self.objects}
        self.furniture = {n: dict(x=x, y=y, top=top, hx=hx, hy=hy)
                          for n, x, y, top, hx, hy in FURNITURE}
        self.pelvis = self.meta.body_ids["pelvis"]
        self.head = self.meta.body_ids["head"]
        self.chest = self.meta.body_ids["chest"]
        self.stimuli = None
        self._setup_stimuli()

    def _setup_stimuli(self):
        if self.complexity is not None and getattr(self.complexity, 'stimuli', False):
            from . import stimuli
            self.stimuli = stimuli.default_stimuli()

    def get_odorant_at(self, point) -> float:
        if self.stimuli is not None:
            return self.stimuli.get_odorant_at(point)
        return 0.0

    def get_tastant_at(self, point) -> float:
        if self.stimuli is not None:
            return self.stimuli.get_tastant_at(point)
        return 0.0

    # ------------------------------------------------------------------
    @property
    def d(self):
        return self.agent.data

    def obj_pos(self, name: str) -> np.ndarray:
        return self.d.geom_xpos[self.geom[name]].copy()

    def obj_radius(self, name: str) -> float:
        o = self.objects[name]
        return float(max(o.size[:3]) if o.kind == "box" else o.size[0])

    def obj_vel(self, name: str) -> np.ndarray:
        # free-joint linear velocity
        a = self.meta.object_qpos_addr[name]
        # qvel address of a free joint = nv offset; derive from the joint
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        va = int(self.m.jnt_dofadr[jid])
        return self.d.qvel[va:va + 3].copy()

    def body_pos(self) -> np.ndarray:
        return self.d.xpos[self.pelvis].copy()

    def heading(self) -> float:
        """Heading of the body (rad); 0 faces -y, positive turns left."""
        R = self.d.xmat[self.pelvis].reshape(3, 3)
        f = -R[:, 1]
        return float(np.arctan2(f[0], -f[1]))

    def head_pos(self) -> np.ndarray:
        return self.d.xpos[self.head].copy() + np.array([0.0, 0.0, 0.09])

    # ------------------------------------------------------------------
    def to_ego(self, p: np.ndarray) -> tuple[float, float, float]:
        """World point -> (forward, left, up) relative to the person."""
        psi = self.heading()
        b = self.body_pos()
        dxy = np.asarray(p[:2]) - b[:2]
        return (float(np.dot(dxy, fwd(psi))), float(np.dot(dxy, left(psi))),
                float(p[2] - 0.0))

    def from_ego(self, forward: float, lft: float, up: float = 0.0) -> np.ndarray:
        psi = self.heading()
        b = self.body_pos()
        xy = b[:2] + forward * fwd(psi) + lft * left(psi)
        return np.array([xy[0], xy[1], up])

    # ------------------------------------------------------------------
    def surface_of(self, p: np.ndarray, r: float = 0.05) -> str | None:
        """Name of the furniture an object at ``p`` is resting on, if any."""
        for n, f in self.furniture.items():
            if abs(p[0] - f["x"]) <= f["hx"] and abs(p[1] - f["y"]) <= f["hy"] \
                    and abs(p[2] - r - f["top"]) < 0.06:
                return n
        if p[2] < r + 0.04:
            return "floor"
        return None

    def surface_point(self, name: str, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        f = self.furniture[name]
        return np.array([f["x"] + dx, f["y"] + dy, f["top"]])

    def nearest_free_spot(self, name: str, near: np.ndarray, r: float) -> np.ndarray:
        """A place on a surface, near ``near``, not on top of other objects."""
        f = self.furniture[name]
        best = None
        rng = np.random.default_rng(3)
        for _ in range(60):
            x = float(np.clip(near[0] + rng.normal(0, 0.12), f["x"] - f["hx"] + 0.08,
                              f["x"] + f["hx"] - 0.08))
            y = float(np.clip(near[1] + rng.normal(0, 0.10), f["y"] - f["hy"] + 0.08,
                              f["y"] + f["hy"] - 0.08))
            ok = True
            for on in self.objects:
                q = self.obj_pos(on)
                if np.hypot(q[0] - x, q[1] - y) < r + self.obj_radius(on) + 0.04 \
                        and abs(q[2] - f["top"]) < 0.2:
                    ok = False
                    break
            if ok:
                best = np.array([x, y, f["top"]])
                break
        if best is None:
            best = np.array([f["x"], f["y"], f["top"]])
        return best

    # ------------------------------------------------------------------
    def in_view(self, p: np.ndarray, half_fov: float = 1.40, rng: float = 6.0) -> bool:
        """Inside the visual field: about +-80 deg horizontally and from 60 deg
        below to 45 deg above the line of sight (human: ~+-100 / -70..+55)."""
        h = self.head_pos()
        R = self.d.xmat[self.head].reshape(3, 3)
        gaze = -R[:, 1]                       # forward
        lft = R[:, 0]
        up = R[:, 2]
        v = np.asarray(p) - h
        dist = float(np.linalg.norm(v))
        if dist < 1e-6 or dist > rng:
            return dist < 1e-6
        f = float(np.dot(v, gaze))
        yaw = math.atan2(float(np.dot(v, lft)), f)
        pitch = math.atan2(float(np.dot(v, up)), math.hypot(f, float(np.dot(v, lft))))
        return abs(yaw) < half_fov and -1.25 < pitch < 0.80

    def describe_position(self, p: np.ndarray) -> str:
        f, l, u = self.to_ego(p)
        dist = float(np.hypot(f, l))
        if dist < 0.05:
            return "right here"
        side = "left" if l > 0 else "right"
        parts = [f"{abs(f):.1f} m {'ahead' if f >= 0 else 'behind'}"]
        if abs(l) >= 0.08:
            parts.append(f"{abs(l):.1f} m to the {side}")
        return ", ".join(parts)

    def visible_objects(self) -> list[dict]:
        out = []
        for name, o in self.objects.items():
            p = self.obj_pos(name)
            if not self.in_view(p):
                continue
            on = self.surface_of(p, self.obj_radius(name))
            held = self.agent.skills.held_by(name) if hasattr(self.agent, "skills") else None
            out.append({"name": name, "label": o.label or name, "pos": p,
                        "where": self.describe_position(p), "on": on, "held": held,
                        "dist": float(np.hypot(*self.to_ego(p)[:2]))})
        out.sort(key=lambda r: r["dist"])
        return out

    def get_odorant_concentration(self, pos) -> float:
        return self.ODORANT_EMISSION if pos[2] > 0.1 else 0.0

    def get_tastant_concentration(self, pos) -> float:
        return self.TASTANT_EMISSION if (abs(pos[0]) < 0.1 and abs(pos[1]) < 0.1
                                              and 0.05 < pos[2] < 0.2) else 0.0

    def visible_furniture(self) -> list[dict]:
        out = []
        for name, f in self.furniture.items():
            p = np.array([f["x"], f["y"], f["top"]])
            if self.in_view(p, rng=9.0):
                out.append({"name": name, "where": self.describe_position(p),
                            "dist": float(np.hypot(*self.to_ego(p)[:2]))})
        return out
