"""
Learning to use its own body: which parts of a behaviour are safe.

Most descriptors in the behaviour space are harmless, a few are not (a deep
crouch while leaning, a fast sweeping reach with the trunk tilted ...).  A body
that has just been born into that space does not know which, so it falls over.
:class:`BodySafety` is a logistic model over the *parts* of a behaviour (head yaw,
arm schema, stance, speed ...) plus a few continuous summaries (how far the arms
are from rest, how fast they are asked to move) that predicts the chance that the
behaviour will unbalance the body.

It is learned in two ways:

* **offline, from many bodies at once** -- ``tools/train_body.py`` runs dozens
  of simulated copies of the person in parallel, each "babbling" random
  behaviours and recording what happened, and fits the model on everything they
  collected (``fit``);
* **online, over the person's own life** -- every behaviour it executes yields an
  outcome (did the centre of mass leave the feet, did it fall) and the weights
  take a small gradient step (``update``), so a body in a new situation keeps
  adapting.

The behaviour selector adds the predicted *hazard* to a candidate's expected
free energy, so unsafe behaviours become unattractive rather than impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .behavior_space import (ARM_VAR, HAND_VAR, N_ARM_SCHEMAS, N_HAND_SHAPES, Behavior,
                             BehaviorSpace, GAZE_PITCH, GAZE_YAW, BLINK_PATTERNS, HEAD_PITCH,
                             HEAD_TILT, HEAD_YAW, INTENT_MODES, JAW_LEVELS, LID_APERTURE,
                             OBJECTS, SELF_ACTIONS, SELF_REGIONS, STANCE_FORWARD,
                             STANCE_HEIGHT, STANCE_LATERAL, STYLE_AMP, STYLE_RHYTHM,
                             STYLE_SPEED, STYLE_TREMOR, TORSO_BEND, TORSO_SIDE, TORSO_TWIST,
                             VOCALS, HOLD_SECONDS, WALK_SPEED, WALK_TURN)

DEFAULT_PATH = Path(__file__).resolve().parent / "body_safety.json"

# (name, size) of every one-hot component; the order defines the feature vector
COMPONENTS = [
    ("head_yaw", len(HEAD_YAW)), ("head_pitch", len(HEAD_PITCH)), ("head_tilt", len(HEAD_TILT)),
    ("gaze_yaw", len(GAZE_YAW)), ("gaze_pitch", len(GAZE_PITCH)),
    ("lid_aperture", len(LID_APERTURE)), ("blink", len(BLINK_PATTERNS)),
    ("jaw", len(JAW_LEVELS)), ("vocal", len(VOCALS)),
    ("torso_bend", len(TORSO_BEND)), ("torso_side", len(TORSO_SIDE)), ("torso_twist", len(TORSO_TWIST)),
    ("l_schema", N_ARM_SCHEMAS), ("r_schema", N_ARM_SCHEMAS),
    ("l_hand", N_HAND_SHAPES), ("r_hand", N_HAND_SHAPES),
    ("stance_h", len(STANCE_HEIGHT)), ("stance_l", len(STANCE_LATERAL)), ("stance_f", len(STANCE_FORWARD)),
    ("speed", len(STYLE_SPEED)), ("amp", len(STYLE_AMP)), ("tremor", len(STYLE_TREMOR)),
    ("rhythm", len(STYLE_RHYTHM)), ("hold", len(HOLD_SECONDS)),
    ("intent_mode", 1 + len(INTENT_MODES)), ("intent_obj", 1 + len(OBJECTS)),
    ("touch_region", 1 + len(SELF_REGIONS)), ("touch_hand", 3), ("touch_action", 1 + len(SELF_ACTIONS)),
    ("walk", 1 + len(WALK_SPEED) * len(WALK_TURN)),
]
OFFSET = {}
_n = 0
for _name, _size in COMPONENTS:
    OFFSET[_name] = _n
    _n += _size
N_ONEHOT = _n
N_CONT = 8
N_FEATURES = N_ONEHOT + N_CONT


class BodySafety:
    """Logistic model: P(behaviour unbalances the body)."""

    def __init__(self, space: BehaviorSpace, path: Path | None = DEFAULT_PATH, *,
                 load: bool = True):
        self.space = space
        self.w = np.zeros(N_FEATURES)
        self.b = -2.2
        self.n_seen = 0
        self.l2 = 0.02
        self.lr = 0.08
        ci = space.channel_index
        self.ci = ci
        self._nom_arm = {s: np.array(space.q_nom[[space.idx[f"sh_{s}_abd"], space.idx[f"sh_{s}_flex"],
                                                  space.idx[f"sh_{s}_rot"], space.idx[f"elbow_{s}"]]])
                         for s in "lr"}
        if load and path is not None and Path(path).exists():
            self.load(path)

    # ------------------------------------------------------------------
    def indices(self, d: np.ndarray) -> list[int]:
        ci = self.ci
        idx = []
        h = np.unravel_index(int(d[ci["head"]]), (len(HEAD_YAW), len(HEAD_PITCH), len(HEAD_TILT)))
        for nm, v in zip(("head_yaw", "head_pitch", "head_tilt"), h):
            idx.append(OFFSET[nm] + int(v))
        g = np.unravel_index(int(d[ci["eyes"]]), (len(GAZE_YAW), len(GAZE_PITCH)))
        idx += [OFFSET["gaze_yaw"] + int(g[0]), OFFSET["gaze_pitch"] + int(g[1])]
        l = np.unravel_index(int(d[ci["lids"]]), (len(LID_APERTURE), len(BLINK_PATTERNS)))
        idx += [OFFSET["lid_aperture"] + int(l[0]), OFFSET["blink"] + int(l[1])]
        m = np.unravel_index(int(d[ci["mouth"]]), (len(JAW_LEVELS), len(VOCALS)))
        idx += [OFFSET["jaw"] + int(m[0]), OFFSET["vocal"] + int(m[1])]
        t = np.unravel_index(int(d[ci["torso"]]), (len(TORSO_BEND), len(TORSO_SIDE), len(TORSO_TWIST)))
        for nm, v in zip(("torso_bend", "torso_side", "torso_twist"), t):
            idx.append(OFFSET[nm] + int(v))
        idx += [OFFSET["l_schema"] + int(d[ci["l_schema"]]), OFFSET["r_schema"] + int(d[ci["r_schema"]]),
                OFFSET["l_hand"] + int(d[ci["l_hand"]]), OFFSET["r_hand"] + int(d[ci["r_hand"]])]
        s = np.unravel_index(int(d[ci["stance"]]), (len(STANCE_HEIGHT), len(STANCE_LATERAL), len(STANCE_FORWARD)))
        for nm, v in zip(("stance_h", "stance_l", "stance_f"), s):
            idx.append(OFFSET[nm] + int(v))
        st = np.unravel_index(int(d[ci["style"]]), (len(STYLE_SPEED), len(STYLE_AMP), len(STYLE_TREMOR), len(STYLE_RHYTHM)))
        for nm, v in zip(("speed", "amp", "tremor", "rhythm"), st):
            idx.append(OFFSET[nm] + int(v))
        idx.append(OFFSET["hold"] + int(d[ci["hold"]]))
        it = int(d[ci["intent"]])
        if it > 0:
            o, mo = divmod(it - 1, len(INTENT_MODES))
            idx += [OFFSET["intent_mode"] + 1 + mo, OFFSET["intent_obj"] + 1 + o]
        else:
            idx += [OFFSET["intent_mode"], OFFSET["intent_obj"]]
        tc = int(d[ci["touch"]])
        if tc > 0:
            r, rest = divmod(tc - 1, 2 * len(SELF_ACTIONS))
            hnd, a = divmod(rest, len(SELF_ACTIONS))
            idx += [OFFSET["touch_region"] + 1 + r, OFFSET["touch_hand"] + 1 + hnd,
                    OFFSET["touch_action"] + 1 + a]
        else:
            idx += [OFFSET["touch_region"], OFFSET["touch_hand"], OFFSET["touch_action"]]
        wk = int(d[ci["walk"]])
        idx.append(OFFSET["walk"] + wk)
        return idx

    def continuous(self, b: Behavior) -> np.ndarray:
        """A few summaries the one-hots cannot express."""
        sp = self.space
        arm = 0.0
        for s in "lr":
            a = np.array([b.joints.get(f"sh_{s}_abd", 0.0), b.joints.get(f"sh_{s}_flex", 0.0),
                          b.joints.get(f"sh_{s}_rot", 0.0), b.joints.get(f"elbow_{s}", 0.0)])
            arm += float(np.sum((a - self._nom_arm[s]) ** 2))
        trunk = abs(b.joints.get("spine_bend", 0.0)) + abs(b.joints.get("chest_bend", 0.0)) \
            + abs(b.joints.get("spine_side", 0.0)) + abs(b.joints.get("spine_twist", 0.0))
        drop = -b.stance[0]
        lat = abs(b.stance[1])
        return np.array([arm * 0.25, arm * 0.25 * b.speed ** 2 * b.amp ** 2 * 0.25, trunk,
                         trunk * drop * 8.0, drop * 8.0 * b.speed * 0.5, lat * 30.0,
                         b.tremor * 10.0, (1.0 if (b.touch or b.intent) else 0.0)])

    def features(self, b: Behavior) -> np.ndarray:
        x = np.zeros(N_FEATURES)
        x[self.indices(b.desc)] = 1.0
        x[N_ONEHOT:] = self.continuous(b)
        return x

    # ------------------------------------------------------------------
    def logit(self, x: np.ndarray) -> float:
        return float(x @ self.w + self.b)

    def risk(self, b: Behavior) -> float:
        z = self.logit(self.features(b))
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))

    def update(self, b: Behavior, y: float, lr: float | None = None) -> float:
        """One online gradient step on the log-loss; returns the prediction error."""
        x = self.features(b)
        p = 1.0 / (1.0 + np.exp(-np.clip(self.logit(x), -30, 30)))
        g = (p - y)
        lr = self.lr if lr is None else lr
        self.w -= lr * (g * x + self.l2 * self.w)
        self.b -= lr * g
        self.n_seen += 1
        return float(g)

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 40, l2: float = 1.5) -> dict:
        """Ridge-regularised logistic regression by Newton / IRLS."""
        n, f = X.shape
        w = np.zeros(f)
        b = float(np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)))
        Xb = np.hstack([X, np.ones((n, 1))])
        wb = np.concatenate([w, [b]])
        R = np.eye(f + 1) * l2
        R[-1, -1] = 0.0
        for _ in range(iters):
            z = Xb @ wb
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            W = p * (1 - p) + 1e-4
            g = Xb.T @ (p - y) + R @ wb
            H = (Xb * W[:, None]).T @ Xb + R
            step = np.linalg.solve(H, g)
            wb -= step
            if np.abs(step).max() < 1e-5:
                break
        self.w, self.b = wb[:-1], float(wb[-1])
        self.n_seen = int(n)
        z = Xb @ wb
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return {"n": int(n), "pos_rate": float(y.mean()),
                "logloss": float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))}

    # ------------------------------------------------------------------
    def top_risks(self, k: int = 12) -> list[tuple[str, float]]:
        names = []
        for nm, size in COMPONENTS:
            for j in range(size):
                names.append(f"{nm}={j}")
        names += [f"cont{j}" for j in range(N_CONT)]
        order = np.argsort(-self.w)[:k]
        return [(names[i], float(self.w[i])) for i in order]

    def save(self, path: Path = DEFAULT_PATH) -> None:
        Path(path).write_text(json.dumps({"w": self.w.tolist(), "b": self.b,
                                          "n_seen": self.n_seen, "n_features": N_FEATURES}))

    def load(self, path: Path = DEFAULT_PATH) -> bool:
        try:
            d = json.loads(Path(path).read_text())
            if int(d.get("n_features", -1)) != N_FEATURES:
                return False
            self.w = np.array(d["w"], float)
            self.b = float(d["b"])
            self.n_seen = int(d.get("n_seen", 0))
            return True
        except (OSError, ValueError, KeyError):
            return False
