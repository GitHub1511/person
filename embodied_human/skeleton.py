"""
Skeleton specification for the embodied human.

This module is pure data: bones, joints, joint limits, actuator torque limits,
segment geometry and masses.  It is deliberately separate from the XML writer
so that the same description can be used to place skin taxels on real
geometric surfaces (see :mod:`embodied_human.skin`).

Conventions
-----------
* Each body's frame origin coincides with its *proximal joint*.
* Limbs (arms, legs) extend along **-z** in their own frame.
* The spine and neck extend along **+z**.
* Hinge axes are expressed in the *parent* body frame, matching MuJoCo.

Degrees of freedom (excluding the 6-DoF floating base)
------------------------------------------------------
spine 3, chest 3, neck 3, jaw 1, eyes 4, arms 2x12, legs 2x7  ->  52 actuated DoF
Total model DoF: 58 (52 hinge + 6 float)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Mass budget for a 75 kg adult (Dempster-style segment fractions)
# --------------------------------------------------------------------------
MASSES = {
    "pelvis": 9.3,
    "abdomen": 7.0,
    "chest": 14.0,
    "neck": 1.2,
    "head": 4.6,
    "jaw": 0.4,
    "eye": 0.03,
    "upper_arm": 2.1,
    "forearm": 1.5,
    "hand": 0.55,
    "fingers": 0.20,
    "thigh": 8.0,
    "shin": 4.2,
    "foot": 1.05,
    "toes": 0.35,
}


@dataclass
class Geom:
    """A collision/visual primitive in body-local coordinates."""
    kind: str                       # capsule | sphere | box | ellipsoid
    a: tuple = (0.0, 0.0, 0.0)      # proximal end (capsule) or centre
    b: tuple = (0.0, 0.0, 0.0)      # distal end (capsule)
    size: tuple = (0.0,)            # (radius,) / (half_z,) / (hx,hy,hz)
    mass: float = 0.0
    rgba: tuple = (0.86, 0.71, 0.62, 1.0)
    friction: tuple = (1.0, 0.02, 0.001)
    contype: int = 1
    conaffinity: int = 1
    name: str = ""

    def midpoint(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.a, float) + np.asarray(self.b, float))

    def axis(self) -> np.ndarray:
        v = np.asarray(self.b, float) - np.asarray(self.a, float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])

    def length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.b, float) - np.asarray(self.a, float)))

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return orthonormal (u, v, axis) with u, v perpendicular to the axis."""
        z = self.axis()
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(z, ref))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(ref, z)
        u /= np.linalg.norm(u)
        v = np.cross(z, u)
        return u, v, z


@dataclass
class Joint:
    """A hinge joint with human-plausible limits and actuator strength."""
    name: str
    axis: tuple
    lo: float
    hi: float
    torque: float                   # Nm, peak isometric
    damping: float = 1.0
    frictionloss: float = 0.6
    armature: float = 0.02
    kp: float = 40.0                # for the position-servo variant
    kind: str = "hinge"             # hinge | slide


@dataclass
class Bone:
    """A rigid body: parent link, mount offset, joints, geometry."""
    name: str
    parent: str
    pos: tuple
    joints: list = field(default_factory=list)
    geoms: list = field(default_factory=list)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    side: str = "c"                 # l | r | c  (left / right / centre)
    rgba: tuple = (0.86, 0.71, 0.62, 1.0)


# --------------------------------------------------------------------------
# Joint constructors (so the tables below stay readable)
# --------------------------------------------------------------------------

def hinge(name, axis, lo, hi, torque, **kw) -> Joint:
    return Joint(name=name, axis=axis, lo=lo, hi=hi, torque=torque, **kw)


def capsule(a, b, r, mass, **kw) -> Geom:
    return Geom("capsule", a=a, b=b, size=(r,), mass=mass, **kw)


def sphere(c, r, mass, **kw) -> Geom:
    return Geom("sphere", a=c, size=(r,), mass=mass, **kw)


def box(c, hx, hy, hz, mass, **kw) -> Geom:
    return Geom("box", a=c, size=(hx, hy, hz), mass=mass, **kw)


# Phalanx lengths (m): proximal + distal ~ the old single 6.2 cm segment
PHALANX_PROX = 0.036
PHALANX_DIST = 0.030

# --------------------------------------------------------------------------
# Skin / tissue colours
# --------------------------------------------------------------------------
SKIN = (0.87, 0.72, 0.63, 1.0)
SKIN_DARK = (0.78, 0.62, 0.54, 1.0)
MUCOUS = (0.83, 0.45, 0.45, 1.0)
SCLERA = (0.95, 0.95, 0.96, 1.0)
IRIS = (0.32, 0.22, 0.14, 1.0)
PUPIL = (0.02, 0.02, 0.02, 1.0)

# Eye socket position in the head frame, and the lid geometry built around it
EYE_X, EYE_Y, EYE_Z = 0.030, -0.0955, 0.078

# --------------------------------------------------------------------------
# Torque limits by joint family (Nm) - from isokinetic dynamometry literature
# --------------------------------------------------------------------------
T = {
    "spine_bend": 120.0, "spine_side": 80.0, "spine_twist": 55.0,
    "neck": 18.0, "jaw": 45.0, "eye": 0.05,
    "sh_flex": 80.0, "sh_abd": 70.0, "sh_rot": 40.0,
    "elbow": 60.0, "wrist_flex": 14.0, "wrist_dev": 10.0,
    # Real finger flexors produce a few newton-metres at the MCP joint; the
    # earlier 18 Nm would crush an apple and registers as phantom pain.
    "finger": 2.6, "thumb": 3.2,
    "hip_flex": 160.0, "hip_abd": 110.0, "hip_rot": 55.0,
    "knee": 200.0, "ankle_flex": 110.0, "ankle_inv": 60.0, "toe": 45.0,
}


# --------------------------------------------------------------------------
# The skeleton
# --------------------------------------------------------------------------
def build_bones() -> list[Bone]:
    """Return the full body tree in parent-before-child order.

    Joint-axis derivation
    ---------------------
    The character faces **-y**, up is **+z**, so its own left is **+x**.
    All limb segments point along **-z** in their own frame, which makes the
    axis semantics:

    * rotation about **+x** swings a distal segment towards **+y** (backwards),
      so *flexion* is **negative** x-rotation for shoulders/hips and
      **positive** x-rotation for knees (the heel goes back);
    * rotation about **+y** swings a distal segment towards **-x**, so frontal
      plane abduction is side-dependent;
    * rotation about **+z** is a twist about the limb's long axis.

    Getting these wrong silently removes the sagittal degree of freedom a
    body needs to stand up, so each joint is annotated with what it means.
    """
    bones: list[Bone] = []

    # ---------------- trunk ------------------------------------------------
    bones.append(Bone(
        "pelvis", "world", (0.0, 0.0, 0.98), side="c", rgba=SKIN,
        joints=[],
        geoms=[capsule((0, 0, 0.02), (0, 0, 0.14), 0.122, MASSES["pelvis"])],
    ))
    bones.append(Bone(
        "abdomen", "pelvis", (0.0, 0.0, 0.14), side="c", rgba=SKIN,
        joints=[
            # positive = trunk flexion (lean forward)
            hinge("spine_bend", (1, 0, 0), -0.35, 0.80, T["spine_bend"], armature=0.25),
            # positive = lean towards -x (character's right)
            hinge("spine_side", (0, 1, 0), -0.35, 0.35, T["spine_side"], armature=0.20),
            hinge("spine_twist", (0, 0, 1), -0.45, 0.45, T["spine_twist"], armature=0.15),
        ],
        geoms=[capsule((0, 0, 0.0), (0, 0, 0.11), 0.132, MASSES["abdomen"])],
    ))
    bones.append(Bone(
        "chest", "abdomen", (0.0, 0.0, 0.11), side="c", rgba=SKIN,
        joints=[
            hinge("chest_bend", (1, 0, 0), -0.25, 0.50, T["spine_bend"] * 0.8, armature=0.25),
            hinge("chest_side", (0, 1, 0), -0.25, 0.25, T["spine_side"] * 0.8, armature=0.20),
            hinge("chest_twist", (0, 0, 1), -0.40, 0.40, T["spine_twist"] * 0.8, armature=0.15),
        ],
        geoms=[capsule((0, 0, 0.0), (0, 0, 0.17), 0.145, MASSES["chest"])],
    ))

    # ---------------- neck & head -----------------------------------------
    bones.append(Bone(
        "neck", "chest", (0.0, 0.0, 0.17), side="c", rgba=SKIN,
        joints=[
            # positive = chin towards chest
            hinge("neck_bend", (1, 0, 0), -0.55, 0.65, T["neck"], armature=0.02,
                  damping=0.4, frictionloss=0.2),
            hinge("neck_side", (0, 1, 0), -0.45, 0.45, T["neck"], armature=0.02,
                  damping=0.4, frictionloss=0.2),
            hinge("neck_twist", (0, 0, 1), -0.90, 0.90, T["neck"], armature=0.02,
                  damping=0.4, frictionloss=0.2),
        ],
        geoms=[capsule((0, 0, 0.0), (0, 0, 0.07), 0.055, MASSES["neck"])],
    ))
    bones.append(Bone(
        "head", "neck", (0.0, 0.0, 0.075), side="c", rgba=SKIN,
        joints=[],
        geoms=[
            sphere((0.0, -0.01, 0.085), 0.098, MASSES["head"] * 0.82, rgba=SKIN),
            box((0.0, -0.075, 0.055), 0.070, 0.030, 0.040, MASSES["head"] * 0.10,
                rgba=SKIN, name="face"),
            sphere((0.0, -0.095, 0.055), 0.016, MASSES["head"] * 0.01,
                   rgba=SKIN_DARK, name="nose"),
        ],
    ))
    bones.append(Bone(
        "jaw", "head", (0.0, -0.045, 0.020), side="c", rgba=SKIN,
        # positive = mouth opens (chin travels down and back)
        joints=[hinge("jaw_open", (1, 0, 0), 0.0, 0.42, T["jaw"], armature=0.005,
                      damping=0.2, frictionloss=0.1)],
        geoms=[
            box((0.0, -0.030, -0.010), 0.052, 0.030, 0.014, MASSES["jaw"] * 0.6,
                rgba=SKIN, name="jaw_bone"),
            # tongue: soft tissue proxy, richly innervated
            box((0.0, -0.030, 0.006), 0.036, 0.026, 0.008, MASSES["jaw"] * 0.4,
                rgba=MUCOUS, name="tongue"),
        ],
    ))

    for s, sx in (("l", 1.0), ("r", -1.0)):
        # The eyes used to sit *inside* the face box (y = -0.070 vs a face front
        # at -0.105) and could not be seen.  They now bulge from the face, with
        # an iris and a pupil (visual only), and the lids in build_model.py.
        bones.append(Bone(
            f"eye_{s}", "head", (EYE_X * sx, EYE_Y, EYE_Z), side=s, rgba=SCLERA,
            joints=[
                hinge(f"eye_{s}_yaw", (0, 0, 1), -0.85, 0.85, T["eye"],
                      armature=1e-5, damping=0.005, frictionloss=0.001),
                # positive = look down
                hinge(f"eye_{s}_pitch", (1, 0, 0), -0.50, 0.50, T["eye"],
                      armature=1e-5, damping=0.005, frictionloss=0.001),
            ],
            geoms=[sphere((0, 0, 0), 0.0122, MASSES["eye"], rgba=SCLERA,
                          name=f"eyeball_{s}"),
                   sphere((0, -0.0085, 0), 0.0070, MASSES["eye"] * 0.02,
                          rgba=IRIS, name=f"vis_iris_{s}"),
                   sphere((0, -0.0128, 0), 0.0030, MASSES["eye"] * 0.01,
                          rgba=PUPIL, name=f"vis_pupil_{s}")],
        ))

    # ---------------- arms -------------------------------------------------
    # The arm hangs with the palm facing medially: -x for the left hand,
    # +x for the right.  Abduction (raising the arm sideways) is therefore
    # negative y-rotation on the left and positive on the right.
    for s, sx in (("l", 1.0), ("r", -1.0)):
        abd_lo, abd_hi = (-2.80, 0.45) if sx > 0 else (-0.45, 2.80)
        bones.append(Bone(
            f"shoulder_{s}", "chest", (0.205 * sx, 0.0, 0.145), side=s, rgba=SKIN,
            joints=[
                hinge(f"sh_{s}_abd", (0, 1, 0), abd_lo, abd_hi, T["sh_abd"], armature=0.10),
                # negative = raise the arm forwards
                hinge(f"sh_{s}_flex", (1, 0, 0), -2.60, 0.90, T["sh_flex"], armature=0.10),
                hinge(f"sh_{s}_rot", (0, 0, 1), -1.60, 1.60, T["sh_rot"], armature=0.06),
            ],
            geoms=[sphere((0, 0, 0), 0.062, MASSES["upper_arm"] * 0.13,
                          rgba=SKIN, name=f"delt_{s}")],
        ))
        bones.append(Bone(
            f"upper_arm_{s}", f"shoulder_{s}", (0.0, 0.0, 0.0), side=s, rgba=SKIN,
            joints=[],
            geoms=[capsule((0, 0, -0.015), (0, 0, -0.245), 0.052,
                           MASSES["upper_arm"] * 0.87)],
        ))
        bones.append(Bone(
            f"forearm_{s}", f"upper_arm_{s}", (0.0, 0.0, -0.255), side=s, rgba=SKIN,
            # negative = elbow flexion (hand travels forwards and up)
            joints=[hinge(f"elbow_{s}", (1, 0, 0), -2.60, 0.02, T["elbow"], armature=0.05)],
            geoms=[capsule((0, 0, 0.0), (0, 0, -0.215), 0.044, MASSES["forearm"] * 0.95)],
        ))
        bones.append(Bone(
            f"hand_{s}", f"forearm_{s}", (0.0, 0.0, -0.230), side=s, rgba=SKIN,
            joints=[
                hinge(f"wrist_{s}_flex", (1, 0, 0), -1.20, 1.20, T["wrist_flex"], armature=0.01),
                hinge(f"wrist_{s}_dev", (0, 1, 0), -0.50, 0.50, T["wrist_dev"], armature=0.01),
            ],
            geoms=[
                box((0.0, 0.0, -0.038), 0.013, 0.042, 0.045, MASSES["hand"] * 0.72,
                    name=f"palm_{s}"),
            ],
        ))
        # Fingers curl towards the palm, i.e. towards -x on the left (positive
        # y-rotation) and towards +x on the right (negative y-rotation).
        mcp = (0.0, 1.45) if sx > 0 else (-1.45, 0.0)
        pip = (0.0, 1.60) if sx > 0 else (-1.60, 0.0)
        # Each digit is two phalanges: a proximal one carrying the MCP joint and
        # a distal one carrying the PIP joint.  (An earlier version put both
        # joints on a single 6 cm capsule, which cannot wrap an object: it
        # sweeps through it instead.)
        for fname, fx, fy in (("thumb", 0.028, -0.042), ("index", 0.018, 0.028),
                              ("fingers", -0.006, 0.030)):
            tq = T["thumb"] if fname == "thumb" else T["finger"]
            m_digit = MASSES["fingers"] * (0.4 if fname == "fingers" else 0.3)
            bones.append(Bone(
                f"{fname}_{s}", f"hand_{s}", (fx * sx, fy, -0.078), side=s, rgba=SKIN,
                joints=[hinge(f"{fname}_{s}_mcp", (0, 1, 0), mcp[0], mcp[1], tq,
                              armature=0.004, damping=0.05, frictionloss=0.04)],
                geoms=[capsule((0, 0, 0.0), (0, 0, -PHALANX_PROX), 0.0115, 0.55 * m_digit,
                               name=f"{fname}_seg_{s}")],
            ))
            bones.append(Bone(
                f"{fname}_d_{s}", f"{fname}_{s}", (0.0, 0.0, -PHALANX_PROX), side=s,
                rgba=SKIN,
                joints=[hinge(f"{fname}_{s}_pip", (0, 1, 0), pip[0], pip[1], tq,
                              armature=0.003, damping=0.05, frictionloss=0.04)],
                geoms=[capsule((0, 0, 0.0), (0, 0, -PHALANX_DIST), 0.0105, 0.45 * m_digit,
                               name=f"{fname}_dseg_{s}")],
            ))

    # ---------------- legs -------------------------------------------------
    for s, sx in (("l", 1.0), ("r", -1.0)):
        # abduction (thigh outwards) is negative y-rotation on the left
        abd_lo, abd_hi = (-0.55, 0.35) if sx > 0 else (-0.35, 0.55)
        bones.append(Bone(
            f"hip_{s}", "pelvis", (0.095 * sx, 0.0, 0.03), side=s, rgba=SKIN,
            joints=[
                # negative = hip flexion (knee travels forwards)
                hinge(f"hip_{s}_flex", (1, 0, 0), -1.45, 0.45, T["hip_flex"], armature=0.18),
                hinge(f"hip_{s}_abd", (0, 1, 0), abd_lo, abd_hi, T["hip_abd"], armature=0.15),
                hinge(f"hip_{s}_rot", (0, 0, 1), -0.50, 0.50, T["hip_rot"], armature=0.10),
            ],
            geoms=[sphere((0, 0, 0), 0.072, MASSES["thigh"] * 0.12, name=f"hipball_{s}")],
        ))
        bones.append(Bone(
            f"thigh_{s}", f"hip_{s}", (0.0, 0.0, 0.0), side=s, rgba=SKIN,
            joints=[],
            geoms=[capsule((0, 0, -0.02), (0, 0, -0.395), 0.075,
                           MASSES["thigh"] * 0.88)],
        ))
        bones.append(Bone(
            f"shin_{s}", f"thigh_{s}", (0.0, 0.0, -0.415), side=s, rgba=SKIN,
            # positive = knee flexion (heel travels backwards) -- the sagittal
            # degree of freedom that makes standing possible at all
            joints=[hinge(f"knee_{s}", (1, 0, 0), -0.02, 2.40, T["knee"], armature=0.12)],
            geoms=[
                capsule((0, 0, 0.0), (0, 0, -0.375), 0.058, MASSES["shin"] * 0.92),
                sphere((0, 0, 0.0), 0.060, MASSES["shin"] * 0.08, name=f"kneecap_{s}"),
            ],
        ))
        bones.append(Bone(
            f"foot_{s}", f"shin_{s}", (0.0, 0.0, -0.395), side=s, rgba=SKIN,
            joints=[
                # negative = dorsiflexion (toes up), positive = plantarflexion
                hinge(f"ankle_{s}_flex", (1, 0, 0), -0.35, 0.60, T["ankle_flex"], armature=0.04),
                hinge(f"ankle_{s}_inv", (0, 1, 0), -0.35, 0.35, T["ankle_inv"], armature=0.03),
            ],
            geoms=[box((0.0, -0.055, -0.030), 0.048, 0.105, 0.030, MASSES["foot"],
                       name=f"footbody_{s}")],
        ))
        bones.append(Bone(
            f"toes_{s}", f"foot_{s}", (0.0, -0.155, -0.030), side=s, rgba=SKIN,
            # negative = toe extension (toes bend up during push-off)
            joints=[hinge(f"toe_{s}", (1, 0, 0), -0.85, 0.05, T["toe"], armature=0.01)],
            geoms=[box((0.0, -0.032, 0.0), 0.047, 0.032, 0.026, MASSES["toes"],
                       name=f"toebox_{s}")],
        ))

    return bones


BONES: list[Bone] = build_bones()
BONE_BY_NAME: dict[str, Bone] = {b.name: b for b in BONES}


def all_joints() -> list[tuple[str, Joint, Bone]]:
    out = []
    for b in BONES:
        for j in b.joints:
            out.append((b.name, j, b))
    return out


def total_mass() -> float:
    return float(sum(g.mass for b in BONES for g in b.geoms))


def n_actuated() -> int:
    return sum(len(b.joints) for b in BONES)


# --------------------------------------------------------------------------
# Nominal standing posture (radians).  Used as the "reference configuration"
# for the posture controller and for the equilibrium point of the stretch
# reflex.
# --------------------------------------------------------------------------
def nominal_posture() -> dict[str, float]:
    q: dict[str, float] = {}
    for b in BONES:
        for j in b.joints:
            q[j.name] = 0.0
    # Arms hang at the sides with a slight elbow bend; legs splay a little for
    # a stable base.  Finger flexion is positive on the left hand and negative
    # on the right because the palms face medially (see build_bones()).
    for s, sx in (("l", 1.0), ("r", -1.0)):
        q[f"sh_{s}_abd"] = 0.0
        q[f"sh_{s}_flex"] = 0.02
        q[f"elbow_{s}"] = -0.30
        q[f"hip_{s}_abd"] = -0.075 * sx
        q[f"knee_{s}"] = 0.10
        q[f"ankle_{s}_flex"] = -0.05
        q[f"ankle_{s}_inv"] = 0.03 * sx
        q[f"thumb_{s}_mcp"] = 0.35 * sx
        q[f"thumb_{s}_pip"] = 0.25 * sx
        q[f"index_{s}_mcp"] = 0.25 * sx
        q[f"index_{s}_pip"] = 0.30 * sx
        q[f"fingers_{s}_mcp"] = 0.55 * sx
        q[f"fingers_{s}_pip"] = 0.65 * sx
    q["spine_bend"] = 0.02
    return q


def posture_vector(joint_order: list[tuple[str, str, Joint]]) -> np.ndarray:
    """Nominal posture as an array in the model's actuator order."""
    nominal = nominal_posture()
    return np.array([nominal.get(j.name, 0.0) for _, _, j in joint_order], float)


def segment_endpoints() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World-agnostic local a/b endpoints of each body's primary capsule."""
    out = {}
    for b in BONES:
        for g in b.geoms:
            if g.kind == "capsule":
                out[b.name] = (np.asarray(g.a, float), np.asarray(g.b, float))
                break
    return out
