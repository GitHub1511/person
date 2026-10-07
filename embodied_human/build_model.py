"""
MuJoCo model generation.

Builds the complete MJCF from the declarative skeleton + skin tables, loads it,
and returns a :class:`ModelMeta` that maps every physiological concept onto a
concrete MuJoCo index (site id, joint id, actuator id, qpos/dof address, ...).

Two design decisions worth stating explicitly
---------------------------------------------
1. **Emission order matters.**  ``qpos`` is laid out in body order, so the human
   is emitted before the scene objects and the root free-joint address is read
   back from the compiled model rather than assumed to be zero.

2. **Selective self-collision.**  A body whose own segments collide
   destructively (a pelvis capsule inside a thigh capsule) cannot stand up.
   Contacts are therefore category-masked:

   ==================  =========  ==============  ============================
   category            contype    conaffinity     collides with
   ==================  =========  ==============  ============================
   limb (general)      1          6 (world,obj)   world + objects only
   hand                9          30              world, objects, other hand,
                                                  torso/head
   torso & head        17         14              world, objects, hands
   world/static        2          1               everything except world
   manipulable object  4          7               everything
   ==================  =========  ==============  ============================

   This gives the agent genuine self-touch -- a hand can feel its own face,
   which is a prerequisite for any body schema -- without the model tearing
   itself apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # MuJoCo is a hard dependency; fail loudly and usefully if absent.
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "MuJoCo is required.  Install it with:  python -m pip install mujoco"
    ) from exc

from . import skin
from .config import SimConfig
from .skeleton import BONE_BY_NAME, BONES, Geom, nominal_posture, total_mass

# --------------------------------------------------------------------------
# Collision filtering
# --------------------------------------------------------------------------
# Contact is enabled between geoms A and B when
#     (contype_A & conaffinity_B) or (contype_B & conaffinity_A)
#
# Bit allocation:
#   bit0 (1)   solid            -- collides with the world and with objects
#   bit1 (2)   world / static
#   bit2 (4)   manipulable objects
#   bit3 (8)   left hand
#   bit4 (16)  trunk and head
#   bit5 (32)  right hand
#   bit6 (64)  limbs
#
# The important consequences:
#   * limbs never collide with limbs or with the trunk, so the body cannot
#     tear itself apart on self-intersection (a pelvis capsule sitting inside a
#     thigh capsule);
#   * geoms of the *same* hand never collide with each other, which removes the
#     deep finger-finger interpenetration that otherwise registers as 370 kPa of
#     spurious nociceptive pressure on the fingers;
#   * the two hands, and either hand against the trunk/head, still collide --
#     self-touch is the whole point of a body schema, and MuJoCo separately
#     filters parent/child pairs so a hand never fights its own forearm.
COLLISION = {
    "limb": (1 | 64, 2 | 4),
    "hand_l": (1 | 8, 2 | 4 | 16 | 32 | 64),
    "hand_r": (1 | 32, 2 | 4 | 16 | 8 | 64),
    "body": (1 | 16, 2 | 4 | 8 | 32),
    "world": (2, 1),
    "object": (4, 7),
}

HAND_PREFIXES = ("hand_", "thumb_", "index_", "fingers_")
BODY_NAMES = {"chest", "abdomen", "pelvis", "head", "jaw", "neck"}


def collision_category(bone_name: str) -> str:
    if bone_name in BODY_NAMES:
        return "body"
    if bone_name.startswith(HAND_PREFIXES):
        return "hand_l" if bone_name.endswith("_l") else "hand_r"
    return "limb"


# --------------------------------------------------------------------------
# Scene objects: things in the world that can be touched, smelled, tasted
# --------------------------------------------------------------------------
@dataclass
class SceneObject:
    name: str
    kind: str                     # sphere | box | cylinder
    pos: tuple
    size: tuple
    rgba: tuple
    mass: float
    friction: tuple = (1.0, 0.02, 0.001)
    temperature: float = 22.0     # degC, surface temperature
    odor: np.ndarray | None = None      # 24-D olfactory signature
    taste: np.ndarray | None = None     # 5-D gustatory signature
    compliance: float = 0.0       # 0 rigid .. 1 very soft
    label: str = ""

    def qpos(self) -> np.ndarray:
        return np.array([self.pos[0], self.pos[1], self.pos[2], 1.0, 0.0, 0.0, 0.0])

    def xml(self) -> str:
        ct, ca = COLLISION["object"]
        # The body sits at the origin and the free joint's qpos (see qpos())
        # carries the position, so the geom must be at the body origin.  An
        # earlier version offset the geom as well, which doubled every object's
        # position and threw them to the floor.
        attrs = [f'name="{self.name}"', f'type="{self.kind}"', 'pos="0 0 0"']
        if self.kind == "box":
            attrs.append(f'size="{self.size[0]} {self.size[1]} {self.size[2]}"')
        elif self.kind == "cylinder":
            attrs.append(f'size="{self.size[0]} {self.size[1]}"')
        else:
            attrs.append(f'size="{self.size[0]}"')
        attrs.append(f'rgba="{self.rgba[0]} {self.rgba[1]} {self.rgba[2]} {self.rgba[3]}"')
        roll = 0.012 if self.kind == "sphere" else self.friction[2]
        attrs.append(f'friction="{self.friction[0]} {self.friction[1]} {roll}"')
        attrs.append(f'mass="{self.mass}"')
        attrs.append(f'contype="{ct}" conaffinity="{ca}"')
        if self.compliance > 0:
            tc = 0.02 + 0.10 * self.compliance
            attrs.append(f'solref="{tc:.4f} 1"')
            attrs.append(f'solimp="{0.9 - 0.7 * self.compliance:.2f} 1"')
            attrs.append('condim="4"')
        elif self.kind == "sphere":
            # rolling friction: a sphere set down on a table should not roll off it
            attrs.append('condim="6"')
        else:
            attrs.append('condim="3"')
        geom = "<geom " + " ".join(attrs) + "/>"
        return (f'<body name="{self.name}_body" pos="0 0 0">'
                f'<freejoint name="{self.name}_joint"/>{geom}</body>')


def _odor(*pairs: tuple[int, float], dim: int = 24) -> np.ndarray:
    v = np.zeros(dim)
    for i, x in pairs:
        v[i % dim] = x
    return v


# --------------------------------------------------------------------------
# Furniture: (name, centre x, centre y, top height, half-x, half-y)
# --------------------------------------------------------------------------
# The workbench is in front of the person (who faces -y); the shelf is across
# the room, far enough that carrying something to it means actually walking.
FURNITURE = [
    ("table", 0.0, -0.82, 0.74, 0.46, 0.30),
    ("shelf", 0.0, -3.40, 0.74, 0.40, 0.26),
]
TABLE_THICKNESS = 0.025


def surface_top(name: str) -> float:
    for n, _, _, top, _, _ in FURNITURE:
        if n == name:
            return top
    return 0.0


def default_scene_objects(complexity=None) -> list[SceneObject]:
    """A small, deliberately mundane world: things a person might touch.

    Everything starts resting on the workbench.
    """
    top = surface_top("table")
    objs = [
        SceneObject(
            name="mug", kind="cylinder", pos=(0.22, -0.60, top + 0.057), size=(0.040, 0.055),
            rgba=(0.92, 0.93, 0.95, 1.0), mass=0.35, temperature=62.0,
            odor=_odor((2, 0.9), (7, 0.5), (13, 0.3)),          # coffee-ish
            taste=_odor((3, 0.85), (2, 0.15), dim=5),           # bitter
            label="hot mug of coffee",
        ),
        SceneObject(
            name="apple", kind="sphere", pos=(-0.20, -0.60, top + 0.044), size=(0.042,),
            rgba=(0.85, 0.18, 0.16, 1.0), mass=0.18, temperature=21.0,
            odor=_odor((1, 0.8), (5, 0.6), (11, 0.4)),
            taste=_odor((0, 0.75), (2, 0.45), dim=5),           # sweet + sour
            label="apple",
        ),
        SceneObject(
            name="stone", kind="box", pos=(0.02, -0.70, top + 0.032),
            size=(0.055, 0.045, 0.030), rgba=(0.45, 0.45, 0.47, 1.0),
            mass=0.9, temperature=17.0, friction=(0.9, 0.02, 0.001),
            odor=_odor((20, 0.2)), label="cold stone",
        ),
        SceneObject(
            name="cushion", kind="box", pos=(0.36, -0.80, top + 0.037),
            size=(0.10, 0.10, 0.035), rgba=(0.30, 0.42, 0.70, 1.0), mass=0.25,
            temperature=24.0, compliance=0.9,
            odor=_odor((9, 0.35), (16, 0.25)), label="soft cushion",
        ),
        SceneObject(
            name="sphere_toy", kind="sphere", pos=(-0.34, -0.98, top + 0.032),
            size=(0.030,), rgba=(0.95, 0.75, 0.15, 1.0), mass=0.05,
            temperature=23.0, odor=_odor((4, 0.5)), label="small ball",
        ),
    ]


# --------------------------------------------------------------------------
# Anatomical landmark sites (IMUs, force plates, rangefinders, gaze)
#
# Entries are (name, bone, local position, local quaternion or None).  A
# rangefinder measures along the site's local +z, so the palm, chest and toe
# proximity sensors carry an explicit orientation; the rest inherit the body
# frame.
# --------------------------------------------------------------------------
LANDMARKS: list[tuple] = [
    ("imu_head", "head", (0.0, 0.0, 0.075), None),
    ("imu_chest", "chest", (0.0, 0.0, 0.10), None),
    ("imu_pelvis", "pelvis", (0.0, 0.0, 0.06), None),
    ("soma_hand_l", "hand_l", (0.0, 0.0, -0.038), None),
    ("soma_hand_r", "hand_r", (0.0, 0.0, -0.038), None),
    ("soma_foot_l", "foot_l", (0.0, -0.055, -0.030), None),
    ("soma_foot_r", "foot_r", (0.0, -0.055, -0.030), None),
    # palm rangefinders look out of the palm: -x on the left, +x on the right
    ("rf_palm_l", "hand_l", (-0.016, 0.0, -0.038), (0.7071068, 0.0, -0.7071068, 0.0)),
    ("rf_palm_r", "hand_r", (0.016, 0.0, -0.038), (0.7071068, 0.0, 0.7071068, 0.0)),
    # chest looks forwards (-y); toes look down (-z)
    ("rf_chest", "chest", (0.0, -0.145, 0.10), (0.7071068, -0.7071068, 0.0, 0.0)),
    ("rf_toe_l", "toes_l", (0.0, -0.032, -0.026), (0.7071068, 0.7071068, 0.0, 0.0)),
    ("rf_toe_r", "toes_r", (0.0, -0.032, -0.026), (0.7071068, 0.7071068, 0.0, 0.0)),
    ("gaze", "head", (0.0, -0.090, 0.075), None),
]

SOMA_SITES = ("imu_head", "soma_hand_l", "soma_hand_r", "soma_foot_l",
              "soma_foot_r", "gaze")


# --------------------------------------------------------------------------
# XML helpers
# --------------------------------------------------------------------------
def _f(x) -> str:
    return " ".join(f"{v:.6g}" for v in np.atleast_1d(np.asarray(x, float)))


def _quat_from_z(z: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) rotating local +z onto ``z``."""
    z = np.asarray(z, float)
    z = z / max(np.linalg.norm(z), 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(ref, z))
    if c > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(ref, z)
    axis /= np.linalg.norm(axis)
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    return np.concatenate([[np.cos(ang / 2)], axis * np.sin(ang / 2)])


# Eyelids: thin skin-coloured boxes in front of each eye, moved at runtime by
# ocular.EyeRig (they are not collision geoms).  Heights are the *open* pose.
LID_Y = -0.1128
LID_HALF_X = 0.0155
LID_UP_HALF_Z = 0.0075
LID_LO_HALF_Z = 0.0035


def eyelid_xml() -> list[str]:
    from .skeleton import EYE_X, EYE_Z, SKIN
    out = []
    rgba = " ".join(f"{v:.3g}" for v in SKIN)
    for s, sx in (("l", 1.0), ("r", -1.0)):
        up_z = EYE_Z + 0.0095 + LID_UP_HALF_Z
        lo_z = EYE_Z - 0.0095 - LID_LO_HALF_Z
        for nm, z, hz in (("up", up_z, LID_UP_HALF_Z), ("lo", lo_z, LID_LO_HALF_Z)):
            out.append(
                f'<geom name="lid_{nm}_{s}" type="box" pos="{EYE_X * sx:.5f} {LID_Y:.5f} {z:.5f}" '
                f'size="{LID_HALF_X} 0.0022 {hz}" rgba="{rgba}" mass="1e-7" '
                f'contype="0" conaffinity="0" group="1"/>')
    return out


def _geom_xml(g: Geom, bone_name: str) -> str:
    a = np.asarray(g.a, float)
    b = np.asarray(g.b, float)
    ct, ca = COLLISION[collision_category(bone_name)]
    if g.name.startswith(("vis_", "eyeball")):
        ct = ca = 0                       # eyes are drawn, not collided
    friction = g.friction
    extra = ""
    if g.name.startswith(("footbody", "toebox")):
        # Torsional friction (condim 4) lets a planted foot hold its yaw, which
        # is what makes turning on the spot possible; with the default condim 3
        # a stance foot spins freely about the vertical axis.
        friction = (g.friction[0], 0.10, g.friction[2])
        extra = ' condim="4"'
    common = (f'mass="{g.mass:.6g}" rgba="{_f(g.rgba)}" '
              f'friction="{_f(friction)}" contype="{ct}" conaffinity="{ca}"{extra}')
    if g.kind == "capsule":
        pos = 0.5 * (a + b)
        half = float(np.linalg.norm(b - a)) * 0.5
        axis = (b - a) / max(np.linalg.norm(b - a), 1e-12)
        quat = _quat_from_z(axis)
        return (f'<geom name="{g.name}" class="soma" type="capsule" '
                f'pos="{_f(pos)}" quat="{_f(quat)}" '
                f'size="{g.size[0]:.6g} {half:.6g}" {common}/>')
    if g.kind == "sphere":
        return (f'<geom name="{g.name}" class="soma" type="sphere" '
                f'pos="{_f(a)}" size="{g.size[0]:.6g}" {common}/>')
    if g.kind == "box":
        return (f'<geom name="{g.name}" class="soma" type="box" pos="{_f(a)}" '
                f'size="{_f(g.size)}" {common}/>')
    raise ValueError(g.kind)


# --------------------------------------------------------------------------
# The MJCF document
# --------------------------------------------------------------------------
def build_xml(cfg: SimConfig, include_touch_sensors: bool = True) -> str:
    """Generate the complete MJCF document as a string."""
    L: list[str] = []
    add = L.append
    wct, wca = COLLISION["world"]

    add('<?xml version="1.0" encoding="utf-8"?>')
    add('<mujoco model="embodied_human">')
    add('  <compiler angle="radian" autolimits="true" inertiafromgeom="true" '
        'balanceinertia="true" discardvisual="false"/>')
    add(f'  <option timestep="{1.0 / cfg.rates.physics:.8f}" '
        'integrator="implicitfast" gravity="0 0 -9.81" solver="Newton" '
        'cone="elliptic" impratio="10" iterations="80" tolerance="1e-9" '
        'o_solref="0.02 1" o_solimp="0.9 0.95 0.001 0.5 2"/>')
    # (the MjData arena is sized from these: 4000/1200 meant ~870 MB per data
    # structure, which made parallel parameter searches run out of memory)
    add('  <size njmax="1500" nconmax="400" nkey="1"/>')
    add('  <visual>')
    add('    <map znear="0.005" zfar="60" shadowclip="1.0" fogstart="12" fogend="55"/>')
    add('    <quality shadowsize="2048" offsamples="0"/>')
    add('    <headlight ambient="0.34 0.34 0.38" diffuse="0.62 0.62 0.60" '
        'specular="0.20 0.20 0.20"/>')
    add('    <global azimuth="132" elevation="-16" fovy="52" '
        'offwidth="1280" offheight="960"/>')
    add('    <rgba haze="0.18 0.21 0.26 1"/>')
    add('  </visual>')
    add('  <statistic meansize="0.06" extent="1.6" center="0 0 0.9"/>')

    # ---------------- defaults -------------------------------------------
    add('  <default>')
    add('    <joint damping="1.2" frictionloss="0.7" armature="0.03" '
        'solreflimit="0.004 1" solimplimit="0.9 0.95 0.001"/>')
    add('    <geom friction="1.1 0.02 0.001" margin="0.0008" condim="3" '
        'solref="0.008 1" solimp="0.92 0.96 0.001"/>')
    add('    <site group="4" rgba="0.95 0.35 0.25 0.35" size="0.012"/>')
    add('    <default class="soma">')
    add('      <geom type="capsule" condim="3" margin="0.001"/>')
    add('    </default>')
    add('    <default class="skin_site">')
    add('      <site group="5" rgba="1 0.45 0.35 0" size="0.012"/>')
    add('    </default>')
    add('    <default class="env">')
    add(f'      <geom contype="{wct}" conaffinity="{wca}" condim="3"/>')
    add('    </default>')
    add('  </default>')

    # ---------------- assets ---------------------------------------------
    add('  <asset>')
    add('    <texture name="sky" type="skybox" builtin="gradient" '
        'rgb1="0.30 0.38 0.50" rgb2="0.06 0.08 0.12" width="256" height="256"/>')
    add('    <texture name="grid" type="2d" builtin="checker" '
        'rgb1="0.30 0.31 0.33" rgb2="0.24 0.25 0.27" width="512" height="512" '
        'markrgb="0.55 0.56 0.58" mark="none"/>')
    add('    <material name="floor" texture="grid" texrepeat="14 14" '
        'reflectance="0.12" shininess="0.15" specular="0.3"/>')
    add('    <material name="skin_mat" rgba="0.87 0.72 0.63 1" '
        'specular="0.25" shininess="0.25"/>')
    add('    <material name="wall_mat" rgba="0.55 0.55 0.58 1"/>')
    add('  </asset>')

    # ---------------- worldbody: human FIRST (qpos ordering) -------------
    add('  <worldbody>')

    def emit_bone(bone, indent: int) -> None:
        pad = "  " * indent
        attrs = [f'name="{bone.name}"', f'pos="{_f(bone.pos)}"']
        if bone.quat != (1.0, 0.0, 0.0, 0.0):
            attrs.append(f'quat="{_f(bone.quat)}"')
        add(f'{pad}<body {" ".join(attrs)}>')
        if bone.name == "pelvis":
            add(f'{pad}  <freejoint name="root"/>')
        for j in bone.joints:
            add(f'{pad}  <joint name="{j.name}" type="{j.kind}" '
                f'axis="{_f(j.axis)}" range="{j.lo:.6g} {j.hi:.6g}" limited="true" '
                f'damping="{j.damping:.6g}" frictionloss="{j.frictionloss:.6g}" '
                f'armature="{j.armature:.6g}"/>')
        for g in bone.geoms:
            add(f'{pad}  ' + _geom_xml(g, bone.name))
        for lname, lbone, lpos, lquat in LANDMARKS:
            if lbone == bone.name:
                qattr = f' quat="{_f(lquat)}"' if lquat is not None else ""
                add(f'{pad}  <site name="{lname}" pos="{_f(lpos)}"{qattr} '
                    f'size="0.008" group="3" rgba="0.2 0.7 1.0 0.5"/>')
        if bone.name == "head":
            # in front of the face (it used to sit inside the face box and
            # rendered the inside of the head)
            add(f'{pad}  <camera name="egocentric" pos="0 -0.121 0.078" '
                f'xyaxes="1 0 0 0 0 1" fovy="70"/>')
            for lid_xml in eyelid_xml():
                add(f'{pad}  {lid_xml}')
        for idx in skin.TAXELS_BY_BONE.get(bone.name, []):
            t = skin.TAXELS[idx]
            add(f'{pad}  <site name="{t.name}" class="skin_site" '
                f'pos="{_f(t.pos)}" size="{cfg.tactile.taxel_radius:.5g}" '
                f'rgba="1 0.45 0.35 0"/>')
        for child in BONES:
            if child.parent == bone.name:
                emit_bone(child, indent + 1)
        add(f'{pad}</body>')

    for bone in BONES:
        if bone.parent == "world":
            emit_bone(bone, 2)

    # ---------------- environment (after the human) ----------------------
    add('    <light name="key" pos="2.4 -3.0 4.2" dir="-0.4 0.5 -1" '
        'directional="false" castshadow="true" cutoff="70" '
        'diffuse="0.85 0.85 0.82" specular="0.3 0.3 0.3" range="14"/>')
    add('    <light name="fill" pos="-3.0 1.6 3.0" dir="0.7 -0.4 -1" '
        'directional="false" castshadow="false" cutoff="70" '
        'diffuse="0.35 0.38 0.45" range="16"/>')
    add('    <light name="amb" pos="0 0 6" dir="0 0 -1" directional="true" '
        'castshadow="false" diffuse="0.24 0.26 0.32"/>')
    add('    <geom name="floor" class="env" type="plane" size="14 14 0.1" '
        'material="floor" friction="1.1 0.02 0.001" '
        'solref="0.01 1" solimp="0.95 0.99 0.001"/>')
    add('    <geom name="wall" class="env" type="box" pos="0 1.9 1.0" '
        'size="3.0 0.06 1.0" material="wall_mat"/>')
    for fname, fx, fy, ftop, hx, hy in FURNITURE:
        th = TABLE_THICKNESS
        suffix = "" if fname == "table" else f"_{fname}"
        add(f'    <geom name="table_top{suffix}" class="env" type="box" '
            f'pos="{fx} {fy} {ftop - th / 2:.4f}" size="{hx} {hy} {th / 2:.4f}" '
            f'material="wall_mat" friction="1.0 0.02 0.001"/>')
        leg_h = (ftop - th) / 2
        for i, (sx_, sy_) in enumerate([(1, 1), (-1, 1), (1, -1), (-1, -1)]):
            add(f'    <geom name="table_leg{suffix}{i}" class="env" type="box" '
                f'pos="{fx + sx_ * (hx - 0.03):.3f} {fy + sy_ * (hy - 0.03):.3f} {leg_h:.3f}" '
                f'size="0.02 0.02 {leg_h:.3f}" material="wall_mat"/>')
    for obj in default_scene_objects():
        add('    ' + obj.xml())
    add('    <camera name="observer" pos="1.9 -2.5 1.75" '
        'xyaxes="0.79 0.61 0 -0.24 0.31 0.92" fovy="48"/>')
    add('    <camera name="closeup" pos="0.55 -1.05 1.15" '
        'xyaxes="0.91 0.41 0 -0.18 0.40 0.90" fovy="40"/>')
    add('  </worldbody>')

    # ---------------- actuators ------------------------------------------
    add('  <actuator>')
    for bone in BONES:
        for j in bone.joints:
            tau = cfg.motor.command_tau if j.torque >= 1.0 else 0.012
            add(f'    <general name="act_{j.name}" joint="{j.name}" '
                f'dyntype="filter" dynprm="{tau:.5g}" gaintype="fixed" '
                f'gainprm="1" biastype="none" '
                f'ctrlrange="{-j.torque:.6g} {j.torque:.6g}" ctrllimited="true" '
                f'gear="1"/>')
    add('  </actuator>')

    # ---------------- grip assist ----------------------------------------
    # One *inactive* soft weld per (hand, object).  The skill system switches a
    # weld on when a grasp is established (palm and several digits touching the
    # object) and off on release, freezing the relative pose at that instant.
    # Rigid fingers 3 cm long with no opposable thumb cannot hold a sphere by
    # friction alone -- it is squeezed out like a pip from a wedge -- so this
    # is the one deliberate shortcut in the manipulation stack, and the README
    # says so.
    add('  <equality>')
    for obj in default_scene_objects():
        for s in ("l", "r"):
            add(f'    <weld name="grip_{s}_{obj.name}" body1="hand_{s}" '
                f'body2="{obj.name}_body" active="false" solref="0.025 1" '
                f'solimp="0.9 0.95 0.001 0.5 2"/>')
    add('  </equality>')

    # ---------------- sensors (grouped contiguously) ---------------------
    add('  <sensor>')
    for bone in BONES:
        for j in bone.joints:
            add(f'    <jointpos name="q_{j.name}" joint="{j.name}"/>')
    for bone in BONES:
        for j in bone.joints:
            add(f'    <jointvel name="qd_{j.name}" joint="{j.name}"/>')
    for bone in BONES:
        for j in bone.joints:
            add(f'    <actuatorfrc name="tau_{j.name}" actuator="act_{j.name}"/>')
    for bone in BONES:
        for j in bone.joints:
            add(f'    <jointactuatorfrc name="jtf_{j.name}" joint="{j.name}"/>')
    add('    <framepos name="root_pos" objtype="body" objname="pelvis"/>')
    add('    <framequat name="root_quat" objtype="body" objname="pelvis"/>')
    add('    <framelinvel name="root_linvel" objtype="body" objname="pelvis"/>')
    add('    <frameangvel name="root_angvel" objtype="body" objname="pelvis"/>')
    for nm in ("head", "chest", "pelvis"):
        add(f'    <accelerometer name="otolith_{nm}" site="imu_{nm}"/>')
    for nm in ("head", "chest", "pelvis"):
        add(f'    <gyro name="canal_{nm}" site="imu_{nm}"/>')
    add('    <velocimeter name="vel_head" site="imu_head"/>')
    add('    <magnetometer name="magneto_head" site="imu_head"/>')
    for nm in ("hand_l", "hand_r", "foot_l", "foot_r"):
        add(f'    <force name="force_{nm}" site="soma_{nm}"/>')
    for nm in ("hand_l", "hand_r", "foot_l", "foot_r"):
        add(f'    <torque name="torque_{nm}" site="soma_{nm}"/>')
    for nm in ("palm_l", "palm_r", "chest", "toe_l", "toe_r"):
        add(f'    <rangefinder name="rf_{nm}" site="rf_{nm}"/>')
    for nm in SOMA_SITES:
        add(f'    <framepos name="fp_{nm}" objtype="site" objname="{nm}"/>')
    for nm in SOMA_SITES:
        add(f'    <framelinvel name="fv_{nm}" objtype="site" objname="{nm}"/>')
    add('    <framequat name="fq_gaze" objtype="site" objname="gaze"/>')
    if include_touch_sensors:
        for t in skin.TAXELS:
            add(f'    <touch name="touch_{t.name}" site="{t.name}"/>')
    add('  </sensor>')

    # ---------------- keyframe (human joints, then object free joints) ---
    nominal = nominal_posture()
    qpos = [0.0, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0]
    for b in BONES:
        for j in b.joints:
            qpos.append(nominal.get(j.name, 0.0))
    for obj in default_scene_objects():
        qpos.extend(obj.qpos().tolist())
    add('  <keyframe>')
    add(f'    <key name="standing" qpos="{_f(qpos)}"/>')
    add('  </keyframe>')
    add('</mujoco>')
    return "\n".join(L)


# --------------------------------------------------------------------------
# Model metadata
# --------------------------------------------------------------------------
@dataclass
class ModelMeta:
    model: "mujoco.MjModel"
    joint_order: list = field(default_factory=list)
    actuator_names: list = field(default_factory=list)
    actuator_index: dict = field(default_factory=dict)
    joint_index: dict = field(default_factory=dict)
    qpos_addr: dict = field(default_factory=dict)
    dof_addr: dict = field(default_factory=dict)
    torque_limit: np.ndarray = field(default_factory=lambda: np.zeros(0))
    taxel_site_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    taxel_bone_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    landmark_site_ids: dict = field(default_factory=dict)
    sensor_addr: dict = field(default_factory=dict)     # uniform base -> (adr,w,n)
    sensor_by_name: dict = field(default_factory=dict)  # full name -> (adr, width)
    geom_ids: dict = field(default_factory=dict)
    body_ids: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)
    touch_sensor_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    name_to_actuator: dict = field(default_factory=dict)
    root_qpos_addr: int = 0
    object_qpos_addr: dict = field(default_factory=dict)
    xml: str = ""

    @property
    def n_actuators(self) -> int:
        return self.model.nu

    @property
    def n_taxels(self) -> int:
        return len(self.taxel_site_ids)

    def sensor_slice(self, base: str) -> slice:
        adr, width, count = self.sensor_addr[base]
        return slice(adr, adr + width * count)

    def read(self, data, name: str) -> np.ndarray:
        adr, dim = self.sensor_by_name[name]
        return np.array(data.sensordata[adr:adr + dim], float)

    def read_many(self, data, names) -> np.ndarray:
        return np.concatenate([self.read(data, n) for n in names]) if len(names) \
            else np.zeros(0)


def _sensor_name_table(model) -> tuple[dict, dict]:
    by_name: dict[str, tuple[int, int]] = {}
    by_base: dict[str, list[int]] = {}
    for i in range(model.nsensor):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        by_name[nm] = (int(model.sensor_adr[i]), int(model.sensor_dim[i]))
        by_base.setdefault(nm.split("_", 1)[0], []).append(i)
    return by_name, by_base


def _validate_sensor_contiguity(model) -> None:
    """Assert each prefix group is contiguous in sensordata."""
    seen: dict[str, list[int]] = {}
    for i in range(model.nsensor):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        w = int(model.sensor_dim[i])
        adr = int(model.sensor_adr[i])
        seen.setdefault(nm.split("_", 1)[0], []).extend(range(adr, adr + w))
    for base, idx in seen.items():
        if idx != list(range(min(idx), min(idx) + len(idx))):
            raise RuntimeError(f"sensor group '{base}' is not contiguous")


def _uniform_group_table(model, by_name: dict, by_base: dict) -> dict:
    """base -> (start, width, count) for groups whose members share a width.

    Mixed-width groups such as ``root`` (3-vector position + 4-vector
    quaternion) are omitted; read those by name instead.
    """
    out = {}
    for base, ids in by_base.items():
        entries = []
        for i in ids:
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            entries.append((by_name[nm][0], by_name[nm][1]))
        entries.sort()
        if len({w for _, w in entries}) != 1:
            continue
        out[base] = (entries[0][0], entries[0][1], len(entries))
    return out


def load_model(cfg: SimConfig, include_touch_sensors: bool = True,
               xml_path: Path | None = None
               ) -> tuple["mujoco.MjModel", "mujoco.MjData", ModelMeta]:
    """Build the MJCF, load it, and assemble the index metadata."""
    xml = build_xml(cfg, include_touch_sensors=include_touch_sensors)
    if xml_path is not None:
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    _validate_sensor_contiguity(model)

    joint_order: list[tuple[str, str, object]] = []
    joint_index: dict[str, int] = {}
    qpos_addr: dict[str, int] = {}
    dof_addr: dict[str, int] = {}
    for b in BONES:
        for j in b.joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
            if jid < 0:
                raise RuntimeError(f"joint {j.name} missing from compiled model")
            joint_order.append((b.name, j.name, j))
            joint_index[j.name] = jid
            qpos_addr[j.name] = int(model.jnt_qposadr[jid])
            dof_addr[j.name] = int(model.jnt_dofadr[jid])

    actuator_index = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }
    torque_limit = np.array([j.torque for _, _, j in joint_order], float)

    taxel_site_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, t.name)
         for t in skin.TAXELS], dtype=int)
    if (taxel_site_ids < 0).any():
        missing = [t.name for t, i in zip(skin.TAXELS, taxel_site_ids) if i < 0][:5]
        raise RuntimeError(f"missing taxel sites, e.g. {missing}")

    landmark_site_ids = {
        nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
        for nm, _, _, _ in LANDMARKS
    }
    geom_ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i
                for i in range(model.ngeom)}
    body_ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i
                for i in range(model.nbody)}

    if include_touch_sensors:
        ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                          f"touch_{t.name}") for t in skin.TAXELS])
        touch_ids = ids[ids >= 0]
    else:
        touch_ids = np.zeros(0, int)

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_jid < 0:
        raise RuntimeError("root free joint not found")
    objects = default_scene_objects()
    object_qpos_addr = {}
    for obj in objects:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.name}_joint")
        object_qpos_addr[obj.name] = int(model.jnt_qposadr[jid])

    by_name, by_base = _sensor_name_table(model)
    meta = ModelMeta(
        model=model,
        joint_order=joint_order,
        actuator_names=[f"act_{n}" for _, n, _ in joint_order],
        actuator_index=actuator_index,
        joint_index=joint_index,
        qpos_addr=qpos_addr,
        dof_addr=dof_addr,
        torque_limit=torque_limit,
        taxel_site_ids=taxel_site_ids,
        taxel_bone_ids=skin.TAXEL_BONE,
        landmark_site_ids=landmark_site_ids,
        sensor_addr=_uniform_group_table(model, by_name, by_base),
        sensor_by_name=by_name,
        geom_ids=geom_ids,
        body_ids=body_ids,
        objects=objects,
        touch_sensor_ids=touch_ids,
        name_to_actuator={jn: actuator_index[f"act_{jn}"] for _, jn, _ in joint_order},
        root_qpos_addr=int(model.jnt_qposadr[root_jid]),
        object_qpos_addr=object_qpos_addr,
        xml=xml,
    )

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    set_standing_state(model, data, meta)
    return model, data, meta


def lowest_foot_z(model, data) -> float:
    """Lowest world z of the foot/toe contact boxes."""
    lowest = np.inf
    for i in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if nm is None or not (nm.startswith("footbody") or nm.startswith("toebox")):
            continue
        hx, hy, hz = model.geom_size[i][:3]
        corners = np.array([[sx * hx, sy * hy, sz * hz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        world = data.geom_xpos[i] + corners @ data.geom_xmat[i].reshape(3, 3).T
        lowest = min(lowest, float(world[:, 2].min()))
    return lowest


def set_standing_state(model, data, meta: ModelMeta) -> None:
    """Place the body so the soles just touch the ground at nominal posture."""
    nominal = nominal_posture()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for name, addr in meta.qpos_addr.items():
        data.qpos[addr] = nominal.get(name, 0.0)
    ra = meta.root_qpos_addr
    data.qpos[ra:ra + 3] = (0.0, 0.0, 1.05)
    data.qpos[ra + 3:ra + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    lowest = lowest_foot_z(model, data)
    if np.isfinite(lowest):
        data.qpos[ra + 2] -= lowest - 0.0015
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0


def body_map_anchor_table() -> dict[str, list[int]]:
    table: dict[str, list[int]] = {}
    for t in skin.TAXELS:
        table.setdefault(t.region, []).append(t.index)
    return table


def save_xml(cfg: SimConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_xml(cfg), encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    import time
    cfg = SimConfig()
    t0 = time.time()
    xml = build_xml(cfg)
    t1 = time.time()
    m, d, meta = load_model(cfg)
    t2 = time.time()
    print(f"xml chars      : {len(xml):,}")
    print(f"generate       : {t1 - t0:.2f} s   compile+forward: {t2 - t1:.2f} s")
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nsite={m.nsite} ngeom={m.ngeom} "
          f"nbody={m.nbody}")
    print(f"nsensor={m.nsensor} nsensordata={m.nsensordata}")
    human_mass = sum(m.body_mass) - sum(o.mass for o in meta.objects)
    print(f"human mass     : {human_mass:.2f} kg (declared {total_mass():.2f})")
    print(f"taxels={meta.n_taxels}  root qpos addr={meta.root_qpos_addr}")
    print(f"standing pelvis z={d.qpos[meta.root_qpos_addr + 2]:.4f}  "
          f"lowest foot z={lowest_foot_z(m, d):.4f}  ncon={d.ncon}")
    print("sensor groups:", {k: v[2] for k, v in meta.sensor_addr.items()})

