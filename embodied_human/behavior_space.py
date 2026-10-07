"""
The space of things the person can *do*.

The original agent chose among 55 hand-written motor programs.  This module
replaces "a list of behaviours" with a **generative behaviour space**: a
behaviour is a *descriptor* -- one choice in each of ~30 factored channels --
and every descriptor compiles to something the body can actually execute.

======================  ==============================================  =========
channel                 what a choice fixes                              choices
======================  ==============================================  =========
head                    neck yaw x pitch x tilt                            693
eyes                    gaze yaw x pitch                                   63
lids                    resting aperture x blink pattern                   35
mouth                   jaw opening x vocalisation                         54
torso                   bend x side-bend x twist                           245
left / right arm        one of 48 arm schemas, then a +-2 step variation   750,000
                        on each of six joints (5^6) -> 48 x 15,625         each
left / right hand       one of 14 hand shapes, +-1 on each digit group     378 each
stance                  crouch depth x weight shift x lean                 75
style                   speed x amplitude x tremor x rhythm                320
hold                    how long the behaviour lasts                       6
object intent           none | {look, point, reach, grasp} x 6 objects     25
self-touch              none | 16 body regions x hand x {rest,rub,tap,     129
                        scratch}
walk                    none | speed x turn                                16
======================  ==============================================  =========

The product of the channel sizes is ~10^40 *descriptors*.  The point is not
the number: it is that (a) any descriptor can be written down, encoded as an
integer, decoded back and *executed*, (b) behaviours are built from parts that
have meaning (a hand-to-eye schema, a squint, a sigh), so that internal state
-- dry eyes, an itch, fear, boredom, cold -- can bias *which parts* are chosen,
and (c) the executed repertoire is large enough that a long run does not repeat
itself (see ``BehaviorStats``).

Honest caveats: most descriptors are physically *similar* to many others (two
arm schemas that differ by 0.1 rad are different descriptors but nearly the same
movement), and the arm schemas and region anchors are hand-authored approximations
-- there is no motion-capture data behind them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Affinity tags: how a part of a behaviour relates to the person's condition
# --------------------------------------------------------------------------
TAGS = ("vigilance", "defensive", "explore", "social", "comfort", "fatigue",
        "ocular", "pain", "itch", "cold", "heat", "hunger", "boredom",
        "energetic", "expressive", "soothe", "object", "relax")
TI = {t: i for i, t in enumerate(TAGS)}
NT = len(TAGS)


def tags(**kw) -> np.ndarray:
    v = np.zeros(NT)
    for k, x in kw.items():
        v[TI[k]] = x
    return v


# --------------------------------------------------------------------------
# Arm schemas, in LEFT-arm coordinates:
#   (sh_abd, sh_flex, sh_rot, elbow, wrist_flex, wrist_dev)
# The right arm mirrors abd, rot and dev.  Angles in radians.
# --------------------------------------------------------------------------
ARM_SCHEMAS: list[tuple[str, tuple, np.ndarray]] = [
    ("rest",            (-0.10, 0.00, 0.00, -0.15, 0.00, 0.00), tags(relax=1.0)),
    ("loose_out",       (-0.35, 0.05, 0.20, -0.25, 0.00, 0.00), tags(relax=0.6, boredom=0.3)),
    ("hand_on_hip",     (-0.60, 0.15, 0.70, -1.45, 0.00, 0.20), tags(expressive=0.6, relax=0.2)),
    ("hand_in_pocket",  (-0.25, 0.30, 0.50, -0.90, 0.10, 0.00), tags(relax=0.7, boredom=0.4)),
    ("behind_back",     (-0.15, 0.55, 0.60, -0.55, 0.30, 0.00), tags(relax=0.5, social=0.2)),
    ("arms_crossed",    (0.35, -0.80, 1.00, -1.90, 0.00, 0.00), tags(defensive=0.5, soothe=0.4, cold=0.5)),
    ("hand_to_chest",   (0.30, -0.75, 0.20, -1.85, 0.00, 0.00), tags(soothe=0.9, social=0.4)),
    ("hand_to_cheek",   (0.30, -1.35, 0.00, -2.05, 0.25, 0.00), tags(soothe=0.6, boredom=0.4, itch=0.4)),
    ("hand_to_mouth",   (0.20, -1.45, 0.10, -2.30, 0.30, 0.00), tags(hunger=0.8, soothe=0.4, vigilance=0.3)),
    ("hand_to_eye",     (0.25, -1.50, -0.20, -2.25, 0.40, 0.00), tags(ocular=1.0, soothe=0.2)),
    ("hand_to_chin",    (0.25, -1.20, 0.20, -2.15, 0.10, 0.00), tags(explore=0.6, boredom=0.4)),
    ("hand_to_neck",    (0.20, -1.55, 0.00, -2.30, 0.50, 0.00), tags(soothe=0.7, itch=0.5, vigilance=0.3)),
    ("scratch_head",    (-0.70, -1.90, 0.00, -2.20, 0.30, 0.00), tags(itch=0.9, boredom=0.4, explore=0.2)),
    ("reach_low",       (-0.10, -0.90, 0.00, -0.60, -0.10, 0.00), tags(object=0.8, explore=0.5)),
    ("reach_mid",       (-0.10, -1.30, 0.00, -0.30, 0.00, 0.00), tags(object=1.0, explore=0.5)),
    ("reach_high",      (-0.20, -1.70, 0.00, -0.15, 0.00, 0.00), tags(object=0.6, explore=0.4, energetic=0.3)),
    ("reach_up",        (-0.40, -2.30, 0.00, -0.25, 0.00, 0.00), tags(energetic=0.7, expressive=0.4, fatigue=0.2)),
    ("reach_side",      (-1.55, 0.00, 0.00, -0.25, 0.00, 0.00), tags(expressive=0.6, energetic=0.5)),
    ("raise_side_high", (-2.30, 0.00, 0.00, -0.30, 0.00, 0.00), tags(expressive=0.7, energetic=0.6)),
    ("wave_pose",       (-1.30, -0.40, 1.50, -1.55, 0.00, 0.00), tags(social=1.0, expressive=0.7, energetic=0.4)),
    ("point_forward",   (-0.15, -1.35, 0.00, -0.20, 0.00, 0.00), tags(object=0.9, social=0.3, explore=0.4)),
    ("point_up",        (-0.30, -2.40, 0.00, -0.20, 0.00, 0.00), tags(expressive=0.6, explore=0.3)),
    ("guard",           (-0.50, -0.45, 0.00, -1.30, 0.00, 0.00), tags(defensive=1.0, vigilance=0.5)),
    ("shield_face",     (-0.20, -1.10, 0.00, -2.20, 0.00, 0.00), tags(defensive=1.0, vigilance=0.4)),
    ("palm_out_offer",  (-0.20, -1.10, -1.40, -0.90, 0.30, 0.00), tags(social=0.8, expressive=0.5)),
    ("beckon",          (-0.15, -1.00, 0.60, -1.40, -0.40, 0.00), tags(social=0.7, expressive=0.7)),
    ("stop_palm",       (-0.20, -1.30, 0.00, -0.40, -0.90, 0.00), tags(defensive=0.6, expressive=0.6, vigilance=0.3)),
    ("shrug_out",       (-0.55, -0.15, 0.90, -1.20, 0.20, 0.00), tags(expressive=0.8, social=0.4)),
    ("cheer",           (-0.50, -2.40, 0.00, -0.40, 0.00, 0.00), tags(energetic=1.0, expressive=0.9, social=0.4)),
    ("stretch_back",    (-2.00, 0.90, 0.00, -0.10, 0.00, 0.00), tags(fatigue=0.8, comfort=0.6, relax=0.2)),
    ("touch_forearm",   (0.10, -0.95, 0.20, -1.95, 0.15, 0.00), tags(soothe=0.7, social=0.2, boredom=0.3)),
    ("clap_ready",      (0.15, -1.00, 0.30, -1.60, 0.00, 0.00), tags(expressive=0.7, energetic=0.6, social=0.5)),
    ("hug_upper",       (0.45, -0.80, 0.70, -2.00, 0.00, 0.00), tags(cold=0.9, soothe=0.8, defensive=0.3)),
    ("hug_waist",       (0.20, -0.40, 0.60, -1.70, 0.00, 0.00), tags(cold=0.6, soothe=0.5, comfort=0.4)),
    ("salute",          (-1.00, -0.60, 0.80, -2.30, 0.10, 0.00), tags(expressive=0.6, social=0.4)),
    ("hand_to_ear",     (-0.90, -1.00, 0.60, -2.10, 0.30, 0.00), tags(vigilance=0.6, explore=0.3, itch=0.2)),
    ("hold_cup",        (-0.10, -0.95, 0.20, -1.60, 0.00, 0.00), tags(hunger=0.5, object=0.4, comfort=0.3)),
    ("drink",           (0.00, -1.20, 0.10, -2.20, 0.35, 0.00), tags(hunger=0.7, comfort=0.3)),
    ("elbow_out",       (-1.20, -0.30, 0.40, -1.90, 0.00, 0.00), tags(heat=0.8, expressive=0.3)),
    ("swing_back",      (0.00, 0.70, 0.00, -0.35, 0.00, 0.00), tags(energetic=0.5, explore=0.2)),
    ("swing_forward",   (0.00, -0.70, 0.00, -0.30, 0.00, 0.00), tags(energetic=0.5, explore=0.2)),
    ("reach_across",    (0.55, -1.20, 0.90, -1.30, 0.00, 0.00), tags(object=0.6, explore=0.4)),
    ("press_down",      (-0.20, -0.80, 0.00, -0.90, -0.60, 0.00), tags(explore=0.4, object=0.5, comfort=0.2)),
    ("cradle",          (0.20, -0.70, 0.30, -1.75, 0.20, 0.00), tags(soothe=0.6, social=0.4)),
    ("offer",           (-0.25, -1.15, -0.30, -0.55, 0.40, 0.00), tags(social=0.9, object=0.4)),
    ("rub_eyes_both",   (0.25, -1.55, -0.30, -2.35, 0.45, 0.00), tags(ocular=0.8, fatigue=0.5, soothe=0.2)),
    ("fidget_low",      (-0.12, -0.25, 0.30, -0.70, 0.10, 0.00), tags(boredom=0.9, relax=0.2)),
]
N_ARM_SCHEMAS = len(ARM_SCHEMAS)
ARM_STEP = np.array([0.14, 0.14, 0.22, 0.14, 0.16, 0.08])     # one variation step per joint
ARM_VAR = 5                                                    # variation levels per joint

# --------------------------------------------------------------------------
# Hand shapes: curl 0..1 for (thumb, index, other fingers)
# --------------------------------------------------------------------------
HAND_SHAPES: list[tuple[str, tuple, np.ndarray]] = [
    ("open",          (0.00, 0.00, 0.00), tags(expressive=0.4, social=0.3)),
    ("relaxed",       (0.30, 0.25, 0.35), tags(relax=1.0)),
    ("fist",          (0.90, 1.00, 1.00), tags(defensive=0.6, energetic=0.4, pain=0.4)),
    ("point",         (0.70, 0.00, 1.00), tags(object=0.7, explore=0.4)),
    ("pinch",         (0.70, 0.70, 0.15), tags(object=0.6, itch=0.4, explore=0.3)),
    ("thumbs_up",     (0.00, 1.00, 1.00), tags(social=0.6, expressive=0.7)),
    ("ok",            (0.60, 0.60, 0.10), tags(social=0.6, expressive=0.6)),
    ("claw",          (0.50, 0.60, 0.60), tags(itch=0.7, energetic=0.3)),
    ("half_curl",     (0.50, 0.50, 0.50), tags(relax=0.5)),
    ("grip_ball",     (0.55, 0.65, 0.65), tags(object=0.6, hunger=0.2)),
    ("tap_ready",     (0.40, 0.20, 0.50), tags(boredom=0.8)),
    ("thumb_out",     (0.00, 0.80, 0.80), tags(expressive=0.4, social=0.3)),
    ("index_curl",    (0.10, 0.90, 0.10), tags(boredom=0.5, itch=0.2)),
    ("loose_fist",    (0.60, 0.70, 0.70), tags(soothe=0.5, cold=0.6, comfort=0.3)),
]
N_HAND_SHAPES = len(HAND_SHAPES)
HAND_VAR = 3
HAND_STEP = 0.16

# --------------------------------------------------------------------------
# Small direct channels
# --------------------------------------------------------------------------
HEAD_YAW = np.linspace(-0.80, 0.80, 11)
HEAD_PITCH = np.linspace(-0.40, 0.50, 9)
HEAD_TILT = np.linspace(-0.32, 0.32, 7)
GAZE_YAW = np.linspace(-0.70, 0.70, 9)
GAZE_PITCH = np.linspace(-0.35, 0.35, 7)
LID_APERTURE = np.array([1.25, 1.0, 0.75, 0.50, 0.20])
LID_APERTURE_NAMES = ("wide", "normal", "squint", "droopy", "nearly_closed")
BLINK_PATTERNS = ("spontaneous", "suppressed", "double", "slow", "flutter",
                  "wink_left", "wink_right")
JAW_LEVELS = np.array([0.0, 0.08, 0.16, 0.24, 0.33, 0.42])
VOCALS = ("none", "hum", "sigh", "cough", "gasp", "laugh", "yawn", "murmur", "whistle")
TORSO_BEND = np.linspace(-0.10, 0.34, 7)
TORSO_SIDE = np.linspace(-0.16, 0.16, 5)
TORSO_TWIST = np.linspace(-0.34, 0.34, 7)
STANCE_HEIGHT = np.array([0.0, -0.02, -0.04, -0.06, -0.08])         # COM drop (m)
STANCE_LATERAL = np.linspace(-0.030, 0.030, 5)
STANCE_FORWARD = np.array([-0.015, 0.0, 0.020])
STYLE_SPEED = np.array([0.35, 0.6, 1.0, 1.6, 2.4])
STYLE_AMP = np.array([0.45, 0.75, 1.0, 1.2])
STYLE_TREMOR = np.array([0.0, 0.010, 0.030, 0.060])
STYLE_RHYTHM = ("steady", "pulse", "sway_slow", "sway_fast")
HOLD_SECONDS = np.array([0.8, 1.5, 2.5, 4.0, 7.0, 12.0])
INTENT_MODES = ("look", "point", "reach", "grasp")
OBJECTS = ("apple", "mug", "stone", "cushion", "ball", "table")
WALK_SPEED = np.array([0.15, 0.30, 0.45])
WALK_TURN = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])

# Self-touch regions: (name, body, local anchor (left-hand-side), tag vector)
SELF_REGIONS: list[tuple[str, str, tuple, np.ndarray]] = [
    ("eyes",       "head",  (0.030, -0.118, 0.078), tags(ocular=1.0, fatigue=0.4)),
    ("cheek",      "head",  (0.050, -0.110, 0.048), tags(soothe=0.5, boredom=0.4, itch=0.4)),
    ("nose",       "head",  (0.000, -0.118, 0.056), tags(itch=0.8, boredom=0.3)),
    ("mouth",      "head",  (0.000, -0.110, 0.030), tags(hunger=0.7, soothe=0.3)),
    ("chin",       "head",  (0.000, -0.102, 0.008), tags(explore=0.6, boredom=0.4)),
    ("forehead",   "head",  (0.000, -0.102, 0.108), tags(fatigue=0.6, soothe=0.4, pain=0.4)),
    ("scalp",      "head",  (0.020, -0.020, 0.190), tags(itch=0.9, boredom=0.4)),
    ("ear",        "head",  (0.098, -0.010, 0.082), tags(itch=0.5, vigilance=0.3)),
    ("neck",       "neck",  (0.030, -0.045, 0.035), tags(soothe=0.7, itch=0.5, pain=0.4)),
    ("chest",      "chest", (0.070, -0.130, 0.090), tags(soothe=0.9, social=0.3, defensive=0.2)),
    ("abdomen",    "abdomen", (0.040, -0.125, 0.050), tags(hunger=0.8, pain=0.4, comfort=0.4)),
    ("shoulder",   "shoulder_l", (0.000, -0.020, 0.000), tags(cold=0.6, pain=0.6, soothe=0.4)),
    ("upper_arm",  "upper_arm_l", (0.000, -0.020, -0.150), tags(cold=0.9, soothe=0.5)),
    ("forearm",    "forearm_l", (0.000, -0.020, -0.120), tags(soothe=0.6, itch=0.5, boredom=0.3)),
    ("thigh",      "thigh_l", (0.000, -0.070, -0.200), tags(boredom=0.7, relax=0.4, itch=0.3)),
    ("hip",        "pelvis", (0.130, -0.040, -0.020), tags(relax=0.4, pain=0.3, expressive=0.2)),
]
SELF_ACTIONS = ("rest", "rub", "tap", "scratch")


# --------------------------------------------------------------------------
# Names for the mind: ``express(head="left", lids="squint", left_arm="wave_pose", ...)``
# --------------------------------------------------------------------------
def _nearest(arr, v) -> int:
    return int(np.argmin(np.abs(np.asarray(arr, float) - float(v))))


_DIRS = {  # name -> (yaw, pitch)   yaw + = to the person's left, pitch + = down
    "forward": (0.0, 0.0), "left": (0.55, 0.0), "right": (-0.55, 0.0), "up": (0.0, -0.35),
    "down": (0.0, 0.40), "left_up": (0.5, -0.30), "right_up": (-0.5, -0.30),
    "left_down": (0.5, 0.35), "right_down": (-0.5, 0.35), "far_left": (0.8, 0.0),
    "far_right": (-0.8, 0.0), "away_left": (0.7, 0.1), "away_right": (-0.7, 0.1),
}
_TORSO = {"upright": (0.05, 0.0, 0.0), "slouch": (0.28, 0.0, 0.0), "lean_forward": (0.18, 0.0, 0.0),
          "lean_back": (-0.08, 0.0, 0.0), "twist_left": (0.05, 0.0, 0.3), "twist_right": (0.05, 0.0, -0.3),
          "lean_left": (0.05, 0.14, 0.0), "lean_right": (0.05, -0.14, 0.0), "bow": (0.34, 0.0, 0.0)}
_STANCE = {"normal": (0.0, 0.0, 0.0), "crouch": (-0.04, 0.0, 0.0), "deep_crouch": (-0.08, 0.0, 0.0),
           "shift_left": (0.0, 0.03, 0.0), "shift_right": (0.0, -0.03, 0.0), "lean_in": (0.0, 0.0, 0.02),
           "lean_back": (0.0, 0.0, -0.015)}
_STYLE = {  # name -> (speed idx, amp idx, tremor idx, rhythm idx)
    "normal": (2, 2, 0, 0), "slow": (0, 2, 0, 0), "fast": (3, 2, 0, 0), "trembling": (2, 1, 3, 0),
    "sway": (1, 2, 0, 2), "pulse": (2, 2, 0, 1), "small": (2, 0, 0, 0), "large": (2, 3, 0, 0),
    "gentle": (1, 1, 0, 0), "restless": (3, 1, 1, 3),
}
_JAW = {"closed": 0, "slightly_open": 1, "open": 3, "wide": 5}


def build_from_names(space: "BehaviorSpace", **kw) -> tuple[np.ndarray, list[str]]:
    """Descriptor from human-readable choices.  Unknown names are reported, not fatal."""
    d = space.neutral()
    ci = space.channel_index
    errors: list[str] = []

    def bad(k, v, options):
        errors.append(f"{k}={v!r} is not one of: {', '.join(options)}")

    for k, v in kw.items():
        try:
            if k == "head":
                if v not in _DIRS:
                    bad(k, v, _DIRS); continue
                y, p = _DIRS[v]
                d[ci["head"]] = np.ravel_multi_index((_nearest(HEAD_YAW, y), _nearest(HEAD_PITCH, p), 3),
                                                     (len(HEAD_YAW), len(HEAD_PITCH), len(HEAD_TILT)))
            elif k == "tilt":
                tl = {"left": 0.28, "right": -0.28, "none": 0.0}
                if v not in tl:
                    bad(k, v, tl); continue
                y, p, _ = np.unravel_index(int(d[ci["head"]]), (len(HEAD_YAW), len(HEAD_PITCH), len(HEAD_TILT)))
                d[ci["head"]] = np.ravel_multi_index((y, p, _nearest(HEAD_TILT, tl[v])),
                                                     (len(HEAD_YAW), len(HEAD_PITCH), len(HEAD_TILT)))
            elif k in ("gaze", "eyes"):
                if v not in _DIRS:
                    bad(k, v, _DIRS); continue
                y, p = _DIRS[v]
                d[ci["eyes"]] = np.ravel_multi_index((_nearest(GAZE_YAW, y), _nearest(GAZE_PITCH, p * 0.8)),
                                                     (len(GAZE_YAW), len(GAZE_PITCH)))
            elif k == "lids":
                if v not in LID_APERTURE_NAMES:
                    bad(k, v, LID_APERTURE_NAMES + ("closed",)); continue
                a = LID_APERTURE_NAMES.index(v)
                _, b = np.unravel_index(int(d[ci["lids"]]), (len(LID_APERTURE), len(BLINK_PATTERNS)))
                d[ci["lids"]] = np.ravel_multi_index((a, b), (len(LID_APERTURE), len(BLINK_PATTERNS)))
            elif k == "blink":
                if v not in BLINK_PATTERNS:
                    bad(k, v, BLINK_PATTERNS); continue
                a, _ = np.unravel_index(int(d[ci["lids"]]), (len(LID_APERTURE), len(BLINK_PATTERNS)))
                d[ci["lids"]] = np.ravel_multi_index((a, BLINK_PATTERNS.index(v)), (len(LID_APERTURE), len(BLINK_PATTERNS)))
            elif k in ("mouth", "voice"):
                j, vo = np.unravel_index(int(d[ci["mouth"]]), (len(JAW_LEVELS), len(VOCALS)))
                if k == "mouth":
                    if v not in _JAW:
                        bad(k, v, _JAW); continue
                    j = _JAW[v]
                else:
                    if v not in VOCALS:
                        bad(k, v, VOCALS); continue
                    vo = VOCALS.index(v)
                d[ci["mouth"]] = np.ravel_multi_index((j, vo), (len(JAW_LEVELS), len(VOCALS)))
            elif k == "torso":
                if v not in _TORSO:
                    bad(k, v, _TORSO); continue
                b_, s_, t_ = _TORSO[v]
                d[ci["torso"]] = np.ravel_multi_index((_nearest(TORSO_BEND, b_), _nearest(TORSO_SIDE, s_),
                                                       _nearest(TORSO_TWIST, t_)),
                                                      (len(TORSO_BEND), len(TORSO_SIDE), len(TORSO_TWIST)))
            elif k in ("left_arm", "right_arm"):
                names = [n for n, _, _ in ARM_SCHEMAS]
                if v not in names:
                    bad(k, v, names); continue
                d[ci[("l" if k == "left_arm" else "r") + "_schema"]] = names.index(v)
            elif k in ("left_hand", "right_hand"):
                names = [n for n, _, _ in HAND_SHAPES]
                if v not in names:
                    bad(k, v, names); continue
                d[ci[("l" if k == "left_hand" else "r") + "_hand"]] = names.index(v)
            elif k == "stance":
                if v not in _STANCE:
                    bad(k, v, _STANCE); continue
                h, l, f = _STANCE[v]
                d[ci["stance"]] = np.ravel_multi_index((_nearest(STANCE_HEIGHT, h), _nearest(STANCE_LATERAL, l),
                                                        _nearest(STANCE_FORWARD, f)),
                                                       (len(STANCE_HEIGHT), len(STANCE_LATERAL), len(STANCE_FORWARD)))
            elif k == "style":
                if v not in _STYLE:
                    bad(k, v, _STYLE); continue
                d[ci["style"]] = np.ravel_multi_index(_STYLE[v], (len(STYLE_SPEED), len(STYLE_AMP),
                                                                  len(STYLE_TREMOR), len(STYLE_RHYTHM)))
            elif k in ("hold", "seconds"):
                d[ci["hold"]] = _nearest(HOLD_SECONDS, v)
            else:
                errors.append(f"unknown part '{k}' (try head tilt gaze lids blink mouth voice torso "
                              f"left_arm right_arm left_hand right_hand stance style hold)")
        except Exception as exc:                       # a malformed value must never crash the body
            errors.append(f"{k}: {exc}")
    return d, errors


def option_names() -> dict:
    """The vocabulary ``express`` understands (for the mind's prompt)."""
    return {
        "head": list(_DIRS), "gaze": list(_DIRS), "lids": list(LID_APERTURE_NAMES),
        "blink": list(BLINK_PATTERNS), "mouth": list(_JAW), "voice": list(VOCALS),
        "torso": list(_TORSO), "arms": [n for n, _, _ in ARM_SCHEMAS],
        "hands": [n for n, _, _ in HAND_SHAPES], "stance": list(_STANCE), "style": list(_STYLE),
    }


@dataclass
class Channel:
    name: str
    size: int
    tag: np.ndarray            # (size, NT) affinity of each choice


def _tag_matrix(n: int, fn) -> np.ndarray:
    m = np.zeros((n, NT))
    for i in range(n):
        m[i] = fn(i)
    return m


class BehaviorSpace:
    """Factored space of behaviours: encode / decode / sample / describe."""

    def __init__(self, joint_names: list[str], joint_lo: np.ndarray, joint_hi: np.ndarray,
                 q_nom: np.ndarray):
        self.names = list(joint_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.lo, self.hi, self.q_nom = joint_lo, joint_hi, q_nom
        self.build_channels()

    # ------------------------------------------------------------------
    def build_channels(self) -> None:
        ch: list[Channel] = []
        n_arm = N_ARM_SCHEMAS
        arm_tags = np.stack([t for _, _, t in ARM_SCHEMAS])
        hand_tags = np.stack([t for _, _, t in HAND_SHAPES])

        def head_tag(i):          # head: facing forward is social/relax, extremes explore
            y, p, t = np.unravel_index(i, (len(HEAD_YAW), len(HEAD_PITCH), len(HEAD_TILT)))
            ay, ap, at = abs(HEAD_YAW[y]), abs(HEAD_PITCH[p] - 0.05), abs(HEAD_TILT[t])
            return tags(explore=1.4 * ay + 0.6 * ap, vigilance=0.7 * ay, social=0.8 * (1 - ay) * (1 - at),
                        relax=0.5 * (1 - ay - ap), fatigue=0.8 * max(HEAD_PITCH[p] - 0.2, 0),
                        expressive=1.2 * at, boredom=0.5 * ay, ocular=0.2 * max(-HEAD_PITCH[p], 0))

        def gaze_tag(i):
            y, p = np.unravel_index(i, (len(GAZE_YAW), len(GAZE_PITCH)))
            ay, ap = abs(GAZE_YAW[y]), abs(GAZE_PITCH[p])
            return tags(explore=1.2 * ay + 0.8 * ap, vigilance=0.9 * ay, social=0.8 * (1 - ay),
                        boredom=0.7 * ay, relax=0.3 * (1 - ay - ap))

        def lid_tag(i):
            a, b = np.unravel_index(i, (len(LID_APERTURE), len(BLINK_PATTERNS)))
            return tags(vigilance=1.0 if a == 0 else 0.0, fatigue=1.0 if a >= 3 else 0.0,
                        ocular=(0.9 if a == 2 else 0.0) + (1.0 if b in (4, 3) else 0.0)
                        + (0.6 if b == 2 else 0.0), relax=0.5 if a == 1 and b == 0 else 0.0,
                        expressive=1.0 if b >= 5 else 0.0, social=0.6 if b >= 5 else 0.0,
                        explore=0.4 if (a == 0 or b == 1) else 0.0)

        def mouth_tag(i):
            j, v = np.unravel_index(i, (len(JAW_LEVELS), len(VOCALS)))
            return tags(fatigue=1.4 if VOCALS[v] == "yawn" else 0.0,
                        expressive=0.8 if VOCALS[v] in ("laugh", "gasp", "hum") else 0.0,
                        relax=0.8 if VOCALS[v] in ("hum", "sigh", "murmur") else 0.0,
                        comfort=0.6 if VOCALS[v] in ("cough", "sigh") else 0.0,
                        boredom=0.6 if VOCALS[v] in ("hum", "whistle") else 0.0,
                        hunger=0.5 if j >= 2 and VOCALS[v] == "none" else 0.0,
                        defensive=0.6 if VOCALS[v] == "gasp" else 0.0,
                        social=0.4 if VOCALS[v] in ("laugh", "hum") else 0.0)

        def torso_tag(i):
            b, s, t = np.unravel_index(i, (len(TORSO_BEND), len(TORSO_SIDE), len(TORSO_TWIST)))
            return tags(fatigue=1.2 * max(TORSO_BEND[b] - 0.1, 0), defensive=0.8 * max(TORSO_BEND[b] - 0.15, 0),
                        explore=0.9 * abs(TORSO_TWIST[t]) + 0.3 * abs(TORSO_SIDE[s]),
                        relax=0.5 * (1 - abs(TORSO_TWIST[t]) - abs(TORSO_SIDE[s])),
                        expressive=1.0 * abs(TORSO_SIDE[s]), energetic=0.8 * max(-TORSO_BEND[b], 0),
                        social=0.5 * max(TORSO_BEND[b] * (1 - abs(TORSO_TWIST[t])), 0), pain=0.5 * abs(TORSO_SIDE[s]))

        def stance_tag(i):
            h, l, f = np.unravel_index(i, (len(STANCE_HEIGHT), len(STANCE_LATERAL), len(STANCE_FORWARD)))
            return tags(defensive=1.3 * (-STANCE_HEIGHT[h]) * 7, vigilance=0.7 * (-STANCE_HEIGHT[h]) * 7,
                        fatigue=0.6 * (STANCE_HEIGHT[h] == 0) * 0.0 + 0.8 * abs(STANCE_LATERAL[l]) * 33 * 0.3,
                        boredom=0.9 * abs(STANCE_LATERAL[l]) * 33, relax=0.4 * (STANCE_HEIGHT[h] == 0),
                        pain=0.5 * abs(STANCE_LATERAL[l]) * 33, explore=0.6 * max(STANCE_FORWARD[f], 0) * 50)

        def style_tag(i):
            sp, am, tr, rh = np.unravel_index(i, (len(STYLE_SPEED), len(STYLE_AMP), len(STYLE_TREMOR), len(STYLE_RHYTHM)))
            return tags(energetic=0.9 * (sp - 2) / 2 + 0.7 * (am - 1.5) / 1.5 + 0.5 * (rh >= 2),
                        fatigue=-0.9 * (sp - 2) / 2 * -1 if sp < 2 else 0.0,
                        vigilance=0.9 * (tr >= 2) + 0.4 * (sp >= 3), defensive=0.5 * (tr >= 2),
                        relax=0.8 * (sp <= 1) * (tr == 0), boredom=0.7 * (rh == 1), expressive=0.6 * (am >= 3),
                        pain=0.5 * (tr >= 2))

        def intent_tag(i):
            if i == 0:
                return tags(relax=0.4)
            o, m = divmod(i - 1, len(INTENT_MODES))
            return tags(object=1.0 + 0.3 * m, explore=0.6 + 0.2 * m, social=0.3 if m == 1 else 0.0,
                        hunger=0.5 if (OBJECTS[o] in ("apple", "mug") and m == 3) else 0.0,
                        comfort=0.5 if (OBJECTS[o] == "cushion" and m >= 2) else 0.0)

        def touch_tag(i):
            if i == 0:
                return tags(relax=0.4)
            r, rest = divmod(i - 1, 2 * len(SELF_ACTIONS))
            h, a = divmod(rest, len(SELF_ACTIONS))
            t = SELF_REGIONS[r][3].copy()
            if SELF_ACTIONS[a] == "scratch":
                t[TI["itch"]] += 0.7
            elif SELF_ACTIONS[a] == "rub":
                t[TI["soothe"]] += 0.3
                t[TI["ocular"]] += 0.3 * (SELF_REGIONS[r][0] == "eyes")
            elif SELF_ACTIONS[a] == "tap":
                t[TI["boredom"]] += 0.5
            return t

        def walk_tag(i):
            if i == 0:
                return tags(relax=0.3)
            return tags(explore=1.0, energetic=0.6, cold=0.3)

        def zeros(n):
            return np.zeros((n, NT))

        sizes = {
            "head": len(HEAD_YAW) * len(HEAD_PITCH) * len(HEAD_TILT),
            "eyes": len(GAZE_YAW) * len(GAZE_PITCH),
            "lids": len(LID_APERTURE) * len(BLINK_PATTERNS),
            "mouth": len(JAW_LEVELS) * len(VOCALS),
            "torso": len(TORSO_BEND) * len(TORSO_SIDE) * len(TORSO_TWIST),
            "stance": len(STANCE_HEIGHT) * len(STANCE_LATERAL) * len(STANCE_FORWARD),
            "style": len(STYLE_SPEED) * len(STYLE_AMP) * len(STYLE_TREMOR) * len(STYLE_RHYTHM),
            "hold": len(HOLD_SECONDS),
            "intent": 1 + len(OBJECTS) * len(INTENT_MODES),
            "touch": 1 + len(SELF_REGIONS) * 2 * len(SELF_ACTIONS),
            "walk": 1 + len(WALK_SPEED) * len(WALK_TURN),
        }
        self.tagm = {
            "head": _tag_matrix(sizes["head"], head_tag),
            "eyes": _tag_matrix(sizes["eyes"], gaze_tag),
            "lids": _tag_matrix(sizes["lids"], lid_tag),
            "mouth": _tag_matrix(sizes["mouth"], mouth_tag),
            "torso": _tag_matrix(sizes["torso"], torso_tag),
            "stance": _tag_matrix(sizes["stance"], stance_tag),
            "style": _tag_matrix(sizes["style"], style_tag),
            "hold": zeros(sizes["hold"]),
            "intent": _tag_matrix(sizes["intent"], intent_tag),
            "touch": _tag_matrix(sizes["touch"], touch_tag),
            "walk": _tag_matrix(sizes["walk"], walk_tag),
            "arm_schema": arm_tags,
            "hand_shape": hand_tags,
        }
        # ordered descriptor layout: (name, size)
        self.layout: list[tuple[str, int]] = [
            ("head", sizes["head"]), ("eyes", sizes["eyes"]), ("lids", sizes["lids"]),
            ("mouth", sizes["mouth"]), ("torso", sizes["torso"]),
            ("l_schema", N_ARM_SCHEMAS)] + [(f"l_arm_{j}", ARM_VAR) for j in range(6)] + [
            ("r_schema", N_ARM_SCHEMAS)] + [(f"r_arm_{j}", ARM_VAR) for j in range(6)] + [
            ("l_hand", N_HAND_SHAPES)] + [(f"l_hand_{j}", HAND_VAR) for j in range(3)] + [
            ("r_hand", N_HAND_SHAPES)] + [(f"r_hand_{j}", HAND_VAR) for j in range(3)] + [
            ("stance", sizes["stance"]), ("style", sizes["style"]), ("hold", sizes["hold"]),
            ("intent", sizes["intent"]), ("touch", sizes["touch"]), ("walk", sizes["walk"])]
        self.channel_index = {n: i for i, (n, _) in enumerate(self.layout)}
        self.n_channels = len(self.layout)
        self.sizes = np.array([s for _, s in self.layout], dtype=object)

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of distinct descriptors (a Python int: it is ~1e40)."""
        n = 1
        for _, s in self.layout:
            n *= int(s)
        return n

    @property
    def per_arm(self) -> int:
        return N_ARM_SCHEMAS * ARM_VAR ** 6

    def encode(self, d: np.ndarray) -> int:
        """Descriptor -> integer id (mixed radix)."""
        v = 0
        for (_, s), x in zip(self.layout, d):
            v = v * int(s) + int(x)
        return v

    def decode(self, v: int) -> np.ndarray:
        d = np.zeros(self.n_channels, dtype=np.int64)
        for k in range(self.n_channels - 1, -1, -1):
            s = int(self.layout[k][1])
            v, r = divmod(int(v), s)
            d[k] = r
        return d

    def neutral(self) -> np.ndarray:
        """The 'do nothing' descriptor: rest everywhere."""
        d = np.zeros(self.n_channels, dtype=np.int64)
        c = self.channel_index
        d[c["head"]] = np.ravel_multi_index((5, 3, 3), (len(HEAD_YAW), len(HEAD_PITCH), len(HEAD_TILT)))
        d[c["eyes"]] = np.ravel_multi_index((4, 3), (len(GAZE_YAW), len(GAZE_PITCH)))
        d[c["lids"]] = np.ravel_multi_index((1, 0), (len(LID_APERTURE), len(BLINK_PATTERNS)))
        d[c["mouth"]] = 0
        d[c["torso"]] = np.ravel_multi_index((1, 2, 3), (len(TORSO_BEND), len(TORSO_SIDE), len(TORSO_TWIST)))
        for s in ("l", "r"):
            d[c[f"{s}_schema"]] = 0
            for j in range(6):
                d[c[f"{s}_arm_{j}"]] = ARM_VAR // 2
            d[c[f"{s}_hand"]] = 1
            for j in range(3):
                d[c[f"{s}_hand_{j}"]] = HAND_VAR // 2
        d[c["stance"]] = np.ravel_multi_index((0, 2, 1), (len(STANCE_HEIGHT), len(STANCE_LATERAL), len(STANCE_FORWARD)))
        d[c["style"]] = np.ravel_multi_index((2, 2, 0, 0), (len(STYLE_SPEED), len(STYLE_AMP), len(STYLE_TREMOR), len(STYLE_RHYTHM)))
        d[c["hold"]] = 2
        return d

    # ------------------------------------------------------------------
    # compile: descriptor -> executable behaviour
    # ------------------------------------------------------------------
    def compile(self, d: np.ndarray) -> "Behavior":
        c = self.channel_index
        b = Behavior(desc=np.array(d, dtype=np.int64))
        q = {}
        ju = self.idx

        def put(name, val):
            if name in ju:
                lo, hi = self.lo[ju[name]], self.hi[ju[name]]
                m = 0.03 * (hi - lo)
                q[name] = float(np.clip(val, lo + m, hi - m))

        # head
        y, p, t = np.unravel_index(int(d[c["head"]]), (len(HEAD_YAW), len(HEAD_PITCH), len(HEAD_TILT)))
        put("neck_twist", HEAD_YAW[y]); put("neck_bend", HEAD_PITCH[p]); put("neck_side", HEAD_TILT[t])
        # torso: split between spine and chest joints
        bi, si, ti = np.unravel_index(int(d[c["torso"]]), (len(TORSO_BEND), len(TORSO_SIDE), len(TORSO_TWIST)))
        put("spine_bend", 0.6 * TORSO_BEND[bi]); put("chest_bend", 0.4 * TORSO_BEND[bi])
        put("spine_side", 0.6 * TORSO_SIDE[si]); put("chest_side", 0.4 * TORSO_SIDE[si])
        put("spine_twist", 0.55 * TORSO_TWIST[ti]); put("chest_twist", 0.45 * TORSO_TWIST[ti])
        # arms and hands
        for s, sx in (("l", 1.0), ("r", -1.0)):
            sch = ARM_SCHEMAS[int(d[c[f"{s}_schema"]])][1]
            var = np.array([(int(d[c[f"{s}_arm_{j}"]]) - ARM_VAR // 2) * ARM_STEP[j] for j in range(6)])
            a = np.array(sch) + var
            put(f"sh_{s}_abd", sx * a[0]); put(f"sh_{s}_flex", a[1]); put(f"sh_{s}_rot", sx * a[2])
            put(f"elbow_{s}", a[3]); put(f"wrist_{s}_flex", a[4]); put(f"wrist_{s}_dev", sx * a[5])
            curl = np.array(HAND_SHAPES[int(d[c[f"{s}_hand"]])][1]) + np.array(
                [(int(d[c[f"{s}_hand_{j}"]]) - HAND_VAR // 2) * HAND_STEP for j in range(3)])
            curl = np.clip(curl, 0.0, 1.0)
            for k, dg in enumerate(("thumb", "index", "fingers")):
                put(f"{dg}_{s}_mcp", sx * 1.30 * curl[k]); put(f"{dg}_{s}_pip", sx * 1.45 * curl[k])
        b.joints = q
        # eyes, lids, mouth
        gy, gp = np.unravel_index(int(d[c["eyes"]]), (len(GAZE_YAW), len(GAZE_PITCH)))
        b.gaze = (float(GAZE_YAW[gy]), float(GAZE_PITCH[gp]))
        la, lb = np.unravel_index(int(d[c["lids"]]), (len(LID_APERTURE), len(BLINK_PATTERNS)))
        b.lid_aperture = float(LID_APERTURE[la]); b.blink = BLINK_PATTERNS[lb]
        mj, mv = np.unravel_index(int(d[c["mouth"]]), (len(JAW_LEVELS), len(VOCALS)))
        b.jaw = float(JAW_LEVELS[mj]); b.vocal = VOCALS[mv]
        # stance, style, hold
        sh, sl, sf = np.unravel_index(int(d[c["stance"]]), (len(STANCE_HEIGHT), len(STANCE_LATERAL), len(STANCE_FORWARD)))
        b.stance = (float(STANCE_HEIGHT[sh]), float(STANCE_LATERAL[sl]), float(STANCE_FORWARD[sf]))
        sp, am, tr, rh = np.unravel_index(int(d[c["style"]]), (len(STYLE_SPEED), len(STYLE_AMP), len(STYLE_TREMOR), len(STYLE_RHYTHM)))
        b.speed = float(STYLE_SPEED[sp]); b.amp = float(STYLE_AMP[am])
        b.tremor = float(STYLE_TREMOR[tr]); b.rhythm = STYLE_RHYTHM[rh]
        b.hold = float(HOLD_SECONDS[int(d[c["hold"]])])
        # intent
        i = int(d[c["intent"]])
        if i > 0:
            o, m = divmod(i - 1, len(INTENT_MODES))
            b.intent = (INTENT_MODES[m], OBJECTS[o])
        # self-touch
        i = int(d[c["touch"]])
        if i > 0:
            r, rest = divmod(i - 1, 2 * len(SELF_ACTIONS))
            h, a = divmod(rest, len(SELF_ACTIONS))
            b.touch = (SELF_REGIONS[r][0], "l" if h == 0 else "r", SELF_ACTIONS[a])
        # walk
        i = int(d[c["walk"]])
        if i > 0:
            k = i - 1
            sidx, tidx = divmod(k, len(WALK_TURN))
            b.walk = (float(WALK_SPEED[sidx]), float(WALK_TURN[tidx]))
        b.name = self.describe_behavior(b)
        return b

    @staticmethod
    def describe_behavior(b: "Behavior") -> str:
        bits = []
        if b.touch:
            bits.append(f"{b.touch[2]} {b.touch[0]} ({b.touch[1]} hand)")
        if b.intent:
            bits.append(f"{b.intent[0]} {b.intent[1]}")
        if b.vocal != "none":
            bits.append(b.vocal)
        if b.blink not in ("spontaneous",):
            bits.append(f"blink:{b.blink}")
        if b.walk:
            bits.append("walk")
        bits.append(f"{b.rhythm}/{b.speed:.1f}x")
        return ", ".join(bits)

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "channels": {n: int(s) for n, s in self.layout},
            "n_channels": self.n_channels,
            "arm_schemas": N_ARM_SCHEMAS, "per_arm": self.per_arm,
            "hand_shapes": N_HAND_SHAPES,
            "descriptors": self.size,
            "descriptors_log10": float(sum(math.log10(int(s)) for _, s in self.layout)),
            "self_touch_regions": len(SELF_REGIONS),
            "objects": list(OBJECTS),
        }


@dataclass
class Behavior:
    """One executable behaviour (the compiled form of a descriptor)."""
    desc: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    name: str = ""
    joints: dict = field(default_factory=dict)       # joint name -> absolute angle (rad)
    gaze: tuple = (0.0, 0.0)
    lid_aperture: float = 1.0
    blink: str = "spontaneous"
    jaw: float = 0.0
    vocal: str = "none"
    stance: tuple = (0.0, 0.0, 0.0)                  # (height drop, lateral, forward) m
    speed: float = 1.0
    amp: float = 1.0
    tremor: float = 0.0
    rhythm: str = "steady"
    hold: float = 2.5
    intent: tuple | None = None                      # (mode, object)
    touch: tuple | None = None                       # (region, hand, action)
    walk: tuple | None = None                        # (speed, turn)
    deliberate: bool = False                         # chosen by the mind: no ambient damping

    def key(self) -> bytes:
        return self.desc.tobytes()
