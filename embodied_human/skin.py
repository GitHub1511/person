"""
Whole-body artificial skin.

The skin is a dense array of *taxels* (tactile pixels) laid out on the real
geometric surfaces of the skeleton in :mod:`embodied_human.skeleton`.  Each
taxel carries its own receptor complement, so the same physical contact
produces a different afferent pattern on a fingertip than on the back --
which is exactly what makes touch feel like touch.

Receptor classes modelled (after the mammalian somatosensory literature)
-----------------------------------------------------------------------
======================  ==========================  =======================
Class                   Receptor cell               Signal
======================  ==========================  =======================
SA-I                    Merkel disc                 sustained pressure, form
SA-II                   Ruffini ending              skin stretch, shear
FA-I                    Meissner corpuscle          flutter, slip, grip
FA-II                   Pacinian corpuscle          vibration, tool use
C-tactile (CT)          C-LTMR                      affective/social touch
A-delta mechanonociceptor  --                       sharp pricking pain
C-mechanonociceptor     --                          dull burning pain
warm / cold             TRPV1-4 / TRPM8             temperature
itch                    MrgprA3+                    histamine itch
hair follicle           lanceolate ending           light air movement
======================  ==========================  =======================

Receptor *densities* differ by four orders of magnitude across the body
(fingertip ~140 units/cm^2, back ~10), which is encoded here per region.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .complexity import C as COMPLEXITY
from .skeleton import BONE_BY_NAME, BONES, Geom

# --------------------------------------------------------------------------
# Receptor density profiles
# --------------------------------------------------------------------------
# Each value is a relative gain applied to that receptor class on that region.
PROFILES: dict[str, dict[str, float]] = {
    # glabrous skin, extremely dense
    "fingertip": dict(sa1=1.00, sa2=0.60, fa1=1.15, fa2=0.95, ct=0.15,
                      noci_mech=0.95, noci_heat=0.80, noci_cold=0.70,
                      warm=0.55, cold=0.75, itch=0.30, hair=0.00,
                      thermal_mass=0.35),
    "palm": dict(sa1=0.85, sa2=0.70, fa1=0.90, fa2=0.75, ct=0.45,
                 noci_mech=0.65, noci_heat=0.55, noci_cold=0.50,
                 warm=0.60, cold=0.70, itch=0.25, hair=0.00,
                 thermal_mass=0.65),
    "sole": dict(sa1=0.95, sa2=0.65, fa1=0.80, fa2=0.70, ct=0.10,
                 noci_mech=0.80, noci_heat=0.45, noci_cold=0.60,
                 warm=0.35, cold=0.55, itch=0.15, hair=0.00,
                 thermal_mass=0.95),
    "mucosa": dict(sa1=1.20, sa2=0.50, fa1=1.00, fa2=0.80, ct=0.90,
                   noci_mech=1.05, noci_heat=1.15, noci_cold=0.30,
                   warm=0.90, cold=0.35, itch=0.20, hair=0.00,
                   thermal_mass=0.25),
    "face": dict(sa1=0.80, sa2=0.55, fa1=0.75, fa2=0.60, ct=0.75,
                 noci_mech=0.85, noci_heat=0.70, noci_cold=0.65,
                 warm=0.70, cold=0.80, itch=0.45, hair=0.35,
                 thermal_mass=0.40),
    "hairy_limb": dict(sa1=0.50, sa2=0.55, fa1=0.55, fa2=0.45, ct=0.85,
                       noci_mech=0.60, noci_heat=0.50, noci_cold=0.60,
                       warm=0.65, cold=0.80, itch=0.60, hair=1.10,
                       thermal_mass=0.60),
    "trunk": dict(sa1=0.42, sa2=0.50, fa1=0.40, fa2=0.35, ct=0.95,
                  noci_mech=0.50, noci_heat=0.45, noci_cold=0.55,
                  warm=0.70, cold=0.85, itch=0.75, hair=1.00,
                  thermal_mass=0.75),
    "back": dict(sa1=0.30, sa2=0.40, fa1=0.30, fa2=0.25, ct=1.05,
                 noci_mech=0.40, noci_heat=0.35, noci_cold=0.45,
                 warm=0.60, cold=0.80, itch=0.90, hair=1.20,
                 thermal_mass=0.85),
    "proximal_limb": dict(sa1=0.45, sa2=0.50, fa1=0.45, fa2=0.40, ct=0.90,
                          noci_mech=0.55, noci_heat=0.45, noci_cold=0.55,
                          warm=0.65, cold=0.85, itch=0.65, hair=1.05,
                          thermal_mass=0.70),
}

REGION_NAME = {
    "fingertip": "Fingertip (glabrous)",
    "palm": "Palm",
    "sole": "Sole",
    "mucosa": "Mucosa / lips / tongue",
    "face": "Face",
    "hairy_limb": "Hairy limb skin",
    "trunk": "Trunk",
    "back": "Back / gluteal",
    "proximal_limb": "Proximal limb",
}


# Mechanical pain threshold by body site, in kPa.
#
# Nociceptors respond to *pressure*, not to total force: a 12 N load on a
# 0.5 cm^2 fingertip taxel is 240 kPa and hurts, while the same 12 N spread
# over a 5.7 cm^2 sole taxel is 21 kPa and is simply standing up.  Thresholding
# on force alone makes every step agonising.  Values are the pressure at which
# a site reports pain; the sole is roughly three times more tolerant than the
# fingertip, matching the literature on pressure pain thresholds.
NOCI_PRESSURE_THRESHOLD_KPA = {
    "fingertip": 250.0,
    "palm": 350.0,
    "sole": 700.0,
    "mucosa": 180.0,
    "face": 300.0,
    "hairy_limb": 400.0,
    "trunk": 450.0,
    "back": 500.0,
    "proximal_limb": 400.0,
}
for _profile, _thr in NOCI_PRESSURE_THRESHOLD_KPA.items():
    PROFILES[_profile]["noci_threshold_kpa"] = _thr


@dataclass
class Taxel:
    index: int
    name: str
    region: str
    bone: str
    geom: str
    profile: str
    pos: np.ndarray            # body-local position
    normal: np.ndarray         # body-local outward normal
    surface: str               # ventral / dorsal / lateral / medial / distal ...
    u: float                   # 2-D body-map coordinate for plotting
    v: float
    area_cm2: float            # receptive field area
    densities: dict = field(default_factory=dict)

    def gains(self, key: str) -> float:
        return float(self.densities.get(key, 0.0))


@dataclass
class SkinPatch:
    """A contiguous sheet of taxels on one geometric surface."""
    region: str                       # body region label (arm, chest, ...)
    bone: str
    geom: str
    profile: str
    kind: str                         # cylinder | sphere | plane
    n_u: int
    n_v: int
    surface: str = "all"
    t0: float = 0.0
    t1: float = 1.0
    theta0: float = 0.0
    theta1: float = 2.0 * np.pi
    closed_theta: bool = True
    phi0: float = 0.15
    phi1: float = np.pi - 0.15
    face: str = "-z"
    inset: float = 0.0
    # Offset applied along the outward normal (mm) so taxels sit just proud of
    # the collision surface -- prevents self-intersection while staying in
    # contact range.
    proud: float = 0.0015
    enabled: bool = True


# --------------------------------------------------------------------------
# The patch table.  This is the single place to make the body more or less
# sensitive, and the main source of the "how many touch vectors" answer.
# --------------------------------------------------------------------------
def default_patches() -> list[SkinPatch]:
    P: list[SkinPatch] = []

    def limb_patches(side: str) -> None:
        sx = 1.0 if side == "l" else -1.0
        # --- upper arm: hairy, 360 deg, 12 x 6
        P.append(SkinPatch(f"upper_arm_{side}", f"upper_arm_{side}", f"upper_arm_{side}",
                           "hairy_limb", "cylinder", 12, 6, t0=0.05, t1=0.98))
        # --- deltoid
        P.append(SkinPatch(f"deltoid_{side}", f"shoulder_{side}", f"delt_{side}",
                           "proximal_limb", "sphere", 10, 5, phi0=0.25, phi1=np.pi - 0.35))
        # --- forearm: 12 x 6, split ventral/dorsal for honest labelling
        P.append(SkinPatch(f"forearm_{side}", f"forearm_{side}", f"forearm_{side}",
                           "hairy_limb", "cylinder", 12, 6, t0=0.03, t1=0.97))
        # --- palm: the hand slab is thin in x, and the palm faces medially
        #     (-x on the left hand, +x on the right), so the palmar sheet and
        #     the dorsum sheet are on opposite x faces.
        palmar_face = "-x" if side == "l" else "+x"
        dorsum_face = "+x" if side == "l" else "-x"
        P.append(SkinPatch(f"palm_{side}", f"hand_{side}", f"palm_{side}",
                           "palm", "plane", 4, 6, face=palmar_face,
                           surface="palmar"))
        P.append(SkinPatch(f"hand_dorsum_{side}", f"hand_{side}", f"palm_{side}",
                           "hairy_limb", "plane", 4, 6, face=dorsum_face,
                           surface="dorsal"))
        # --- fingers: dense cylindrical wrap
        for fname in ("thumb", "index", "fingers"):
            P.append(SkinPatch(f"{fname}_{side}", f"{fname}_{side}", f"{fname}_seg_{side}",
                               "fingertip", "cylinder", 10, 3, t0=0.0, t1=1.0))
            # distal phalanx: the fingertip proper
            P.append(SkinPatch(f"{fname}_tip_{side}", f"{fname}_d_{side}",
                               f"{fname}_dseg_{side}", "fingertip", "cylinder",
                               10, 3, t0=0.0, t1=1.0))

        # --- thigh: 14 x 8
        P.append(SkinPatch(f"thigh_{side}", f"thigh_{side}", f"thigh_{side}",
                           "hairy_limb", "cylinder", 14, 8, t0=0.02, t1=0.98))
        # --- shin: 12 x 7
        P.append(SkinPatch(f"shin_{side}", f"shin_{side}", f"shin_{side}",
                           "hairy_limb", "cylinder", 12, 7, t0=0.03, t1=0.94))
        # --- kneecap
        P.append(SkinPatch(f"knee_{side}", f"shin_{side}", f"kneecap_{side}",
                           "proximal_limb", "sphere", 9, 4, phi0=0.3, phi1=np.pi - 0.3))
        # --- foot: sole + dorsum
        P.append(SkinPatch(f"sole_{side}", f"foot_{side}", f"footbody_{side}",
                           "sole", "plane", 4, 7, face="-z", surface="plantar"))
        P.append(SkinPatch(f"foot_dorsum_{side}", f"foot_{side}", f"footbody_{side}",
                           "hairy_limb", "plane", 4, 7, face="+z", surface="dorsal"))
        # --- toes
        P.append(SkinPatch(f"toes_{side}", f"toes_{side}", f"toebox_{side}",
                           "sole", "plane", 4, 3, face="-z", surface="plantar"))
        # --- hip ball
        P.append(SkinPatch(f"hip_{side}", f"hip_{side}", f"hipball_{side}",
                           "proximal_limb", "sphere", 9, 4, phi0=0.35, phi1=np.pi - 0.35))

    limb_patches("l")
    limb_patches("r")

    # ---------------- trunk ------------------------------------------------
    P.append(SkinPatch("chest", "chest", "chest", "trunk", "cylinder", 18, 6,
                       t0=0.05, t1=0.96))
    P.append(SkinPatch("abdomen", "abdomen", "abdomen", "trunk", "cylinder",
                       16, 5, t0=0.05, t1=0.95))
    P.append(SkinPatch("pelvis", "pelvis", "pelvis", "back", "cylinder",
                       14, 4, t0=0.15, t1=0.98))
    P.append(SkinPatch("neck", "neck", "neck", "hairy_limb", "cylinder",
                       10, 3, t0=0.05, t1=0.95))

    # ---------------- head & face -----------------------------------------
    P.append(SkinPatch("scalp", "head", "head", "face", "sphere", 16, 7,
                       phi0=0.18, phi1=np.pi - 0.18))
    P.append(SkinPatch("face", "head", "face", "face", "plane", 6, 4,
                       face="-y", surface="anterior"))
    P.append(SkinPatch("nose", "head", "nose", "face", "sphere", 7, 4,
                       phi0=0.4, phi1=np.pi - 0.4))
    P.append(SkinPatch("jaw", "jaw", "jaw_bone", "face", "plane", 5, 2,
                       face="-y", surface="anterior"))
    P.append(SkinPatch("tongue", "jaw", "tongue", "mucosa", "plane", 5, 4,
                       face="+z", surface="dorsal"))
    P.append(SkinPatch("lips", "jaw", "jaw_bone", "mucosa", "plane", 4, 2,
                       face="-y", surface="anterior"))

    # Finer quantisation of every sheet: the whole patch table is the 1x body, and
    # ``skin_density`` multiplies the sampling in both directions (so the taxel
    # count grows with its square).  Glabrous skin keeps its relative advantage.
    f = float(COMPLEXITY.skin_density)
    if abs(f - 1.0) > 1e-9:
        for p in P:
            p.n_u = max(2, int(round(p.n_u * f)))
            p.n_v = max(2, int(round(p.n_v * f)))
    return [p for p in P if p.enabled]


# --------------------------------------------------------------------------
# Taxonomy helpers
# --------------------------------------------------------------------------
# Coarse "dermatome-like" grouping used for body-map plots and for computing
# region-level aggregate afferent vectors.
REGION_GROUPS = {
    "head_face": ("scalp", "face", "nose", "jaw", "tongue", "lips"),
    "neck": ("neck",),
    "torso_front": ("chest", "abdomen"),
    "torso_back": ("chest", "abdomen", "pelvis"),
    "pelvis": ("pelvis",),
    "left_arm": ("deltoid_l", "upper_arm_l", "forearm_l", "palm_l",
                 "hand_dorsum_l", "thumb_l", "index_l", "fingers_l",
                 "thumb_tip_l", "index_tip_l", "fingers_tip_l"),
    "right_arm": ("deltoid_r", "upper_arm_r", "forearm_r", "palm_r",
                  "hand_dorsum_r", "thumb_r", "index_r", "fingers_r",
                  "thumb_tip_r", "index_tip_r", "fingers_tip_r"),
    "left_hand": ("palm_l", "hand_dorsum_l", "thumb_l", "index_l", "fingers_l",
                  "thumb_tip_l", "index_tip_l", "fingers_tip_l"),
    "right_hand": ("palm_r", "hand_dorsum_r", "thumb_r", "index_r", "fingers_r",
                   "thumb_tip_r", "index_tip_r", "fingers_tip_r"),
    "left_leg": ("hip_l", "thigh_l", "shin_l", "knee_l", "sole_l",
                 "foot_dorsum_l", "toes_l"),
    "right_leg": ("hip_r", "thigh_r", "shin_r", "knee_r", "sole_r",
                  "foot_dorsum_r", "toes_r"),
    "left_foot": ("sole_l", "foot_dorsum_l", "toes_l"),
    "right_foot": ("sole_r", "foot_dorsum_r", "toes_r"),
}

SURFACE_DESCRIPTOR = {
    "all": "circumferential",
    "palmar": "palmar (glabrous)",
    "dorsal": "dorsal (hairy)",
    "plantar": "plantar (glabrous)",
    "anterior": "anterior",
}


# --------------------------------------------------------------------------
# Geometry sampling
# --------------------------------------------------------------------------
def _geom_of(bone_name: str, geom_name: str) -> Geom:
    bone = BONE_BY_NAME[bone_name]
    for g in bone.geoms:
        if (g.name or bone_name) == geom_name:
            return g
    # fall back to the first capsule / named geometry
    for g in bone.geoms:
        if g.kind == "capsule":
            return g
    if not bone.geoms:
        raise KeyError(f"bone {bone_name} has no geometry")
    return bone.geoms[0]


def _sample_cylinder(g: Geom, n_u: int, n_v: int, t0: float, t1: float,
                     theta0: float, theta1: float, closed: bool) -> list[tuple]:
    u_ax, v_ax, ax = g.frame()
    a = np.asarray(g.a, float)
    L = g.length()
    r = g.size[0]
    n_theta = n_u if closed else n_u
    thetas = (np.linspace(theta0, theta1, n_theta, endpoint=not closed))
    ts = np.linspace(t0, t1, n_v)
    out = []
    for j, t in enumerate(ts):
        centre = a + t * L * ax
        for i, th in enumerate(thetas):
            nrm = np.cos(th) * u_ax + np.sin(th) * v_ax
            out.append((centre + r * nrm, nrm, i / max(n_theta - 1, 1), j / max(n_v - 1, 1),
                        float(th)))
    return out


def _sample_sphere(g: Geom, n_u: int, n_v: int, phi0: float, phi1: float) -> list[tuple]:
    c = np.asarray(g.a, float)
    r = g.size[0]
    thetas = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    phis = np.linspace(phi0, phi1, n_v)
    out = []
    for j, phi in enumerate(phis):
        for i, th in enumerate(thetas):
            nrm = np.array([np.sin(phi) * np.cos(th),
                            np.sin(phi) * np.sin(th),
                            np.cos(phi)])
            out.append((c + r * nrm, nrm, i / n_u, j / max(n_v - 1, 1), float(th)))
    return out


_FACE_NORMAL = {
    "+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
    "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]),
    "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0]),
}


def _sample_plane(g: Geom, n_u: int, n_v: int, face: str) -> list[tuple]:
    c = np.asarray(g.a, float)
    hx, hy, hz = (list(g.size) + [0, 0, 0])[:3]
    nrm = _FACE_NORMAL[face]
    # pick two in-plane axes
    if face in ("+x", "-x"):
        e1, e2, h1, h2 = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), hy, hz
    elif face in ("+y", "-y"):
        e1, e2, h1, h2 = np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), hx, hz
    else:
        e1, e2, h1, h2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), hx, hy
    half = {"+x": hx, "-x": hx, "+y": hy, "-y": hy, "+z": hz, "-z": hz}[face]
    centre = c + nrm * half
    us = np.linspace(-0.86, 0.86, n_u)
    vs = np.linspace(-0.86, 0.86, n_v)
    out = []
    for j, sv in enumerate(vs):
        for i, su in enumerate(us):
            p = centre + e1 * (su * h1) + e2 * (sv * h2)
            out.append((p, nrm, i / max(n_u - 1, 1), j / max(n_v - 1, 1), 0.0))
    return out


# --------------------------------------------------------------------------
# Body-map layout: arrange regions into a schematic 2-D map so the taxel
# population can be drawn as a sensory homunculus.
# --------------------------------------------------------------------------
BODY_MAP_ANCHORS: dict[str, tuple[float, float, float, float]] = {
    # region: (x0, y0, width, height) in an arbitrary 0..1 layout space
    "scalp": (0.40, 0.86, 0.20, 0.14), "face": (0.40, 0.80, 0.20, 0.07),
    "nose": (0.47, 0.785, 0.06, 0.03), "jaw": (0.43, 0.775, 0.14, 0.03),
    "tongue": (0.45, 0.760, 0.10, 0.02), "lips": (0.44, 0.768, 0.12, 0.015),
    "neck": (0.42, 0.745, 0.16, 0.03),
    "chest": (0.36, 0.60, 0.28, 0.145), "abdomen": (0.38, 0.50, 0.24, 0.10),
    "pelvis": (0.39, 0.42, 0.22, 0.08),
    "deltoid_l": (0.26, 0.68, 0.09, 0.07), "deltoid_r": (0.65, 0.68, 0.09, 0.07),
    "upper_arm_l": (0.25, 0.55, 0.10, 0.13), "upper_arm_r": (0.65, 0.55, 0.10, 0.13),
    "forearm_l": (0.26, 0.40, 0.09, 0.14), "forearm_r": (0.65, 0.40, 0.09, 0.14),
    "palm_l": (0.26, 0.33, 0.09, 0.06), "palm_r": (0.65, 0.33, 0.09, 0.06),
    "hand_dorsum_l": (0.26, 0.33, 0.09, 0.06), "hand_dorsum_r": (0.65, 0.33, 0.09, 0.06),
    "thumb_l": (0.22, 0.30, 0.04, 0.05), "thumb_r": (0.74, 0.30, 0.04, 0.05),
    "index_l": (0.27, 0.29, 0.04, 0.06), "index_r": (0.69, 0.29, 0.04, 0.06),
    "fingers_l": (0.29, 0.27, 0.06, 0.06), "fingers_r": (0.65, 0.27, 0.06, 0.06),
    "thumb_tip_l": (0.22, 0.26, 0.04, 0.04), "thumb_tip_r": (0.74, 0.26, 0.04, 0.04),
    "index_tip_l": (0.27, 0.23, 0.04, 0.04), "index_tip_r": (0.69, 0.23, 0.04, 0.04),
    "fingers_tip_l": (0.29, 0.21, 0.06, 0.04), "fingers_tip_r": (0.65, 0.21, 0.06, 0.04),
    "hip_l": (0.40, 0.40, 0.06, 0.05), "hip_r": (0.54, 0.40, 0.06, 0.05),
    "thigh_l": (0.39, 0.28, 0.10, 0.13), "thigh_r": (0.51, 0.28, 0.10, 0.13),
    "knee_l": (0.40, 0.245, 0.08, 0.035), "knee_r": (0.52, 0.245, 0.08, 0.035),
    "shin_l": (0.40, 0.12, 0.08, 0.12), "shin_r": (0.52, 0.12, 0.08, 0.12),
    "foot_dorsum_l": (0.38, 0.04, 0.10, 0.07), "foot_dorsum_r": (0.52, 0.04, 0.10, 0.07),
    "sole_l": (0.38, 0.04, 0.10, 0.07), "sole_r": (0.52, 0.04, 0.10, 0.07),
    "toes_l": (0.36, 0.005, 0.11, 0.04), "toes_r": (0.52, 0.005, 0.11, 0.04),
}


def build_taxels(patches: list[SkinPatch] | None = None) -> list[Taxel]:
    """Instantiate every taxel in the patch table."""
    if patches is None:
        patches = default_patches()
    taxels: list[Taxel] = []
    idx = 0
    for p in patches:
        g = _geom_of(p.bone, p.geom)
        geom_tag = p.geom
        if p.kind == "cylinder":
            samples = _sample_cylinder(g, p.n_u, p.n_v, p.t0, p.t1,
                                       p.theta0, p.theta1, p.closed_theta)
        elif p.kind == "sphere":
            samples = _sample_sphere(g, p.n_u, p.n_v, p.phi0, p.phi1)
        elif p.kind == "plane":
            samples = _sample_plane(g, p.n_u, p.n_v, p.face)
        else:
            raise ValueError(f"unknown patch kind {p.kind}")

        anchor = BODY_MAP_ANCHORS.get(p.region, (0.45, 0.45, 0.1, 0.1))
        dens = PROFILES[p.profile]
        for k, (pos, nrm, uu, vv, theta) in enumerate(samples):
            # place the taxel just proud of the collision surface
            pos = np.asarray(pos, float) + p.proud * np.asarray(nrm, float)
            if p.kind == "cylinder" and p.closed_theta:
                surface = _cylinder_surface_name(theta)
            else:
                surface = p.surface
            if p.kind == "sphere":
                surface = _sphere_surface_name(np.asarray(nrm, float))
            taxels.append(Taxel(
                index=idx,
                name=f"tx_{p.region}_{k:03d}",
                region=p.region,
                bone=p.bone,
                geom=geom_tag,
                profile=p.profile,
                pos=pos,
                normal=np.asarray(nrm, float),
                surface=surface,
                u=anchor[0] + uu * anchor[2],
                v=anchor[1] + vv * anchor[3],
                area_cm2=_taxel_area(p, g),
                densities=dict(dens),
            ))
            idx += 1
    return taxels


def _taxel_area(p: SkinPatch, g: Geom) -> float:
    """Approximate per-taxel surface area in cm^2 (for density realism)."""
    if p.kind == "cylinder":
        r = g.size[0]
        L = g.length() * (p.t1 - p.t0)
        circ = 2.0 * np.pi * r * ((p.theta1 - p.theta0) / (2.0 * np.pi))
        return float(circ * L / max(p.n_u * p.n_v, 1)) * 1e4
    if p.kind == "sphere":
        r = g.size[0]
        area = 2.0 * np.pi * r * r * (np.cos(p.phi0) - np.cos(p.phi1))
        return float(area / max(p.n_u * p.n_v, 1)) * 1e4
    hx, hy, hz = (list(g.size) + [0, 0, 0])[:3]
    dims = {"+x": (hy, hz), "-x": (hy, hz), "+y": (hx, hz), "-y": (hx, hz),
            "+z": (hx, hy), "-z": (hx, hy)}[p.face]
    return float(4.0 * dims[0] * dims[1] * 0.86 * 0.86 / max(p.n_u * p.n_v, 1)) * 1e4


def _cylinder_surface_name(theta: float) -> str:
    """Map a circumferential angle to an anatomical descriptor.

    For limbs pointing along -z the sampler yields u=+y (dorsal) at theta=0 and
    v=+x (lateral) at theta=pi/2.
    """
    t = float(theta) % (2.0 * np.pi)
    if t < np.pi / 4 or t >= 7 * np.pi / 4:
        return "dorsal"
    if t < 3 * np.pi / 4:
        return "lateral"
    if t < 5 * np.pi / 4:
        return "ventral"
    return "medial"


def _sphere_surface_name(n: np.ndarray) -> str:
    z = float(n[2])
    if z > 0.6:
        return "superior"
    if z < -0.6:
        return "inferior"
    if float(n[1]) < -0.4:
        return "anterior"
    if float(n[1]) > 0.4:
        return "posterior"
    return "lateral" if float(n[0]) > 0 else "medial"


# --------------------------------------------------------------------------
# Convenience containers built once at import
# --------------------------------------------------------------------------
PATCHES = default_patches()
TAXELS = build_taxels(PATCHES)
N_TAXELS = len(TAXELS)

_BONE_INDEX = {b.name: i for i, b in enumerate(BONES)}
TAXEL_BONE = np.array([_BONE_INDEX[t.bone] for t in TAXELS], dtype=int)
TAXEL_POS = np.stack([t.pos for t in TAXELS]) if TAXELS else np.zeros((0, 3))
TAXEL_NORMAL = np.stack([t.normal for t in TAXELS]) if TAXELS else np.zeros((0, 3))
TAXEL_REGION = [t.region for t in TAXELS]

# per-body index lists for fast contact assignment
TAXELS_BY_BONE: dict[str, list[int]] = {}
for _t in TAXELS:
    TAXELS_BY_BONE.setdefault(_t.bone, []).append(_t.index)

# per-receptor gain matrices, shape (n_taxels,)
RECEPTOR_KEYS = ("sa1", "sa2", "fa1", "fa2", "ct", "noci_mech", "noci_heat",
                 "noci_cold", "warm", "cold", "itch", "hair", "thermal_mass",
                 "noci_threshold_kpa")
GAINS = {k: np.array([t.gains(k) for t in TAXELS], float) for k in RECEPTOR_KEYS}

# patch-region label per taxel (e.g. "sole_l", "chest")
TAXEL_PATCH = []
for _p in PATCHES:
    g = _geom_of(_p.bone, _p.geom)
    nsamp = {"cylinder": _p.n_u * _p.n_v, "sphere": _p.n_u * _p.n_v,
             "plane": _p.n_u * _p.n_v}[_p.kind]
    TAXEL_PATCH.extend([_p.region] * nsamp)
TAXEL_PATCH = np.array(TAXEL_PATCH)

# integer patch index per taxel, for broadcasting patch-level (organ-level)
# quantities such as local blood flow onto the taxels
PATCH_NAMES = sorted(set(TAXEL_PATCH.tolist()))
_PATCH_ID = {n: i for i, n in enumerate(PATCH_NAMES)}
TAXEL_PATCH_IDX = np.array([_PATCH_ID[n] for n in TAXEL_PATCH.tolist()], dtype=int)
N_PATCHES = len(PATCH_NAMES)


def summary() -> dict:
    """Human-readable description of the skin."""
    from collections import Counter
    regions = Counter(TAXEL_PATCH.tolist())
    profiles = Counter(t.profile for t in TAXELS)
    surfaces = Counter(t.surface for t in TAXELS)
    return {
        "n_taxels": N_TAXELS,
        "n_patches": len(PATCHES),
        "n_body_regions": len(regions),
        "taxels_per_region": dict(sorted(regions.items(), key=lambda kv: -kv[1])),
        "receptor_profiles": dict(profiles),
        "surfaces": dict(surfaces),
        "total_skin_area_cm2": float(sum(t.area_cm2 for t in TAXELS)),
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(summary(), indent=2))
