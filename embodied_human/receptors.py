"""
Peripheral transduction: physics -> receptor potentials.

This is where the simulation stops being a robot and starts being a body.

Tactile bank
------------
Every taxel reports a **26-dimensional** receptor vector, so the full
somatosensory frame is ``n_taxels x 26`` numbers (~48 000 values):

=====  ==================================================  ==========
 idx   channel                                             unit
=====  ==================================================  ==========
  0    normal force                                        N
  1    tangential force (surface u)                        N
  2    tangential force (surface v)                        N
  3    contact pressure                                    kPa
  4    shear magnitude                                     N
  5    indentation depth                                   mm
  6    contact area                                        cm^2
  7    normal force rate (dF/dt)                           N/s
  8    vibration energy (broadband, 5-400 Hz)              a.u.
  9    FA-I band energy (5-50 Hz, flutter)                 a.u.
 10    FA-II band energy (50-400 Hz, tool/vibration)       a.u.
 11    SA ripple energy (<5 Hz, sustained pressure)        a.u.
 12    SA-I (Merkel) activation                            a.u.
 13    SA-II (Ruffini) activation                          a.u.
 14    FA-I (Meissner) activation                          a.u.
 15    FA-II (Pacinian) activation                         a.u.
 16    C-tactile (affective touch) activation              a.u.
 17    mechanonociceptor activation                        a.u.
 18    heat nociceptor (TRPV1) activation                  a.u.
 19    cold nociceptor (TRPM8/TRPA1) activation            a.u.
 20    itch (MrgprA3+) activation                           a.u.
 21    warmth receptor activation                          a.u.
 22    cold receptor activation                            a.u.
 23    local skin temperature                              degC
 24    slip probability                                    a.u.
 25    friction utilisation (|F_t| / mu F_n)               a.u.
=====  ==================================================  ==========

Other modalities
----------------
* **Proprioception** - joint angles/velocities, muscle spindle Ia (dynamic)
  and II (static) afferents, Golgi tendon organs, joint-limit afferents,
  efference copy, Cartesian limb positions.
* **Vestibular** - 3 semicircular canals (with cupula adaptation) and 2
  otolith organs (utricle/saccule), plus magnetoception.
* **Vision** - a 2-channel retina (luminance + motion energy) rendered from
  the egocentric camera, with foveal/peripheral weighting, saccades, blinks
  and a pupil that responds to light and arousal.
* **Audition** - a 24-band cochlear filterbank driven by bone-conducted
  contact transients and self-generated motor noise.
* **Chemoreception** - 24 olfactory channels driven by proximity to scene
  objects with odour signatures, and 5 gustatory channels when the tongue
  touches something.
* **Nociception** - aggregated from the tactile bank plus visceral inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from . import skin
from .complexity import C as COMPLEXITY
from .config import SimConfig
from .senses_ext import (Cochlea, GustatoryBank, OlfactoryBank, RetinaBank,
                         SpindleBank, VestibularBank)
from .state import BodyState

# Tactile channel layout -----------------------------------------------------
TACTILE_CHANNELS = (
    "normal_force", "shear_u", "shear_v", "pressure", "shear_mag",
    "indentation", "contact_area", "force_rate",
    "vib_broadband", "vib_fa1", "vib_fa2", "vib_sa",
    "sa1", "sa2", "fa1", "fa2", "ct",
    "noci_mech", "noci_heat", "noci_cold", "itch", "warm", "cold",
    "temperature", "slip", "friction_util",
)
# Extended somatosensory bank: receptor classes and tissue states the 26-channel
# bank did not have.  Appended, so the original indices are unchanged.
EXTENDED_TACTILE_CHANNELS = (
    "hair_deflection",      # lanceolate endings: air movement, light touch
    "stretch_u", "stretch_v", "stretch_energy",   # Ruffini-like skin stretch
    "edge_gradient",        # Merkel edge response (contrast with the patch)
    "tickle",               # knismesis: light moving touch on hairy skin
    "wetness",              # skin moisture (sweat, contact)
    "blood_flow_local",     # dermal perfusion of this patch
    "piloerection",         # arrector pili state
    "sensitisation",        # peripheral sensitisation (hyperalgesia)
    "local_inflammation",
    "pruritogen",
    "receptor_fatigue",
    "noci_a_delta",         # fast pricking pain
    "noci_c_poly",          # slow polymodal (burning/aching) pain
    "irritant",             # TRPA1: chemical / drying irritation
    "ischemia",             # pressure-induced local ischaemia (the urge to shift)
)
N_BASE_TACTILE = len(TACTILE_CHANNELS)
if COMPLEXITY.extended_tactile:
    TACTILE_CHANNELS = TACTILE_CHANNELS + EXTENDED_TACTILE_CHANNELS
N_TACTILE_CH = len(TACTILE_CHANNELS)
CH = {n: i for i, n in enumerate(TACTILE_CHANNELS)}

# Proprioceptive channel layout (per joint) ---------------------------------
PROPRIO_CHANNELS = (
    "q", "qd", "qdd", "spindle_ia", "spindle_ii", "golgi_ib",
    "efference", "tau_measured", "tau_error", "limit_proximity",
    "muscle_length", "muscle_velocity", "effort",
)
N_PROPRIO_CH = len(PROPRIO_CHANNELS)

# Vestibular
VESTIBULAR_CHANNELS = (
    "canal_x", "canal_y", "canal_z",           # adapted angular velocity
    "canal_raw_x", "canal_raw_y", "canal_raw_z",
    "otolith_x", "otolith_y", "otolith_z",     # linear accel incl. gravity
    "gravity_x", "gravity_y", "gravity_z",     # gravity direction in head frame
    "tilt", "tilt_rate", "yaw",                # derived orientation
    "magneto_x", "magneto_y", "magneto_z",
)

# Visual (global, not per-pixel)
VISUAL_CHANNELS = (
    "luminance_mean", "luminance_sd", "contrast", "motion_energy",
    "foveal_luminance", "foveal_contrast", "peripheral_motion",
    "pupil_diameter", "blink", "saccade_active", "gaze_x", "gaze_y",
    "red", "green", "blue", "edge_density", "depth_min",
)

# Auditory
AUDITORY_CHANNELS = ("loudness", "pitch", "onset", "spectral_centroid",
                     "roughness", "self_generated")

# Olfactory / gustatory
CHEMO_CHANNELS = ("odor_intensity", "odor_novelty", "odor_valence_proxy",
                  "airflow")
GUSTATORY_CHANNELS = ("sweet", "salty", "sour", "bitter", "umami")


@dataclass
class ReceptorFrame:
    """Raw receptor activity for one instant (before nerve conduction)."""
    t: float = 0.0

    # Somatosensory
    tactile: np.ndarray = field(default_factory=lambda: np.zeros((0, N_TACTILE_CH)))
    tactile_by_region: dict = field(default_factory=dict)   # region -> (n, CH)
    skin_temperature: np.ndarray = field(default_factory=lambda: np.zeros(0))
    contact_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    # Proprioceptive
    proprio: np.ndarray = field(default_factory=lambda: np.zeros((0, N_PROPRIO_CH)))
    proprio_by_joint: dict = field(default_factory=dict)

    # Other senses
    vestibular: np.ndarray = field(default_factory=lambda: np.zeros(len(VESTIBULAR_CHANNELS)))
    visual: np.ndarray = field(default_factory=lambda: np.zeros(len(VISUAL_CHANNELS)))
    auditory: np.ndarray = field(default_factory=lambda: np.zeros(len(AUDITORY_CHANNELS)))
    olfactory: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gustatory: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chemo_summary: np.ndarray = field(default_factory=lambda: np.zeros(len(CHEMO_CHANNELS)))

    # Nociceptive summary (whole body)
    pain_mech: float = 0.0
    pain_heat: float = 0.0
    pain_cold: float = 0.0
    pain_total: float = 0.0
    itch_total: float = 0.0
    affective_touch: float = 0.0
    touch_intensity: float = 0.0
    contact_count: int = 0

    # Cell populations (spindles, hair cells, receptor types, retina) -- name ->
    # flat array of firing rates / activations.  See senses_ext.py.
    ext: dict = field(default_factory=dict)

    def n_scalars(self) -> int:
        """How many sensory numbers this frame carries."""
        n = int(self.tactile.size) + int(self.proprio.size) + len(self.vestibular) \
            + len(self.visual) + len(self.auditory) + len(self.olfactory) \
            + len(self.gustatory) + len(self.chemo_summary)
        n += sum(int(v.size) for v in self.ext.values())
        return n

    def summary_vector(self) -> np.ndarray:
        """A compact, fixed-length feature vector for learning."""
        return np.concatenate([
            self.proprio[:, :8].ravel(),
            self.tactile_by_region.get("_summary", np.zeros(0)),
            self.vestibular,
            self.visual,
            self.auditory,
            self.chemo_summary,
            np.array([self.pain_mech, self.pain_heat, self.pain_cold,
                      self.pain_total, self.itch_total,
                      self.affective_touch, self.touch_intensity]),
        ])


# ==========================================================================
# Tactile
# ==========================================================================
class TactileSystem:
    """Reconstructs a dense tactile field from MuJoCo contact physics.

    Approach
    --------
    MuJoCo's native ``touch`` sensor is gated by ``site_size`` and sums whole
    contact forces, which blurs the field.  Here every contact is instead
    splatted onto the taxels of the two contacting bodies with a Gaussian
    receptive field, which gives a smooth, gap-free, spatially honest image --
    and it is what makes the fingertip feel different from the back.
    """

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.meta = meta
        self.n_taxels = skin.N_TAXELS
        self.ch = CH
        self.radius = float(cfg.tactile.contact_radius)
        self._sigma2 = self.radius ** 2

        # taxel geometry in body-local frame
        self.local_pos = skin.TAXEL_POS
        self.local_normal = skin.TAXEL_NORMAL
        self.gains = skin.GAINS
        self.patch = skin.TAXEL_PATCH

        # per-body taxel index arrays for fast splatting
        self.by_body: dict[int, np.ndarray] = {}
        for bone_name, idxs in skin.TAXELS_BY_BONE.items():
            bid = meta.body_ids.get(bone_name)
            if bid is not None:
                self.by_body[bid] = np.asarray(idxs, dtype=int)

        # derived per-taxel constants
        self.area = np.array([max(t.area_cm2, 1e-4) for t in skin.TAXELS])
        self.thermal_mass = self.gains["thermal_mass"] * 0.6 + 0.2

        # state (ODEs and history)
        self.temperature = np.full(self.n_taxels, cfg.tactile.__dict__.get("_t0", 33.0))
        self.temperature[:] = 33.0
        self.prev_normal = np.zeros(self.n_taxels)
        self.vib_state = np.zeros((self.n_taxels, 4))
        self.adapt_sa1 = np.zeros(self.n_taxels)
        self.adapt_sa2 = np.zeros(self.n_taxels)
        self.adapt_fa1 = np.zeros(self.n_taxels)
        self.adapt_fa2 = np.zeros(self.n_taxels)
        self.adapt_ct = np.zeros(self.n_taxels)
        self.adapt_noci = np.zeros(self.n_taxels)
        self.itch_state = np.zeros(self.n_taxels)
        self.histamine = np.zeros(self.n_taxels)
        self._brush_velocity = np.zeros(self.n_taxels)
        self._prev_pos = self.local_pos.copy()

        # ---- extended bank: tissue states and the organ-level inputs they
        # ---- take from the internal world (see inner_world.py) --------------
        self.extended = bool(COMPLEXITY.extended_tactile)
        self.patch_idx = skin.TAXEL_PATCH_IDX
        self.n_patches = skin.N_PATCHES
        self._patch_count = np.maximum(np.bincount(self.patch_idx,
                                                   minlength=self.n_patches), 1)
        n = self.n_taxels
        self.stretch_u = np.zeros(n)
        self.stretch_v = np.zeros(n)
        self.wet = np.zeros(n)
        self.pilo = np.zeros(n)
        self.sens = np.zeros(n)            # peripheral sensitisation
        self.rfatigue = np.zeros(n)
        self.ischemia = np.zeros(n)
        self.pruri = np.zeros(n)
        # organ-level inputs, one value per skin patch; the internal world
        # overwrites these, the defaults describe a healthy resting body
        self.inner = {
            "blood_flow": np.ones(self.n_patches),
            "inflammation": np.zeros(self.n_patches),
            "sweat": np.zeros(self.n_patches),
            "irritant": np.zeros(self.n_patches),
            "damage": np.zeros(self.n_patches),
            "pruritogen": np.zeros(self.n_patches),
            "humidity": 0.45,
            "airflow": 0.0,
        }

    def set_inner(self, **arrays) -> None:
        """Let the internal world drive patch-level tissue state."""
        for k, v in arrays.items():
            self.inner[k] = v

    def _patch_mean(self, x: np.ndarray) -> np.ndarray:
        return (np.bincount(self.patch_idx, weights=x, minlength=self.n_patches)
                / self._patch_count)

    # ------------------------------------------------------------------
    def sense(self, model, data, meta, state: BodyState,
              blood_flow: float = 1.0,
              ambient: float = 22.0) -> tuple[np.ndarray, dict]:
        """Return (tactile[taxels, CH], region_aggregates)."""
        cfg = self.cfg
        T = cfg.tactile
        A = cfg.afferent
        dt = 1.0 / cfg.rates.receptor

        out = np.zeros((self.n_taxels, N_TACTILE_CH))
        normal = out[:, CH["normal_force"]]
        su = out[:, CH["shear_u"]]
        sv = out[:, CH["shear_v"]]

        # ---- world geometry of every taxel (free from the engine) --------
        xpos = data.site_xpos[meta.taxel_site_ids]
        xmat = data.site_xmat[meta.taxel_site_ids].reshape(-1, 3, 3)
        world_normal = np.einsum("nij,nj->ni", xmat, self.local_normal)

        contact_any = np.zeros(self.n_taxels, bool)

        # ---- splat each contact onto the taxels of its two bodies --------
        for c in state.contacts:
            f = c.force
            # contact-frame force: normal along local x of the contact frame
            fn = float(f[0])
            ft1, ft2 = float(f[1]), float(f[2])
            for bid, sign in ((c.body1, 1.0), (c.body2, -1.0)):
                idxs = self.by_body.get(bid)
                if idxs is None or idxs.size == 0:
                    continue
                d = xpos[idxs] - c.pos
                dist2 = np.einsum("ij,ij->i", d, d)
                w = np.exp(-dist2 / (2.0 * self._sigma2))
                # only taxels facing the contact point respond
                facing = np.einsum("ij,j->i", world_normal[idxs], c.normal * sign) > -0.1
                w = w * facing
                if not np.any(w > 1e-4):
                    continue
                contact_any[idxs] |= w > 0.15
                # conductive heat exchange with the opposite surface
                other_t = c.temp_for(bid)
                near = w > 0.25
                if near.any():
                    rate = min(dt / 1.2, 0.5)
                    self.temperature[idxs[near]] += rate * (
                        other_t - self.temperature[idxs[near]]) * 0.30
                # distribute the contact force; w is normalised per contact
                wn = w / max(w.sum(), 1e-9)
                normal[idxs] += sign * fn * wn
                # Tangential force resolved in each taxel's own surface frame.
                # Written out explicitly rather than with np.cross: this inner
                # loop runs once per contact per body on every transduction
                # tick, and np.cross dominated the tactile cost.
                tn = world_normal[idxs]
                tx, ty, tz = tn[:, 0], tn[:, 1], tn[:, 2]
                t1x, t1y = ty, -tx                       # t1 = t x z_hat
                n1 = np.sqrt(t1x * t1x + t1y * t1y) + 1e-9
                t1x, t1y = t1x / n1, t1y / n1
                t2x = -tz * t1y                          # t2 = t x t1
                t2y = tz * t1x
                su[idxs] += (ft1 * t1x + ft2 * t2x) * wn
                sv[idxs] += (ft1 * t1y + ft2 * t2y) * wn

        normal = np.abs(normal)
        area = np.maximum(self.area, 1e-4)

        # ---- pressure, shear, indentation, area -------------------------
        pressure_pa = normal / (area * 1e-4)                 # N / m^2
        shear_mag = np.hypot(su, sv)
        # a soft-body contact model: indentation ~ sqrt(pressure)
        indentation_mm = 3.2 * np.sqrt(np.maximum(pressure_pa, 0.0) / 1e4)
        contact_area = area * fclip(indentation_mm / 4.0, 0.0, 1.0) ** 0.5
        force_rate = (normal - self.prev_normal) / max(dt, 1e-6)

        # ---- vibration: leaky band-pass on the force derivative ---------
        drive = fclip(force_rate, -1e4, 1e4)
        a_fa1 = dt / (dt + A.tau_fa1)
        a_fa2 = dt / (dt + A.tau_fa2)
        a_sa = dt / (dt + A.tau_sa1)
        self.vib_state[:, 0] += a_fa1 * (drive - self.vib_state[:, 0])       # 5-50 Hz
        self.vib_state[:, 1] += a_fa2 * (drive - self.vib_state[:, 1])       # 50-400 Hz
        self.vib_state[:, 2] += a_sa * (normal - self.vib_state[:, 2])       # sustained
        self.vib_state[:, 3] = np.abs(drive)                                  # broadband
        band_fa1 = np.abs(self.vib_state[:, 0])
        band_fa2 = np.abs(self.vib_state[:, 1])
        band_sa = np.abs(self.vib_state[:, 2])
        vib_broad = self.vib_state[:, 3]

        # ---- skin temperature -------------------------------------------
        # perfusion brings core heat; contact conducts towards the object
        target = 33.0 * blood_flow + ambient * (1.0 - blood_flow) * 0.35
        tau_therm = 22.0 / np.maximum(self.thermal_mass, 1e-3)
        self.temperature += (dt / tau_therm) * (target - self.temperature)
        # conductive contact cooling/warming is applied by the caller via
        # `apply_contact_temperature` (needs the object temperatures)

        # ---- receptor activations ---------------------------------------
        g = self.gains
        # slowly adapting: sustained force, with adaptation
        drive_sa1 = fclip(normal, 0, None) * 0.06 + band_sa * 0.004
        drive_sa2 = shear_mag * 0.05 + fclip(normal, 0, None) * 0.02
        drive_fa1 = band_fa1 * 0.010 + shear_mag * 0.02
        drive_fa2 = band_fa2 * 0.016
        # C-tactile: responds to slow stroking within a narrow velocity band
        self._brush_velocity = 0.7 * self._brush_velocity + 0.3 * np.abs(force_rate) * 0.0
        vel_term = np.exp(-0.5 * ((self._brush_velocity - T.ct_optimal_velocity)
                                  / T.ct_velocity_width) ** 2)
        drive_ct = 0.25 * band_sa + 0.10 * normal * vel_term

        self.adapt_sa1 += (dt / A.tau_sa1) * (drive_sa1 - self.adapt_sa1)
        self.adapt_sa2 += (dt / A.tau_sa2) * (drive_sa2 - self.adapt_sa2)
        self.adapt_fa1 += (dt / A.tau_fa1) * (drive_fa1 - self.adapt_fa1)
        self.adapt_fa2 += (dt / A.tau_fa2) * (drive_fa2 - self.adapt_fa2)
        self.adapt_ct += (dt / A.tau_ct) * (drive_ct - self.adapt_ct)

        sa1 = g["sa1"] * T.sa1_gain * np.tanh(self.adapt_sa1 * 0.9)
        sa2 = g["sa2"] * T.sa2_gain * np.tanh(self.adapt_sa2 * 0.9)
        fa1 = g["fa1"] * T.fa1_gain * np.tanh(self.adapt_fa1 * 1.4)
        fa2 = g["fa2"] * T.fa2_gain * np.tanh(self.adapt_fa2 * 2.2)
        ct = g["ct"] * T.ct_gain * np.tanh(self.adapt_ct * 3.0)
        if self.extended:
            # receptors that have been driven hard for a while respond less
            fat = 1.0 - 0.30 * self.rfatigue
            sa1 = sa1 * fat
            fa1 = fa1 * fat

        # ---- nociceptors -------------------------------------------------
        # Mechanical nociceptors are thresholded on *pressure*, with a
        # site-specific threshold: a sole taxel carries 50 N every time the
        # agent stands up and must not report that as pain, whereas the same
        # pressure on a fingertip is aversive.
        pressure_kpa = pressure_pa / 1000.0
        threshold_kpa = g["noci_threshold_kpa"]
        if self.extended:
            # sensitised tissue hurts at a lower pressure (hyperalgesia)
            threshold_kpa = threshold_kpa * (1.0 - 0.45 * self.sens)
        mech_over = fclip(pressure_kpa - threshold_kpa, 0.0, None)
        noci_mech_drive = mech_over / 400.0
        heat_over = fclip(self.temperature - T.noci_heat_threshold, 0.0, None)
        noci_heat_drive = heat_over / 8.0 + fclip(
            pressure_kpa / np.maximum(threshold_kpa, 1.0) - 1.6, 0, None) * 0.5
        cold_over = fclip(T.noci_cold_threshold - self.temperature, 0.0, None)
        noci_cold_drive = cold_over / 10.0
        self.adapt_noci += (dt / A.tau_noci) * (
            np.maximum.reduce([noci_mech_drive, noci_heat_drive, noci_cold_drive])
            - self.adapt_noci)
        noci = g["noci_mech"] * np.tanh(self.adapt_noci * 1.5)
        noci_heat = g["noci_heat"] * np.tanh(self.adapt_noci * 1.5 * (heat_over > 0))
        noci_cold = g["noci_cold"] * np.tanh(self.adapt_noci * 1.5 * (cold_over > 0))

        # ---- itch (histamine-driven, provoked by light repeated touch) ---
        itch_drive = 0.05 * band_sa + 0.02 * normal
        self.histamine += (dt / 60.0) * (itch_drive - self.histamine)
        self.itch_state += (dt / 8.0) * (np.tanh(self.histamine * 4.0) - self.itch_state)
        itch = g["itch"] * self.itch_state

        # ---- thermoreceptors --------------------------------------------
        warm_drive = fclip(self.temperature - 34.0, 0.0, None) / 6.0
        cold_drive = fclip(34.0 - self.temperature, 0.0, None) / 10.0
        warm = g["warm"] * np.tanh(warm_drive)
        cold = g["cold"] * np.tanh(cold_drive)

        # ---- slip / friction utilisation --------------------------------
        mu = T.slip_friction_ratio
        util = np.divide(shear_mag, np.maximum(mu * normal, 1e-6))
        slip = fclip((util - 0.8) / 0.4, 0.0, 1.0) * (normal > 0.2)
        hair = g["hair"] * np.tanh(vib_broad * 0.08)

        # ---- write the bank ---------------------------------------------
        out[:, CH["normal_force"]] = normal
        out[:, CH["shear_u"]] = su
        out[:, CH["shear_v"]] = sv
        out[:, CH["pressure"]] = pressure_pa / 1000.0
        out[:, CH["shear_mag"]] = shear_mag
        out[:, CH["indentation"]] = indentation_mm
        out[:, CH["contact_area"]] = contact_area
        out[:, CH["force_rate"]] = force_rate
        out[:, CH["vib_broadband"]] = vib_broad
        out[:, CH["vib_fa1"]] = band_fa1
        out[:, CH["vib_fa2"]] = band_fa2
        out[:, CH["vib_sa"]] = band_sa
        out[:, CH["sa1"]] = sa1
        out[:, CH["sa2"]] = sa2
        out[:, CH["fa1"]] = fa1
        out[:, CH["fa2"]] = fa2
        out[:, CH["ct"]] = ct
        out[:, CH["noci_mech"]] = noci
        out[:, CH["noci_heat"]] = noci_heat
        out[:, CH["noci_cold"]] = noci_cold
        out[:, CH["itch"]] = itch
        out[:, CH["warm"]] = warm
        out[:, CH["cold"]] = cold
        out[:, CH["temperature"]] = self.temperature
        out[:, CH["slip"]] = slip
        out[:, CH["friction_util"]] = util * (normal > 0.05)

        if self.extended:
            self._extended_channels(out, dt, normal, su, sv, shear_mag, force_rate,
                                    pressure_kpa, threshold_kpa, vib_broad, hair,
                                    noci_mech_drive, world_normal, state,
                                    blood_flow, cold_drive, band_sa)

        self.prev_normal = normal
        return out, self._aggregate(out, contact_any)

    # ------------------------------------------------------------------
    def _extended_channels(self, out, dt, normal, su, sv, shear_mag, force_rate,
                           pressure_kpa, threshold_kpa, vib_broad, hair,
                           noci_mech_drive, world_normal, state, blood_flow,
                           cold_drive, band_sa) -> None:
        """Tissue states and the receptor classes the 26-channel bank lacks."""
        inner = self.inner
        pidx = self.patch_idx
        g = self.gains

        # --- air movement on hairy skin: walking makes a wind -----------------
        vel = np.asarray(getattr(state, "com_vel", np.zeros(3)), float)
        speed = float(np.linalg.norm(vel[:2]))
        wind = float(inner.get("airflow", 0.0)) + 0.9 * speed
        if speed > 1e-3:
            facing = np.clip(-(world_normal[:, :2] @ (vel[:2] / speed)), 0.0, 1.0)
        else:
            facing = np.full(self.n_taxels, 0.25)
        hair_defl = g["hair"] * np.tanh(0.45 * wind * (0.3 + facing) + vib_broad * 0.08)
        hair_defl = np.maximum(hair_defl, hair)

        # --- skin stretch: leaky integral of shear (Ruffini) -------------------
        a = dt / (dt + 2.0)
        self.stretch_u += a * (0.5 * su * 4.0 - self.stretch_u)
        self.stretch_v += a * (0.5 * sv * 4.0 - self.stretch_v)
        stretch_e = np.hypot(self.stretch_u, self.stretch_v)

        # --- edge / contrast: how much more loaded than its own patch ----------
        edge = np.abs(normal - self._patch_mean(normal)[pidx])

        # --- tickle: light, moving touch on hairy skin -------------------------
        light = np.exp(-normal / 1.5)
        tickle = hair_defl * light * np.tanh(np.abs(force_rate) * 0.05 + 0.2 * wind)

        # --- moisture: sweat from the patch, evaporating with the air ----------
        sweat = inner["sweat"][pidx]
        humid = float(inner.get("humidity", 0.45))
        evap = (1.0 - humid) * (1.0 + 0.8 * wind) / 90.0
        self.wet += dt * (0.05 * sweat - evap * self.wet)
        self.wet = np.clip(self.wet, 0.0, 1.0)
        # evaporation cools the skin a little
        self.temperature -= dt * 0.35 * evap * self.wet * 10.0

        # --- local perfusion, piloerection ------------------------------------
        bf = inner["blood_flow"][pidx] * blood_flow
        arousal = float(inner.get("arousal", 0.25))
        pilo_drive = np.clip(cold_drive * 1.6 + 0.35 * arousal, 0, 1) * g["hair"]
        self.pilo += (dt / 12.0) * (pilo_drive - self.pilo)

        # --- inflammation, sensitisation, itch --------------------------------
        infl = np.clip(inner["inflammation"][pidx] + inner["damage"][pidx], 0.0, 1.5)
        self.sens += dt * (0.03 * infl + 0.004 * np.tanh(self.adapt_noci * 2.0)
                           - self.sens / 900.0)
        self.sens = np.clip(self.sens, 0.0, 1.0)
        prur = inner["pruritogen"][pidx]
        self.pruri += (dt / 40.0) * (np.tanh(self.histamine * 3.0) + prur - self.pruri)

        # --- receptor fatigue, pressure ischaemia ------------------------------
        self.rfatigue += dt * (0.5 * np.tanh(self.adapt_sa1 + self.adapt_fa1)
                               - self.rfatigue / 25.0)
        self.rfatigue = np.clip(self.rfatigue, 0.0, 1.0)
        ratio = pressure_kpa / np.maximum(threshold_kpa, 1.0)
        self.ischemia += dt * (0.004 * np.maximum(ratio - 0.08, 0.0) / np.maximum(bf, 0.2)
                               - self.ischemia / 300.0)
        self.ischemia = np.clip(self.ischemia, 0.0, 1.0)

        # --- nociceptor sub-classes -------------------------------------------
        a_delta = np.tanh(2.0 * noci_mech_drive) * g["noci_mech"]
        c_poly = np.tanh(1.2 * self.adapt_noci + 0.6 * infl + 0.5 * self.ischemia) \
            * g["noci_mech"]
        irritant = np.clip(inner["irritant"][pidx] + 0.3 * self.wet * (humid < 0.2), 0, 1.5)

        base = N_BASE_TACTILE
        out[:, base + 0] = hair_defl
        out[:, base + 1] = self.stretch_u
        out[:, base + 2] = self.stretch_v
        out[:, base + 3] = stretch_e
        out[:, base + 4] = edge
        out[:, base + 5] = tickle
        out[:, base + 6] = self.wet
        out[:, base + 7] = bf
        out[:, base + 8] = self.pilo
        out[:, base + 9] = self.sens
        out[:, base + 10] = infl
        out[:, base + 11] = self.pruri
        out[:, base + 12] = self.rfatigue
        out[:, base + 13] = a_delta
        out[:, base + 14] = c_poly
        out[:, base + 15] = irritant
        out[:, base + 16] = self.ischemia

    # ------------------------------------------------------------------
    def _aggregate(self, out: np.ndarray, mask: np.ndarray) -> dict:
        """Per-region summaries plus a whole-body summary vector.

        Two views are produced because they answer different questions:
        ``agg[region]`` is the *mean* 26-channel receptor vector of that body
        region (what the region feels like), while ``agg["_region5"]`` is a
        compact (n_regions x 5) table of intensity measures -- total force,
        peak pressure, strongest SA-I and C-tactile drive, and mean
        temperature -- which is what the body-map figures and the CSV logs
        need, and which does not survive averaging.
        """
        agg: dict[str, np.ndarray] = {}
        region5 = np.zeros((len(REGION_ORDER), 5))
        for i, region in enumerate(REGION_ORDER):
            idxs = _REGION_INDEX[region]
            sub = out[idxs]
            agg[region] = sub.mean(axis=0)
            region5[i, 0] = float(sub[:, CH["normal_force"]].sum())
            region5[i, 1] = float(sub[:, CH["pressure"]].max())
            region5[i, 2] = float(sub[:, CH["sa1"]].max())
            region5[i, 3] = float(sub[:, CH["ct"]].max())
            region5[i, 4] = float(sub[:, CH["temperature"]].mean())
        agg["_region5"] = region5
        # summary: for each channel, the max and the sum over the body
        ch_max = out.max(axis=0)
        ch_sum = out.sum(axis=0)
        active_frac = mask.mean() if mask.size else 0.0
        agg["_summary"] = np.concatenate([ch_max, np.log1p(fclip(ch_sum, 0, None)),
                                          [active_frac]])
        agg["_mask"] = mask
        return agg

    # ------------------------------------------------------------------
    def apply_contact_temperature(self, state: BodyState, dt: float) -> None:
        """Deprecated no-op.

        Conductive heat exchange is applied inside :meth:`sense`, spatially
        local to each contact point.  Doing it here as well would heat every
        taxel of a contacting limb and double-count the exchange.
        """
        return None

    def reset(self) -> None:
        self.temperature[:] = 33.0
        self.prev_normal[:] = 0.0
        self.vib_state[:] = 0.0
        for nm in ("adapt_sa1", "adapt_sa2", "adapt_fa1", "adapt_fa2",
                   "adapt_ct", "adapt_noci", "itch_state", "histamine",
                   "stretch_u", "stretch_v", "wet", "pilo", "sens", "rfatigue",
                   "ischemia", "pruri"):
            getattr(self, nm)[:] = 0.0


_REGION_INDEX: dict[str, np.ndarray] = {}
for _r in sorted(set(skin.TAXEL_PATCH.tolist())):
    _REGION_INDEX[_r] = np.where(skin.TAXEL_PATCH == _r)[0]
# coarse groups too
for _g, _members in skin.REGION_GROUPS.items():
    _sel = np.where(np.isin(skin.TAXEL_PATCH, list(_members)))[0]
    if _sel.size:
        _REGION_INDEX[f"grp_{_g}"] = _sel

REGION_NAMES = sorted(_REGION_INDEX.keys())
# the 40 contiguous skin patches, in a stable order, for the summary table
REGION_ORDER = sorted(set(skin.TAXEL_PATCH.tolist()))


# ==========================================================================
# Proprioception
# ==========================================================================
class ProprioceptiveSystem:
    """Muscle spindles, Golgi tendon organs, joint afferents, efference copy.

    The muscle model is deliberately simple but *directionally correct*:
    spindle Ia reports length **and** rate of change (so it is silent during
    a static hold but fires on movement), spindle II reports length only, and
    the Golgi tendon organ reports force.
    """

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.meta = meta
        self.n = meta.n_actuators
        self.names = [n for _, n, _ in meta.joint_order]
        self.limit_lo = np.zeros(self.n)
        self.limit_hi = np.zeros(self.n)
        for i, (_, _, j) in enumerate(meta.joint_order):
            self.limit_lo[i] = j.lo
            self.limit_hi[i] = j.hi
        # moment arms: crude but monotone in joint strength
        self.moment_arm = 0.02 + 0.10 * np.tanh(meta.torque_limit / 120.0)
        self.prev_q = np.zeros(self.n)
        self.prev_qd = np.zeros(self.n)
        self._q_init = False
        self.spindle_adapt = np.zeros(self.n)

    def sense(self, model, data, meta, state: BodyState) -> np.ndarray:
        q, qd = state.q, state.qd
        if not self._q_init:
            self.prev_q = q.copy()
            self.prev_qd = qd.copy()
            self._q_init = True
        dt = 1.0 / self.cfg.rates.receptor
        qdd = (qd - self.prev_qd) / max(dt, 1e-6)

        # muscle length/velocity: joint angle mapped through a moment arm
        length = 1.0 + self.moment_arm * q
        velocity = self.moment_arm * qd

        # Spindle Ia: dynamic (length + velocity), with a small adaptation
        ia = 0.55 + 1.6 * (length - 1.0) + 0.22 * velocity
        self.spindle_adapt += (dt / 0.25) * (ia - self.spindle_adapt)
        ia = np.tanh(ia - 0.25 * self.spindle_adapt)
        # Spindle II: static length only
        ii = np.tanh(0.9 * (length - 1.0))
        # Golgi tendon organ: force, with a high threshold
        ib = np.tanh(np.abs(state.tau) / (0.35 * self.cfg.motor.__dict__.get(
            "_x", 1.0) + 0.55 * np.maximum(self.meta.torque_limit, 1.0)) * 1.2)
        # Efference copy: what the brain told the muscle to do
        efference = np.tanh(state.ctrl / np.maximum(self.meta.torque_limit, 1e-6))
        tau_err = state.tau - state.ctrl
        # joint-limit afferents: proximity to the mechanical stop
        span = np.maximum(self.limit_hi - self.limit_lo, 1e-6)
        prox = np.maximum(
            (q - (self.limit_hi - 0.10 * span)) / (0.10 * span),
            ((self.limit_lo + 0.10 * span) - q) / (0.10 * span))
        prox = fclip(prox, 0.0, 1.0)
        effort = np.abs(state.tau) / np.maximum(self.meta.torque_limit, 1e-6)

        out = np.zeros((self.n, N_PROPRIO_CH))
        out[:, 0] = q
        out[:, 1] = qd
        out[:, 2] = qdd
        out[:, 3] = ia
        out[:, 4] = ii
        out[:, 5] = ib
        out[:, 6] = efference
        out[:, 7] = state.tau
        out[:, 8] = tau_err
        out[:, 9] = prox
        out[:, 10] = length
        out[:, 11] = velocity
        out[:, 12] = effort

        self.prev_q = q.copy()
        self.prev_qd = qd.copy()
        return out

    def reset(self) -> None:
        self._q_init = False
        self.spindle_adapt[:] = 0.0


# ==========================================================================
# Vestibular
# ==========================================================================
class VestibularSystem:
    """Semicircular canals with cupula adaptation, plus otolith organs.

    The otolith organs cannot distinguish gravity from linear acceleration,
    which is exactly why tilt and translation are ambiguous -- a real
    perceptual problem we reproduce rather than solve.
    """

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.meta = meta
        self.canal_state = np.zeros(3)
        self.otolith_state = np.zeros(3)
        self.prev_lin = np.zeros(3)
        self._init = False
        self.yaw = 0.0
        self.tilt = 0.0

    def sense(self, model, data, meta, state: BodyState) -> np.ndarray:
        V = self.cfg.vestibular
        dt = 1.0 / self.cfg.rates.receptor

        gyro = meta.read(data, "canal_head")
        accel = meta.read(data, "otolith_head")
        vel = meta.read(data, "vel_head")
        mag = meta.read(data, "magneto_head")

        canal_raw = gyro
        # cupula adaptation: the canal signal decays during sustained rotation
        self.canal_state += (dt / V.canal_tau) * (canal_raw - self.canal_state)
        canal = (canal_raw - self.canal_state) * V.canal_gain

        # otoliths report specific force (linear accel - gravity) in head frame
        if not self._init:
            self.prev_lin = vel.copy()
            self._init = True
        lin_acc = (vel - self.prev_lin) / max(dt, 1e-6)
        self.prev_lin = vel.copy()
        self.otolith_state += (dt / V.otolith_tau) * (accel - self.otolith_state)
        otolith = self.otolith_state * V.otolith_gain

        # gravity direction in the head frame: specific force at rest is -g
        grav = -accel / max(np.linalg.norm(accel) + 1e-9, 1e-9)
        tilt = float(np.arctan2(grav[1], grav[2]))
        self.tilt += 0.4 * (tilt - self.tilt)
        # yaw from the magnetometer (only the heading component is usable)
        yaw_meas = float(np.arctan2(mag[1], mag[0]))
        dyaw = np.arctan2(np.sin(yaw_meas - self.yaw), np.cos(yaw_meas - self.yaw))
        self.yaw += 0.25 * dyaw
        tilt_rate = float(np.hypot(canal[0], canal[1]))

        out = np.zeros(len(VESTIBULAR_CHANNELS))
        out[0:3] = canal
        out[3:6] = canal_raw
        out[6:9] = otolith
        out[9:12] = grav
        out[12] = self.tilt
        out[13] = tilt_rate
        out[14] = self.yaw
        out[15:18] = mag
        return out

    def reset(self) -> None:
        self.canal_state[:] = 0
        self.otolith_state[:] = 0
        self._init = False
        self.yaw = self.tilt = 0.0


# ==========================================================================
# Vision
# ==========================================================================
class VisualSystem:
    """A tiny retina driven by MuJoCo's offscreen renderer.

    Only ~1 700 photoreceptor "pixels" survive, but the important structure is
    preserved: foveal acuity, a motion channel, saccades that move the fovea,
    blinks, and a pupil that reacts to both light and arousal.
    """

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.meta = meta
        self.V = cfg.vision
        self.enabled = bool(cfg.vision.enabled)
        self.renderer = None
        self.prev_gray = None
        self.pupil = 4.0
        self.blink_t = 0.0
        self.next_blink = 3.0
        self.saccade_t = 0.0
        self.next_saccade = 0.3
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.rng = np.random.default_rng(cfg.seed + 11)
        self._last_update = -1.0
        self._cached = np.zeros(len(VISUAL_CHANNELS))
        self._last_img = None
        self._since_render = 1e9
        self.retina = RetinaBank(self.V.retina_h, self.V.retina_w)
        self.last_retina = None
        # set from outside by the agent: the state of the eyes (see ocular.py).
        # None -> the original random blink generator.
        self.eye_state: dict | None = None
        if self.enabled:
            try:
                self.renderer = mujoco_renderer(meta.model, self.V.retina_h,
                                                self.V.retina_w)
            except Exception:
                self.enabled = False

    def sense(self, model, data, meta, state: BodyState, arousal: float = 0.3,
              luminance: float = 1.0) -> np.ndarray:
        if not self.enabled or self.renderer is None:
            return self._cached.copy()
        dt = 1.0 / self.cfg.rates.receptor
        V = self.V

        # ---- pupil: light reflex + arousal dilation ---------------------
        target = V.pupil_min + (V.pupil_max - V.pupil_min) * (
            0.65 * (1.0 - fclip(luminance, 0, 1)) + 0.35 * fclip(arousal, 0, 1))
        self.pupil += (dt / 0.6) * (target - self.pupil)

        # ---- blink -------------------------------------------------------
        blur = 0.0
        if self.eye_state is not None:
            # the ocular surface decides: lid aperture and tear-film blur
            blink = float(1.0 - self.eye_state["aperture"])
            blur = float(self.eye_state["blur"])
        else:
            self.blink_t -= dt
            blink = 0.0
            if self.blink_t > 0:
                blink = 1.0
            else:
                self.next_blink -= dt
                if self.next_blink <= 0:
                    self.blink_t = V.blink_duration
                    self.next_blink = float(self.rng.uniform(*V.blink_interval))

        # ---- saccades ----------------------------------------------------
        self.saccade_t -= dt
        saccade = 0.0
        if self.saccade_t > 0:
            saccade = 1.0
        else:
            self.next_saccade -= dt
            if self.next_saccade <= 0:
                self.saccade_t = 0.04
                self.next_saccade = float(self.rng.uniform(*V.saccade_interval))
                self.gaze_x = float(self.rng.normal(0, 0.35))
                self.gaze_y = float(self.rng.normal(0, 0.25))

        # ---- render, but only at the retina's own refresh rate -----------
        # Transduction runs at 250-500 Hz; the retina does not.  Re-rendering
        # on every transduction tick is the single most expensive mistake in
        # this file (it costs ~40 % of total runtime for no extra information).
        self._since_render += dt
        period = 1.0 / max(V.update_hz, 1.0)
        if self._last_img is None or self._since_render >= period:
            self._since_render = 0.0
            img = self._render(model, data)
            if img is None:
                return self._cached.copy()
            self._last_img = img
        else:
            img = self._last_img
        eff = img
        if blur > 0.02:
            # an irregular tear film scatters light: mix in a defocused copy
            bl = (np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge"))
            soft = (bl[:-2, :-2] + bl[:-2, 1:-1] + bl[:-2, 2:] + bl[1:-1, :-2]
                    + bl[1:-1, 1:-1] + bl[1:-1, 2:] + bl[2:, :-2] + bl[2:, 1:-1]
                    + bl[2:, 2:]) / 9.0
            m = 0.85 * min(blur, 1.0)
            eff = (1.0 - m) * img + m * soft
        if blink > 0:
            eff = eff * (1.0 - 0.95 * min(blink, 1.0))
        gray = eff.mean(axis=2)
        self.last_retina = self.retina.sense(eff, pupil_mm=self.pupil)

        h, w = gray.shape
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            motion = np.zeros_like(gray)
        else:
            motion = np.abs(gray - self.prev_gray)
        self.prev_gray = gray

        # fovea: a central crop, shifted by the current gaze offset
        fw, fh = max(int(w * V.fovea_frac), 3), max(int(h * V.fovea_frac), 3)
        cx = int(fclip(w / 2 + self.gaze_x * w * 0.25, fw / 2, w - fw / 2))
        cy = int(fclip(h / 2 + self.gaze_y * h * 0.25, fh / 2, h - fh / 2))
        fov = gray[cy - fh // 2:cy + fh // 2, cx - fw // 2:cx + fw // 2]
        per_mask = np.ones_like(gray, bool)
        per_mask[cy - fh // 2:cy + fh // 2, cx - fw // 2:cx + fw // 2] = False

        gx = np.abs(np.diff(gray, axis=1)).mean() if w > 1 else 0.0
        color = eff.reshape(-1, 3).mean(axis=0)

        out = np.zeros(len(VISUAL_CHANNELS))
        out[0] = float(gray.mean())
        out[1] = float(gray.std())
        out[2] = float(gray.std() / max(gray.mean(), 1e-3))
        out[3] = float(motion.mean())
        out[4] = float(fov.mean())
        out[5] = float(fov.std() / max(fov.mean(), 1e-3))
        out[6] = float(motion[per_mask].mean()) if per_mask.any() else 0.0
        out[7] = self.pupil
        out[8] = blink
        out[9] = saccade
        out[10] = self.gaze_x
        out[11] = self.gaze_y
        out[12:15] = color
        out[15] = float(gx)
        out[16] = float(1.0 / (1.0 + gray.mean()))
        self._cached = out
        return out

    def _render(self, model, data):
        try:
            self.renderer.update_scene(data, camera="egocentric")
            return self.renderer.render().astype(np.float32) / 255.0
        except Exception:
            self.enabled = False
            return None

    def reset(self) -> None:
        self.prev_gray = None
        self._last_img = None
        self._since_render = 1e9


def mujoco_renderer(model, h, w):
    import mujoco
    return mujoco.Renderer(model, h, w)


# ==========================================================================
# Audition
# ==========================================================================
class AuditorySystem:
    """A cochlear filterbank excited by contact transients and self-motion.

    MuJoCo has no acoustic field, so sound is synthesised from the physics
    that *would* make noise: impacts (contact force rate), sliding (shear),
    and the body's own movement (bone conduction + muscle noise).
    """

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.A = cfg.audio
        self.n_bands = cfg.audio.n_bands
        self.bands = np.zeros(self.n_bands)
        self.prev_loudness = 0.0
        self.onset = 0.0
        self.centres = np.geomspace(cfg.audio.f_min, cfg.audio.f_max, self.n_bands)
        # log-spaced band weights over the audible range
        self.tilt = 1.0 / (1.0 + self.centres / 2000.0)
        self.rng = np.random.default_rng(cfg.seed + 23)

    def sense(self, model, data, meta, state: BodyState) -> np.ndarray:
        A = self.A
        # impact excitation: total contact force rate
        impact = 0.0
        slide = 0.0
        for c in state.contacts:
            impact += abs(float(c.force[0]))
            slide += float(np.hypot(c.force[1], c.force[2]))
        self_noise = float(np.abs(state.tau).mean()) * A.motor_noise_gain
        bone = float(np.abs(state.com_acc).max()) * A.bone_conduction_gain

        drive = impact * 0.02 + slide * 0.004 + self_noise + bone
        drive += abs(self.rng.normal(0, 0.004))
        # spread the drive across the filterbank with a spectral tilt
        target = drive * self.tilt
        self.bands += (target - self.bands) * 0.35
        loudness = float(np.sqrt((self.bands ** 2).mean()))
        # crude pitch: energy-weighted mean band index
        w = np.maximum(self.bands, 0.0)
        pitch = float((w * np.arange(self.n_bands)).sum() / max(w.sum(), 1e-9))
        centroid = float((w * self.centres).sum() / max(w.sum(), 1e-9))
        roughness = float(np.std(self.bands) / max(np.mean(self.bands), 1e-6))
        self.onset = max(0.0, loudness - self.prev_loudness) * 12.0
        self.prev_loudness = loudness

        return np.array([loudness, pitch, self.onset, centroid, roughness,
                         self_noise + bone])

    def reset(self) -> None:
        self.bands[:] = 0.0
        self.prev_loudness = 0.0


# ==========================================================================
# Chemoreception
# ==========================================================================
class ChemoSystem:
    """Olfaction from proximity to odorous objects, gustation on contact.

    Each scene object carries a 24-D odour signature and a 5-D taste
    signature.  Concentration falls off with distance from the nose; taste
    requires actual contact between the tongue and the object.
    """

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.C = cfg.chemo
        self.n_olf = cfg.chemo.n_olfactory_channels
        self.objects = meta.objects
        self.odor_conc = np.zeros(self.n_olf)
        self.prev_odor = np.zeros(self.n_olf)
        self.novelty = 0.0
        self._seen = np.zeros(self.n_olf)
        self.tongue_site = meta.landmark_site_ids.get("gaze")

    def sense(self, model, data, meta, state: BodyState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nose = state.site_pos.get("gaze", state.gaze_pos)
        odor = np.zeros(self.n_olf)
        taste = np.zeros(5)
        tongue_hit = 0.0
        for obj in self.objects:
            jid = meta.object_qpos_addr.get(obj.name)
            if jid is None:
                continue
            p = data.qpos[jid:jid + 3]
            d = float(np.linalg.norm(nose - p))
            if obj.odor is not None and d < self.C.olfactory_range:
                # 1/r falloff, clipped
                conc = np.exp(-(d / self.C.olfactory_range) ** 2 * 2.0)
                odor += obj.odor * conc
        # adaptation: receptors stop responding to a steady odour
        self.odor_conc += (odor - self.odor_conc) * 0.12
        self._seen += (self.odor_conc - self._seen) * 0.01
        novelty = float(np.abs(self.odor_conc - self._seen).sum())
        self.novelty += 0.1 * (novelty - self.novelty)

        # gustation: tongue-object contact
        for c in state.contacts:
            for bid, other in ((c.body1, c.name2), (c.body2, c.name1)):
                bname = _body_name_of(meta, bid)
                if bname != "jaw":
                    continue
                for obj in self.objects:
                    if obj.name in other and obj.taste is not None:
                        taste = np.maximum(taste, obj.taste)
                        tongue_hit = 1.0
        summary = np.array([
            float(self.odor_conc.sum()),
            self.novelty,
            float(np.dot(self.odor_conc, np.linspace(-1, 1, self.n_olf))),
            float(np.linalg.norm(state.com_vel)) * 0.1 + 0.2,
        ])
        return self.odor_conc.copy(), taste, summary

    def reset(self) -> None:
        self.odor_conc[:] = 0
        self._seen[:] = 0
        self.novelty = 0.0


def _body_name_of(meta, body_id: int) -> str:
    import mujoco
    if body_id < 0:
        return ""
    return mujoco.mj_id2name(meta.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


# ==========================================================================
# Aggregator
# ==========================================================================
class ReceptorSystem:
    """Runs every transduction organ and returns one :class:`ReceptorFrame`."""

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.meta = meta
        self.tactile = TactileSystem(cfg, meta)
        self.proprio = ProprioceptiveSystem(cfg, meta)
        self.vestibular = VestibularSystem(cfg, meta)
        self.visual = VisualSystem(cfg, meta)
        self.auditory = AuditorySystem(cfg, meta)
        self.chemo = ChemoSystem(cfg, meta)
        # ---- cell populations (see senses_ext.py) ----------------------------
        seed = cfg.seed
        self.spindles = SpindleBank(self.proprio.n, self.proprio.moment_arm,
                                    meta.torque_limit, seed + 101)
        self.vest_cells = VestibularBank(seed + 102)
        self.cochlea = Cochlea(cfg.audio.n_bands, cfg.audio.f_min, cfg.audio.f_max,
                               seed + 103)
        self.olfactory = OlfactoryBank(cfg.chemo.n_olfactory_channels, seed + 104)
        self.gustatory = GustatoryBank(seed + 105)
        self.head_id = meta.body_ids.get("head")
        # set from outside (the agent / skills) -- see EmbodiedHuman.step
        self.self_voice = 0.0          # loudness of the person's own voice
        self.resp_rate = 13.0          # breaths per minute, for sniffing
        self.fusimotor = 0.35

    def sense(self, model, data, meta, state: BodyState,
              world=None, world_nose_pos=None, world_mouth_pos=None,
              *, arousal: float = 0.3, blood_flow: float = 1.0) -> ReceptorFrame:
        dt = 1.0 / self.cfg.rates.receptor
        self.tactile.apply_contact_temperature(state, dt)
        tact, agg = self.tactile.sense(model, data, meta, state,
                                       blood_flow=blood_flow)
        prop = self.proprio.sense(model, data, meta, state)
        vest = self.vestibular.sense(model, data, meta, state)
        vis = self.visual.sense(model, data, meta, state, arousal=arousal)
        aud = self.auditory.sense(model, data, meta, state)
        # ---- olfactory / gustatory world emission -----------------------
        nose_pos = world_nose_pos if world_nose_pos is not None else (
            state.site_pos.get("gaze", state.gaze_pos) if hasattr(state, 'site_pos') else np.zeros(3))
        mouth_pos = world_mouth_pos if world_mouth_pos is not None else (
            state.site_pos.get("mouth", state.gaze_pos) if hasattr(state, 'site_pos') else np.zeros(3))
        if world is not None:
            olf[:] = world.get_odorant_concentration(nose_pos) * np.ones_like(olf)
            gus[:] = world.get_tastant_concentration(mouth_pos) * np.ones_like(gus)

        olf, gus, chemo_sum = self.chemo.sense(model, data, meta, state)
        # override with world emission if present
        if world is not None:
            olf[:] = world.get_odorant_concentration(nose_pos) * np.ones_like(olf)
            gus[:] = world.get_tastant_concentration(mouth_pos) * np.ones_like(gus)

        # ---- populations ------------------------------------------------------
        ext: dict[str, np.ndarray] = {}
        ext["spindle"] = self.spindles.sense(state.q, state.qd, state.tau, dt,
                                             fusimotor=self.fusimotor + 0.4 * arousal)
        ext["vestibular_cells"] = self.vest_cells.sense(vest[3:6], vest[6:9], dt)
        hid = self.head_id
        head_pos = data.xpos[hid].copy() if hid is not None else np.zeros(3)
        head_R = data.xmat[hid].reshape(3, 3).copy() if hid is not None else np.eye(3)
        cps, cas = [], []
        if state.contacts:
            loud = sorted(state.contacts, key=lambda c: -abs(float(c.force[0])))[:6]
            for c in loud:
                a = abs(float(c.force[0])) * 0.02
                if a > 1e-4:
                    cps.append(c.pos)
                    cas.append(a)
        ext["cochlea"] = self.cochlea.sense(
            state.t, dt, head_pos, head_R, self.auditory.bands,
            np.array(cps) if cps else np.zeros((0, 3)), np.array(cas),
            self_voice=self.self_voice)
        ext["olfactory"] = self.olfactory.sense(self.chemo.odor_conc, dt,
                                                resp_rate=self.resp_rate,
                                                novelty=self.chemo.novelty)
        ext["gustatory"] = self.gustatory.sense(gus, dt)
        if self.visual.last_retina is not None:
            ext["retina"] = self.visual.last_retina

        pain_mech = float((tact[:, CH["noci_mech"]]).max()) if len(tact) else 0.0
        pain_heat = float((tact[:, CH["noci_heat"]]).max()) if len(tact) else 0.0
        pain_cold = float((tact[:, CH["noci_cold"]]).max()) if len(tact) else 0.0
        itch = float((tact[:, CH["itch"]]).max()) if len(tact) else 0.0
        ct = float(np.sort(tact[:, CH["ct"]])[-50:].mean()) if len(tact) else 0.0
        touch_int = float(np.sort(tact[:, CH["normal_force"]])[-200:].mean()) \
            if len(tact) else 0.0
        mask = agg.get("_mask", np.zeros(0, bool))

        return ReceptorFrame(
            t=state.t,
            tactile=tact,
            tactile_by_region=agg,
            skin_temperature=self.tactile.temperature.copy(),
            contact_mask=mask,
            proprio=prop,
            proprio_by_joint={n: prop[i] for i, n in enumerate(self.proprio.names)},
            vestibular=vest,
            visual=vis,
            auditory=aud,
            olfactory=olf,
            gustatory=gus,
            chemo_summary=chemo_sum,
            pain_mech=pain_mech,
            pain_heat=pain_heat,
            pain_cold=pain_cold,
            pain_total=float(max(pain_mech, pain_heat, pain_cold)),
            itch_total=itch,
            affective_touch=ct,
            touch_intensity=touch_int,
            contact_count=int(mask.sum()) if mask.size else 0,
            ext=ext,
        )

    def reset(self) -> None:
        for sub in (self.tactile, self.proprio, self.vestibular, self.visual,
                    self.auditory, self.chemo):
            sub.reset()

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "tactile": {
                "n_taxels": self.tactile.n_taxels,
                "channels_per_taxel": N_TACTILE_CH,
                "channel_names": list(TACTILE_CHANNELS),
                "total_scalar_channels": self.tactile.n_taxels * N_TACTILE_CH,
            },
            "proprioceptive": {
                "n_joints": self.proprio.n,
                "channels_per_joint": N_PROPRIO_CH,
                "channel_names": list(PROPRIO_CHANNELS),
                "total_scalar_channels": self.proprio.n * N_PROPRIO_CH,
            },
            "vestibular": {"channels": list(VESTIBULAR_CHANNELS)},
            "visual": {"channels": list(VISUAL_CHANNELS),
                       "retina": [self.cfg.vision.retina_h, self.cfg.vision.retina_w],
                       "enabled": self.visual.enabled},
            "auditory": {"bands": self.auditory.n_bands,
                         "channels": list(AUDITORY_CHANNELS)},
            "olfactory": {"channels": self.cfg.chemo.n_olfactory_channels},
            "gustatory": {"channels": list(GUSTATORY_CHANNELS)},
        }
