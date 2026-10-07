"""
The ocular surface: why the eyes need to blink.

Each eye carries a tear film that thins and evaporates between blinks, a
meniscus reservoir fed by the lacrimal gland, a lipid layer that slows
evaporation, a mucin layer that keeps the film wetted, and a population of
corneal nerve terminals that report the consequences: evaporative cooling (cold
thermoreceptors), hyperosmolarity and dry patches (polymodal nociceptors) and
lid friction over a thin film (mechano-nociceptors).  The *sensation of
dryness* that results drives the blink, and -- unlike a timer -- it also reaches
into the rest of the person:

* **vision**    an irregular film scatters light (blur) and a blink blanks it,
* **affect**    discomfort and irritability (``ocular_discomfort`` -> appraisal),
* **pain**      chronic irritation sensitises central pain processing,
* **drives**    an ``ocular_comfort`` drive, and an urge to rub the eyes,
* **attention** staring (reading, concentrating) suppresses blinking, which is
                exactly what makes the eyes dry; speaking, arousal and
                tiredness change the rate too,
* **the body**  tears when the person is sad, dehydration dries the eyes, the
                lids droop with sleepiness and narrow in bright light.

Everything here is a toy model with plausible orders of magnitude (film ~3 um,
tear break-up time 10-20 s, osmolarity 300 -> 320+ mOsm/L when dry, ~15 blinks a
minute while talking, ~4 while reading); it is not a clinical simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .complexity import C as COMPLEXITY

H_REF = 3.5          # um, healthy film thickness
H_CRIT = 1.0         # um, film break-up threshold
V_REF = 7.0          # uL, tear reservoir
OSM_REF = 300.0      # mOsm/L


@dataclass
class OcularInputs:
    """What the rest of the person tells the eyes."""
    humidity: float = 0.45            # 0..1 relative humidity of the room
    airflow: float = 0.0              # m/s of air over the eyes (walking adds some)
    luminance: float = 0.6            # 0..1 scene brightness
    gaze_up: float = 0.0              # -1 (down) .. +1 (up); looking up exposes more
    attention: float = 0.3            # 0..1 visual concentration (suppresses blinking)
    arousal: float = 0.25
    sadness: float = 0.0              # drives emotional tearing
    anxiety: float = 0.0
    fatigue: float = 0.0              # 0..1
    sleepiness: float = 0.2
    hydration: float = 1.0            # 1 = well hydrated
    speaking: float = 0.0             # 0..1 (speech raises blink rate)
    startle: float = 0.0              # sudden stimulus -> reflex blink
    hand_near: float = 0.0            # hand close to the face -> protective blink
    dopamine: float = 0.35
    sensitisation: float = 0.0        # central sensitisation (0..1)
    parasympathetic: float = 0.5      # lacrimal drive from the autonomic system
    temperature: float = 34.0         # ocular surface temperature (degC)
    wipe: float = 0.0                 # hand rubbing the eye (1 = rubbing now)


@dataclass
class OcularOutputs:
    aperture: np.ndarray = field(default_factory=lambda: np.ones(2))   # 1 open .. 0 closed
    blur: float = 0.0
    dryness: float = 0.0              # perceived, 0..1+
    burning: float = 0.0
    grit: float = 0.0
    discomfort: float = 0.0           # aversive feeling, feeds affect
    ocular_pain: float = 0.0          # nociceptive contribution (feeds the pain system)
    blink_rate: float = 0.0           # per minute, smoothed
    blinking: bool = False
    rub_urge: float = 0.0
    tearing: float = 0.0              # visible tears (0..1)
    redness: float = 0.0
    tbut: float = 0.0                 # s since last blink (tear break-up clock)
    osmolarity: float = OSM_REF
    reservoir: float = V_REF
    breakup_frac: float = 0.0
    photophobia: float = 0.0
    n_blinks: int = 0
    attention_cost: float = 0.0       # how much dryness degrades visual precision


class OcularSurface:
    """Two eyes' worth of tear film, corneal nerves and blink control."""

    def __init__(self, seed: int = 0):
        C = COMPLEXITY
        self.S = int(C.tear_film_sectors)
        self.U = int(C.corneal_units)
        rng = np.random.default_rng(seed + 991)
        self.rng = rng
        S, U = self.S, self.U
        # sectors run from the top of the cornea (v = +1) to the bottom (v = -1)
        self.v = np.linspace(1.0, -1.0, S)
        self.central = (np.abs(self.v) < 0.45)
        # ---- tear film state, per eye -----------------------------------------
        self.h = np.full((2, S), H_REF)            # film thickness (um)
        self.mucin = np.ones((2, S))               # wettability 0..1
        self.lipid = np.array([0.82, 0.8])         # lipid layer quality 0..1
        self.V = np.array([V_REF, V_REF])          # reservoir (uL)
        self.osm = np.array([OSM_REF, OSM_REF])
        self.since_blink = np.zeros(2)
        # ---- corneal nerves ---------------------------------------------------
        self.kind = rng.choice(3, size=U, p=[0.40, 0.35, 0.25])   # 0 cold, 1 polymodal, 2 mechano
        self.sector = rng.integers(0, S, size=U)
        self.thr = rng.uniform(0.05, 0.55, size=U)
        self.gain = rng.lognormal(0.0, 0.3, size=U)
        self.tau_ad = rng.uniform(1.0, 8.0, size=U)
        self.adapt = np.zeros((2, U))
        self.rate = np.zeros((2, U))
        self.rest = rng.uniform(1.0, 6.0, size=U)
        # ---- perception -------------------------------------------------------
        self.dryness = 0.0
        self.burning = 0.0
        self.grit = 0.0
        self.redness = 0.0
        self.rub_urge = 0.0
        self.chronic = 0.0                          # slow irritation memory (hours)
        # ---- aperture and the blink generator ---------------------------------
        self.aperture = np.ones(2)
        self.base_aperture = 1.0
        self.urge = 0.0
        self.urge_thr = 1.0
        self.phase = "open"                         # open | closing | hold | opening
        self.phase_t = 0.0
        self.blink_depth = 1.0
        self.n_blinks = 0
        self.rate_smooth = 15.0
        self._blink_times: list[float] = []
        self._pending_voluntary: list[tuple] = []   # (depth, speed)
        self._wink = np.zeros(2)                    # per-eye extra closure (winks)
        self.t = 0.0
        self.tearing = 0.0
        self.out = OcularOutputs()
        self.size = (2 * S * 2 + 2 * U * 2 + 14)    # number of state variables

    # ------------------------------------------------------------------
    @property
    def n_state(self) -> int:
        return int(self.h.size + self.mucin.size + self.adapt.size + self.rate.size
                   + self.lipid.size + self.V.size + self.osm.size + 16)

    def blink_now(self, depth: float = 1.0, speed: float = 1.0) -> None:
        """A voluntary or commanded blink (also used for winks via ``wink``)."""
        self._pending_voluntary.append((float(depth), float(speed)))

    def wink(self, eye: int, amount: float = 1.0) -> None:
        self._wink[eye] = float(np.clip(amount, 0, 1))

    # ------------------------------------------------------------------
    def update(self, dt: float, inp: OcularInputs, aperture_cmd: float | None = None
               ) -> OcularOutputs:
        S, U = self.S, self.U
        self.t += dt
        ap = float(np.mean(self.aperture))
        self.since_blink += dt

        # ---- how much of each sector is exposed to the air --------------------
        gaze = float(np.clip(inp.gaze_up, -1, 1))
        edge_top = ap * 1.0 + 0.0
        expo = 1.0 / (1.0 + np.exp(-(1.0 * ap - np.abs(self.v - 0.18 * gaze)) / 0.07))
        expo = np.broadcast_to(expo, (2, S))

        # ---- evaporation and thinning ----------------------------------------
        lip = np.clip(self.lipid, 0.05, 1.0)[:, None]
        e0 = 0.85
        wind = float(np.clip(inp.airflow, 0, 3.0))
        warm = 1.0 + 0.045 * (float(inp.temperature) - 34.0)
        evap = (e0 * (1.0 - float(np.clip(inp.humidity, 0, 0.98)))
                * (1.0 + 0.9 * wind) * (1.0 - 0.8 * lip) * warm
                * (1.0 + 0.6 * np.maximum(gaze, 0.0)) * (1.0 + 0.6 * (1.0 - self.mucin)) * expo)
        refill = (H_REF * (0.6 + 0.4 * self.V[:, None] / V_REF) - self.h) / 25.0
        self.h = np.maximum(self.h + dt * (refill - evap), 0.0)
        broken = self.h < H_CRIT
        # dry spots lose their mucin coat
        self.mucin = np.clip(self.mucin + dt * (np.where(broken, -0.15, 0.01)), 0.0, 1.0)
        evap_mean = (evap * expo).sum(axis=1) / np.maximum(expo.sum(axis=1), 1e-6)

        # ---- osmolarity: evaporation concentrates, turnover dilutes ------------
        turnover = 1.0 / (45.0 * (0.5 + self.V / V_REF))
        self.osm += dt * (3.2 * evap_mean * (0.6 + 0.4 * (V_REF / np.maximum(self.V, 1.0)))
                          - (self.osm - OSM_REF) * turnover)
        self.osm = np.clip(self.osm, 280.0, 400.0)

        # ---- reservoir: lacrimal secretion, drainage, evaporation, overflow ----
        irritation = float(self.dryness + 0.7 * self.burning + 0.6 * self.grit)
        reflex = float(np.clip(irritation, 0.0, 2.0))
        emo = 3.0 * float(np.clip(inp.sadness, 0, 1.5))
        para = 0.5 + float(np.clip(inp.parasympathetic, 0, 1.0))
        secretion = (0.012 * para * (1.0 + 2.2 * reflex + emo)
                     * (1.0 - 0.45 * (1.0 - float(np.clip(inp.hydration, 0, 1.2))))
                     * (1.0 - 0.25 * float(inp.fatigue)))
        drain = self.V / 380.0
        self.V = np.clip(self.V + dt * (secretion - drain - 0.0006 * evap_mean), 0.5, 14.0)
        overflow = np.maximum(self.V - 11.0, 0.0)
        self.V -= overflow * min(dt * 2.0, 1.0)
        self.tearing += (dt / 2.0) * (float(overflow.max() > 0.01) * 1.0
                                      + 0.3 * float(self.V.max() > 9.0) - self.tearing)

        # ---- lipid layer: expressed by full blinks, degraded slowly -----------
        self.lipid = np.clip(self.lipid - dt * (1.0 / 2400.0) * (1.0 + 0.5 * wind), 0.2, 1.0)
        if inp.wipe > 0.5:
            # rubbing smears the film and irritates the surface
            self.h = np.minimum(self.h + dt * 2.0, H_REF)
            self.mucin = np.clip(self.mucin - dt * 0.02, 0, 1)
            self.grit = min(self.grit + dt * 0.1, 1.5)

        # ---- corneal nerves ----------------------------------------------------
        hs = self.h[:, self.sector]                       # (2, U)
        bk = (hs < H_CRIT).astype(float)
        cooling = np.clip(evap[:, self.sector] / 0.17, 0.0, 3.0) * 0.5
        hyper = np.clip((self.osm[:, None] - 308.0) / 18.0, 0.0, 3.0)
        # lid friction over a thin film, only while the lid is moving
        lid_speed = abs(float(np.mean(self.aperture)) - ap)
        friction = np.clip((1.0 - hs / H_REF), 0.0, 1.0) * (lid_speed / max(dt, 1e-6)) * 0.02
        kind = self.kind[None, :]
        drive = np.where(kind == 0, cooling,
                         np.where(kind == 1, 0.55 * hyper + 0.9 * bk + 0.2 * (1 - self.mucin[:, self.sector]),
                                  friction + 0.25 * bk + 0.3 * float(inp.wipe)))
        drive = drive * (1.0 + 0.9 * float(np.clip(inp.sensitisation, 0, 1)))
        a = dt / (dt + self.tau_ad)
        self.adapt += a[None, :] * (drive - self.adapt)
        eff = np.maximum(drive - 0.6 * self.adapt - self.thr[None, :], 0.0)
        self.rate = np.clip(self.rest[None, :] + 60.0 * self.gain[None, :] * np.tanh(eff * 1.4), 0, 200)
        act = np.tanh(eff * 1.4) * self.gain[None, :]
        w = {0: 0.0, 1: 0.0, 2: 0.0}
        n_k = np.maximum(np.bincount(self.kind, minlength=3), 1)
        pop = [float(act[:, self.kind == k].mean()) for k in range(3)]

        # ---- perception: central gain depends on mood, attention and sensitisation
        gain_c = ((1.0 + 0.7 * float(inp.anxiety) + 0.8 * float(inp.sensitisation)
                   + 0.35 * float(inp.fatigue)) * (0.75 + 0.5 * (1.0 - 0.5 * float(inp.attention))))
        dry_raw = 1.9 * pop[0] + 1.3 * pop[1]
        burn_raw = 2.2 * pop[1]
        grit_raw = 2.4 * pop[2]
        k = dt / 3.0
        self.dryness += k * (dry_raw * gain_c - self.dryness)
        self.burning += k * (burn_raw * gain_c - self.burning)
        self.grit += (dt / 4.0) * (grit_raw * gain_c - self.grit)
        self.chronic += (dt / 1800.0) * (irritation - self.chronic)
        self.redness += (dt / 60.0) * (np.clip(0.8 * irritation + 0.5 * float(inp.wipe)
                                               + 0.3 * float(inp.sadness), 0, 1.2) - self.redness)
        discomfort = float(np.clip(0.6 * self.dryness + 0.9 * self.burning + 0.8 * self.grit
                                   + 0.4 * self.chronic, 0, 2.0))
        self.rub_urge += (dt / 5.0) * (np.clip(0.8 * self.grit + 0.7 * self.burning + 0.6 * self.dryness
                                               - 0.2, 0, 1.5) - self.rub_urge)

        # ---- blink generator --------------------------------------------------
        blinking = self.phase != "open"
        attn = float(np.clip(inp.attention, 0, 1))
        f_total = (1.0 + 4.5 * self.dryness + 2.0 * self.grit - 0.72 * attn
                   + 0.5 * float(inp.arousal) + 0.9 * float(inp.speaking)
                   + 0.45 * float(inp.fatigue) + 0.5 * (float(inp.dopamine) - 0.35)
                   + 0.6 * float(inp.anxiety))
        f_total = float(np.clip(f_total, 0.10, 6.0))
        if not blinking:
            self.urge += dt * f_total / 4.0
        photo = float(np.clip(0.8 * (float(inp.luminance) - 0.75) / 0.25 + 0.5 * self.burning
                              + 0.4 * self.chronic, 0.0, 1.0))
        trigger = False
        depth = 1.0
        speed = 1.0
        if self._pending_voluntary and not blinking:
            depth, speed = self._pending_voluntary.pop(0)
            trigger = True
        elif (float(inp.startle) > 0.55 or float(inp.hand_near) > 0.7) and not blinking \
                and self.phase_t <= 0.0:
            trigger, speed = True, 1.6              # reflex blinks are fast and complete
        elif self.urge >= self.urge_thr and not blinking:
            trigger = True
            # incomplete blinks: concentrating, tired or dry eyes blink shallow
            depth = float(np.clip(self.rng.normal(0.93, 0.07)
                                  - 0.35 * attn * attn - 0.15 * self.dryness
                                  - 0.1 * float(inp.fatigue), 0.25, 1.0))
        if trigger:
            self.phase, self.phase_t = "closing", 0.0
            self.blink_depth = depth
            self._speed = speed
            self.urge = 0.0
            self.urge_thr = float(np.clip(self.rng.normal(1.0, 0.25), 0.5, 1.8))
            self._blink_times.append(self.t)
            self.n_blinks += 1
        # a resting aperture: squint in bright light or irritation, droop when sleepy
        base = 1.0 - 0.22 * photo - 0.18 * float(np.clip(inp.sleepiness - 0.3, 0, 1)) \
            - 0.10 * float(np.clip(inp.fatigue, 0, 1)) - 0.12 * float(min(self.burning, 1.0))
        if aperture_cmd is not None:
            base = float(np.clip(aperture_cmd, 0.0, 1.35))
        self.base_aperture += min(dt / 0.25, 1.0) * (base - self.base_aperture)
        # lid kinematics: closing ~75 ms, brief hold, opening ~200 ms
        sp = getattr(self, "_speed", 1.0)
        a_now = self.base_aperture
        if self.phase == "closing":
            self.phase_t += dt * sp
            x = min(self.phase_t / 0.075, 1.0)
            a_now = self.base_aperture * (1.0 - (1.0 - 0.0) * self.blink_depth * x * x * (3 - 2 * x))
            if x >= 1.0:
                self.phase, self.phase_t = "hold", 0.0
        elif self.phase == "hold":
            self.phase_t += dt * sp
            a_now = self.base_aperture * (1.0 - self.blink_depth)
            if self.phase_t >= 0.02:
                self.phase, self.phase_t = "opening", 0.0
                # a full blink spreads a fresh film and expresses the lipid layer
                comp = self.blink_depth
                reset = np.clip((self.v < 1.0 - 2.0 * (1.0 - comp)).astype(float), 0, 1)
                spread = 0.55 + 0.4 * comp
                self.h = self.h + (reset[None, :] * (H_REF * (0.7 + 0.3 * self.V[:, None] / V_REF) - self.h)) * spread
                self.mucin = np.clip(self.mucin + comp * 0.8, 0, 1)
                self.lipid = np.clip(self.lipid + 0.015 * comp, 0, 1)
                self.osm -= 0.30 * comp * (self.osm - OSM_REF)
                self.V = np.maximum(self.V - 0.05 * comp, 0.5)         # the lacrimal pump
                self.since_blink[:] = 0.0
        elif self.phase == "opening":
            self.phase_t += dt * sp
            x = min(self.phase_t / 0.20, 1.0)
            a_now = self.base_aperture * (1.0 - self.blink_depth * (1.0 - x * x * (3 - 2 * x)))
            if x >= 1.0:
                self.phase, self.phase_t = "open", -0.3        # refractory
        else:
            self.phase_t = min(self.phase_t + dt, 0.0) if self.phase_t < 0 else self.phase_t
        self.aperture = np.clip(np.array([a_now, a_now]) * (1.0 - 0.9 * self._wink), 0.0, 1.35)
        self._wink *= max(0.0, 1.0 - dt / 0.6)

        # ---- the clock and the rate -------------------------------------------
        self._blink_times = [t for t in self._blink_times if self.t - t < 30.0]
        window = min(self.t, 30.0)
        if window > 3.0:
            r = len(self._blink_times) / window * 60.0
            self.rate_smooth += (dt / 5.0) * (r - self.rate_smooth)

        # ---- the film's effect on the image -----------------------------------
        cf = self.central
        thin = 1.0 - float(np.mean(self.h[:, cf])) / H_REF
        blur = float(np.clip(1.1 * float(np.mean(broken[:, cf])) + 0.45 * max(thin, 0.0)
                             + 0.15 * float(max(0.0, np.mean(self.osm) - 312.0)) / 20.0, 0.0, 1.0))
        o = self.out
        o.aperture = self.aperture.copy()
        o.blur = blur
        o.dryness, o.burning, o.grit = float(self.dryness), float(self.burning), float(self.grit)
        o.discomfort = discomfort
        o.ocular_pain = float(np.clip(0.55 * self.burning + 0.5 * self.grit + 0.15 * self.dryness, 0, 1.5))
        o.blink_rate = float(self.rate_smooth)
        o.blinking = self.phase != "open"
        o.rub_urge = float(self.rub_urge)
        o.tearing = float(np.clip(self.tearing + 0.25 * float(inp.sadness), 0, 1))
        o.redness = float(np.clip(self.redness, 0, 1))
        o.tbut = float(self.since_blink.mean())
        o.osmolarity = float(self.osm.mean())
        o.reservoir = float(self.V.mean())
        o.breakup_frac = float(np.mean(broken))
        o.photophobia = photo
        o.n_blinks = self.n_blinks
        o.attention_cost = float(np.clip(0.5 * blur + 0.4 * discomfort, 0, 1))
        return o

    # ------------------------------------------------------------------
    def state_vector(self) -> np.ndarray:
        """Everything that is dynamic, flat (for logging / the inner-world count)."""
        return np.concatenate([self.h.ravel(), self.mucin.ravel(), self.adapt.ravel(),
                               self.rate.ravel(), self.lipid, self.V, self.osm,
                               [self.dryness, self.burning, self.grit, self.redness,
                                self.rub_urge, self.chronic, self.urge, self.base_aperture,
                                float(self.aperture.mean()), self.rate_smooth, self.tearing,
                                float(self.n_blinks), self.phase_t, self.blink_depth,
                                float(self._wink.sum()), self.t]])

    def summary_features(self) -> np.ndarray:
        o = self.out
        return np.array([o.dryness, o.burning, o.grit, o.discomfort, o.blur,
                         o.blink_rate / 30.0, float(np.mean(o.aperture)), o.rub_urge,
                         o.tearing, o.redness, (o.osmolarity - OSM_REF) / 30.0,
                         o.reservoir / V_REF, o.breakup_frac, o.photophobia,
                         min(o.tbut / 20.0, 2.0), o.ocular_pain])


# ==========================================================================
# The rig: lids, pupils and sclera colour on the MuJoCo model
# ==========================================================================
class EyeRig:
    """Moves the visual eye geoms (lids, pupils, redness).  Purely cosmetic."""

    def __init__(self, model):
        import mujoco
        from .build_model import LID_UP_HALF_Z, LID_LO_HALF_Z
        from .skeleton import EYE_Z, SCLERA
        self.m = model
        self.ok = True
        self.gid = {}
        for s in "lr":
            for nm in (f"lid_up_{s}", f"lid_lo_{s}", f"vis_pupil_{s}", f"eyeball_{s}"):
                g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
                self.gid[nm] = g
                if g < 0:
                    self.ok = False
        self.up_hz = LID_UP_HALF_Z
        self.lo_hz = LID_LO_HALF_Z
        self.eye_z = EYE_Z
        self.sclera = np.array(SCLERA, float)
        if self.ok:
            self.up_open = self.eye_z + 0.0095 + self.up_hz
            self.up_closed = self.eye_z - 0.0080 + self.up_hz
            self.lo_open = self.eye_z - 0.0095 - self.lo_hz
            self.lo_closed = self.eye_z - 0.0085 - self.lo_hz
            self.pupil_front = -0.0158

    def apply(self, aperture, pupil_mm: float = 4.0, redness: float = 0.0) -> None:
        if not self.ok:
            return
        m = self.m
        for i, s in enumerate("lr"):
            a = float(np.clip(aperture[i], 0.0, 1.0))
            c = 1.0 - a
            m.geom_pos[self.gid[f"lid_up_{s}"]][2] = self.up_open + (self.up_closed - self.up_open) * c
            m.geom_pos[self.gid[f"lid_lo_{s}"]][2] = self.lo_open + (self.lo_closed - self.lo_open) * (c ** 1.5)
            g = self.gid[f"vis_pupil_{s}"]
            r = 0.00085 * float(np.clip(pupil_mm, 1.5, 8.5))        # 4 mm -> 3.4 mm radius
            m.geom_size[g][0] = r
            m.geom_pos[g][1] = self.pupil_front + r
            m.geom_rgba[self.gid[f"eyeball_{s}"]][:3] = self.sclera[:3] * (1 - 0.5 * redness) \
                + np.array([0.85, 0.15, 0.15]) * 0.5 * redness
