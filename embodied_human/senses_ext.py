"""
Populations of sensory cells, not single channels.

The original receptor layer reports one number per modality (or per joint, or
per taxel).  A nervous system does not: every muscle holds dozens of spindles,
every ear thousands of hair cells, the nose hundreds of receptor *types*, each
with its own threshold, gain, adaptation and tuning.  This module generates
those populations from the :mod:`complexity` profile and transduces the same
physics the base layer already computes through them:

=====================  ===========================================  ==========
population             what varies cell to cell                      cells (x)
=====================  ===========================================  ==========
muscle spindles        threshold, dynamic gain, adaptation, Ia / II    4,160
Golgi tendon organs    force threshold, gain                           (incl.)
canal afferents        side, regular / irregular, gain, adaptation     1,152
otolith hair cells     preferred direction, striola type, adaptation   (incl.)
cochlear hair cells    best frequency, ear, adaptation                 ~400
olfactory receptors    odorant tuning vector, adaptation, sniff gate   ~640
taste cells            tuning to sweet/salty/sour/bitter/umami/fat     ~50
retinal cells          cone type, ON / OFF, centre-surround            ~48 000
=====================  ===========================================  ==========

(x) counts at the ``extreme`` complexity level.

None of this is a model of any particular cell type's biophysics.  Each cell
is a thresholded, saturating, adapting unit with randomised parameters drawn
from distributions that follow the right *shape* (spread of thresholds, push-
pull between ears, broad overlapping odorant tuning), which is what makes the
population code behave like a population code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._fast import fclip
from .complexity import C as COMPLEXITY


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ==========================================================================
# Muscle spindles and Golgi tendon organs
# ==========================================================================
class SpindleBank:
    """Two antagonist muscles per joint, each with its own spindles and GTOs."""

    def __init__(self, n_joints: int, moment_arm: np.ndarray, torque_limit: np.ndarray,
                 seed: int):
        C = COMPLEXITY
        self.nj = n_joints
        self.ns = int(C.spindles_per_muscle)
        self.ng = int(C.gto_per_muscle)
        self.ma = moment_arm
        self.tlim = np.maximum(torque_limit, 1.0)
        rng = np.random.default_rng(seed)
        shp = (n_joints, 2, self.ns)
        # primary (Ia): dynamic, high gain; secondary (II): static
        self.theta = rng.uniform(-0.35, 0.35, shp)           # length threshold
        self.gain_len = rng.lognormal(0.0, 0.35, shp) * 4.0
        self.gain_vel = rng.lognormal(0.0, 0.5, shp) * 1.2
        self.tau_adapt = rng.uniform(0.15, 1.2, shp)
        self.gamma_sens = rng.uniform(0.3, 1.0, shp)          # fusimotor sensitivity
        self.adapt_ia = np.zeros(shp)
        gshp = (n_joints, 2, self.ng)
        self.gto_thr = rng.uniform(0.02, 0.5, gshp)
        self.gto_gain = rng.lognormal(0.0, 0.3, gshp) * 3.0
        self.size = n_joints * 2 * (2 * self.ns + self.ng)

    def sense(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, dt: float,
              fusimotor: float = 0.4) -> np.ndarray:
        # agonist stretches with +q, antagonist with -q
        sign = np.array([1.0, -1.0])
        length = sign[None, :] * (self.ma * q)[:, None]                 # (nj, 2)
        vel = sign[None, :] * (self.ma * qd)[:, None]
        L = length[:, :, None]
        V = vel[:, :, None]
        gam = 0.5 + fusimotor * self.gamma_sens                          # (nj,2,ns)
        drive_ia = self.gain_len * (L - self.theta) * gam + self.gain_vel * V * 6.0
        a = dt / (dt + self.tau_adapt)
        self.adapt_ia += a * (drive_ia - self.adapt_ia)
        ia = _sig(drive_ia - 0.45 * self.adapt_ia)
        ii = _sig(self.gain_len * (L - self.theta) * 1.3)
        # force: the muscle that is shortening/producing torque loads its tendon
        f = (np.abs(tau) / self.tlim)
        agon = np.where(tau >= 0, f, 0.12 * f)
        antag = np.where(tau < 0, f, 0.12 * f)
        force = np.stack([agon, antag], axis=1)[:, :, None]               # (nj,2,1)
        gto = np.tanh(self.gto_gain * np.maximum(force - self.gto_thr, 0.0))
        return np.concatenate([ia.ravel(), ii.ravel(), gto.ravel()])


# ==========================================================================
# Vestibular hair cells
# ==========================================================================
class VestibularBank:
    """Canal afferents (push-pull between the ears) and otolith hair cells."""

    def __init__(self, seed: int):
        C = COMPLEXITY
        rng = np.random.default_rng(seed)
        self.nc = int(C.canal_afferents)
        self.no = int(C.otolith_hair_cells)
        # canals: axis (3) x ear (2) x afferents.  The ear sets the sign of the
        # response to rotation about that axis (push-pull).
        shp = (3, 2, self.nc)
        self.irregular = rng.random(shp) < 0.3           # irregular: phasic, high gain
        self.c_gain = np.where(self.irregular, rng.uniform(1.5, 3.0, shp),
                               rng.uniform(0.5, 1.2, shp))
        self.c_tau = np.where(self.irregular, rng.uniform(0.05, 0.3, shp),
                              rng.uniform(1.0, 6.0, shp))
        self.c_rest = rng.uniform(40, 110, shp)
        self.c_state = np.zeros(shp)
        self.c_sign = np.array([1.0, -1.0])[None, :, None]
        # otoliths: utricle (horizontal plane) and saccule (sagittal-vertical)
        oshp = (2, 2, self.no)                            # organ, ear, cell
        phi = rng.uniform(0, 2 * np.pi, oshp)
        self.o_dir = np.stack([np.cos(phi), np.sin(phi)], axis=-1)       # (2,2,no,2)
        self.o_striola = rng.random(oshp) < 0.25
        self.o_gain = np.where(self.o_striola, rng.uniform(1.5, 3.0, oshp),
                               rng.uniform(0.6, 1.3, oshp))
        self.o_tau = np.where(self.o_striola, rng.uniform(0.1, 0.6, oshp),
                              rng.uniform(1.0, 5.0, oshp))
        self.o_state = np.zeros(oshp)
        self.o_rest = rng.uniform(30, 90, oshp)
        self.size = 3 * 2 * self.nc + 2 * 2 * self.no

    def sense(self, omega: np.ndarray, spec_force: np.ndarray, dt: float) -> np.ndarray:
        """omega: head angular velocity (rad/s); spec_force: otolith input (m/s^2)."""
        w = np.asarray(omega, float)[:, None, None] * self.c_sign             # (3,2,1)
        a = dt / (dt + self.c_tau)
        self.c_state += a * (w - self.c_state)
        drive = self.c_gain * (w - np.where(self.irregular, 0.0, self.c_state)) * 6.0
        canal = fclip(self.c_rest * (1.0 + np.tanh(drive)), 0.0, 400.0)
        # utricle: (x, y) of the head; saccule: (y, z)
        sf = np.asarray(spec_force, float)
        plane = np.stack([sf[[0, 1]], sf[[1, 2]]])                            # (2,2)
        proj = np.einsum("oekd,od->oek", self.o_dir, plane)
        a2 = dt / (dt + self.o_tau)
        self.o_state += a2 * (proj - self.o_state)
        d2 = self.o_gain * (proj - np.where(self.o_striola, self.o_state, 0.0) * 0.5) * 0.35
        oto = fclip(self.o_rest * (1.0 + np.tanh(d2)), 0.0, 350.0)
        return np.concatenate([canal.ravel(), oto.ravel()])


# ==========================================================================
# Cochlea: two ears, a place code, and where sounds come from
# ==========================================================================
@dataclass
class SoundEvent:
    pos: np.ndarray
    amp: float
    t0: float
    dur: float
    slope: float = 1.0       # spectral tilt: >1 = more low frequency
    kind: str = "contact"


class Cochlea:
    """Left and right hair-cell arrays driven by sound sources in space."""

    def __init__(self, base_bands: int, f_min: float, f_max: float, seed: int):
        self.nb = int(COMPLEXITY.cochlear_bands)
        self.centres = np.geomspace(f_min, f_max, self.nb)
        rng = np.random.default_rng(seed)
        self.rest = rng.uniform(5, 60, (2, self.nb))
        self.tau_ad = rng.uniform(0.02, 0.6, (2, self.nb))
        self.adapt = np.zeros((2, self.nb))
        self.level = np.zeros((2, self.nb))
        self.base_bands = base_bands
        self.events: list[SoundEvent] = []
        self.ear_offset = 0.085
        self.size = 2 * self.nb + 8
        self.last_itd = 0.0
        self.last_ild = 0.0

    def add_event(self, pos, amp: float, t0: float, dur: float = 0.25,
                  slope: float = 1.0, kind: str = "sound") -> None:
        self.events.append(SoundEvent(np.asarray(pos, float), float(amp), t0, dur,
                                      slope, kind))
        if len(self.events) > 64:
            self.events = self.events[-64:]

    def sense(self, t: float, dt: float, head_pos: np.ndarray, head_R: np.ndarray,
              base_drive: np.ndarray, contact_pos: np.ndarray, contact_amp: np.ndarray,
              self_voice: float = 0.0) -> np.ndarray:
        left_ear = head_pos + head_R[:, 0] * self.ear_offset
        right_ear = head_pos - head_R[:, 0] * self.ear_offset
        ears = (left_ear, right_ear)
        lat_axis = head_R[:, 0]                            # points to the person's left
        amp = np.zeros(2)
        itd_num = 0.0
        itd_den = 1e-9
        tilt_lo = 1.0 / (1.0 + self.centres / 1500.0)
        spec = np.zeros((2, self.nb))
        # --- sources ---------------------------------------------------------
        srcs = [(e.pos, e.amp * (1.0 if (t - e.t0) < e.dur else 0.0), e.slope)
                for e in self.events if t - e.t0 < e.dur + 1.0]
        self.events = [e for e in self.events if t - e.t0 < e.dur + 1.0]
        for k in range(len(contact_amp)):
            srcs.append((contact_pos[k], float(contact_amp[k]), 1.0))
        # the person's own voice is at the mouth, close to both ears
        if self_voice > 1e-4:
            mouth = head_pos + head_R[:, 1] * 0.03 - head_R[:, 2] * 0.06
            srcs.append((mouth, self_voice, 1.6))
        for pos, a, slope in srcs:
            if a <= 1e-6:
                continue
            for e_i, ear in enumerate(ears):
                v = pos - ear
                d = float(np.linalg.norm(v)) + 0.05
                # head shadow: sources on the far side are attenuated at high f
                side = float(np.dot(v / d, lat_axis)) * (1.0 if e_i == 0 else -1.0)
                shadow = 0.5 * (1.0 + np.tanh(2.0 * side))            # 1 near, 0 far
                g = a / (d * d + 0.05)
                hf = 0.25 + 0.75 * shadow
                spec[e_i] += g * tilt_lo ** slope * (1.0 - (1.0 - hf) * (self.centres > 1500))
                amp[e_i] += g * (0.55 + 0.45 * shadow)
            # arrival-time difference (ms), positive when the left ear hears first
            dl = float(np.linalg.norm(pos - ears[0]))
            dr = float(np.linalg.norm(pos - ears[1]))
            w = a / (dl * dl + 0.05) + a / (dr * dr + 0.05)
            itd_num += (dr - dl) / 343.0 * 1000.0 * w
            itd_den += w
        # ambient: the synthesized body noise from the base auditory model,
        # resampled onto the denser place axis
        if base_drive.size:
            xb = np.linspace(0, 1, base_drive.size)
            xn = np.linspace(0, 1, self.nb)
            amb = np.interp(xn, xb, base_drive)
            spec += amb[None, :] * 0.5
        # --- hair-cell adaptation -------------------------------------------
        self.level += 0.5 * (spec - self.level)
        a = dt / (dt + self.tau_ad)
        self.adapt += a * (self.level - self.adapt)
        drive = np.tanh(2.2 * (self.level - 0.6 * self.adapt))
        rates = fclip(self.rest + 280.0 * np.maximum(drive, 0.0), 0.0, 400.0)
        self.last_itd = itd_num / itd_den
        loud = np.log1p(amp * 20.0)
        self.last_ild = float(loud[0] - loud[1])
        az = float(np.arctan2(self.last_itd, 0.7)) if (amp.sum() > 1e-4) else 0.0
        extra = np.array([loud[0], loud[1], self.last_itd, self.last_ild, az,
                          float(len(srcs)), float(rates[0].mean()), float(rates[1].mean())])
        return np.concatenate([rates.ravel(), extra])


# ==========================================================================
# Olfaction and taste
# ==========================================================================
class OlfactoryBank:
    """Hundreds of receptor types with broad, overlapping odorant tuning."""

    def __init__(self, odor_dim: int, seed: int):
        C = COMPLEXITY
        self.n = int(C.olfactory_receptors)
        rng = np.random.default_rng(seed)
        self.odor_dim = odor_dim
        # broad tuning: each receptor responds to ~3 odorant dimensions
        W = np.zeros((self.n, odor_dim))
        for i in range(self.n):
            k = rng.choice(odor_dim, size=min(3, odor_dim), replace=False)
            W[i, k] = rng.uniform(0.3, 1.0, size=k.size)
        self.W = W
        self.thresh = rng.uniform(0.02, 0.25, self.n)
        self.tau_ad = rng.uniform(2.0, 9.0, self.n)
        self.adapt = np.zeros(self.n)
        self.glom = np.zeros(self.n)
        # glomerular lateral inhibition between neighbours
        self.phase = 0.0
        self.size = 2 * self.n + 6

    def sense(self, conc: np.ndarray, dt: float, resp_rate: float = 13.0,
              novelty: float = 0.0) -> np.ndarray:
        # sniffing: respiration gates airflow; novelty drives bursts of sniffs
        f = resp_rate / 60.0 * (1.0 + 3.0 * np.tanh(novelty * 4.0))
        self.phase = (self.phase + 2 * np.pi * f * dt) % (2 * np.pi)
        flow = 0.25 + 0.75 * max(np.sin(self.phase), 0.0)
        c = np.zeros(self.odor_dim)
        m = min(conc.size, self.odor_dim)
        c[:m] = conc[:m]
        drive = (self.W @ c) * flow
        a = dt / (dt + self.tau_ad)
        self.adapt += a * (drive - self.adapt)
        resp = np.tanh(np.maximum(drive - 0.7 * self.adapt - self.thresh, 0.0) * 4.0)
        # lateral inhibition on a ring of glomeruli
        lat = 0.5 * (np.roll(resp, 1) + np.roll(resp, -1))
        self.glom = np.maximum(resp - 0.35 * lat, 0.0)
        extra = np.array([flow, self.phase, float(resp.sum()), float(self.glom.sum()),
                          float((self.glom > 0.1).sum()), f])
        return np.concatenate([resp, self.glom, extra])


class GustatoryBank:
    """Taste-cell types tuned to mixtures of the five basic qualities (+ fat, etc.)."""

    def __init__(self, seed: int):
        self.n = int(COMPLEXITY.taste_cell_types)
        rng = np.random.default_rng(seed)
        # sweet, salty, sour, bitter, umami: each cell leans on one or two
        W = np.zeros((self.n, 5))
        for i in range(self.n):
            main = i % 5 if i < 5 else rng.integers(0, 5)
            W[i, main] = rng.uniform(0.6, 1.0)
            if rng.random() < 0.4:
                W[i, rng.integers(0, 5)] += rng.uniform(0.1, 0.4)
        self.W = W
        self.thr = rng.uniform(0.02, 0.2, self.n)
        self.adapt = np.zeros(self.n)
        self.saliva = 0.2
        self.size = self.n + 2

    def sense(self, taste: np.ndarray, dt: float) -> np.ndarray:
        t = np.zeros(5)
        t[:min(5, taste.size)] = taste[:5]
        drive = self.W @ t
        a = dt / (dt + 6.0)
        self.adapt += a * (drive - self.adapt)
        resp = np.tanh(np.maximum(drive - 0.5 * self.adapt - self.thr, 0.0) * 3.0)
        self.saliva += dt * (0.4 * float(t.sum() > 0.05) - 0.12 * (self.saliva - 0.2))
        return np.concatenate([resp, [self.saliva, float(resp.sum())]])


# ==========================================================================
# Retina
# ==========================================================================
class RetinaBank:
    """Cones (L, M, S), rods, ON / OFF bipolar cells and magnocellular ganglion
    cells computed from the rendered egocentric image."""

    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.prev = None
        self.size = h * w * 7

    @staticmethod
    def _blur(x: np.ndarray) -> np.ndarray:
        p = np.pad(x, 1, mode="edge")
        return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 1:-1]
                + p[1:-1, 2:] + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0

    def sense(self, img: np.ndarray, pupil_mm: float = 4.0) -> np.ndarray:
        """img: (h, w, 3) float RGB in [0, 1]."""
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        gain = (4.0 / max(pupil_mm, 1.5)) ** 2 * 0.5     # a larger pupil admits more light
        L = np.clip(0.60 * R + 0.35 * G + 0.05 * B, 0, 1) * gain
        M = np.clip(0.25 * R + 0.65 * G + 0.10 * B, 0, 1) * gain
        S = np.clip(0.05 * R + 0.15 * G + 0.80 * B, 0, 1) * gain
        lum = (L + M) * 0.5
        rod = np.clip(1.4 * lum ** 0.6 * (1.0 - lum), 0, 1)      # rods saturate in light
        surround = self._blur(lum)
        on = np.clip(lum - surround, 0, 1) * 4.0
        off = np.clip(surround - lum, 0, 1) * 4.0
        if self.prev is None or self.prev.shape != lum.shape:
            magno = np.zeros_like(lum)
        else:
            magno = np.abs(lum - self.prev) * 6.0
        self.prev = lum
        return np.stack([L, M, S, rod, on, off, magno]).astype(np.float32).ravel()
