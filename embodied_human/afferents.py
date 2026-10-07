"""
Peripheral nerve: receptor potentials -> afferent traffic.

Three things happen here that a naive sensor pipeline omits, and each of them
is load-bearing for the feeling of embodiment:

1. **Conduction delay.**  Aδ and C fibres are slow (60 ms / 250 ms).  The brain
   therefore acts on a *stale* image of the body, which is precisely why a
   forward model is necessary.

2. **Adaptation and rate coding.**  Receptors report *changes* far better than
   steady states.  A shirt you put on an hour ago is no longer felt.

3. **Reafference cancellation.**  When you make a movement, a copy of the
   motor command (corollary discharge) is used to predict the self-generated
   sensory inflow and subtract it, so that touching your own arm does not feel
   like being touched by someone else.

Fibre populations
-----------------
======================  ================  ==========================
population              delay             source
======================  ================  ==========================
Aβ SA-I / SA-II         18 ms             sustained force, stretch
Aβ FA-I / FA-II         18 ms             flutter, vibration
Aβ proprioceptive Ia/II 18 ms             muscle spindles
Aβ Golgi Ib             22 ms             tendon force
C-tactile (CT)          250 ms            affective stroking
Aδ mechanonociceptor    60 ms             pricking pain, cold
C nociceptor / itch     250 ms            burning pain, itch
efferent (motor)        30 ms             descending command
======================  ================  ==========================
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from .config import SimConfig
from .receptors import CH, N_PROPRIO_CH, ReceptorFrame

# Fibre populations reported as population firing rates (Hz)
FIBRE_POPULATIONS = (
    "Abeta_SA1", "Abeta_SA2", "Abeta_FA1", "Abeta_FA2",
    "Abeta_Ia", "Abeta_II", "Abeta_Ib",
    "Ctactile", "Adelta_noci", "Adelta_cold",
    "C_noci", "C_itch", "C_warm",
)


@dataclass
class AfferentFrame:
    """Delayed, rate-coded afferent traffic."""
    t: float = 0.0

    # per-taxel firing rates (Hz)
    rates_tactile: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    rates_noci: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rates_ct: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rates_itch: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # per-joint proprioceptive rates (Hz)
    rates_proprio: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    # population totals (Hz, summed over fibres then log-compressed)
    population: np.ndarray = field(
        default_factory=lambda: np.zeros(len(FIBRE_POPULATIONS)))
    population_raw: np.ndarray = field(
        default_factory=lambda: np.zeros(len(FIBRE_POPULATIONS)))

    # delay-free reference (what a "perfect" sensor would have reported)
    immediate_tactile: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    # corollary discharge / efference copy, delayed like a real motor command
    efference: np.ndarray = field(default_factory=lambda: np.zeros(0))
    predicted_reafference: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    reafference_residual: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    # aggregate somatosensory percept
    touch_rate_total: float = 0.0
    pain_rate_total: float = 0.0
    itch_rate_total: float = 0.0
    affective_rate_total: float = 0.0
    proprio_rate_total: float = 0.0
    total_rate: float = 0.0
    sensation_age: float = 0.0

    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.population,
            [self.touch_rate_total, self.pain_rate_total, self.itch_rate_total,
             self.affective_rate_total, self.proprio_rate_total, self.total_rate],
        ])


class DelayLine:
    """Fixed-length ring buffer implementing axonal conduction delay."""

    def __init__(self, n_samples: int, shape: tuple, dtype=float):
        self.n = max(int(n_samples), 1)
        self.buf = np.zeros((self.n,) + tuple(shape), dtype=dtype)
        self.i = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.buf[self.i] = x
        self.i = (self.i + 1) % self.n
        return self.buf[self.i].copy()

    def reset(self) -> None:
        self.buf[:] = 0.0
        self.i = 0


class AfferentSystem:
    """Applies delays, adaptation, noise and reafference cancellation."""

    def __init__(self, cfg: SimConfig, meta, n_taxels: int, n_joints: int):
        self.cfg = cfg
        self.A = cfg.afferent
        self.n_taxels = n_taxels
        self.n_joints = n_joints
        rate = cfg.rates.afferent
        dt = 1.0 / rate

        def samples(sec: float) -> int:
            return max(int(round(sec * rate)), 1)

        # --- conduction delays -----------------------------------------
        self.dly_abeta = DelayLine(samples(self.A.delay_abeta), (n_taxels, 4))
        self.dly_ct = DelayLine(samples(self.A.delay_c), (n_taxels,))
        self.dly_adelta = DelayLine(samples(self.A.delay_adelta), (n_taxels, 2))
        self.dly_c = DelayLine(samples(self.A.delay_c), (n_taxels, 3))
        self.dly_proprio = DelayLine(samples(self.A.delay_abeta), (n_joints, 4))
        self.dly_efferent = DelayLine(samples(self.A.delay_efferent),
                                      (meta.n_actuators,))

        # --- rate coding state -----------------------------------------
        self.rate_tactile = np.zeros((n_taxels, 4))
        self.rate_noci = np.zeros((n_taxels, 3))
        self.rate_ct = np.zeros(n_taxels)
        self.rate_itch = np.zeros(n_taxels)
        self.rate_proprio = np.zeros((n_joints, 4))
        self._rate_alpha = dt / (dt + self.A.rate_tau)

        # --- reafference model -----------------------------------------
        # learned linear map: efference copy -> expected self-touch
        self.reaff_w = np.zeros((n_taxels, 4, meta.n_actuators))
        self.reaff_ridge = 200.0
        self._reaff_learned = False
        self.reafference_r2 = 0.0
        self._reaff_buffer: deque = deque(maxlen=400)
        self.rng = np.random.default_rng(cfg.seed + 31)

        # adaptation is already handled at the receptor level; here we add
        # a slower habituation of the afferent population
        self.habituation = np.ones(n_taxels)

    # ------------------------------------------------------------------
    def _rates(self, drive: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """Rate code with a first-order lag, spontaneous activity and noise."""
        A = self.A
        target = A.spontaneous_hz + (A.rate_max - A.spontaneous_hz) * fclip(drive, 0, 1)
        r = prev + self._rate_alpha * (target - prev)
        r = r * (1.0 + self.rng.normal(0, A.noise_sd_frac, r.shape))
        return fclip(r, 0.0, A.rate_max)

    # ------------------------------------------------------------------
    def process(self, frame: ReceptorFrame, state, ctrl: np.ndarray) -> AfferentFrame:
        T = frame.tactile
        P = frame.proprio
        n = self.n_taxels

        # ---- immediate (undelayed) drive ------------------------------
        drive = np.stack([
            T[:, CH["sa1"]], T[:, CH["sa2"]],
            T[:, CH["fa1"]], T[:, CH["fa2"]],
        ], axis=1)
        noci_drive = np.stack([
            T[:, CH["noci_mech"]], T[:, CH["noci_heat"]], T[:, CH["noci_cold"]],
        ], axis=1)
        ct_drive = T[:, CH["ct"]]
        itch_drive = T[:, CH["itch"]]

        # ---- habituation: steady contact fades over ~30 s -------------
        contact = T[:, CH["normal_force"]] > 0.05
        self.habituation += (1.0 / self.cfg.rates.afferent) * (
            (1.0 if False else 0.0) - 0.0)
        tau_hab = 30.0
        target_hab = np.where(contact, 0.35, 1.0)
        self.habituation += (1.0 / self.cfg.rates.afferent) / tau_hab * (
            target_hab - self.habituation)
        drive = drive * self.habituation[:, None]

        # ---- reafference cancellation ---------------------------------
        eff = self.dly_efferent(ctrl)
        predicted = self._predict_reafference(eff)
        residual = drive - predicted
        # keep the raw drive for learning, cancel only for perception
        cancelled = fclip(drive - self.A.reafference_gain * predicted, 0.0, None)

        # ---- rate coding ----------------------------------------------
        self.rate_tactile = self._rates(cancelled, self.rate_tactile)
        self.rate_noci = self._rates(noci_drive, self.rate_noci)
        self.rate_ct = self._rates(ct_drive, self.rate_ct)
        self.rate_itch = self._rates(itch_drive, self.rate_itch)

        pdrive = np.stack([
            fclip(P[:, 3], 0, 1),      # Ia
            fclip(P[:, 4], 0, 1),      # II
            fclip(P[:, 5], 0, 1),      # Ib
            fclip(P[:, 2] * 0.02 + P[:, 12], 0, 1),   # acceleration + effort
        ], axis=1)
        self.rate_proprio = self._rates(pdrive, self.rate_proprio)

        # ---- conduction delays ----------------------------------------
        d_abeta = self.dly_abeta(self.rate_tactile)
        d_ct = self.dly_ct(self.rate_ct)
        d_adelta = self.dly_adelta(self.rate_noci[:, :2])
        # C fibres carry slow heat pain, cold pain and itch
        d_c = self.dly_c(np.stack([self.rate_noci[:, 1], self.rate_noci[:, 2],
                                   self.rate_itch], axis=1))
        d_prop = self.dly_proprio(self.rate_proprio)
        d_itch = d_c[:, 2]

        # ---- population totals ----------------------------------------
        pop_raw = np.array([
            d_abeta[:, 0].sum(), d_abeta[:, 1].sum(),
            d_abeta[:, 2].sum(), d_abeta[:, 3].sum(),
            d_prop[:, 0].sum(), d_prop[:, 1].sum(), d_prop[:, 2].sum(),
            d_ct.sum(),
            d_adelta[:, 0].sum(), d_adelta[:, 1].sum(),
            d_c[:, 0].sum(), d_itch.sum(), d_c[:, 1].sum(),
        ])
        # normalise by population size so numbers stay interpretable, then
        # compress with a logarithm (as a real nervous system does)
        sizes = np.array([n, n, n, n, self.n_joints, self.n_joints, self.n_joints,
                          n, n, n, n, n, n], float)
        pop = np.log1p(pop_raw / np.maximum(sizes, 1) * 10.0)

        touch_total = float(d_abeta.sum() / max(n, 1))
        pain_total = float((d_adelta[:, 0] + d_c[:, 0]).sum() / max(n, 1))
        itch_total = float(d_itch.sum() / max(n, 1))
        ct_total = float(d_ct.sum() / max(n, 1))
        prop_total = float(d_prop.sum() / max(self.n_joints, 1))

        # ---- store for learning --------------------------------------
        self._reaff_buffer.append((eff.copy(), drive.copy()))

        return AfferentFrame(
            t=frame.t,
            rates_tactile=d_abeta,
            rates_noci=np.stack([d_adelta[:, 0], d_c[:, 0], d_adelta[:, 1]], axis=1),
            rates_ct=d_ct,
            rates_itch=d_c[:, 2] if d_c.shape[1] > 2 else self.rate_itch,
            rates_proprio=d_prop,
            population=pop,
            population_raw=pop_raw,
            immediate_tactile=self.rate_tactile.copy(),
            efference=eff,
            predicted_reafference=self.A.reafference_gain * predicted,
            reafference_residual=residual,
            touch_rate_total=touch_total,
            pain_rate_total=pain_total,
            itch_rate_total=itch_total,
            affective_rate_total=ct_total,
            proprio_rate_total=prop_total,
            total_rate=touch_total + pain_total + itch_total + ct_total + prop_total,
            sensation_age=float(self.A.delay_abeta),
        )

    # ------------------------------------------------------------------
    def _predict_reafference(self, eff: np.ndarray) -> np.ndarray:
        """Expected self-generated tactile drive from the efference copy."""
        if not self._reaff_learned:
            return np.zeros_like(self.rate_tactile)
        return np.maximum(self.reaff_w @ eff * 0.02, 0.0) \
            if False else self._matmul_reaff(eff)

    def _matmul_reaff(self, eff: np.ndarray) -> np.ndarray:
        return np.maximum(np.tensordot(self.reaff_w, eff, axes=([2], [0])) * 0.02, 0.0)

    # ------------------------------------------------------------------
    def learn_reafference(self) -> float:
        """Fit the corollary-discharge model on recently buffered data.

        Returns the fraction of self-generated tactile variance explained
        (1.0 = the agent perfectly predicts the touch it causes itself).
        """
        if len(self._reaff_buffer) < 40:
            return 0.0
        E = np.stack([e for e, _ in self._reaff_buffer])      # (S, n_act)
        Y = np.stack([y for _, y in self._reaff_buffer])      # (S, n_tax, 4)
        # ridge regression: W = (E'E + lI)^-1 E'Y  ->  (n_tax, 4, n_act)
        lam = self.reaff_ridge * np.eye(E.shape[1])
        G = E.T @ E + lam
        try:
            W = np.linalg.solve(G, E.T @ Y.reshape(len(E), -1))
        except np.linalg.LinAlgError:
            return 0.0
        self.reaff_w = W.reshape(Y.shape[1], Y.shape[2], E.shape[1])
        self._reaff_learned = True
        pred = np.tensordot(self.reaff_w, E, axes=([2], [1]))   # (n_tax, 4, S)
        pred = np.transpose(pred, (2, 0, 1))                    # (S, n_tax, 4)
        err = float(np.mean((pred - Y) ** 2))
        base = float(np.mean(Y ** 2)) + 1e-9
        return float(fclip(1.0 - err / base, -1.0, 1.0))

    def reset(self) -> None:
        for d in (self.dly_abeta, self.dly_ct, self.dly_adelta, self.dly_c,
                  self.dly_proprio, self.dly_efferent):
            d.reset()
        self.rate_tactile[:] = 0
        self.rate_noci[:] = 0
        self.rate_ct[:] = 0
        self.rate_itch[:] = 0
        self.rate_proprio[:] = 0
        self.habituation[:] = 1.0
        self._reaff_buffer.clear()

    def describe(self) -> dict:
        return {
            "populations": list(FIBRE_POPULATIONS),
            "delays_ms": {
                "Abeta": self.A.delay_abeta * 1e3,
                "Adelta": self.A.delay_adelta * 1e3,
                "C": self.A.delay_c * 1e3,
                "efferent": self.A.delay_efferent * 1e3,
            },
            "rate_range_hz": [self.A.rate_min, self.A.rate_max],
            "reafference_cancellation": True,
        }
