"""
The brain-side inner world.

=================  ====================================================  ===========
system             what it is                                             states
=================  ====================================================  ===========
CircadianClock     ~200 coupled oscillators (an SCN), peripheral clocks,  ~900
                   a sleep-pressure process; entrained by light and meals
NeuralMass         ~12,000 rate units in 20 populations (sensory and      ~25,000
                   motor cortex, insula, PFC, ACC, amygdala, hippocampus,
                   striatum, thalamus, brainstem arousal nuclei ...) with
                   structured sparse connectivity, adaptation, noise and
                   neuromodulator gain control
EpisodicMemory     a ring buffer of a few thousand episodes (sensory /     ~400,000
                   affective / behavioural snapshots) recalled by
                   similarity: familiarity, and the mood the past carries
Conditioning       associative strengths between stimuli (body regions,    ~130
                   objects, sounds, smells) and valence, learned online
InteroModel        a one-step predictor of the interoceptive state with    ~300
                   learned precision: interoceptive surprise
=================  ====================================================  ===========

The neural mass is *not* a model of any particular circuit's physiology.  It is a
large structured recurrent network whose populations are driven by what the body
senses and feels and whose summary activity (an "amygdala" threat tone, a
"default-mode" mind-wandering tone, an arousal tone, a "control" tone ...) is fed
back into affect, attention and behaviour.  Its value here is high-dimensional,
spontaneous, history-dependent internal activity that the rest of the person
actually listens to.
"""

from __future__ import annotations

import numpy as np

from .complexity import C as COMPLEXITY


# ==========================================================================
class CircadianClock:
    def __init__(self, seed: int, start_hour: float = 10.0, time_scale: float = 1.0):
        n = int(COMPLEXITY.circadian_oscillators)
        rng = np.random.default_rng(seed)
        self.n = n
        self.omega = 2 * np.pi / 86400.0 * (1.0 + rng.normal(0, 0.03, n))   # rad / s
        self.theta = 2 * np.pi * (start_hour / 24.0) + rng.normal(0, 0.25, n)
        self.K = 3.0 / 86400.0 * 2 * np.pi * 8.0
        self.time_scale = float(time_scale)
        self.periph = np.full(24, 2 * np.pi * (start_hour / 24.0))
        self.periph_lag = rng.uniform(0.2, 1.5, 24)
        self.sleep_pressure = 0.3
        self.t_awake = 0.0
        self.R = 0.9
        self.psi = float(2 * np.pi * start_hour / 24.0)
        self.light_ema = 0.6
        self.size = 2 * n + 2 * 24 + 4

    def update(self, dt, light, meal, asleep=False):
        dt_eff = dt * self.time_scale
        self.light_ema += (dt / 120.0) * (light - self.light_ema)
        z = np.exp(1j * self.theta).mean()
        self.R, self.psi = float(np.abs(z)), float(np.angle(z))
        # light advances / delays the clock depending on phase (a crude PRC)
        prc = -np.sin(self.theta - 2 * np.pi * 3.0 / 24.0) * 0.5
        dth = self.omega + self.K * self.R * np.sin(self.psi - self.theta) \
            + 2 * np.pi / 86400.0 * 20.0 * (self.light_ema - 0.4) * prc
        self.theta += dt_eff * dth
        dp = self.omega.mean() + (2 * np.pi / 86400.0) * 6.0 * meal * np.sin(self.psi - self.periph) \
            + (self.psi - self.periph) * 1.5e-4 / self.periph_lag
        self.periph += dt_eff * dp
        # sleep homeostat: pressure builds while awake, discharges in sleep
        if asleep:
            self.sleep_pressure += dt_eff * (-self.sleep_pressure / 7200.0)
            self.t_awake = 0.0
        else:
            self.sleep_pressure += dt_eff * ((1.0 - self.sleep_pressure) / 64800.0)
            self.t_awake += dt_eff
        self.sleep_pressure = float(np.clip(self.sleep_pressure, 0, 1))

    @property
    def hour(self) -> float:
        return float((self.psi / (2 * np.pi) * 24.0) % 24.0)

    @property
    def alertness_drive(self) -> float:
        c = 0.5 + 0.5 * np.cos(self.psi - 2 * np.pi * 16.0 / 24.0)      # peaks late afternoon
        return float(np.clip(0.55 * c * self.R + 0.45 * (1.0 - self.sleep_pressure), 0, 1))

    def state(self):
        return np.concatenate([np.sin(self.theta), np.cos(self.theta), np.sin(self.periph),
                               np.cos(self.periph), [self.R, self.psi, self.sleep_pressure, self.t_awake / 3600]])


# ==========================================================================
POPULATIONS = [
    # name, share of units, fraction excitatory
    ("visual_ctx", 0.05, 0.8), ("auditory_ctx", 0.03, 0.8), ("somatosensory_ctx", 0.06, 0.8),
    ("insula", 0.04, 0.8), ("motor_ctx", 0.06, 0.8), ("premotor_ctx", 0.04, 0.8),
    ("pfc_dorsolateral", 0.07, 0.8), ("pfc_ventromedial", 0.05, 0.8), ("acc", 0.04, 0.8),
    ("parietal_ctx", 0.05, 0.8), ("temporal_ctx", 0.05, 0.8), ("default_mode", 0.05, 0.8),
    ("hippocampus", 0.05, 0.8), ("amygdala", 0.04, 0.75), ("striatum", 0.06, 0.2),
    ("thalamus", 0.06, 0.85), ("hypothalamus", 0.03, 0.7), ("brainstem_arousal", 0.03, 0.7),
    ("periaqueductal_gray", 0.02, 0.7), ("cerebellum", 0.12, 0.85),
]
POP_NAMES = [p[0] for p in POPULATIONS]
PI = {n: i for i, n in enumerate(POP_NAMES)}
# directed projections (src, dst, strength)
EDGES = [
    ("thalamus", "visual_ctx", 1.0), ("thalamus", "auditory_ctx", 1.0), ("thalamus", "somatosensory_ctx", 1.0),
    ("thalamus", "insula", 0.6), ("thalamus", "pfc_dorsolateral", 0.5), ("thalamus", "motor_ctx", 0.5),
    ("visual_ctx", "parietal_ctx", 0.9), ("visual_ctx", "temporal_ctx", 0.9), ("auditory_ctx", "temporal_ctx", 0.9),
    ("somatosensory_ctx", "parietal_ctx", 0.8), ("somatosensory_ctx", "insula", 0.8), ("somatosensory_ctx", "motor_ctx", 0.6),
    ("parietal_ctx", "premotor_ctx", 0.9), ("premotor_ctx", "motor_ctx", 0.9), ("parietal_ctx", "pfc_dorsolateral", 0.7),
    ("temporal_ctx", "pfc_ventromedial", 0.6), ("temporal_ctx", "hippocampus", 0.8), ("temporal_ctx", "amygdala", 0.7),
    ("insula", "acc", 0.9), ("insula", "amygdala", 0.7), ("insula", "hypothalamus", 0.6),
    ("acc", "pfc_dorsolateral", 0.7), ("acc", "pfc_ventromedial", 0.6), ("acc", "striatum", 0.5),
    ("pfc_dorsolateral", "motor_ctx", 0.5), ("pfc_dorsolateral", "premotor_ctx", 0.6), ("pfc_dorsolateral", "amygdala", -0.7),
    ("pfc_ventromedial", "amygdala", -0.8), ("pfc_ventromedial", "default_mode", 0.8), ("default_mode", "hippocampus", 0.8),
    ("default_mode", "pfc_ventromedial", 0.7), ("hippocampus", "default_mode", 0.7), ("hippocampus", "pfc_ventromedial", 0.5),
    ("amygdala", "hypothalamus", 0.8), ("amygdala", "periaqueductal_gray", 0.7), ("amygdala", "brainstem_arousal", 0.7),
    ("amygdala", "insula", 0.5), ("hypothalamus", "brainstem_arousal", 0.6), ("brainstem_arousal", "thalamus", 0.7),
    ("brainstem_arousal", "pfc_dorsolateral", 0.4), ("striatum", "thalamus", -0.6), ("striatum", "motor_ctx", 0.3),
    ("motor_ctx", "striatum", 0.6), ("premotor_ctx", "striatum", 0.5), ("motor_ctx", "cerebellum", 0.8),
    ("cerebellum", "thalamus", 0.6), ("periaqueductal_gray", "brainstem_arousal", 0.4),
    ("somatosensory_ctx", "periaqueductal_gray", 0.3), ("pfc_dorsolateral", "default_mode", -0.6),
    ("default_mode", "pfc_dorsolateral", -0.4),
]


class NeuralMass:
    """Structured sparse recurrent rate network with neuromodulated gain."""

    N_SENSORY_INPUTS = 160

    def __init__(self, seed: int):
        C = COMPLEXITY
        N = int(C.neural_units)
        K = int(C.neural_indegree)
        rng = np.random.default_rng(seed)
        share = np.array([p[1] for p in POPULATIONS])
        share /= share.sum()
        counts = np.maximum((share * N).astype(int), 4)
        counts[-1] += N - counts.sum()
        self.N = int(counts.sum())
        self.K = K
        self.pop_of = np.repeat(np.arange(len(POPULATIONS)), counts)
        self.start = np.concatenate([[0], np.cumsum(counts)])
        # excitatory or inhibitory units
        efrac = np.array([p[2] for p in POPULATIONS])
        self.exc = rng.random(self.N) < efrac[self.pop_of]
        # presynaptic partners
        pre = np.empty((self.N, K), dtype=np.int32)
        w = np.empty((self.N, K), dtype=np.float32)
        incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(POPULATIONS))}
        for s, d, st in EDGES:
            incoming[PI[d]].append((PI[s], st))
        for i in range(len(POPULATIONS)):
            lo, hi = self.start[i], self.start[i + 1]
            n_i = hi - lo
            srcs = incoming[i]
            n_local = int(round(0.55 * K)) if srcs else K
            n_remote = K - n_local
            pre[lo:hi, :n_local] = rng.integers(lo, hi, size=(n_i, n_local))
            w[lo:hi, :n_local] = rng.uniform(0.3, 1.0, (n_i, n_local))
            if n_remote:
                strengths = np.array([abs(st) for _, st in srcs])
                pick = rng.choice(len(srcs), size=(n_i, n_remote), p=strengths / strengths.sum())
                for j in range(len(srcs)):
                    mask = pick == j
                    sp_, st = srcs[j]
                    cnt = int(mask.sum())
                    if cnt == 0:
                        continue
                    rows, cols = np.where(mask)
                    pre[lo + rows, n_local + cols] = rng.integers(self.start[sp_], self.start[sp_ + 1], size=cnt)
                    w[lo + rows, n_local + cols] = rng.uniform(0.3, 1.0, cnt) * np.sign(st) * abs(st)
        # Dale's law for local connections; remote sign comes from the projection sign
        sign_pre = np.where(self.exc[pre], 1.0, -1.8).astype(np.float32)
        remote_neg = w < 0
        w = np.abs(w) * np.where(remote_neg, -1.0, 1.0).astype(np.float32)
        w = np.where(remote_neg, w, w * sign_pre)
        self.pre = pre
        self.w = (w * (0.9 / np.sqrt(K)) * 1.6).astype(np.float32)
        self.tau = rng.uniform(0.05, 0.15, self.N).astype(np.float32)
        self.tau_a = rng.uniform(0.6, 2.5, self.N).astype(np.float32)
        self.b_adapt = rng.uniform(0.1, 0.8, self.N).astype(np.float32)
        self.x = (0.05 * rng.standard_normal(self.N)).astype(np.float32)
        self.a = np.zeros(self.N, np.float32)
        self.bias = (rng.normal(0, 0.1, self.N)).astype(np.float32)
        # sensory input projection: 160 features -> sensory & interoceptive populations
        self.n_in = self.N_SENSORY_INPUTS
        sens_pops = [PI["visual_ctx"], PI["auditory_ctx"], PI["somatosensory_ctx"], PI["insula"],
                     PI["hypothalamus"], PI["brainstem_arousal"], PI["periaqueductal_gray"]]
        self.W_in = np.zeros((self.N, self.n_in), np.float32)
        for p in sens_pops:
            lo, hi = self.start[p], self.start[p + 1]
            m = rng.random((hi - lo, self.n_in)) < 0.08
            self.W_in[lo:hi] = m * rng.normal(0, 1.0, (hi - lo, self.n_in)).astype(np.float32)
        self.gain = np.ones(len(POPULATIONS), np.float32)
        self.drive = np.zeros(len(POPULATIONS), np.float32)
        self.base = np.zeros(len(POPULATIONS), np.float32)          # EMA baseline of pop means
        self.pop_mean = np.zeros(len(POPULATIONS), np.float32)
        self.rng = rng
        self.n_updates = 0
        self.size = 2 * self.N

    def update(self, dt, u: np.ndarray, drive: np.ndarray, gain: np.ndarray, noise: float = 0.35):
        """u: sensory feature vector (n_in,), drive: per-population tonic input,
        gain: per-population multiplicative gain."""
        g = gain[self.pop_of]
        inp = self.W_in @ u.astype(np.float32) + drive[self.pop_of] + self.bias
        rec = np.einsum("ik,ik->i", self.w, self.x[self.pre])
        z = g * (rec + inp) - self.a + noise * np.sqrt(dt / 0.02) * self.rng.standard_normal(self.N).astype(np.float32) * 0.5
        self.x += (dt / self.tau) * (-self.x + np.tanh(z))
        self.a += (dt / self.tau_a) * (self.b_adapt * self.x - self.a)
        self.pop_mean = np.add.reduceat(self.x, self.start[:-1]) / np.diff(self.start)
        self.base += 0.002 * (self.pop_mean - self.base)
        self.n_updates += 1

    def deviation(self, name: str) -> float:
        i = PI[name]
        return float(self.pop_mean[i] - self.base[i])

    def activity(self, name: str) -> float:
        return float(self.pop_mean[PI[name]])

    def state(self):
        return np.concatenate([self.x, self.a])


# ==========================================================================
class EpisodicMemory:
    def __init__(self, seed: int):
        C = COMPLEXITY
        self.cap = int(C.episodic_capacity)
        self.dim = int(C.episodic_dim)
        self.M = np.zeros((self.cap, self.dim), np.float32)
        self.val = np.zeros(self.cap, np.float32)
        self.sal = np.zeros(self.cap, np.float32)
        self.t_ep = np.zeros(self.cap, np.float32)
        self.n = 0
        self.ptr = 0
        rng = np.random.default_rng(seed)
        self.R = (rng.standard_normal((256, self.dim)) / np.sqrt(256)).astype(np.float32)
        self.familiarity = 0.0
        self.recalled_valence = 0.0
        self.recalled_n = 0
        self.size = self.cap * self.dim + 3 * self.cap

    def embed(self, feat: np.ndarray) -> np.ndarray:
        f = np.zeros(256, np.float32)
        m = min(len(feat), 256)
        f[:m] = feat[:m]
        v = f @ self.R
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 1e-8 else v

    def store(self, feat: np.ndarray, valence: float, salience: float, t: float) -> None:
        e = self.embed(feat)
        # near-duplicate of a recent memory: strengthen it instead of storing again
        if self.n > 0:
            sims = self.M[:self.n] @ e
            j = int(np.argmax(sims))
            if sims[j] > 0.985:
                self.val[j] = 0.8 * self.val[j] + 0.2 * valence
                self.sal[j] = max(self.sal[j], salience)
                return
        self.M[self.ptr] = e
        self.val[self.ptr] = valence
        self.sal[self.ptr] = salience
        self.t_ep[self.ptr] = t
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def recall(self, feat: np.ndarray) -> tuple[float, float]:
        if self.n == 0:
            self.familiarity, self.recalled_valence = 0.0, 0.0
            return 0.0, 0.0
        e = self.embed(feat)
        sims = self.M[:self.n] @ e
        k = min(6, self.n)
        top = np.argpartition(-sims, k - 1)[:k]
        s = sims[top]
        wts = np.exp(8.0 * (s - s.max()))
        wts /= wts.sum()
        self.recalled_valence = float((wts * self.val[top] * self.sal[top].clip(0.3, 1.5)).sum())
        self.familiarity = float(np.clip(s.max(), 0.0, 1.0))
        self.recalled_n = int((s > 0.8).sum())
        return self.familiarity, self.recalled_valence

    def state(self):
        return self.M.ravel()


# ==========================================================================
class Conditioning:
    """Rescorla-Wagner associations between stimulus features and valence."""

    N_STIM = 64

    def __init__(self):
        self.V = np.zeros(self.N_STIM)
        self.trace = np.zeros(self.N_STIM)
        self.alpha = 0.05
        self.pred = 0.0
        self.rpe = 0.0
        self.size = 2 * self.N_STIM

    def update(self, dt, stimuli: np.ndarray, outcome: float) -> None:
        s = np.zeros(self.N_STIM)
        m = min(len(stimuli), self.N_STIM)
        s[:m] = stimuli[:m]
        self.trace += (dt / 1.5) * (s - self.trace)
        self.pred = float(self.V @ self.trace)
        self.rpe = float(outcome - self.pred)
        self.V += self.alpha * self.rpe * self.trace * min(dt * 10.0, 1.0)
        self.V *= (1.0 - dt / 1800.0)                 # slow extinction

    def state(self):
        return np.concatenate([self.V, self.trace])


# ==========================================================================
class InteroModel:
    """One-step predictor of the interoceptive state with learned precision."""

    def __init__(self, n: int):
        self.n = n
        self.a = np.full(n, 0.995)
        self.b = np.zeros(n)
        self.var = np.full(n, 1e-3)
        self.prev = None
        self.pred = None
        self.surprise = 0.0
        self.err = np.zeros(n)
        self.size = 4 * n

    def update(self, x: np.ndarray) -> None:
        if self.prev is not None:
            e = x - self.pred
            scale = np.maximum(np.abs(x), 1.0)
            en = e / scale
            self.var += 0.01 * (en ** 2 - self.var)
            # LMS update of the AR(1) coefficients
            xs = self.prev / scale
            self.a += 0.002 * en * xs
            self.b += 0.002 * en
            prec = 1.0 / np.maximum(self.var, 1e-5)
            self.surprise = float(np.sqrt(np.mean(prec * en ** 2) / 50.0))
            self.err = en
        self.pred = self.a * x + self.b
        self.prev = x.copy()

    def state(self):
        return np.concatenate([self.a, self.b, self.var, self.err])
