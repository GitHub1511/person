"""
Predictive coding: forward model, inverse model, body schema, precision.

A body is not a set of sensors; it is a **model that predicts its own sensory
consequences**.  Three learned objects implement that here:

**Forward model** ``p(s_{t+1} | s_t, a_t)``
    Learned online by recursive least squares (RLS) with a forgetting factor,
    so it tracks a body that changes (fatigue, growth, injury).

**Inverse model** ``a_t = f(s_t, s*_{t+1})``
    What command would produce the sensory change I want.  This is what makes
    a movement *intended* rather than merely caused.

**Body schema** ``J = d(proprioception)/d(action)``
    A learned Jacobian.  Its row norms say which sensory dimensions are
    *controllable*, which is the operational definition of "this is my body".
    A dimension that is perfectly predicted by efference copy is *mine*; one
    that is not is *external*.  That judgement is the same computation that
    produces the rubber-hand illusion.

**Precision** is the inverse variance of each prediction error.  Attention
allocates precision, so a channel that is currently unpredictable is
down-weighted rather than allowed to dominate learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from .config import SimConfig


class RecursiveLeastSquares:
    """Online linear regression with forgetting and ridge regularisation."""

    def __init__(self, n_in: int, n_out: int, forget: float = 0.999,
                 ridge: float = 1.0):
        self.n_in = n_in
        self.n_out = n_out
        self.lam = float(forget)
        self.W = np.zeros((n_out, n_in))
        self.P = np.eye(n_in) * float(ridge)
        self.n_updates = 0
        self.err_ema = np.zeros(n_out)
        self.err_var = np.ones(n_out)
        self._alpha = 0.02

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x

    def update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        if denom < 1e-12:
            return np.zeros(self.n_out)
        k = Px / denom
        pred = self.W @ x
        err = y - pred
        self.W += np.outer(err, k)
        self.P = (self.P - np.outer(k, Px)) / self.lam
        self.n_updates += 1
        if self.n_updates % 64 == 0:                 # keep P symmetric
            self.P = 0.5 * (self.P + self.P.T)
        # running error statistics -> precision
        self.err_ema += self._alpha * (err - self.err_ema)
        self.err_var += self._alpha * ((err - self.err_ema) ** 2 - self.err_var)
        return err

    def precision(self) -> np.ndarray:
        return 1.0 / np.maximum(self.err_var, 1e-6)

    def r2(self) -> np.ndarray:
        """Per-output coefficient of determination of the recent errors."""
        return fclip(1.0 - self.err_var / (self.err_var + 1e-6 + self.err_ema ** 2
                                            + 1.0), 0.0, 1.0)

    def reset(self) -> None:
        self.W[:] = 0.0
        self.P = np.eye(self.n_in)
        self.n_updates = 0
        self.err_ema[:] = 0.0
        self.err_var[:] = 1.0


class KalmanStateEstimator:
    """Assumed-density (diagonal covariance) filter for the sensory latent.

    Provides the posterior state estimate that the rest of the agent treats as
    "where my body is", which is *not* the same as the raw afferent stream --
    it is a filtered, predicted version of it.

    The covariance is kept diagonal.  A full 410x410 covariance would need a
    matrix inverse on every cognitive tick (~0.45 s per call here, 19 % of
    total runtime), while the diagonal form is exact for independent channels
    and costs O(n).  Sensory channels are near enough to independent for this
    to be fair, and the forward model carries the cross-channel structure.
    """

    def __init__(self, dim: int, process_noise: float = 0.02,
                 obs_noise: float = 0.08):
        self.dim = dim
        self.q = float(process_noise)
        self.r = float(obs_noise)
        self.x = np.zeros(dim)
        self.p = np.ones(dim)
        self._init = False

    def step(self, obs: np.ndarray, predicted_next: np.ndarray | None = None
             ) -> np.ndarray:
        if not self._init:
            self.x = obs.copy()
            self._init = True
            return self.x
        if predicted_next is not None:
            self.x = predicted_next
            self.p = self.p + self.q
        k = self.p / (self.p + self.r)          # scalar per channel
        self.x = self.x + k * (obs - self.x)
        self.p = (1.0 - k) * self.p
        return self.x

    def reset(self) -> None:
        self._init = False
        self.p[:] = 1.0


@dataclass
class PredictionFrame:
    t: float = 0.0
    prediction: np.ndarray = field(default_factory=lambda: np.zeros(0))
    error: np.ndarray = field(default_factory=lambda: np.zeros(0))
    precision: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weighted_error: np.ndarray = field(default_factory=lambda: np.zeros(0))
    free_energy: float = 0.0
    free_energy_norm: float = 0.0      # per sensory dimension
    complexity: float = 0.0
    inaccuracy: float = 0.0
    surprise: float = 0.0
    body_ownership: np.ndarray = field(default_factory=lambda: np.zeros(0))
    controllability: np.ndarray = field(default_factory=lambda: np.zeros(0))
    proprioceptive_drift: float = 0.0
    inverse_error: float = 0.0
    model_confidence: float = 0.0
    jacobian_norm: float = 0.0
    empowerment: float = 0.0


class PredictiveSystem:
    """Forward model + inverse model + body schema + precision weighting."""

    def __init__(self, cfg: SimConfig, latent_dim: int, action_dim: int,
                 n_taxels: int = 0):
        self.cfg = cfg
        self.P = cfg.predictive
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        n_in = latent_dim + action_dim
        self.forward = RecursiveLeastSquares(n_in, latent_dim,
                                             forget=self.P.forward_forget,
                                             ridge=self.P.forward_ridge)
        # Prior: "nothing changes unless I act".  Without this the untrained
        # model predicts a zero state, so every candidate policy looks equally
        # bad and policy selection is vacuous until the model has learned.
        self.forward.W[:, :latent_dim] = np.eye(latent_dim)
        self.inverse = RecursiveLeastSquares(latent_dim * 2, action_dim,
                                             forget=self.P.inverse_forget,
                                             ridge=self.P.inverse_ridge)
        self.kalman = KalmanStateEstimator(
            latent_dim, self.P.kalman_process_noise, self.P.kalman_obs_noise)

        # body schema: d(proprio)/d(action), learned as a linear map
        self.schema = RecursiveLeastSquares(n_in, latent_dim,
                                            forget=0.9995,
                                            ridge=self.P.body_schema_ridge)

        self.prev_latent: np.ndarray | None = None
        self.prev_action: np.ndarray | None = None
        self.precision = np.ones(latent_dim)
        self.ownership = np.zeros(latent_dim)
        self.drift = 0.0
        self.n_updates = 0
        self.history: list = []
        self._prev_error = np.zeros(latent_dim)
        self._free_energy_ema = 0.0

    # ------------------------------------------------------------------
    def latent_input(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([latent, action])

    # ------------------------------------------------------------------
    def observe(self, latent: np.ndarray, action: np.ndarray) -> PredictionFrame:
        """One predictive-coding cycle: predict, compare, learn."""
        x = self.latent_input(self.prev_latent if self.prev_latent is not None
                             else latent, self.prev_action if self.prev_action
                             is not None else action)
        if self.prev_latent is None:
            self.prev_latent = latent.copy()
            self.prev_action = action.copy()
            return PredictionFrame(t=0.0, prediction=latent.copy(),
                                   error=np.zeros_like(latent),
                                   precision=np.ones_like(latent))

        # ---- predict the present from the past ------------------------
        pred = self.forward.predict(x)
        err = latent - pred
        self.forward.update(x, latent)

        # ---- precision: inverse error variance, smoothed -------------
        p_raw = self.forward.precision()
        p_norm = p_raw / max(p_raw.mean(), 1e-9)
        self.precision += (1.0 / self.P.precision_tau) * 0.05 * (p_norm - self.precision)
        self.precision = fclip(self.precision, 0.05, 20.0)
        weighted = err * self.precision

        # ---- free energy: inaccuracy + complexity ---------------------
        inaccuracy = 0.5 * float(np.sum((err ** 2) * self.precision))
        # complexity = KL(q || prior) approximated by the deviation of the
        # forward weights from zero (an unlearned model is a strong prior)
        complexity = 0.5 * float(np.sum(self.forward.W ** 2)) / max(
            self.forward.n_out * self.forward.n_in, 1) * 1e3
        free_energy = inaccuracy + 0.05 * complexity
        self._free_energy_ema += 0.05 * (free_energy - self._free_energy_ema)
        surprise = -0.5 * float(np.sum(np.log(np.maximum(self.forward.err_var, 1e-6))
                                       + err ** 2 / np.maximum(
                                           self.forward.err_var, 1e-6)))

        # ---- body schema + ownership ----------------------------------
        self.schema.update(x, latent)
        J = self.schema.W[:, self.latent_dim:]          # d(latent)/d(action)
        row_norm = np.linalg.norm(J, axis=1)
        controllability = row_norm / max(row_norm.max(), 1e-9)
        # ownership: how much of this dimension's variance the efference
        # copy explains (R^2 of the forward model on that dimension)
        ownership = self.forward.r2()
        self.ownership += 0.1 * (ownership - self.ownership)
        controllability = np.nan_to_num(controllability)

        # proprioceptive drift: slow accumulation of unexplained
        # proprioceptive error (as in vibration-induced illusion)
        prop_err = float(np.mean(np.abs(err[:min(64, len(err))])))
        self.drift += 0.002 * (prop_err - self.drift)

        # ---- inverse model --------------------------------------------
        target = latent - self.prev_latent
        inv_x = np.concatenate([self.prev_latent, target])
        a_pred = self.inverse.predict(inv_x)
        inv_err = float(np.mean((a_pred - self.prev_action) ** 2))
        self.inverse.update(inv_x, self.prev_action)

        # ---- state estimation -----------------------------------------
        self.kalman.step(latent, pred)

        self.prev_latent = latent.copy()
        self.prev_action = action.copy()
        self.n_updates += 1

        # empowerment ~ controllability x dimensionality
        empowerment = float(np.mean(controllability) * np.log1p(J.shape[0]))

        frame = PredictionFrame(
            prediction=pred,
            error=err,
            precision=self.precision.copy(),
            weighted_error=weighted,
            free_energy=free_energy,
            free_energy_norm=free_energy / max(self.latent_dim, 1),
            complexity=complexity,
            inaccuracy=inaccuracy,
            surprise=surprise,
            body_ownership=self.ownership.copy(),
            controllability=controllability,
            proprioceptive_drift=float(self.drift),
            inverse_error=inv_err,
            model_confidence=float(fclip(1.0 / (1.0 + self._free_energy_ema * 0.05),
                                           0, 1)),
            jacobian_norm=float(np.linalg.norm(J)),
            empowerment=empowerment,
        )
        return frame

    # ------------------------------------------------------------------
    def imagine(self, latent: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One-step roll-out of the forward model (no learning)."""
        return self.forward.predict(self.latent_input(latent, action))

    def imagine_sequence(self, latent: np.ndarray, actions: list,
                         horizon: int) -> np.ndarray:
        s = latent.copy()
        out = []
        for h in range(horizon):
            a = actions[min(h, len(actions) - 1)]
            s = self.imagine(s, a)
            out.append(s.copy())
        return np.stack(out) if out else np.zeros((0, len(latent)))

    def inverse_action(self, current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        return self.inverse.predict(np.concatenate([current, desired - current]))

    def reset(self) -> None:
        self.forward.reset()
        self.inverse.reset()
        self.schema.reset()
        self.kalman.reset()
        self.prev_latent = None
        self.prev_action = None
        self.precision[:] = 1.0
        self.ownership[:] = 0.0
        self.drift = 0.0

    def describe(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "forward_params": int(self.forward.W.size),
            "inverse_params": int(self.inverse.W.size),
            "schema_params": int(self.schema.W.size),
            "learning_rule": f"RLS forgetting={self.P.forward_forget}",
        }
