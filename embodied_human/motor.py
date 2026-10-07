"""
Motor system: from intention to torque.

The body is commanded the way a real body is -- through an **equilibrium
point**.  The controller does not specify torques; it specifies a virtual
posture, and the physics discovers the forces.  Everything below is layered on
that single idea:

* **Voluntary set point** - the posture active inference chose.
* **Posture controller** - PD towards the nominal configuration, scaled by
  each muscle's strength, with enough damping to hold an inverted pendulum.
* **Balance strategies** - ankle, hip and toe strategies driven by the centre
  of mass relative to the support polygon.  These are the same three
  strategies humans use, and they engage in that order as the perturbation
  grows.
* **Reflexes** - stretch (monosynaptic, with conduction delay), Golgi tendon
  (protects tendon from overload), withdrawal (nociceptive flexion), righting
  and vestibulo-collic.
* **Protective stepping** - when the centre of mass leaves the support
  polygon, no ankle torque can save you.  The only correct answer is to take a
  step, so the body triggers a swing-and-place motor program.
* **Central pattern generator** - a coupled oscillator pair for locomotion.
* **Actuator realism** - torque limits, rate limits, first-order activation
  lag (in MJCF), signal-dependent noise, and fatigue that reduces the
  achievable torque.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from .config import SimConfig
from .skeleton import nominal_posture
from .state import BodyState

# Reflex arcs, in the order they are summed
REFLEX_ARCS = ("stretch", "golgi", "withdrawal", "righting", "vestibulo_collic",
               "startle", "crossed_extensor")

# Balance strategies
STRATEGIES = ("ankle", "hip", "toe", "knee")


@dataclass
class MotorFrame:
    t: float = 0.0
    torque: np.ndarray = field(default_factory=lambda: np.zeros(0))
    voluntary: np.ndarray = field(default_factory=lambda: np.zeros(0))
    posture: np.ndarray = field(default_factory=lambda: np.zeros(0))
    balance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reflex: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cpg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    effort: float = 0.0
    mechanical_power: float = 0.0
    jerk: float = 0.0
    strategy_weight: np.ndarray = field(default_factory=lambda: np.zeros(len(STRATEGIES)))
    stepping: str = "none"
    balance_error: float = 0.0
    reflex_activity: np.ndarray = field(default_factory=lambda: np.zeros(len(REFLEX_ARCS)))
    cpg_phase: float = 0.0
    joint_limit_force: float = 0.0


class MotorSystem:
    """Full spinal + supraspinal motor apparatus."""

    def __init__(self, cfg: SimConfig, meta):
        self.cfg = cfg
        self.M = cfg.motor
        self.meta = meta
        self.names = [n for _, n, _ in meta.joint_order]
        self.n = len(self.names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.limits = meta.torque_limit.copy()
        # body mass (the human only, excluding the manipulable scene objects)
        scene_mass = sum(o.mass for o in getattr(meta, "objects", []))
        self.mass = float(sum(meta.model.body_mass) - scene_mass)
        nominal = nominal_posture()
        self.q_nom = np.array([nominal.get(n, 0.0) for n in self.names])

        # Posture gains scaled by muscle strength.  The damping ratio is not a
        # free parameter: the actuators are first-order lagged by
        # ``command_tau``, and a PD loop driving a lagged actuator needs a
        # derivative time constant of roughly twice that lag to stay stable.
        # With kd/kp = 0.06 s (half the 45 ms lags) the legs oscillate and the
        # body folds; at kd/kp = 0.12 s it stands indefinitely.
        self.kp = 2.0 * self.limits
        self.kd = 0.24 * self.limits
        # The ankle is shared between two controllers: the posture loop wants a
        # particular joint angle while the balance loop wants a particular
        # *torque*.  With equal authority the posture spring wins, the centre of
        # pressure never travels, and the COM drifts away until the body falls.
        # Balance therefore gets priority on the ankle, which is also what a
        # person does when standing still.
        for i, jn in enumerate(self.names):
            if jn.startswith("ankle"):
                self.kp[i] *= 0.45
                self.kd[i] *= 0.45
        weak = self.limits < 1.0                      # extraocular muscles
        self.kp[weak] = 0.010
        self.kd[weak] = 0.0009

        # muscle fatigue multiplier (from interoception)
        self.fatigue_scale = np.ones(self.n)

        # --- reflex state ---------------------------------------------
        self.reflex_delay_qd = np.zeros(self.n)
        self.reflex_delay_cmd = np.zeros(self.n)
        self.reflex_activity = np.zeros(len(REFLEX_ARCS))
        self.startle = 0.0
        self.pain_flexion = np.zeros(self.n)

        # --- equilibrium-point / voluntary ----------------------------
        self.target = self.q_nom.copy()
        self.prev_target = self.q_nom.copy()

        # --- balance --------------------------------------------------
        self.strategy = np.zeros(len(STRATEGIES))
        self.com_prev = None
        self.balance_error = 0.0
        self.postural_priority = 1.0
        self.prio_gain = 30.0
        self.prio_thr = 0.012
        # Which joints postural prioritisation protects.  The trunk and legs
        # carry the body, so their voluntary displacement is attenuated when
        # balance is threatened; arms, hands, head and eyes keep full
        # authority, which is what lets the agent gesture and reach while it
        # stands.
        self.prio_weight = np.array([
            0.0 if (n.startswith(("sh_", "elbow", "wrist", "thumb", "index",
                                  "fingers", "eye", "jaw", "neck")))
            else 1.0
            for n in self.names])

        # --- CPG ------------------------------------------------------
        self.cpg_phase = 0.0
        self.cpg_freq = 0.0
        self.cpg_amp = 0.0
        self.cpg_active = False

        # --- protective stepping --------------------------------------
        self.step_state = "none"
        self.step_timer = 0.0
        self.step_leg = 0
        self.step_dir = 0.0
        self.step_count = 0

        # --- bookkeeping ----------------------------------------------
        self.prev_tau = np.zeros(self.n)
        self.prev_qd = np.zeros(self.n)
        self.jerk = 0.0
        self.rng = np.random.default_rng(cfg.seed + 67)

        # joint ranges for limit avoidance
        self.q_lo = np.array([j.lo for _, _, j in meta.joint_order])
        self.q_hi = np.array([j.hi for _, _, j in meta.joint_order])

    # ------------------------------------------------------------------
    def set_voluntary_target(self, target: np.ndarray) -> None:
        self.target = fclip(np.asarray(target, float), self.q_lo, self.q_hi)

    # ------------------------------------------------------------------
    def compute(self, state: BodyState, dt: float, *,
                voluntary: np.ndarray | None = None,
                arousal: float = 0.3, fear: float = 0.0, pain: float = 0.0,
                fatigue: float = 0.0, cpg_command: float = 0.0,
                gait_tau: np.ndarray | None = None,
                gait_mask: np.ndarray | None = None,
                exempt: np.ndarray | None = None
                ) -> MotorFrame:
        M, n = self.M, self.n
        q, qd = state.q, state.qd

        if voluntary is not None:
            self.set_voluntary_target(voluntary)
        target = self.target

        # ---------------- 0. balance error (needed before posture) -----
        com = state.com
        support = state.support_center
        # COM velocity by finite difference: MuJoCo never populates the world
        # body's subtree velocity, and a balance loop without the derivative
        # term is an undamped spring on an inverted pendulum.
        if self.com_prev is None:
            comv = np.zeros(3)
        else:
            comv = (com - self.com_prev) / max(dt, 1e-9)
        self.com_prev = com.copy()
        state.com_vel = comv
        ex = float(com[0] - support[0])
        ey = float(com[1] - support[1])
        self.balance_error = float(np.hypot(ex, ey))

        # ---------------- 1. posture (equilibrium point) --------------
        # Postural prioritisation: when balance is threatened, voluntary
        # movement is attenuated towards the nominal configuration.  Humans do
        # this too -- a reaching movement is truncated when the support surface
        # moves, because posture is protected first.
        #
        # The threat is measured with the **capture point** (COM plus its
        # momentum divided by the pendulum frequency), not the raw COM offset.
        # The capture point is what actually decides whether the current state
        # is recoverable, so gating on it suppresses a destabilising movement
        # *before* the COM has already left the support polygon -- a purely
        # reactive gate on the COM offset is always one pendulum time constant
        # too late.
        h_com0 = max(float(com[2]) - 0.05, 0.35)
        omega0 = float(np.sqrt(9.81 / h_com0))
        cap_y = ey + comv[1] / omega0
        cap_x = ex + comv[0] / omega0
        risk = max(abs(cap_y) / 0.150, abs(cap_x) / 0.075)
        priority = 1.0 / (1.0 + self.prio_gain * max(0.0, risk - self.prio_thr) ** 2)
        self.postural_priority = priority
        # Prioritisation is *segmental*, as it is in a real body.  What
        # threatens balance is movement of the trunk and legs; an arm or the
        # head can move almost freely, because shifting a 4 kg arm displaces
        # the whole-body COM by only a centimetre or two.  Applying a single
        # global gain to every joint (the obvious implementation) leaves the
        # agent unable to gesture or reach whenever it is standing -- which is
        # always -- and the body then looks frozen rather than embodied.
        # factor = 1 for the arms/head (weight 0), = priority for trunk and legs.
        # (The earlier expression multiplied by the weight itself, which gave
        # the arms *zero* authority -- they never left the nominal posture --
        # the opposite of what the comment above says.)
        pw = self.prio_weight
        if exempt is not None:
            # joints a deliberate skill is using (a reaching lean) are not
            # attenuated; the skill watches the balance itself
            pw = np.where(exempt, 0.0, pw)
        factor = 1.0 - pw * (1.0 - priority)
        target_eff = self.q_nom + factor * (target - self.q_nom)
        self.fatigue_scale = 1.0 - 0.45 * fclip(fatigue, 0.0, 1.0)
        kp = self.kp * self.fatigue_scale
        kd = self.kd
        posture = kp * (target_eff - q) - kd * qd

        # ---------------- 2. balance strategies -----------------------
        # The ankle strategy is formulated on the **centre of pressure**, which
        # is the quantity the foot can actually control and which is bounded by
        # the foot's geometry.  A proportional torque law (an earlier version of
        # this code) has no such bound: it over-drives the ankle into its joint
        # stop, at which point the constraint -- not the muscle -- determines
        # the motion, and the body topples about the toe.
        #
        # Linear inverted pendulum about the ankle:
        #       x'' = omega^2 (x - p),   omega = sqrt(g / h_com)
        # Choosing   p = x + Kp*(x - x_ref) + Kd*x'
        # gives a stable second-order response with
        #       omega_n = omega*sqrt(Kp),  zeta = omega*Kd / (2*sqrt(Kp)).
        foot_front = 0.150      # CoP may travel this far in front of the base
        foot_back = 0.105       # ... and this far behind
        ankle_offset_y = 0.055  # ankle joint sits behind the base centre
        ankle_offset_x = 0.0
        h_com = max(float(com[2]) - 0.05, 0.35)
        omega = float(np.sqrt(9.81 / h_com))
        kp_cop = 1.0
        kd_cop = 2.0 * 0.90 / omega        # damping ratio 0.9

        cop_y = ey + kp_cop * ey + kd_cop * comv[1]
        cop_y = float(fclip(cop_y, -foot_front, foot_back))
        # Foot statics: the ankle torque holding the centre of pressure at
        # ``cop`` is  tau = -m g (cop - a),  where ``a`` is the ankle's own
        # offset from the support centre.  Balancing moments about the ankle
        # first for the foot and then for everything above it eliminates the
        # COM and leaves the lever arm measured from the *joint*.  At
        # equilibrium the CoP sits under the COM and the torque is m g (x - a),
        # which is exactly the moment gravity applies about the ankle.
        tau_ankle_y = -(cop_y - ankle_offset_y) * self.mass * 9.81

        cop_x = ex + kp_cop * ex + kd_cop * comv[0]
        cop_x = float(fclip(cop_x, -0.038, 0.038))   # foot half-width
        tau_ankle_x = -(cop_x - ankle_offset_x) * self.mass * 9.81

        # how much of the available CoP range is being used
        use_y = abs(cop_y) / foot_front
        use_x = abs(cop_x) / 0.038
        excursion = float(max(use_x, use_y, 0.0)) - 1.0
        self.strategy[0] = 1.0                       # the ankle always works
        self.strategy[1] = float(fclip(1.8 * excursion, 0.0, 1.5))   # hip adds
        self.strategy[2] = float(fclip(-cop_y / 0.08, 0.0, 1.2))     # toe roll-off
        self.strategy[3] = float(fclip(0.2 + excursion, 0.0, 1.0))   # knee absorbs

        balance = np.zeros(n)
        # hip strategy: rotate the trunk to move the COM itself, which is the
        # only option once the CoP is pinned at the edge of the foot
        hip_cmd = fclip(-M.hip_kp_gain * ey - M.hip_kd_gain * comv[1],
                          -0.6 * M.hip_kp_gain, 0.6 * M.hip_kp_gain)
        # Sign flip relative to the hip: both joints rotate about the same axis
        # but hip flexion is negative rotation while knee flexion is positive,
        # so the knee needs the opposite torque to push the body the same way.
        knee_cmd = fclip(M.knee_strategy_gain * ey, -50.0, 50.0)
        for side in ("l", "r"):
            balance[self.idx[f"ankle_{side}_flex"]] += 0.5 * tau_ankle_y
            balance[self.idx[f"ankle_{side}_inv"]] += 0.5 * tau_ankle_x
            balance[self.idx[f"hip_{side}_flex"]] += (
                0.5 * M.hip_strategy_gain * hip_cmd * self.strategy[1])
            balance[self.idx[f"toe_{side}"]] += (
                -M.toe_strategy_gain * tau_ankle_y * self.strategy[2] * 0.5)
            balance[self.idx[f"knee_{side}"]] += (
                0.20 * knee_cmd * self.strategy[3])
        # Standing-height gate: below ~0.65 m COM there is no inverted
        # pendulum to balance (lying/kneeling), and the saturated ankle/toe
        # torques only steal budget from deliberate motion such as get-up
        # skills. Exactly 1.0 at normal standing height, so standing is
        # bit-for-bit unaffected.
        stand_gain = float(fclip((float(com[2]) - 0.35) / 0.30, 0.0, 1.0))
        balance *= stand_gain

        # ---------------- 3. reflexes --------------------------------
        reflex = np.zeros(n)
        # stretch reflex: delayed velocity feedback on the agonist
        self.reflex_delay_qd = 0.85 * self.reflex_delay_qd + 0.15 * qd
        stretch = -M.stretch_reflex_gain * self.limits * self.reflex_delay_qd
        # Golgi tendon organ: unload a muscle that is producing too much force
        golgi = -M.golgi_gain * np.sign(state.tau) * np.maximum(
            np.abs(state.tau) - 0.75 * self.limits, 0.0)
        # withdrawal: nociceptive flexion of the affected limb
        if pain > M.withdrawal_threshold:
            mag = (pain - M.withdrawal_threshold) * M.withdrawal_gain
            for i, nm in enumerate(self.names):
                if "flex" in nm or "elbow" in nm or "knee" in nm:
                    reflex[i] += mag * self.limits[i] * 0.35
        # righting reflex: keep the head and trunk upright (gated by
        # standing height like the balance strategies: while prone it would
        # drag every arch back to nominal)
        righting = np.zeros(n)
        for nm in ("spine_bend", "chest_bend", "neck_bend"):
            if nm in self.idx:
                righting[self.idx[nm]] = -M.righting_gain * (
                    q[self.idx[nm]] - self.q_nom[self.idx[nm]]) * self.limits[
                        self.idx[nm]] * 0.5 * stand_gain
        # vestibulo-collic: head stabilisation against trunk rotation
        vc = np.zeros(n)
        if "neck_bend" in self.idx:
            vc[self.idx["neck_bend"]] = -M.vestibulo_collic_gain * self.limits[
                self.idx["neck_bend"]] * float(state.root_angvel[0])
        # startle: a fast, non-specific whole-body flexion after a sudden event
        self.startle = max(0.0, self.startle - dt / 0.8)
        startle_t = fclip(fear * 1.4, 0, 1.5)
        startle = np.zeros(n)
        for i, nm in enumerate(self.names):
            if any(k in nm for k in ("flex", "bend")):
                startle[i] = -self.startle * self.limits[i] * 0.12

        reflex += stretch + golgi + righting + vc + startle
        self.reflex_activity = np.array([
            float(np.abs(stretch).mean() / max(self.limits.mean(), 1e-6)),
            float(np.abs(golgi).mean() / max(self.limits.mean(), 1e-6)),
            float(np.abs(reflex).mean() / max(self.limits.mean(), 1e-6)) * (pain > 0.3),
            float(np.abs(righting).mean() / max(self.limits.mean(), 1e-6)),
            float(np.abs(vc).mean() / max(self.limits.mean(), 1e-6)),
            float(self.startle),
            0.0,
        ])

        # ---------------- 4. CPG -------------------------------------
        cpg = np.zeros(n)
        self.cpg_freq = M.cpg_freq_walk * cpg_command
        self.cpg_active = cpg_command > 0.05
        if self.cpg_active:
            self.cpg_phase = (self.cpg_phase + dt * self.cpg_freq * 2 * np.pi) % (
                2 * np.pi)
            self.cpg_amp += (dt / 0.5) * (M.cpg_amplitude * cpg_command - self.cpg_amp)
            for side, phase_off in (("l", 0.0), ("r", np.pi)):
                ph = self.cpg_phase + phase_off
                swing = np.sin(ph)
                cpg[self.idx[f"hip_{side}_flex"]] += -self.cpg_amp * swing * 90.0
                cpg[self.idx[f"knee_{side}"]] += self.cpg_amp * 60.0 * max(0.0, swing)
                cpg[self.idx[f"ankle_{side}_flex"]] += self.cpg_amp * 30.0 * max(
                    0.0, -swing)
        else:
            self.cpg_amp *= (1.0 - dt / 0.5)

        # ---------------- 5. protective stepping ---------------------
        step_torque = self._stepping(state, dt, ex, ey, comv, fear)
        if self.step_state != "none":
            self.step_count += 1

        # ---------------- 6. sum, limit, add noise ------------------
        tau = posture + balance + reflex + cpg + step_torque
        if gait_mask is not None and gait_tau is not None:
            # The walking controller owns the leg joints while it is active;
            # the posture / balance / reflex terms for those joints are
            # replaced rather than added to, because they fight a gait.
            tau = np.where(gait_mask, gait_tau, tau)
        tau = fclip(tau, -self.limits, self.limits)

        # signal-dependent noise grows with effort (Weber's law in muscle)
        effort = np.abs(tau) / np.maximum(self.limits, 1e-6)
        noise_sd = M.motor_noise_sd * (0.5 + effort) * (1.0 + 0.8 * arousal)
        tau += self.rng.normal(0, 1.0, n) * noise_sd * self.limits

        # Rate limit approximating the activation filter.  This must be a
        # *rise time*, not a fixed increment: a fixed Nm-per-tick limit makes a
        # strong muscle slower than a weak one, and left the knee needing 1.5 s
        # to reach full torque -- longer than the entire balance response.
        rise = max(getattr(M, "torque_rise_time", 0.05), 1e-3)
        max_delta = self.limits * (dt / rise)
        tau = self.prev_tau + fclip(tau - self.prev_tau, -max_delta, max_delta)
        tau = fclip(tau, -self.limits, self.limits)

        # mechanical power and jerk (for intrinsic motivation)
        power = float(np.dot(tau, qd))
        qdd = (qd - self.prev_qd) / max(dt, 1e-9)
        self.jerk = float(np.mean(np.abs(qdd - (qd - self.prev_qd) / max(dt, 1e-9))))
        self.prev_qd = qd.copy()
        self.prev_tau = tau.copy()

        self.prev_target = target.copy()

        return MotorFrame(
            torque=tau,
            voluntary=(kp * (target - q)),
            posture=posture,
            balance=balance,
            reflex=reflex,
            cpg=cpg + step_torque,
            effort=float(np.mean(effort)),
            mechanical_power=power,
            jerk=self.jerk,
            strategy_weight=self.strategy.copy(),
            stepping=self.step_state,
            balance_error=self.balance_error,
            reflex_activity=self.reflex_activity.copy(),
            cpg_phase=float(self.cpg_phase),
            joint_limit_force=float(np.abs(balance).max()),
        )

    # ------------------------------------------------------------------
    def _stepping(self, state: BodyState, dt: float, ex: float, ey: float,
                  comv: np.ndarray, fear: float) -> np.ndarray:
        """Trigger and run a protective step when balance is unrecoverable.

        Once the capture point leaves the support polygon no joint torque can
        save the body -- the only correct action is to move the polygon.  The
        step must go in the direction of the fall, so the swing sign is taken
        from the *sagittal* excursion (an earlier version always swung the leg
        forwards, which rescues a forward fall and accelerates a backward one).
        """
        M = self.M
        out = np.zeros(self.n)
        # No protective stepping while down: a swing leg from lying is not a
        # rescue step, it is ~170 Nm of asymmetric flailing that fights any
        # get-up skill. Steps only make sense from a standing-height body.
        if bool(state.fallen) or float(state.com[2]) < 0.40:
            self.step_state = "none"
            self.step_timer = 0.0
            return out
        support_x = 0.5 * abs(state.site_pos.get("soma_foot_l", np.zeros(3))[0]
                              - state.site_pos.get("soma_foot_r", np.zeros(3))[0]) + 0.038
        # capture point: where the COM would come to rest under the CoP limit
        h_com = max(float(state.com[2]) - 0.05, 0.35)
        omega = float(np.sqrt(9.81 / h_com))
        predicted = np.array([ex + comv[0] / omega, ey + comv[1] / omega])
        need = (abs(predicted[0]) > support_x * 1.02) or (abs(predicted[1]) > 0.125)
        if need and self.step_state == "none" and self.step_timer <= 0.0:
            self.step_state = "unload"
            self.step_timer = 0.12
            # step with the leg on the side the COM is falling towards
            self.step_leg = 1 if predicted[0] > 0 else -1
            # +1 : falling forwards (predicted y negative) -> swing forwards
            self.step_dir = 1.0 if predicted[1] < 0 else -1.0
            self.step_mag = float(fclip(max(abs(predicted[1]) / 0.14,
                                              abs(predicted[0]) / 0.10), 0.6, 1.8))
            self.startle = max(self.startle, 0.5 + 0.5 * fear)

        self.step_timer -= dt
        if self.step_state == "none":
            return out

        side = "l" if self.step_leg > 0 else "r"
        hip = f"hip_{side}_flex"
        knee = f"knee_{side}"
        ank = f"ankle_{side}_flex"
        # hip flexion torque is negative; swinging forwards therefore needs a
        # negative command and swinging backwards a positive one
        sgn = self.step_dir
        mag = getattr(self, "step_mag", 1.0)

        if self.step_state == "unload":
            # shift the weight off the swing leg, then lift it
            out[self.idx[hip]] += -35.0 * sgn
            out[self.idx[knee]] += 60.0
            out[self.idx[ank]] += -15.0
            if self.step_timer <= 0:
                self.step_state = "swing"
                self.step_timer = 0.26
        elif self.step_state == "swing":
            # carry the foot in the direction of the fall
            out[self.idx[hip]] += -95.0 * mag * sgn
            out[self.idx[knee]] += 70.0
            out[self.idx[ank]] += -10.0
            if self.step_timer <= 0:
                self.step_state = "place"
                self.step_timer = 0.22
        elif self.step_state == "place":
            out[self.idx[hip]] += 22.0 * sgn
            out[self.idx[knee]] += 85.0
            out[self.idx[ank]] += 18.0
            if self.step_timer <= 0:
                self.step_state = "recover"
                self.step_timer = 0.40
        elif self.step_state == "recover":
            out[self.idx[hip]] += 20.0
            out[self.idx[knee]] -= 22.0
            out[self.idx[ank]] += 26.0
            if self.step_timer <= 0:
                self.step_state = "none"
                self.step_timer = 0.22
        return out

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.target = self.q_nom.copy()
        self.com_prev = None
        self.prev_tau[:] = 0.0
        self.prev_qd[:] = 0.0
        self.cpg_phase = 0.0
        self.cpg_amp = 0.0
        self.step_state = "none"
        self.step_timer = 0.0
        self.startle = 0.0

    def describe(self) -> dict:
        return {
            "n_actuators": self.n,
            "posture_gain_multiple": 2.0,
            "damping_multiple": 0.12,
            "balance": {"ankle_kp": self.M.balance_kp * 10.0,
                        "ankle_kd": self.M.balance_kd * 10.0,
                        "hip_gain": self.M.hip_strategy_gain,
                        "toe_gain": self.M.toe_strategy_gain},
            "reflex_arcs": list(REFLEX_ARCS),
            "strategies": list(STRATEGIES),
            "cpg": {"freq_walk_hz": self.M.cpg_freq_walk},
            "protective_stepping": True,
        }
