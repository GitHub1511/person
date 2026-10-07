"""
The inner world: every internal system, run at its own rate and coupled to the
rest of the person.

The original agent's internal life was 67 interoceptive scalars, 28 emotions, 25
neuromodulators and 16 drives (~136 numbers).  This module adds, on top of those
(which are all still there and still drive everything they used to):

* organ systems with their own compartments (``inner_organs``),
* a brain-side layer: a circadian clock, a ~12,000-unit neural mass, episodic
  memory, associative conditioning and an interoceptive prediction model
  (``inner_brain``),
* the eyes (``ocular``),

and closes the loops back into the body and the mind:

====================  ====================================================
inner -> outside      what it changes
====================  ====================================================
skin                  per-patch blood flow, sweat, inflammation, damage,
                      itch and irritation (-> hyperalgesia, wetness ...)
affect                a threat tone (neural-mass "amygdala"), rumination,
                      interoceptive surprise, the mood memory carries
drives                five new drives: ocular comfort, muscle soreness,
                      gut discomfort, mental fatigue, the urge to shift
behaviour             desire (mind-wandering -> explore / boredom, fatigue,
                      soreness -> stretching and guarding ...)
latent                two new blocks the active-inference layer sees
====================  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import skin
from ._fast import fclip
from .complexity import C as COMPLEXITY
from .inner_brain import (POP_NAMES, PI, CircadianClock, Conditioning, EpisodicMemory,
                          InteroModel, NeuralMass)
from .inner_organs import (Chemistry, GutMicrobiome, Immune, Kidney, Liver, Lungs, MuscleBank,
                           SkinThermo, VascularBeds)
from .interoception import IDX, N_INTEROCEPTION
from .receptors import CH

N_INNER_SUMMARY = 128
N_OCULAR_SUMMARY = 16
EXTRA_DRIVES = ("ocular_comfort", "muscle_soreness", "gut_discomfort", "mental_fatigue",
                "shift_urge")


@dataclass
class InnerFrame:
    """What the inner world tells the rest of the person."""
    threat_tone: float = 0.0
    rumination: float = 0.0
    arousal_tone: float = 0.0
    control_tone: float = 0.0
    interoceptive_surprise: float = 0.0
    memory_valence: float = 0.0
    familiarity: float = 0.0
    mind_wandering: float = 0.0
    extra_drives: np.ndarray = field(default_factory=lambda: np.zeros(len(EXTRA_DRIVES)))
    summary: np.ndarray = field(default_factory=lambda: np.zeros(N_INNER_SUMMARY))
    clock_hour: float = 10.0
    sleep_pressure: float = 0.3
    muscle_fatigue: float = 0.0
    muscle_soreness: float = 0.0
    gut_discomfort: float = 0.0
    systemic_inflammation: float = 0.0
    cortisol: float = 0.0
    glucose: float = 5.2
    pop_means: np.ndarray = field(default_factory=lambda: np.zeros(len(POP_NAMES)))


class InnerWorld:
    RATES = {"brain": 50.0, "muscle": 20.0, "fast": 10.0, "slow": 2.0, "clock": 1.0,
             "memory_store": 1.0, "memory_recall": 4.0, "assoc": 10.0}

    def __init__(self, agent, seed: int | None = None, start_hour: float = 10.0,
                 time_scale: float = 1.0):
        self.ag = agent
        seed = agent.cfg.seed if seed is None else seed
        self.seed = seed
        meta = agent.meta
        nj = meta.n_actuators
        self.patch_names = skin.PATCH_NAMES
        npatch = len(self.patch_names)
        self.vasc = VascularBeds(npatch, seed + 201)
        self.lungs = Lungs(seed + 202)
        self.kidney = Kidney(seed + 203)
        self.liver = Liver(seed + 204)
        self.gut = GutMicrobiome(seed + 205)
        self.muscle = MuscleBank(nj, meta.torque_limit, seed + 206)
        self.thermo = SkinThermo(self.patch_names, seed + 207)
        self.immune = Immune(npatch, seed + 208)
        self.chem = Chemistry(seed + 209)
        self.clock = CircadianClock(seed + 210, start_hour, time_scale)
        self.brain = NeuralMass(seed + 211)
        self.memory = EpisodicMemory(seed + 212)
        self.cond = Conditioning()
        self.intero = InteroModel(N_INTEROCEPTION)
        self.acc = {k: 0.0 for k in self.RATES}
        self.out = InnerFrame()
        self.t = 0.0
        # derived maps
        nm = self.muscle.nm
        self.bed_of_muscle = npatch + (np.arange(nm) % max(self.vasc.n - npatch, 1))
        rng = np.random.default_rng(seed + 213)
        self.bed_weight = rng.uniform(0.3, 1.5, self.vasc.n)
        self._patch_count = np.maximum(np.bincount(skin.TAXEL_PATCH_IDX, minlength=npatch), 1)
        self._prev_fullness = 0.5
        self._meal = 0.0
        self.sym = 0.3
        self._u = np.zeros(NeuralMass.N_SENSORY_INPUTS)
        self._drive = np.zeros(len(POP_NAMES), np.float32)
        self._gain = np.ones(len(POP_NAMES), np.float32)
        self._factors = np.zeros(9)
        self._last_valence = 0.0
        self._intero_sp = np.array(agent.latent_spec._intero_setpoints(), float)
        self._intero_sc = np.maximum(np.abs(self._intero_sp) * 0.15, 0.2)
        self.fall_count = 0

    # ------------------------------------------------------------------
    @property
    def n_state(self) -> int:
        """Number of dynamic state variables in the internal world (excluding the
        episodic store, which is data rather than dynamics)."""
        n = (self.vasc.size + self.lungs.size + self.kidney.size + self.liver.size
             + self.gut.size + self.muscle.size + self.thermo.size + self.immune.size
             + self.chem.size + self.clock.size + self.brain.size + self.cond.size
             + self.intero.size)
        return int(n) + int(self.ag.ocular.n_state)

    def counts(self) -> dict:
        return {
            "neural_mass_units": self.brain.N,
            "motor_units": int(self.muscle.nm * self.muscle.nu),
            "chemistry_analytes": self.chem.n,
            "vascular_beds": self.vasc.n, "alveoli": self.lungs.n,
            "nephron_groups": self.kidney.n, "liver_zones": self.liver.n,
            "gut_segments": self.gut.ns, "microbiome_taxa": self.gut.nt,
            "cytokines": self.immune.ny, "immune_populations": self.immune.nc,
            "circadian_oscillators": self.clock.n,
            "corneal_nerve_units": int(self.ag.ocular.U * 2),
            "dynamic_state_variables": self.n_state,
            "episodic_store_values": int(self.memory.size),
        }

    # ------------------------------------------------------------------
    def _sensory_vector(self, ag) -> np.ndarray:
        """Compress what the body feels into the neural mass's input layer."""
        u = self._u
        u[:] = 0.0
        f = ag.frame
        if f is None:
            return u
        tsum = f.tactile_by_region.get("_summary", np.zeros(0))
        m = min(len(tsum), 48)
        u[0:m] = np.tanh(0.25 * tsum[:m])
        v = f.vestibular
        u[48:48 + min(len(v), 13)] = np.tanh(0.2 * v[:13])
        vis = f.visual
        u[61:61 + min(len(vis), 17)] = np.tanh(vis[:17])
        aud = f.auditory
        u[78:84] = np.tanh(aud[:6])
        coch = f.ext.get("cochlea")
        if coch is not None and len(coch) >= 8:
            u[84:92] = np.tanh(0.01 * coch[-8:])
        u[92:96] = np.tanh(f.chemo_summary[:4])
        olf = f.ext.get("olfactory")
        if olf is not None and len(olf) >= 6:
            u[96:102] = np.tanh(olf[-6:])
        p = f.proprio
        if p.size:
            u[102:108] = np.tanh(np.array([np.abs(p[:, 1]).mean(), np.abs(p[:, 2]).mean() * 0.01,
                                           p[:, 3].mean(), p[:, 5].mean(), p[:, 12].mean(),
                                           p[:, 9].mean()]))
        s = (ag.interoception.s - self._intero_sp) / self._intero_sc
        u[108:138] = np.tanh(0.5 * s[:30])
        a = ag.affect_frame
        d = ag.drive_frame
        if a is not None and d is not None:
            u[138:142] = [a.valence, a.arousal, a.dominance, a.stress]
            u[142:158] = d.level[:16]
        oc = ag.ocular.out
        u[158] = oc.discomfort
        u[159] = oc.blur
        return u

    # ------------------------------------------------------------------
    def step(self, dt: float, ag) -> None:
        """Called every physics step by the agent."""
        self.t += dt
        A = self.acc
        for k in A:
            A[k] += dt
        if A["fast"] >= 1.0 / self.RATES["fast"]:
            self._fast(A["fast"], ag); A["fast"] = 0.0
        if A["muscle"] >= 1.0 / self.RATES["muscle"]:
            self._muscle(A["muscle"], ag); A["muscle"] = 0.0
        if A["slow"] >= 1.0 / self.RATES["slow"]:
            self._slow(A["slow"], ag); A["slow"] = 0.0
        if A["clock"] >= 1.0:
            self._clock(A["clock"], ag); A["clock"] = 0.0
        if A["brain"] >= 1.0 / self.RATES["brain"]:
            self._brain(A["brain"], ag); A["brain"] = 0.0
        if A["assoc"] >= 1.0 / self.RATES["assoc"]:
            self._assoc(A["assoc"], ag); A["assoc"] = 0.0
        if A["memory_recall"] >= 1.0 / self.RATES["memory_recall"]:
            self._recall(ag); A["memory_recall"] = 0.0
        if A["memory_store"] >= 1.0 / self.RATES["memory_store"]:
            self._store(ag); A["memory_store"] = 0.0

    # ---- 10 Hz: circulation, lungs, skin -----------------------------------
    def _fast(self, dt, ag) -> None:
        s = ag.interoception.s
        g = IDX
        a = ag.affect_frame
        arousal = float(a.arousal) if a is not None else 0.25
        stress = float(a.stress) if a is not None else 0.1
        pain = float(ag.frame.pain_total) if ag.frame is not None else 0.0
        exertion = float(np.clip(ag.state.torque_effort * 4.0, 0, 1)) if ag.state is not None else 0.0
        self.sym += 0.2 * (np.clip(0.2 + 0.6 * arousal + 0.4 * stress + 0.5 * pain + 0.6 * exertion, 0, 1.5) - self.sym)
        demand = 1.0 + 1.6 * exertion * self.bed_weight
        demand[:len(self.patch_names)] = 1.0
        self.vasc.update(dt, self.sym, float(s[g["core_temp"]]), float(s[g["thermal_discomfort"]]),
                         demand, float(s[g["heart_rate"]]), float(s[g["stroke_volume"]]))
        self.lungs.update(dt, float(s[g["minute_ventilation"]]), float(s[g["cardiac_output"]]),
                          float(s[g["metabolic_rate"]]))
        self.thermo.update(dt, float(s[g["core_temp"]]), ag.cfg.intero.__dict__.get("_ambient", 22.0),
                           self.sym, arousal * 0.5 + 0.3 * stress, self.vasc.flow)
        # ---- skin coupling ---------------------------------------------------
        tac = ag.receptors.tactile
        if ag.frame is not None and ag.frame.tactile.size:
            T = ag.frame.tactile
            from .receptors import CH
            pidx = skin.TAXEL_PATCH_IDX
            n = len(self.patch_names)
            load = np.bincount(pidx, weights=T[:, CH["noci_mech"]] * 6.0 + T[:, CH["noci_heat"]] * 5.0
                               + T[:, CH["noci_cold"]] * 3.0, minlength=n) / self._patch_count
            self._patch_load = load
        else:
            self._patch_load = np.zeros(len(self.patch_names))
        bf = np.clip(self.vasc.flow[:len(self.patch_names)] / max(float(np.mean(self.vasc.flow)), 0.2), 0.2, 2.0)
        tac.set_inner(blood_flow=bf, inflammation=self.immune.infl, sweat=self.thermo.sweat,
                      irritant=np.zeros(len(self.patch_names)), damage=self.immune.damage,
                      pruritogen=self.immune.pruri, humidity=ag.ambient_humidity,
                      airflow=ag.ambient_airflow, arousal=arousal)

    # ---- 20 Hz: motor units ----------------------------------------------------
    def _muscle(self, dt, ag) -> None:
        tau = ag.state.tau if ag.state is not None else np.zeros(ag.meta.n_actuators)
        perf = np.clip(self.vasc.flow[self.bed_of_muscle], 0.2, 3.0)
        self.muscle.update(dt, tau, perf)

    # ---- 2 Hz: chemistry, organs, immune ------------------------------------------
    def _slow(self, dt, ag) -> None:
        s = ag.interoception.s
        g = IDX
        a = ag.affect_frame
        # nutrient intake: a rise in gastric fullness is a meal
        full = float(s[g["gastric_fullness"]])
        d_full = full - self._prev_fullness
        self._prev_fullness = full
        meal = max(d_full, 0.0)
        self._meal += 0.5 * (meal * 20.0 - self._meal) * min(dt, 1.0)
        self.gut.update(dt, np.array([0.6, 0.2, 0.15, 0.5]) * meal * 4.0, self.t)
        ab = self.gut.absorbed
        glu = self.chem.value("glucose")
        ins = self.chem.value("insulin")
        gcg = self.chem.value("glucagon")
        cort = self.chem.value("cortisol") / 300.0
        self.liver.update(dt, glu, ins / 6.0, gcg / 60.0, cort, self._meal)
        osm = float(s[g["osmolality"]])
        self.kidney.update(dt, osm, float(s[g["hydration"]]), self.vasc.map_,
                           self.chem.z[self.chem.idx["aldosterone"]], self.sym)
        pain = float(ag.frame.pain_total) if ag.frame is not None else 0.0
        self.immune.update(dt, getattr(self, "_patch_load", np.zeros(len(self.patch_names))),
                           self.vasc.flow[:len(self.patch_names)], cort,
                           self.clock.sleep_pressure * 0.5)
        ex = float(np.clip(ag.state.torque_effort * 4.0, 0, 1)) if ag.state is not None else 0.0
        st = float(a.stress) if a is not None else 0.1
        F = self._factors
        F[0] = ex
        F[1] = min(self._meal + float(ab.sum()) * 3.0, 1.5)
        F[2] = st + 0.5 * pain
        F[3] = np.cos(self.clock.psi - 2 * np.pi * 8.0 / 24.0)
        F[4] = max(0.0, 1.0 - float(s[g["hydration"]])) * 2.0
        F[5] = self.immune.systemic * 2.0
        F[6] = float(s[g["central_fatigue"]])
        F[7] = max(0.0, float(s[g["core_temp"]]) - 36.8)
        F[8] = self.clock.sleep_pressure
        self.chem.update(dt, F)
        # chemistry back into the shared hormonal state: cortisol feeds the immune and
        # affective layers through their own use of ``cort`` above and below
        z = self.chem.z
        self.out.cortisol = float(z[self.chem.idx["cortisol"]])
        self.out.glucose = glu
        # the two-way link with the original interoception: a meal's glucose rises
        # in the panel too
        self.chem.z[self.chem.idx["glucose"]] += 0.3 * (float(s[g["glucose"]]) - self.chem.value("glucose")) \
            / self.chem.scale[self.chem.idx["glucose"]] * min(dt, 1.0)

    def _clock(self, dt, ag) -> None:
        light = float(np.clip(ag.luminance, 0.0, 1.0)) if ag.receptors.visual.enabled else 0.6
        self.clock.update(dt, light, float(self._meal > 0.05),
                          asleep=bool(ag.interoception.s[IDX["sleepiness"]] > 1.5))

    # ---- 50 Hz: the neural mass -------------------------------------------------------
    def _brain(self, dt, ag) -> None:
        a = ag.affect_frame
        d = ag.drive_frame
        u = self._sensory_vector(ag)
        nm = a.neuromodulators if a is not None else np.zeros(25)
        from .affect import NM
        ne = float(nm[NM["norepinephrine"]]) if a is not None else 0.3
        ach = float(nm[NM["acetylcholine"]]) if a is not None else 0.4
        sero = float(nm[NM["serotonin"]]) if a is not None else 0.55
        dop = float(nm[NM["dopamine_tonic"]]) if a is not None else 0.35
        orx = float(nm[NM["orexin"]]) if a is not None else 0.55
        gaba = float(nm[NM["gaba"]]) if a is not None else 0.45
        fear = float(a.emotion("fear")) if a is not None else 0.0
        arousal = float(a.arousal) if a is not None else 0.25
        gain = self._gain
        gain[:] = 1.0 + 0.6 * (ne - 0.3) + 0.3 * (ach - 0.4) - 0.3 * (gaba - 0.45)
        gain[PI["amygdala"]] *= 1.0 - 0.5 * (sero - 0.55)
        gain[PI["striatum"]] *= 1.0 + 0.8 * (dop - 0.35)
        gain[PI["pfc_dorsolateral"]] *= 1.0 + 0.5 * (dop - 0.35)
        gain[PI["thalamus"]] *= 1.0 - 0.4 * self.clock.sleep_pressure + 0.3 * (self.clock.alertness_drive - 0.5)
        dr = self._drive
        dr[:] = 0.0
        dr[PI["brainstem_arousal"]] = 0.5 * (arousal - 0.3) + 0.4 * (orx - 0.55)
        dr[PI["amygdala"]] = 0.9 * fear + 0.4 * float(ag.frame.pain_total if ag.frame is not None else 0.0)
        if d is not None:
            dr[PI["hypothalamus"]] = 0.4 * float(d.level[0] + d.level[1] + d.level[3] + d.level[4])
            dr[PI["striatum"]] = 0.5 * float(d.level[13])
            dr[PI["acc"]] = 0.5 * float(d.level[5])
        dr[PI["default_mode"]] = 0.25 * (1.0 - arousal)
        motor = float(np.abs(ag.voluntary_target - ag.motor.q_nom).mean())
        dr[PI["motor_ctx"]] = 2.0 * motor
        dr[PI["premotor_ctx"]] = 1.5 * motor
        self.brain.update(dt, u, dr, gain)
        # ---- read out ---------------------------------------------------------------------
        b = self.brain
        o = self.out
        o.pop_means = b.pop_mean.copy()
        o.threat_tone = float(np.clip(1.6 * b.deviation("amygdala") - 0.6 * b.deviation("pfc_ventromedial"), -0.5, 1.5))
        o.arousal_tone = float(np.clip(2.0 * b.deviation("brainstem_arousal"), -1, 1.5))
        o.control_tone = float(np.clip(2.0 * b.deviation("pfc_dorsolateral"), -1, 1.5))
        sens = (b.deviation("visual_ctx") + b.deviation("somatosensory_ctx") + b.deviation("auditory_ctx")) / 3.0
        o.mind_wandering = float(np.clip(2.2 * b.deviation("default_mode") - 1.2 * sens, -1, 1.5))
        o.rumination = float(np.clip(o.mind_wandering * max(0.0, o.threat_tone + 0.3), 0, 1.5))

    # ---- 10 Hz: conditioning and the interoceptive model -----------------------------------------
    def _assoc(self, dt, ag) -> None:
        f = ag.frame
        stim = np.zeros(Conditioning.N_STIM)
        if f is not None and f.tactile.size:
            from .receptors import CH
            pidx = skin.TAXEL_PATCH_IDX
            n = len(self.patch_names)
            cn = np.bincount(pidx, weights=f.tactile[:, CH["normal_force"]], minlength=n)
            stim[:n] = (cn > 0.5).astype(float)
        sk = ag.skills
        for i, name in enumerate(("apple", "mug", "stone", "cushion", "sphere_toy")):
            try:
                if sk.world.in_view(sk.world.obj_pos(name)):
                    stim[46 + i] = 1.0
            except Exception:
                pass
        if f is not None:
            stim[52] = float(f.auditory[0] > 0.2)
            stim[53] = float(f.chemo_summary[0] > 0.3)
        stim[54] = float(ag.ocular.out.discomfort > 0.3)
        stim[55] = float(sk.social_pulse > 0.2)
        stim[56] = float(sk.touching is not None)
        a = ag.affect_frame
        outcome = 0.0
        if a is not None:
            outcome = float(a.valence) - 0.4 * float(f.pain_total if f is not None else 0.0)
        self.cond.update(dt, stim, outcome)
        s = ag.interoception.s
        self.intero.update(s)
        self.out.interoceptive_surprise = float(self.intero.surprise)

    # ---- memory -------------------------------------------------------------------------------------
    def _memory_features(self, ag) -> np.ndarray:
        a = ag.affect_frame
        d = ag.drive_frame
        parts = []
        if a is not None:
            parts += [a.valence, a.arousal, a.dominance, a.tension, a.stress]
            parts += list(a.emotions)
        if d is not None:
            parts += list(d.level)
        parts += list(ag.ocular.summary_features())
        parts += list(self.brain.pop_mean)
        beh = ag.behavior.current.desc
        if beh is not None and len(beh):
            e = np.zeros(32)
            for i, v in enumerate(beh):
                e[(i * 7 + int(v)) % 32] += 1.0
            parts += list(e / max(np.linalg.norm(e), 1e-6))
        f = ag.frame
        if f is not None:
            parts += list(np.tanh(0.25 * f.tactile_by_region.get("_summary", np.zeros(0))[:40]))
        return np.array(parts, np.float32)

    def _recall(self, ag) -> None:
        fam, val = self.memory.recall(self._memory_features(ag))
        self.out.familiarity = fam
        self.out.memory_valence = val

    def _store(self, ag) -> None:
        a = ag.affect_frame
        if a is None:
            return
        v = float(a.valence)
        sal = abs(v - self._last_valence) * 4.0 + 0.6 * float(a.arousal) + 0.5 * float(ag.frame.pain_total if ag.frame is not None else 0.0)
        self._last_valence = v
        if sal > 0.35 or self.memory.n < 8:
            self.memory.store(self._memory_features(ag), v, min(sal, 1.5), self.t)

    # ---- outputs ----------------------------------------------------------------------------------------------
    def finalize(self, ag) -> InnerFrame:
        """Assemble the summary vector and the extra drives (call at the cognitive rate)."""
        o = self.out
        mus = self.muscle.summary()
        o.muscle_fatigue = mus["fatigue"]
        o.muscle_soreness = mus["soreness"]
        o.gut_discomfort = self.gut.discomfort
        o.systemic_inflammation = self.immune.systemic
        o.clock_hour = self.clock.hour
        o.sleep_pressure = self.clock.sleep_pressure
        f = ag.frame
        shift = 0.0
        if f is not None and f.tactile.size and "ischemia" in CH:
            shift = float(np.clip(np.percentile(f.tactile[:, CH["ischemia"]], 99) * 8.0, 0, 1.3))
        oc = ag.ocular.out
        o.extra_drives = np.array([
            float(np.clip(1.1 * oc.discomfort + 0.5 * oc.rub_urge, 0, 1.4)),
            float(np.clip(2.0 * mus["soreness"] + 0.6 * mus["fatigue"], 0, 1.4)),
            float(np.clip(o.gut_discomfort, 0, 1.4)),
            float(np.clip(0.7 * float(ag.interoception.s[IDX["central_fatigue"]]) + 0.5 * o.sleep_pressure
                          + 0.4 * max(o.mind_wandering, 0), 0, 1.4)),
            shift,
        ])
        o.summary = self._summary(ag)
        return o

    def _summary(self, ag) -> np.ndarray:
        v = []
        v += list(np.tanh(self.chem.group_summary()))                      # 42
        v += [(self.vasc.map_ - 90) / 20.0, float(self.vasc.tone.mean()) - 0.5,
              float(self.vasc.flow[:len(self.patch_names)].mean()) - 1.0, float(self.vasc.flow.std())]
        v += [(self.lungs.sao2 - 97) / 5.0, (self.lungs.paco2 - 40) / 6.0,
              float(self.lungs.po2.std()) / 20.0, float(self.lungs.vq.std())]
        v += [self.kidney.urine - 1.0, self.kidney.adh - 0.5, float(self.liver.glycogen.mean()) - 0.5,
              self.liver.glucose_out]
        v += [self.gut.discomfort, float(self.gut.motility.mean()) - 0.4, self.gut.scfa - 0.5,
              float(self.gut.pool.sum()), float(-(self.gut.abund * np.log(self.gut.abund + 1e-9)).sum()) / 4.0,
              float(self.gut.absorbed.sum())]
        mus = self.muscle.summary()
        v += [mus["fatigue"], mus["soreness"], mus["capacity_min"] - 0.8, mus["temp"] - 34.0, mus["active"]]
        v += [float(self.thermo.sweat.mean()), float(self.thermo.temp.mean()) - 33.0,
              float(self.thermo.wet.mean())]
        v += [float(self.immune.cyto.mean()), float(self.immune.cells.mean()) - 1.0,
              float(self.immune.damage.mean()), float(self.immune.infl.mean())]
        v += [np.sin(self.clock.psi), np.cos(self.clock.psi), self.clock.R, self.clock.sleep_pressure,
              self.clock.alertness_drive, self.clock.t_awake / 36000.0]
        v += list(self.brain.pop_mean)                                      # 20
        o = self.out
        v += [o.threat_tone, o.rumination, o.arousal_tone, o.control_tone, o.mind_wandering,
              o.interoceptive_surprise]
        v += [self.memory.familiarity, self.memory.recalled_valence, self.memory.n / self.memory.cap]
        v += [self.cond.pred, self.cond.rpe, float(np.abs(self.cond.V).sum()) * 0.2]
        out = np.zeros(N_INNER_SUMMARY)
        m = min(len(v), N_INNER_SUMMARY)
        out[:m] = np.nan_to_num(np.array(v[:m], float), nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(out, -6, 6)
