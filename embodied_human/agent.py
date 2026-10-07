"""
The embodied human.

Wires every subsystem into one multi-rate loop.  The scheduler mirrors the
nervous system's own hierarchy of rates:

    1000 Hz   physics           MuJoCo
     500 Hz   receptors         transduction
     200 Hz   afferents         conduction, rate coding, reafference
     100 Hz   interoception     organ systems, autonomic
      50 Hz   affect            neuromodulators, appraisal, emotion
      10 Hz   cognition         active inference, policy selection
       1 Hz   mood              allostatic slow state

Everything the agent does flows through one loop: feel -> interpret -> want ->
predict -> choose -> move -> feel again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    raise ImportError("MuJoCo is required: python -m pip install mujoco") from exc

from . import receptors as R
from .active_inference import ActiveInference, default_policies
from .affect import AffectFrame, AffectInputs, AffectSystem, N_EMOTIONS, EMOTIONS
from .afferents import AfferentSystem, AfferentFrame
from .build_model import ModelMeta, load_model, lowest_foot_z
from .config import SimConfig
from .drives import DRIVES, DriveFrame, DriveSystem, SETPOINTS
from .interoception import (IDX, INTERO_NAMES, InteroInputs, InteroceptiveSystem,
                            N_INTEROCEPTION)
from .behavior_exec import BehaviorExecutor
from .complexity import C as COMPLEXITY
from .inner_world import (EXTRA_DRIVES, N_INNER_SUMMARY, N_OCULAR_SUMMARY, InnerWorld)
from .intrinsic import IntrinsicMotivation, RewardFrame
from .locomotion import Gait
from .ocular import EyeRig, OcularInputs, OcularSurface
from .skills import SkillSystem
from .motor import MotorFrame, MotorSystem
from .predictive import PredictionFrame, PredictiveSystem
from .receptors import ReceptorFrame, ReceptorSystem
from .skeleton import BONES, nominal_posture
from .state import BodyState, ContactInfo

HUMAN_BODY_IDS: set[int] = set()
for _b in BONES:
    HUMAN_BODY_IDS.add(_b.name)


# ==========================================================================
# Latent sensory space
# ==========================================================================
TACTILE_LATENT_CHANNELS = ("sum_normal", "max_noci", "max_ct")


class LatentSpec:
    """Builds the compact sensory latent **and** the preferred latent.

    The two must share a layout, because the whole point of active inference is
    that risk is measured as the distance between the predicted sensory state
    and the *preferred* one.  Building them together guarantees they can never
    drift out of alignment.
    """

    def __init__(self, cfg: SimConfig, meta, receptor_system: ReceptorSystem):
        self.cfg = cfg
        self.n_joints = meta.n_actuators
        self.joint_names = [n for _, n, _ in meta.joint_order]
        nominal = nominal_posture()
        self.q_nom = np.array([nominal.get(n, 0.0) for n in self.joint_names])
        self.regions = [r for r in R.REGION_NAMES]
        self.n_regions = len(self.regions)

        # ---- layout ---------------------------------------------------
        self.blocks: list[tuple[str, int]] = []
        self.blocks.append(("q", self.n_joints))
        self.blocks.append(("qd", self.n_joints))
        self.blocks.append(("tactile", 3 * self.n_regions))
        self.blocks.append(("vestibular", len(R.VESTIBULAR_CHANNELS)))
        self.blocks.append(("visual", len(R.VISUAL_CHANNELS)))
        self.blocks.append(("auditory", len(R.AUDITORY_CHANNELS)))
        self.blocks.append(("chem", cfg.chemo.n_olfactory_channels
                            + len(R.GUSTATORY_CHANNELS)))
        self.blocks.append(("intero", N_INTEROCEPTION))
        self.blocks.append(("affect", 10))
        # appended after everything that already existed, so no earlier offset moves
        self.blocks.append(("ocular", N_OCULAR_SUMMARY))
        self.blocks.append(("inner", N_INNER_SUMMARY))
        self.offset: dict[str, int] = {}
        self.dim = 0
        for name, size in self.blocks:
            self.offset[name] = self.dim
            self.dim += size

    # ------------------------------------------------------------------
    def build(self, frame: ReceptorFrame, aff, intero: InteroceptiveSystem,
              affect: AffectFrame, drives: DriveFrame,
              state: BodyState, pred: PredictionFrame | None,
              ocular=None, inner=None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (latent, preferred, precision)."""
        L = np.zeros(self.dim)
        P = np.zeros(self.dim)
        W = np.ones(self.dim)

        # ---- proprioception -----------------------------------------
        o = self.offset["q"]
        n = self.n_joints
        L[o:o + n] = state.q
        P[o:o + n] = self.q_nom                     # prefer the rest posture
        W[o:o + n] = 0.35
        o = self.offset["qd"]
        L[o:o + n] = fclip(state.qd, -8, 8)
        P[o:o + n] = 0.0                            # prefer stillness
        W[o:o + n] = 0.12

        # ---- tactile, one triple per body region --------------------
        o = self.offset["tactile"]
        T = frame.tactile
        for i, r in enumerate(self.regions):
            idxs = R._REGION_INDEX[r]
            sub = T[idxs]
            sum_normal = float(sub[:, R.CH["normal_force"]].sum())
            max_noci = float(np.maximum(
                sub[:, R.CH["noci_mech"]].max(),
                max(sub[:, R.CH["noci_heat"]].max(),
                    sub[:, R.CH["noci_cold"]].max())))
            max_ct = float(sub[:, R.CH["ct"]].max())
            L[o + 3 * i] = np.log1p(sum_normal)
            L[o + 3 * i + 1] = max_noci
            L[o + 3 * i + 2] = max_ct
            # prefer no pain anywhere; prefer no contact except on the feet,
            # where the support force is what keeps the body up
            feet = r.endswith("_l") or r.endswith("_r")
            is_foot = any(k in r for k in ("sole", "toes", "foot"))
            P[o + 3 * i] = np.log1p(sum_normal) if is_foot else 0.0
            P[o + 3 * i + 1] = 0.0
            P[o + 3 * i + 2] = 0.35 if is_foot else 0.0
            W[o + 3 * i] = 0.25
            W[o + 3 * i + 1] = 3.0 + 5.0 * float(drives.level[5])   # pain
            W[o + 3 * i + 2] = 0.4 + 1.5 * max(0.0, affect.valence)

        # ---- vestibular ---------------------------------------------
        o = self.offset["vestibular"]
        nv = len(R.VESTIBULAR_CHANNELS)
        L[o:o + nv] = fclip(frame.vestibular, -40, 40)
        P[o:o + nv] = frame.vestibular
        P[o + 0:o + 6] = 0.0            # prefer no rotation
        P[o + 9:o + 12] = frame.vestibular[9:12]   # gravity stays as it is
        W[o:o + nv] = 0.10
        W[o + 0:o + 3] = 0.5

        # ---- vision --------------------------------------------------
        o = self.offset["visual"]
        nvi = len(R.VISUAL_CHANNELS)
        L[o:o + nvi] = frame.visual
        P[o:o + nvi] = frame.visual
        P[o + 2] = min(0.9, frame.visual[2] + 0.15)   # prefer contrast
        P[o + 3] = frame.visual[3]                    # motion: no preference
        W[o:o + nvi] = 0.05
        W[o + 2] = 0.30
        W[o + 7] = 0.05                                # pupil: not a goal

        # ---- audition ------------------------------------------------
        o = self.offset["auditory"]
        na = len(R.AUDITORY_CHANNELS)
        L[o:o + na] = frame.auditory
        P[o:o + na] = frame.auditory
        P[o + 0] = 0.0                                 # prefer quiet
        W[o:o + na] = 0.08

        # ---- chemoreception ------------------------------------------
        o = self.offset["chem"]
        nc = self.cfg.chemo.n_olfactory_channels
        L[o:o + nc] = frame.olfactory
        P[o:o + nc] = 0.0
        W[o:o + nc] = 0.05
        L[o + nc:o + nc + 5] = frame.gustatory
        P[o + nc:o + nc + 5] = 0.0
        W[o + nc:o + nc + 5] = 0.05 * (1.0 + drives.level[0])   # hunger

        # ---- interoception -------------------------------------------
        o = self.offset["intero"]
        L[o:o + N_INTEROCEPTION] = intero.s
        pref = np.array(self._intero_setpoints(), float)
        P[o:o + N_INTEROCEPTION] = pref
        w = np.ones(N_INTEROCEPTION) * 0.02
        # drive urgency sharpens attention on the relevant variables
        w[IDX["pain_intensity"]] = 6.0 * (0.2 + drives.level[5])
        w[IDX["nociceptive_input"]] = 4.0
        w[IDX["core_temp"]] = 1.2 * (0.3 + drives.level[3] + drives.level[4])
        w[IDX["co2_arterial"]] = 0.5 * (0.2 + drives.level[6])
        w[IDX["air_hunger"]] = 3.0
        w[IDX["glucose"]] = 0.25 * (0.2 + drives.level[0])
        w[IDX["osmolality"]] = 0.25 * (0.2 + drives.level[1])
        w[IDX["hydration"]] = 0.25 * (0.2 + drives.level[1])
        w[IDX["sleepiness"]] = 0.4 * (0.2 + drives.level[2])
        w[IDX["heart_rate"]] = 0.06
        w[IDX["spo2"]] = 0.10
        w[IDX["lactate"]] = 0.10
        w[IDX["peripheral_fatigue"]] = 0.2
        w[IDX["nausea"]] = 1.0
        w[IDX["bladder_fullness"]] = 0.5 * (0.2 + drives.level[8])
        w[IDX["sense_of_effort"]] = 0.05
        W[o:o + N_INTEROCEPTION] = w

        # ---- affect ---------------------------------------------------
        o = self.offset["affect"]
        L[o:o + 10] = [affect.valence, affect.arousal, affect.dominance,
                       affect.tension, affect.emotion("fear"),
                       affect.emotion("joy"), affect.emotion("anger"),
                       affect.emotion("sadness"), affect.emotion("disgust"),
                       affect.stress]
        P[o:o + 10] = [0.60, 0.45, 0.70, 0.08, 0.0, 0.45, 0.0, 0.0, 0.0, 0.08]
        W[o:o + 10] = [1.0, 0.2, 0.3, 0.6, 0.8, 0.2, 0.4, 0.3, 0.5, 0.6]

        # ---- the eyes ------------------------------------------------------
        o = self.offset["ocular"]
        if ocular is not None:
            L[o:o + N_OCULAR_SUMMARY] = ocular
            # prefer comfortable, well-wetted eyes with an ordinary blink rate
            P[o:o + N_OCULAR_SUMMARY] = [0, 0, 0, 0, 0, 0.5, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0.5, 0]
            W[o:o + N_OCULAR_SUMMARY] = [1.2, 1.2, 1.2, 1.6, 0.8, 0.1, 0.1, 0.8, 0.1, 0.5, 0.2,
                                         0.1, 0.5, 0.2, 0.1, 1.5]
        # ---- the inner world: observed, no particular preference -----------
        o = self.offset["inner"]
        if inner is not None:
            L[o:o + N_INNER_SUMMARY] = inner
            P[o:o + N_INNER_SUMMARY] = inner
            W[o:o + N_INNER_SUMMARY] = 0.02

        # ---- precision from the predictive layer ---------------------
        if pred is not None and len(pred.precision) == self.dim:
            W = W * fclip(pred.precision, 0.05, 4.0)

        return L, P, fclip(W, 0.0, 50.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _intero_setpoints() -> list[float]:
        s = np.zeros(N_INTEROCEPTION)
        g = IDX
        s[g["heart_rate"]] = 62.0
        s[g["hrv_rmssd"]] = 55.0
        s[g["stroke_volume"]] = 70.0
        s[g["systolic_bp"]] = 118.0
        s[g["diastolic_bp"]] = 74.0
        s[g["perfusion_skin"]] = 1.0
        s[g["perfusion_muscle"]] = 1.0
        s[g["resp_rate"]] = 13.0
        s[g["tidal_volume"]] = 0.5
        s[g["spo2"]] = 98.0
        s[g["co2_arterial"]] = 40.0
        s[g["o2_arterial"]] = 97.0
        s[g["air_hunger"]] = 0.02
        s[g["glucose"]] = 5.2
        s[g["glycogen"]] = 1.0
        s[g["lactate"]] = 1.0
        s[g["atp_reserve"]] = 1.0
        s[g["core_temp"]] = 36.8
        s[g["skin_temp_mean"]] = 33.0
        s[g["thermal_discomfort"]] = 0.0
        s[g["gastric_fullness"]] = 0.55
        s[g["ghrelin"]] = 0.30
        s[g["leptin"]] = 0.60
        s[g["nausea"]] = 0.0
        s[g["hydration"]] = 1.0
        s[g["osmolality"]] = 290.0
        s[g["bladder_fullness"]] = 0.20
        s[g["sodium"]] = 140.0
        s[g["potassium"]] = 4.2
        s[g["peripheral_fatigue"]] = 0.05
        s[g["central_fatigue"]] = 0.05
        s[g["adenosine"]] = 0.20
        s[g["sleepiness"]] = 0.15
        s[g["alertness"]] = 0.85
        s[g["cytokine"]] = 0.05
        s[g["inflammation"]] = 0.03
        s[g["sickness_behavior"]] = 0.0
        s[g["tissue_damage"]] = 0.0
        s[g["nociceptive_input"]] = 0.0
        s[g["pain_intensity"]] = 0.0
        s[g["pain_unpleasantness"]] = 0.0
        s[g["central_sensitisation"]] = 0.0
        s[g["descending_inhibition"]] = 0.6
        s[g["sense_of_effort"]] = 0.0
        s[g["breathlessness"]] = 0.0
        s[g["motor_command_magnitude"]] = 0.0
        s[g["muscle_tone"]] = 0.2
        s[g["adrenaline_h"]] = 0.12
        s[g["noradrenaline_h"]] = 0.30
        s[g["cortisol_h"]] = 0.28
        s[g["insulin_sensitivity"]] = 1.0
        return s.tolist()


# ==========================================================================
# The agent
# ==========================================================================
@dataclass
class AgentStep:
    """One logged tick of everything the agent is."""
    t: float = 0.0
    # body
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_z: float = 0.0
    fallen: bool = False
    n_contact: int = 0
    self_touch: bool = False
    # senses
    touch_intensity: float = 0.0
    pain: float = 0.0
    affective_touch: float = 0.0
    heart_rate: float = 0.0
    core_temp: float = 0.0
    # affect
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    stress: float = 0.0
    dominant_emotion: str = ""
    emotions: np.ndarray = field(default_factory=lambda: np.zeros(N_EMOTIONS))
    neuromodulators: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # cognition
    policy: str = ""
    free_energy: float = 0.0
    reward: float = 0.0
    drive_pressure: float = 0.0
    most_urgent_drive: str = ""
    stepping: str = "none"


class EmbodiedHuman:
    """A simulated person in MuJoCo."""

    def __init__(self, cfg: SimConfig | None = None, *, vision: bool | None = None,
                 touch_sensors: bool = True):
        self.cfg = cfg or SimConfig()
        if vision is not None:
            self.cfg.vision.enabled = bool(vision)
        self.wall0 = time.perf_counter()

        model, data, meta = load_model(self.cfg,
                                       include_touch_sensors=touch_sensors,
                                       xml_path=self.cfg.out_dir / "models" / "human.xml")
        self.model, self.data, self.meta = model, data, meta
        self.dt = float(model.opt.timestep)

        # ---- subsystems ---------------------------------------------
        self.receptors = ReceptorSystem(self.cfg, meta)
        self.latent_spec = LatentSpec(self.cfg, meta, self.receptors)
        self.afferents = AfferentSystem(self.cfg, meta, self.receptors.tactile.n_taxels,
                                        meta.n_actuators)
        self.interoception = InteroceptiveSystem(self.cfg, meta)
        self.affect = AffectSystem(self.cfg)
        self.drives = DriveSystem(self.cfg)
        self.predict = PredictiveSystem(self.cfg, self.latent_spec.dim,
                                        meta.n_actuators,
                                        n_taxels=self.receptors.tactile.n_taxels)
        self.inference = ActiveInference(self.cfg, meta)
        self.motor = MotorSystem(self.cfg, meta)
        self.motivation = IntrinsicMotivation(self.cfg)
        self.gait = Gait(model, meta)
        # The whole-body controller holds the stance by default.  The older
        # posture/balance system (motor.py) cannot keep its footing once the
        # arms really move -- which, since a bug in the postural-priority term was
        # fixed (the arms used to have *no* authority), they now do.  Set to False
        # to get the original standing controller back.
        self.gait.hold_stance = True
        self.skills = SkillSystem(self)
        # When False the active-inference layer still observes and learns but
        # does not move the body, so an external controller (the mind, a test
        # script) has the body to itself.
        self.autonomous = True
        # ---- the eyes, and the open-ended behaviour repertoire -----------------
        self.ocular = OcularSurface(self.cfg.seed)
        self.eye_rig = EyeRig(model)
        self.behavior = BehaviorExecutor(self, seed=self.cfg.seed)
        self.ambient_humidity = 0.45     # relative humidity of the room (0..1)
        self.ambient_airflow = 0.05      # m/s over the eyes with no walking
        self._behavior_on = True
        # the internal world (organs, circadian clock, neural mass, memory ...)
        self.inner = InnerWorld(self) if COMPLEXITY.inner_world else None

        # geom -> surface temperature, and geom -> scene-object name
        self.geom_temp = np.full(model.ngeom, 22.0, float)
        self.geom_objname = [""] * model.ngeom
        for obj in meta.objects:
            gid = meta.geom_ids.get(obj.name)
            if gid is not None:
                self.geom_temp[gid] = obj.temperature
                self.geom_objname[gid] = obj.name
        # Name lookups are resolved once here instead of per contact per step:
        # mj_id2name is a string lookup and dominated _gather_state.
        self.geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                           for i in range(model.ngeom)]
        self.body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
                           for i in range(model.nbody)]
        self.geom_is_human = np.zeros(model.ngeom, bool)
        for name in HUMAN_BODY_IDS:
            bid = meta.body_ids.get(name)
            if bid is not None:
                self.geom_is_human |= (model.geom_bodyid == bid)
        self.torque_limit = meta.torque_limit.copy()
        # index arrays so the per-step state gather is three fancy-indexing calls
        self._q_idx = np.array([meta.qpos_addr[n] for _, n, _ in meta.joint_order])
        self._d_idx = np.array([meta.dof_addr[n] for _, n, _ in meta.joint_order])
        self._a_idx = np.array([meta.name_to_actuator[n] for _, n, _ in meta.joint_order])

        # ---- runtime state ------------------------------------------
        self.t = 0.0
        self.step_count = 0
        self.acc = {k: 0.0 for k in ("receptor", "afferent", "interoception",
                                     "affect", "cognition", "mood", "ocular")}
        self.frame: ReceptorFrame | None = None
        self.aff: AfferentFrame | None = None
        self.affect_frame: AffectFrame | None = None
        self.drive_frame: DriveFrame | None = None
        self.pred_frame: PredictionFrame | None = None
        self.motor_frame: MotorFrame | None = None
        self.reward_frame: RewardFrame | None = None
        self.state: BodyState | None = None
        self.latent = np.zeros(self.latent_spec.dim)
        self.preferred = np.zeros(self.latent_spec.dim)
        self.precision = np.ones(self.latent_spec.dim)
        self.action = np.array(self.motor.q_nom, float)
        self.voluntary_target = self.motor.q_nom.copy()
        self.jaw_target = 0.0
        self.gaze_target = (0.0, 0.0)
        self.blink_rate = 0.0
        self.luminance = 1.0
        self.total_energy_j = 0.0
        self.step_durations: list[float] = []
        self.max_speed = 0.0
        self.last_step_time = time.perf_counter()
        self.last_inference_info: dict = {}

        self._init_gaze()

    # ------------------------------------------------------------------
    def _init_gaze(self) -> None:
        for s in ("l", "r"):
            for jn, val in ((f"eye_{s}_yaw", 0.0), (f"eye_{s}_pitch", 0.0)):
                a = self.meta.qpos_addr.get(jn)
                if a is not None:
                    self.data.qpos[a] = val
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------
    def _gather_state(self) -> BodyState:
        m, d, meta = self.model, self.data, self.meta
        q = d.qpos[self._q_idx]
        qd = d.qvel[self._d_idx]
        tau = d.actuator_force[self._a_idx]

        root_a = meta.root_qpos_addr
        pelvis_id = meta.body_ids["pelvis"]
        com = d.subtree_com[pelvis_id].copy()          # the *human* COM
        site_pos = {nm: d.site_xpos[sid].copy()
                    for nm, sid in meta.landmark_site_ids.items() if sid >= 0}
        site_vel = {}
        for nm in ("imu_head", "soma_hand_l", "soma_hand_r", "soma_foot_l",
                   "soma_foot_r"):
            sid = meta.landmark_site_ids.get(nm)
            if sid is not None and sid >= 0:
                site_vel[nm] = np.array(d.site_xpos[sid], float)

        # ---- contacts -------------------------------------------------
        contacts: list[ContactInfo] = []
        self_touch = False
        for ci in range(d.ncon):
            c = d.contact[ci]
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, f)
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            n1 = self.geom_names[c.geom1]
            n2 = self.geom_names[c.geom2]
            is_self = bool(self.geom_is_human[c.geom1] and self.geom_is_human[c.geom2])
            if is_self:
                self_touch = True
            contacts.append(ContactInfo(
                geom1=int(c.geom1), geom2=int(c.geom2), body1=b1, body2=b2,
                name1=n1, name2=n2,
                pos=np.array(c.pos, float), normal=np.array(c.frame[:3], float),
                force=f, dist=float(c.dist), is_self=is_self,
                other_temp=float(self.geom_temp[c.geom2]),
                other_object=self.geom_objname[c.geom2],
                friction=float(m.geom_friction[c.geom1][0]),
            ))
            contacts[-1].temp1 = float(self.geom_temp[c.geom1])
            contacts[-1].temp2 = float(self.geom_temp[c.geom2])

        # ---- support polygon centre ----------------------------------
        fl = site_pos.get("soma_foot_l", np.zeros(3))
        fr = site_pos.get("soma_foot_r", np.zeros(3))
        support = np.array([0.5 * (fl[0] + fr[0]), 0.5 * (fl[1] + fr[1]), 0.0])
        com_xy = com[:2] - support[:2]

        # ---- fall detection ------------------------------------------
        up = np.zeros(3)
        quat = np.array(d.qpos[root_a + 3:root_a + 7], float)
        mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), quat)
        upright = float(up[2])
        fallen = bool(com[2] < 0.55 or upright < 0.55)

        effort = float(np.mean(np.abs(tau) / np.maximum(meta.torque_limit, 1e-9)))
        st = BodyState(
            t=self.t, step=self.step_count,
            q=q, qd=qd, tau=tau, ctrl=self.data.ctrl.copy(),
            joint_limit_proximity=fclip(
                np.maximum((q - (self.motor.q_hi - 0.1 * (self.motor.q_hi - self.motor.q_lo)))
                           / np.maximum(0.1 * (self.motor.q_hi - self.motor.q_lo), 1e-6),
                           ((self.motor.q_lo + 0.1 * (self.motor.q_hi - self.motor.q_lo)) - q)
                           / np.maximum(0.1 * (self.motor.q_hi - self.motor.q_lo), 1e-6)),
                0.0, 1.0),
            q_min=self.motor.q_lo.copy(), q_max=self.motor.q_hi.copy(),
            root_pos=np.array(d.qpos[root_a:root_a + 3], float),
            root_quat=quat,
            root_linvel=np.array(d.qvel[0:3], float),
            root_angvel=np.array(d.qvel[3:6], float),
            com=com, support_center=support, com_over_support=com_xy,
            contacts=contacts, n_contact=len(contacts), self_touch=self_touch,
            fallen=fallen, torque_effort=effort,
            site_pos=site_pos, site_vel=site_vel,
            gaze_pos=site_pos.get("gaze", np.zeros(3)),
            joint_names=self.motor.names,
        )
        self.total_energy_j += abs(float(np.dot(tau, qd))) * self.dt
        st.actuator_power = float(np.dot(tau, qd))
        st.mechanical_power = st.actuator_power
        return st

    # ------------------------------------------------------------------
    # One physics step
    # ------------------------------------------------------------------
    def step(self) -> None:
        cfg, m, d = self.cfg, self.model, self.data
        dt = self.dt

        state = self._gather_state()
        self.state = state

        # ---- 500 Hz : transduction -----------------------------------
        self.acc["receptor"] += dt
        if self.acc["receptor"] >= 1.0 / cfg.rates.receptor:
            self.acc["receptor"] = 0.0
            blood = float(fclip(
                self.interoception.s[IDX["perfusion_skin"]], 0.1, 1.8))
            self.frame = self.receptors.sense(
                m, d, self.meta, state,
                arousal=float(self.affect.arousal), blood_flow=blood)
            self.luminance = float(fclip(self.frame.visual[0] * 1.6, 0.05, 1.0))

        if self.frame is None:
            self.frame = self.receptors.sense(m, d, self.meta, state,
                                              arousal=float(self.affect.arousal))

        # ---- 200 Hz : afferents --------------------------------------
        self.acc["afferent"] += dt
        if self.acc["afferent"] >= 1.0 / cfg.rates.afferent:
            self.acc["afferent"] = 0.0
            self.aff = self.afferents.process(self.frame, state, self.data.ctrl)

        # ---- 100 Hz : the eyes -----------------------------------------
        self.acc["ocular"] += dt
        if self.acc["ocular"] >= 0.01:
            dt_o = self.acc["ocular"]
            self.acc["ocular"] = 0.0
            lid = self.behavior.current.lid_aperture if self._behavior_on else None
            oc = self.ocular.update(dt_o, self._ocular_inputs(), aperture_cmd=lid)
            self.receptors.visual.eye_state = {"aperture": float(np.mean(oc.aperture)),
                                               "blur": oc.blur}
            self.eye_rig.apply(oc.aperture, self.receptors.visual.pupil, oc.redness)

        # ---- 100 Hz : interoception ----------------------------------
        self.acc["interoception"] += dt
        if self.acc["interoception"] >= 1.0 / cfg.rates.interoception:
            self.acc["interoception"] = 0.0
            inp = InteroInputs(
                exertion=state.torque_effort,
                mechanical_power=state.actuator_power,
                arousal=float(self.affect.arousal),
                stress=float(self.affect.stress),
                pain_afferent=self.frame.pain_total + 0.5 * self.ocular.out.ocular_pain,
                itch_afferent=self.frame.itch_total,
                affective_touch=self.frame.affective_touch,
                touch_intensity=self.frame.touch_intensity,
                ambient_temp=self.cfg.intero.__dict__.get("_ambient", 22.0),
                contact_heat=self._contact_thermal(+1),
                contact_cold=self._contact_thermal(-1),
                taste_sweet=float(self.frame.gustatory[0]) if len(self.frame.gustatory) else 0.0,
                taste_bitter=float(self.frame.gustatory[3]) if len(self.frame.gustatory) > 3 else 0.0,
                odor_food=float(self.frame.chemo_summary[0]) if len(self.frame.chemo_summary) else 0.0,
                vestibular_discomfort=float(np.linalg.norm(self.frame.vestibular[0:3])) * 0.2,
                endorphin=float(self.affect.neuromodulator("endorphin")),
                falling=1.0 if state.fallen else 0.0,
                sleep_debt=0.0,
            )
            self.interoception.update(1.0 / cfg.rates.interoception, inp)

        # ---- 50 Hz : affect ------------------------------------------
        # Affect and drives are mutually dependent (mood colours how urgent a
        # need feels; an urgent need colours mood), so the loop is closed with
        # a one-tick delay: appraisal -> affect -> drives -> next appraisal.
        self.acc["affect"] += dt
        if self.acc["affect"] >= 1.0 / cfg.rates.affect:
            self.acc["affect"] = 0.0
            dt_aff = 1.0 / cfg.rates.affect
            self.affect_frame = self.affect.update(dt_aff, self._affect_inputs())
            self.drive_frame = self.drives.update(
                dt_aff, self.interoception, self.affect,
                afferent_pain=self.frame.pain_total if self.frame else 0.0,
                afferent_itch=self.frame.itch_total if self.frame else 0.0,
                social_contact=float(self.frame.affective_touch) * 2.0
                if self.frame else 0.0,
                novelty=float(self.pred_frame.free_energy_norm) * 0.5
                if self.pred_frame else 0.0,
                balance_error=self.motor.balance_error,
                fallen=1.0 if state.fallen else 0.0,
                extra=self.inner.out.extra_drives if self.inner is not None else None)

        if self.affect_frame is None:
            self.affect_frame = self.affect.update(0.0, AffectInputs())
        if self.drive_frame is None:
            self.drive_frame = self.drives.update(
                0.0, self.interoception, self.affect)

        # ---- the internal world (each part at its own rate) --------------------
        if self.inner is not None:
            self.inner.step(dt, self)

        # ---- 50 Hz : behaviour ---------------------------------------------
        self.acc.setdefault("behavior", 0.0)
        self.acc["behavior"] += dt
        if self.acc["behavior"] >= 0.02:
            dt_b = self.acc["behavior"]
            self.acc["behavior"] = 0.0
            if self._behavior_on and self.latent is not None:
                # autonomous: these behaviours *are* what the person does.
                # otherwise: ambient -- involuntary and expressive behaviour only
                self.behavior.ambient = not self.autonomous
                self.behavior.step(dt_b, self.t)

        # ---- 10 Hz : cognition ---------------------------------------
        self.acc["cognition"] += dt
        if self.acc["cognition"] >= 1.0 / cfg.rates.cognition:
            self.acc["cognition"] = 0.0
            self._cognitive_tick()

        # ---- 1 Hz : mood / allostasis --------------------------------
        self.acc["mood"] += dt
        if self.acc["mood"] >= 1.0:
            self.acc["mood"] = 0.0
            self.afferents.learn_reafference()

        # ---- motor: every physics step -------------------------------
        # The CPG oscillates the legs without any balance coupling, so it is
        # gated by stability: a body that is not upright does not walk.
        cpg_cmd = float(self.inference.cpg_command()) * float(fclip(
            1.0 - 8.0 * self.motor.balance_error, 0.0, 1.0))
        gait_tau = gait_mask = None
        self.skills.update(dt)
        voluntary = self.skills.override_target(self.voluntary_target)
        if self.gait.cmd_active or self.gait.active or self.gait.hold_stance:
            cpg_cmd = 0.0
            voluntary = voluntary.copy()
            for s, off in self.gait.arm_swing_targets().items():
                if not self.skills.arm_busy(s):
                    voluntary[self.motor.idx[f"sh_{s}_flex"]] += off
            gait_tau, gait_mask = self.gait.compute(
                d, dt, q_ref=fclip(voluntary, self.motor.q_lo, self.motor.q_hi))
            if not gait_mask.any():
                gait_tau = gait_mask = None
        mf = self.motor.compute(
            state, dt,
            voluntary=voluntary,
            arousal=float(self.affect_frame.arousal),
            fear=float(self.affect_frame.emotions[EMOTIONS.index("fear")]),
            pain=float(self.frame.pain_total),
            fatigue=float(self.interoception.s[IDX["peripheral_fatigue"]]),
            cpg_command=cpg_cmd,
            gait_tau=gait_tau, gait_mask=gait_mask,
            exempt=self.skills.lean_exempt(),
        )
        self.motor_frame = mf
        self.data.ctrl[:] = mf.torque
        self.skills.mute_jaw_actuator(self.data.ctrl)

        # ---- intrinsic reward (every step, cheap) --------------------
        self._reward_tick(mf)

        # ---- integrate -----------------------------------------------
        mujoco.mj_step(m, d)
        self.t += dt
        self.step_count += 1

    # ------------------------------------------------------------------
    def _ocular_inputs(self) -> OcularInputs:
        """What the rest of the person tells its eyes."""
        s = self.interoception.s
        a = self.affect_frame
        sk = self.skills
        g = IDX
        em = (lambda n: float(a.emotion(n))) if a is not None else (lambda n: 0.0)
        ea = self.meta.qpos_addr
        eye_p = float(self.data.qpos[ea["eye_l_pitch"]]) if "eye_l_pitch" in ea else 0.0
        neck_b = float(self.data.qpos[ea["neck_bend"]]) if "neck_bend" in ea else 0.0
        gaze_up = float(np.clip(-(eye_p + 0.5 * neck_b) / 0.6, -1.0, 1.0))
        # visual concentration: acting on an object or looking at something holds attention
        attn = 0.25 + 0.25 * em("curiosity")
        if sk.gaze is not None:
            attn = max(attn, 0.60)
        if sk.busy:
            attn = max(attn, 0.85)
        speaking = 1.0 if (sk.speech.speaking or self.behavior.voice > 0.15) else 0.0
        walk = float(self.gait.diag.speed) if self.gait.active else 0.0
        # hands near the eyes: a protective blink on approach, wiping when touching
        eyes_mid = self.data.xpos[self.meta.body_ids["head"]] + self.data.xmat[
            self.meta.body_ids["head"]].reshape(3, 3) @ np.array([0.0, -0.11, 0.078])
        dist = min(float(np.linalg.norm(sk.arm[sd].site_world(self.data)[0] - eyes_mid))
                   for sd in "lr")
        touching = sk.touching
        wipe = 1.0 if (touching and touching[0] == "eyes" and touching[3] == "act") else 0.0
        hand_near = 0.0 if wipe else float(np.clip((0.14 - dist) / 0.07, 0.0, 1.0))
        arousal = float(a.arousal) if a is not None else 0.25
        lum = float(np.clip(self.luminance, 0.05, 1.0)) if self.receptors.visual.enabled else 0.6
        return OcularInputs(
            humidity=self.ambient_humidity,
            airflow=self.ambient_airflow + 0.9 * walk,
            luminance=lum,
            gaze_up=gaze_up,
            attention=attn,
            arousal=arousal,
            sadness=em("sadness"),
            anxiety=em("anxiety"),
            fatigue=float(np.clip(s[g["central_fatigue"]], 0, 1)),
            sleepiness=float(np.clip(s[g["sleepiness"]], 0, 1)),
            hydration=float(s[g["hydration"]]),
            speaking=speaking,
            startle=1.0 if (self.state is not None and self.state.fallen) else 0.0,
            hand_near=hand_near,
            dopamine=float(a.neuromodulator("dopamine_tonic")) if a is not None else 0.35,
            sensitisation=float(np.clip(s[g["central_sensitisation"]], 0, 1)),
            parasympathetic=float(np.clip(0.85 - 0.6 * arousal, 0.1, 1.0)),
            temperature=float(s[g["skin_temp_mean"]]) + 1.0,
            wipe=wipe,
        )

    # ------------------------------------------------------------------
    def _contact_thermal(self, sign: int) -> float:
        """Mean temperature mismatch for warming (+) or cooling (-) contact.

        Only counts contacts between the human and something *else*; heat
        exchange with your own body is not a thermal stimulus.
        """
        if not self.state:
            return 0.0
        warm = cool = 0.0
        n = 0
        for c in self.state.contacts:
            h1 = bool(self.geom_is_human[c.geom1])
            h2 = bool(self.geom_is_human[c.geom2])
            if h1 == h2:                       # both human, or neither
                continue
            body = c.body1 if h1 else c.body2
            t_other = c.temp_for(body)
            warm += max(0.0, t_other - 33.0)
            cool += max(0.0, 33.0 - t_other)
            n += 1
        if n == 0:
            return 0.0
        return float((warm if sign > 0 else cool) / n) / 20.0

    # ------------------------------------------------------------------
    def _affect_inputs(self) -> AffectInputs:
        s = self.interoception.s
        g = IDX
        frame = self.frame
        st = self.state
        pred = self.pred_frame
        rw = self.reward_frame
        drv = self.drive_frame
        a = self.affect_frame
        return AffectInputs(
            rpe=rw.rpe if rw else 0.0,
            reward=max(0.0, rw.total) * 0.1 if rw else 0.0,
            punishment=max(0.0, -rw.total) * 0.1 if rw else 0.0,
            threat=float(fclip(
                self.motor.balance_error * 1.5
                + (1.0 if st and st.fallen else 0.0)
                + 0.4 * float(fclip(s[g["lactate"]] - 4, 0, 12)) / 12.0, 0, 1.5)),
            pain=float(frame.pain_total) if frame else 0.0,
            itch=float(frame.itch_total) if frame else 0.0,
            affective_touch=float(frame.affective_touch) if frame else 0.0,
            social_contact=(2.0 if st and st.self_touch else 0.0)
            + (float(frame.affective_touch) if frame else 0.0)
            + self.skills.social_pulse,
            novelty=float(fclip(pred.free_energy_norm * 0.8, 0, 1.5)) if pred else 0.0,
            ocular_discomfort=float(self.ocular.out.discomfort),
            inner_threat=float(self.inner.out.threat_tone) if self.inner is not None else 0.0,
            rumination=float(self.inner.out.rumination) if self.inner is not None else 0.0,
            intero_surprise=float(self.inner.out.interoceptive_surprise) if self.inner is not None else 0.0,
            memory_valence=float(self.inner.out.memory_valence) if self.inner is not None else 0.0,
            familiarity=float(self.inner.out.familiarity) if self.inner is not None else 0.0,
            control=float(fclip(0.9 - 0.6 * self.motor.balance_error
                                  - (0.4 if st and st.fallen else 0.0), 0, 1)),
            safety=float(fclip(1.0 - self.motor.balance_error * 1.2, 0, 1)),
            effort=float(st.torque_effort) if st else 0.0,
            balance_error=float(fclip(self.motor.balance_error, 0, 1.5)),
            fell=1.0 if (st and st.fallen) else 0.0,
            thermal_discomfort=float(s[g["thermal_discomfort"]]),
            hunger=float(drv.level[0]) if drv else 0.0,
            thirst=float(drv.level[1]) if drv else 0.0,
            air_hunger=float(s[g["air_hunger"]]),
            sleepiness=float(s[g["sleepiness"]]),
            nausea=float(s[g["nausea"]]),
            bladder=float(drv.level[8]) if drv else 0.0,
            sleep_pressure=float(s[g["adenosine"]]),
            inflammation=float(s[g["inflammation"]]),
            goal_progress=float(rw.total * 0.2) if rw else 0.0,
            predictability=float(fclip(1.0 - (pred.free_energy_norm * 0.2 if pred else 0),
                                         0, 1)),
            self_evaluation=float(a.valence) if a else 0.0,
            motor_error=float(pred.inverse_error) if pred else 0.0,
        )

    # ------------------------------------------------------------------
    def _cognitive_tick(self) -> None:
        """10 Hz: learn the body model, then choose what to do."""
        inner_vec = None
        if self.inner is not None:
            inner_vec = self.inner.finalize(self).summary
        latent, preferred, precision = self.latent_spec.build(
            self.frame, self.aff, self.interoception, self.affect_frame,
            self.drive_frame, self.state, self.pred_frame,
            ocular=self.ocular.summary_features(), inner=inner_vec)
        self.latent = latent
        self.preferred = preferred
        self.precision = precision

        self.pred_frame = self.predict.observe(latent, self.action)

        if self._behavior_on and self.behavior.enabled:
            # The open-ended behaviour layer owns the body (see behavior_exec.py);
            # it decides at ~2 Hz on its own clock.  The learned forward model is
            # still fed what was actually commanded.
            self.action = self.behavior.cmd.copy()
            self.last_inference_info = self.behavior.summary()
            return

        temperature = float(fclip(
            0.6 + 1.2 * self.affect_frame.arousal
            - 0.4 * self.affect_frame.dominance, 0.15, 3.0))
        policy, G, info = self.inference.evaluate(
            latent, preferred, self.predict, precision,
            q_slice=slice(self.latent_spec.offset["q"],
                          self.latent_spec.offset["q"] + self.latent_spec.n_joints),
            policy_temperature=temperature,
            valence=float(self.affect_frame.valence),
            arousal=float(self.affect_frame.arousal),
            fear=float(self.affect_frame.emotions[EMOTIONS.index("fear")]),
            pain=float(self.frame.pain_total),
            curiosity=float(self.drive_frame.level[13]),
            social_need=float(self.drive_frame.level[11]),
            balance_error=float(self.motor.balance_error),
            drives_level=self.drive_frame.level,
            action_tendency=self.affect_frame.action_tendency,
            current_action=self.voluntary_target,
        )
        self.last_inference_info = info
        if not self.autonomous:
            return
        target = self.inference.commit(policy, 1.0 / self.cfg.rates.cognition)
        # smooth towards the chosen equilibrium point
        alpha = 0.35
        self.voluntary_target = (1 - alpha) * self.voluntary_target + alpha * target
        self.voluntary_target = fclip(self.voluntary_target,
                                        self.motor.q_lo, self.motor.q_hi)
        self.action = self.voluntary_target.copy()

        gz = self.inference.gaze_command()
        self.gaze_target = gz
        for s in ("l", "r"):
            for jn, val in ((f"eye_{s}_yaw", gz[0]), (f"eye_{s}_pitch", gz[1])):
                a = self.meta.qpos_addr.get(jn)
                if a is not None:
                    q = self.data.qpos[a]
                    self.data.qpos[a] = q + 0.35 * (val - q)
        self.jaw_target = self.inference.jaw_command()
        ja = self.meta.qpos_addr.get("jaw_open")
        if ja is not None:
            self.data.qpos[ja] += 0.3 * (self.jaw_target - self.data.qpos[ja])

    # ------------------------------------------------------------------
    def _reward_tick(self, mf: MotorFrame) -> None:
        pred = self.pred_frame
        frame = self.frame
        contact_rate = 0.0
        if frame is not None and self.aff is not None:
            contact_rate = float(fclip(
                frame.touch_intensity / 20.0, 0, 1))
        self.reward_frame = self.motivation.compute(
            jerk=mf.jerk,
            effort=mf.effort,
            power=mf.mechanical_power,
            balance_error=self.motor.balance_error,
            free_energy=(pred.free_energy_norm if pred else 0.0),
            surprise=pred.surprise if pred else 0.0,
            body_ownership=pred.body_ownership if pred else None,
            touch_intensity=frame.touch_intensity if frame else 0.0,
            contact_rate=contact_rate,
            pain=frame.pain_total if frame else 0.0,
            drive_pressure=self.drive_frame.total_pressure if self.drive_frame else 0.0,
            valence=self.affect_frame.valence if self.affect_frame else 0.0,
            tension=self.affect_frame.tension if self.affect_frame else 0.0,
            empowerment=pred.empowerment if pred else 0.0,
        )

    # ------------------------------------------------------------------
    def snapshot(self) -> AgentStep:
        a = self.affect_frame
        f = self.frame
        st = self.state
        return AgentStep(
            t=self.t,
            com=st.com.copy() if st else np.zeros(3),
            root_z=float(st.root_pos[2]) if st else 0.0,
            fallen=bool(st.fallen) if st else False,
            n_contact=st.n_contact if st else 0,
            self_touch=bool(st.self_touch) if st else False,
            touch_intensity=float(f.touch_intensity) if f else 0.0,
            pain=float(f.pain_total) if f else 0.0,
            affective_touch=float(f.affective_touch) if f else 0.0,
            heart_rate=float(self.interoception.s[IDX["heart_rate"]]),
            core_temp=float(self.interoception.s[IDX["core_temp"]]),
            valence=float(a.valence) if a else 0.0,
            arousal=float(a.arousal) if a else 0.0,
            dominance=float(a.dominance) if a else 0.0,
            stress=float(a.stress) if a else 0.0,
            dominant_emotion=a.dominant_emotion if a else "",
            emotions=a.emotions.copy() if a else np.zeros(N_EMOTIONS),
            neuromodulators=a.neuromodulators.copy() if a else np.zeros(0),
            policy=(self.behavior.current_name() if self._behavior_on
                    else self.inference.current.name),
            free_energy=float(self.pred_frame.free_energy) if self.pred_frame else 0.0,
            reward=float(self.reward_frame.total) if self.reward_frame else 0.0,
            drive_pressure=float(self.drive_frame.total_pressure) if self.drive_frame else 0.0,
            most_urgent_drive=self.drive_frame.most_urgent if self.drive_frame else "",
            stepping=self.motor.step_state,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        from .build_model import set_standing_state
        set_standing_state(self.model, self.data, self.meta)
        self._init_gaze()
        self.receptors.reset()
        self.afferents.reset()
        self.interoception = InteroceptiveSystem(self.cfg, self.meta)
        self.ocular = OcularSurface(self.cfg.seed)
        self.behavior.stats = type(self.behavior.stats)(self.behavior.space)
        self.affect.reset()
        self.drives = DriveSystem(self.cfg)
        self.predict.reset()
        self.inference.reset()
        self.motor.reset()
        self.motivation.reset()
        self.t = 0.0
        self.step_count = 0
        self.acc = {k: 0.0 for k in self.acc}
        self.frame = None
        self.aff = None
        self.affect_frame = None
        self.drive_frame = None
        self.pred_frame = None
        self.motor_frame = None
        self.reward_frame = None
        self.voluntary_target = self.motor.q_nom.copy()

    # ------------------------------------------------------------------
    def run(self, duration: float | None = None, *, log_every: int | None = None,
            progress: bool = False) -> list[AgentStep]:
        cfg = self.cfg
        duration = cfg.duration if duration is None else duration
        log_every = cfg.log_every if log_every is None else log_every
        n_steps = int(round(duration / self.dt))
        steps: list[AgentStep] = []
        t0 = time.perf_counter()
        for i in range(n_steps):
            s0 = time.perf_counter()
            self.step()
            self.step_durations.append(time.perf_counter() - s0)
            if i % log_every == 0:
                steps.append(self.snapshot())
            if progress and i % max(n_steps // 20, 1) == 0:
                print(f"  t={self.t:5.2f}s  policies={self.inference.current.name:16s} "
                      f"val={self.affect_frame.valence:+.2f} "
                      f"aro={self.affect_frame.arousal:+.2f} "
                      f"FE={self.pred_frame.free_energy if self.pred_frame else 0:7.2f} "
                      f"z={self.state.root_pos[2]:.3f}")
        self.last_step_time = time.perf_counter() - t0
        return steps

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "model": {
                "nq": self.model.nq, "nv": self.model.nv, "nu": self.model.nu,
                "nbody": self.model.nbody, "ngeom": self.model.ngeom,
                "nsite": self.model.nsite, "nsensor": self.model.nsensor,
                "nsensordata": self.model.nsensordata,
                "human_mass_kg": float(sum(self.model.body_mass)
                                       - sum(o.mass for o in self.meta.objects)),
                "timestep": self.dt,
            },
            "rates_hz": self.cfg.all_rates(),
            "senses": self.receptors.describe(),
            "afferents": self.afferents.describe(),
            "interoception": self.interoception.describe(),
            "affect": self.affect.describe(),
            "drives": self.drives.describe(),
            "prediction": self.predict.describe(),
            "inference": self.inference.describe(),
            "motor": self.motor.describe(),
            "motivation": self.motivation.describe(),
            "latent": {
                "dim": self.latent_spec.dim,
                "blocks": self.latent_spec.blocks,
            },
            "total_scalar_state": self._count_state(),
        }

    def _count_state(self) -> dict:
        tax = self.receptors.tactile.n_taxels
        nj = self.meta.n_actuators
        rc = self.receptors
        cells = (rc.spindles.size + rc.vest_cells.size + rc.cochlea.size + rc.olfactory.size
                 + rc.gustatory.size + (rc.visual.retina.size if rc.visual.enabled else 0))
        inner = self.inner.n_state if self.inner is not None else 0
        base = self._count_state_base(tax, nj)
        base["cell_populations"] = int(cells)
        base["inner_world_dynamic_state"] = int(inner)
        base["total"] += int(cells) + int(inner)
        return base

    def _count_state_base(self, tax: int, nj: int) -> dict:
        return {
            "tactile_channels": tax * R.N_TACTILE_CH,
            "proprioceptive_channels": nj * R.N_PROPRIO_CH,
            "vestibular": len(R.VESTIBULAR_CHANNELS),
            "visual": len(R.VISUAL_CHANNELS),
            "auditory": self.cfg.audio.n_bands + len(R.AUDITORY_CHANNELS),
            "olfactory": self.cfg.chemo.n_olfactory_channels,
            "gustatory": len(R.GUSTATORY_CHANNELS),
            "interoceptive": N_INTEROCEPTION,
            "emotions": N_EMOTIONS,
            "neuromodulators": len(self.affect.nm),
            "appraisal": len(self.affect.appraisal),
            "drives": len(DRIVES),
            "total": (tax * R.N_TACTILE_CH + nj * R.N_PROPRIO_CH
                      + len(R.VESTIBULAR_CHANNELS) + len(R.VISUAL_CHANNELS)
                      + self.cfg.audio.n_bands + len(R.AUDITORY_CHANNELS)
                      + self.cfg.chemo.n_olfactory_channels
                      + len(R.GUSTATORY_CHANNELS) + N_INTEROCEPTION
                      + N_EMOTIONS + len(self.affect.nm)
                      + len(self.affect.appraisal) + len(DRIVES)),
        }
