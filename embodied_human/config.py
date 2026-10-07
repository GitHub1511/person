"""
Global configuration for the Embodied Human simulation.

Everything that controls *rates*, *dims* and *numerical scales* lives here so
that the rest of the package can be read as physiology rather than plumbing.

Design principle
----------------
A real nervous system runs many loops at many rates.  We reproduce that:

    physics     1000 Hz   (MuJoCo integration)
    receptors    500 Hz   (transduction)
    afferents    200 Hz   (delays, spikes, adaptation)
    intero       100 Hz   (slow organ systems, autonomic)
    affect        50 Hz   (neuromodulators, appraisal, emotion)
    cognition     10 Hz   (active inference / policy selection)
    mood           1 Hz   (allostatic slow state)

All sub-systems are integrated with their own dt and their own noise, and the
whole stack is embedded in the physics loop via an accumulator scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .complexity import C as COMPLEXITY, load as load_complexity

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
OUT_DIR = PROJECT_DIR / "out"
MODEL_DIR = PROJECT_DIR / "models"

# --------------------------------------------------------------------------
# Global physical constants (SI)
# --------------------------------------------------------------------------
GRAVITY = 9.81
BODY_MASS = 75.0          # kg, total
BODY_HEIGHT = 1.75        # m
SKIN_TEMPERATURE = 33.0   # degC, comfortable glabrous skin
CORE_TEMPERATURE = 36.8   # degC
AMBIENT_TEMPERATURE = 22.0

# --------------------------------------------------------------------------
# Sensor / rate configuration
# --------------------------------------------------------------------------


@dataclass
class RateConfig:
    """Update rates of the nested control loops (Hz)."""

    physics: float = 1000.0
    receptor: float = COMPLEXITY.tactile_hz   # dense skin is transduced more slowly
    afferent: float = min(200.0, 2.0 * COMPLEXITY.tactile_hz)
    interoception: float = 100.0
    affect: float = 50.0
    cognition: float = 10.0
    mood: float = 1.0

    def dt(self, name: str) -> float:
        return 1.0 / getattr(self, name)


@dataclass
class AfferentConfig:
    """Properties of the peripheral nerve layer."""

    # Conduction delays (seconds) for different fibre classes.
    delay_abeta: float = 0.018     # fast, myelinated touch / proprioception
    delay_adelta: float = 0.060    # sharp pain, cold
    delay_c: float = 0.250         # slow pain, warmth, affective touch
    delay_efferent: float = 0.030  # motor command down to muscle

    # Firing-rate model
    rate_min: float = 0.0          # Hz
    rate_max: float = 300.0        # Hz
    rate_tau: float = 0.012        # Hz, first-order lag of rate coding
    refractory: float = 0.0015     # s

    # Receptor adaptation time constants (seconds)
    tau_fa1: float = 0.008         # Meissner, rapidly adapting
    tau_fa2: float = 0.004         # Pacinian, very rapidly adapting
    tau_sa1: float = 1.200         # Merkel, slowly adapting
    tau_sa2: float = 3.000         # Ruffini, very slowly adapting
    tau_ct: float = 2.500          # C-tactile, affective
    tau_noci: float = 0.500

    # Noise
    noise_sd_frac: float = 0.02    # multiplicative noise on receptor output
    spontaneous_hz: float = 1.5    # baseline firing

    # Efference copy / corollary discharge gain
    reafference_gain: float = 0.85


@dataclass
class TactileConfig:
    """Tactile transduction parameters."""

    contact_radius: float = 0.028      # m, receptive-field radius of a taxel
    taxel_radius: float = 0.010        # m, MuJoCo site radius
    fa1_gain: float = 1.0
    fa2_gain: float = 0.7
    sa1_gain: float = 1.1
    sa2_gain: float = 0.8
    ct_gain: float = 0.5
    # Nociceptor thresholds (N of normal force or degC deviation)
    noci_mech_threshold: float = 12.0
    noci_heat_threshold: float = 43.0
    noci_cold_threshold: float = 12.0
    # Thermal
    tau_warm: float = 3.5
    tau_cold: float = 2.0
    # Slip detection
    slip_friction_ratio: float = 0.85
    # Affective touch: optimal stroking velocity (C-tactile)
    ct_optimal_velocity: float = 0.03  # m/s
    ct_velocity_width: float = 0.05


@dataclass
class VestibularConfig:
    canal_tau: float = 5.0        # s, cupula adaptation
    otolith_tau: float = 0.6
    canal_noise: float = 0.004    # rad/s
    otolith_noise: float = 0.02   # m/s^2
    canal_gain: float = 1.0
    otolith_gain: float = 1.0


@dataclass
class VisionConfig:
    enabled: bool = True
    retina_w: int = COMPLEXITY.retina_w
    retina_h: int = COMPLEXITY.retina_h
    fovea_frac: float = 0.30      # central fraction rendered at full detail
    update_hz: float = 60.0
    saccade_interval: tuple[float, float] = (0.15, 0.55)
    blink_interval: tuple[float, float] = (2.0, 6.0)
    blink_duration: float = 0.12
    pupil_min: float = 2.0        # mm
    pupil_max: float = 8.0
    n_luminance_channels: int = 4
    n_edge_channels: int = 4


@dataclass
class AudioConfig:
    n_bands: int = 24                 # the base bank; the cochlea (senses_ext) is denser
    f_min: float = 20.0
    f_max: float = 16000.0
    tau_onset: float = 0.010
    bone_conduction_gain: float = 0.35
    motor_noise_gain: float = 0.08


@dataclass
class ChemoConfig:
    n_olfactory_channels: int = 24
    n_gustatory_channels: int = 5     # sweet salty sour bitter umami
    olfactory_range: float = 0.35     # m
    tau_chemo: float = 0.8


@dataclass
class InteroceptionConfig:
    # Cardiovascular
    hr_rest: float = 62.0          # bpm
    hr_max: float = 190.0
    hrv_tau: float = 8.0
    # Respiratory
    rr_rest: float = 13.0          # breaths/min
    tidal_volume: float = 0.5      # L
    # Metabolic
    glucose_rest: float = 5.2      # mmol/L
    lactate_rest: float = 1.0      # mmol/L
    atp_rest: float = 1.0          # normalised
    glycogen_rest: float = 1.0
    # Hydration / osmolality
    osmolality_rest: float = 290.0  # mOsm/kg
    # Thermoregulation
    core_temp_rest: float = 36.8
    shivering_threshold: float = 35.9
    sweating_threshold: float = 37.2
    # GI
    gastric_emptying_tau: float = 3600.0
    # Fatigue
    tau_peripheral_fatigue: float = 240.0
    tau_central_fatigue: float = 1800.0
    tau_adenosine: float = 5400.0     # ~90 min half-ish sleep pressure
    # Immune
    tau_cytokine: float = 900.0
    # Bladder
    bladder_fill_rate: float = 1.0 / 7200.0   # full in 2 h


@dataclass
class AffectConfig:
    """Emotional system configuration."""

    # Neuromodulator dynamics
    nm_tau: dict = field(default_factory=lambda: {
        "dopamine": 0.35, "serotonin": 900.0, "norepinephrine": 1.2,
        "acetylcholine": 4.0, "oxytocin": 300.0, "vasopressin": 300.0,
        "endorphin": 120.0, "enkephalin": 120.0, "endocannabinoid": 200.0,
        "cortisol": 1800.0, "adrenaline": 4.0, "noradrenaline_h": 12.0,
        "testosterone": 3600.0, "estrogen": 3600.0, "progesterone": 3600.0,
        "prolactin": 900.0, "melatonin": 3600.0, "adenosine": 5400.0,
        "histamine": 60.0, "gaba": 2.0, "glutamate": 0.5, "substance_p": 30.0,
        "bdnf": 7200.0, "orexin": 600.0,
    })
    # Core affect (Russell / Mehrabian PAD)
    tau_valence: float = 12.0
    tau_arousal: float = 6.0
    tau_dominance: float = 25.0
    # Emotion decay
    tau_emotion_fast: float = 3.0
    tau_emotion_slow: float = 40.0
    # Mood (slow allostatic)
    tau_mood: float = 900.0
    # Stress
    tau_allostatic_load: float = 3600.0
    stress_gain: float = 1.0
    # Appraisal sensitivity
    appraisal_gain: float = 1.0
    reappraisal_rate: float = 0.35
    # Arousal modulates policy temperature and motor noise
    arousal_temp_gain: float = 1.6
    arousal_noise_gain: float = 0.9
    fear_freeze_threshold: float = 0.55
    anger_aggression_threshold: float = 0.45
    sadness_withdrawal_threshold: float = 0.40


@dataclass
class PredictiveConfig:
    """Forward / inverse model and body-schema configuration."""
    latent_dim: int = 192            # compact sensory latent for learning
    forward_ridge: float = 30.0
    inverse_ridge: float = 40.0
    forward_forget: float = 0.9995   # RLS forgetting factor
    inverse_forget: float = 0.9990
    body_schema_ridge: float = 50.0
    precision_tau: float = 1.5
    kalman_process_noise: float = 0.02
    kalman_obs_noise: float = 0.08
    update_every: int = 4            # learn every N cognitive ticks
    jacobian_eps: float = 1e-3


@dataclass
class ActiveInferenceConfig:
    n_policies: int = 12
    horizon: int = 3
    temperature: float = 1.0
    prior_weight: float = 1.0
    ambiguity_weight: float = 0.7
    risk_weight: float = 1.0
    exploration_weight: float = 0.35
    kl_clip: float = 25.0


@dataclass
class MotorConfig:
    # Reflexes
    stretch_reflex_gain: float = 0.12
    stretch_reflex_delay: float = 0.030
    golgi_gain: float = 0.08
    withdrawal_gain: float = 0.55
    withdrawal_threshold: float = 0.35
    righting_gain: float = 0.35
    vestibulo_collic_gain: float = 0.25
    # Posture / balance strategies
    balance_kp: float = 70.0          # ankle strategy, Nm per metre of COM excursion
    balance_kd: float = 6.0           # ankle strategy, Nm per m/s
    ankle_strategy_gain: float = 1.0
    hip_strategy_gain: float = 1.0
    hip_kp_gain: float = 150.0        # hip strategy, Nm per metre
    hip_kd_gain: float = 20.0         # hip strategy, Nm per m/s
    toe_strategy_gain: float = 0.35
    knee_strategy_gain: float = 140.0
    # CPG
    cpg_freq_rest: float = 0.0
    cpg_freq_walk: float = 1.9
    cpg_coupling: float = 0.9
    cpg_amplitude: float = 0.35
    # General
    motor_noise_sd: float = 0.012
    torque_rise_time: float = 0.05    # s to develop full torque (activation)
    command_tau: float = 0.045        # first-order muscle activation lag
    effort_cost_gain: float = 1.0
    # Voluntary posture targets (reference configuration, radians)
    posture_gain: float = 0.55


@dataclass
class IntrinsicConfig:
    """Weights of the intrinsic-motivation reward decomposition."""
    w_jerk: float = 0.030
    w_energy: float = 0.012
    w_stability: float = 0.045
    w_novelty: float = 0.060
    w_tactile: float = 0.025
    w_comfort: float = 0.040
    w_affective_balance: float = 0.080
    w_empowerment: float = 0.020
    w_pain: float = 0.250
    w_smoothness_contact: float = 0.020


@dataclass
class SimConfig:
    """Top-level configuration object."""

    rates: RateConfig = field(default_factory=RateConfig)
    afferent: AfferentConfig = field(default_factory=AfferentConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    vestibular: VestibularConfig = field(default_factory=VestibularConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    chemo: ChemoConfig = field(default_factory=ChemoConfig)
    intero: InteroceptionConfig = field(default_factory=InteroceptionConfig)
    affect: AffectConfig = field(default_factory=AffectConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)
    inference: ActiveInferenceConfig = field(default_factory=ActiveInferenceConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)

    # Episode
    duration: float = 12.0
    seed: int = 7
    render: bool = False
    viewer: bool = False
    log_every: int = 5
    out_dir: Path = OUT_DIR

    def all_rates(self) -> dict[str, float]:
        return {
            k: getattr(self.rates, k)
            for k in ("receptor", "afferent", "interoception", "affect",
                      "cognition", "mood")
        }
