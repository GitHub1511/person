"""
Affect: neuromodulators, appraisal, core affect, emotions, mood, temperament.

The emotional state is not a decoration bolted onto a controller -- it is the
**gain-and-setpoint layer** that the rest of the agent runs through:

* appraisal converts events into a small number of evaluative dimensions
  (Scherer's Component Process Model);
* emotions are dynamical states driven by appraisal, with their own decay
  times, so fear persists after the threat has gone;
* neuromodulators set the *gains*: dopamine scales learning rate, noradrenaline
  scales arousal and policy temperature, oxytocin scales the value of social
  touch, endorphins close the pain gate, cortisol shifts the body towards
  catabolism and vigilance;
* core affect (valence / arousal / dominance) is the low-dimensional summary
  that colours everything;
* mood and temperament are the slow states that make the agent *disposed* to
  react one way rather than another.

Without this layer an agent optimises.  With it, it *prefers*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._fast import fclip

from .config import SimConfig

# ==========================================================================
# 1. Appraisal dimensions (Scherer's CPM, condensed to ten)
# ==========================================================================
APPRAISAL_DIMS = (
    "novelty",              # sudden, unexpected, unfamiliar
    "intrinsic_pleasantness",   # is the event itself pleasant?
    "goal_relevance",       # does it matter for what I want?
    "goal_congruence",      # does it help (+) or hinder (-)?
    "agency_self",          # caused by me (+) or by someone/something else (-)
    "coping_potential",     # can I deal with it?
    "certainty",            # do I understand what is happening?
    "norm_compatibility",   # does it fit my standards?
    "urgency",              # does it demand immediate action?
    "social_evaluation",    # am I admired (+) or judged (-)?
)
N_APPRAISAL = len(APPRAISAL_DIMS)
A = {n: i for i, n in enumerate(APPRAISAL_DIMS)}


# ==========================================================================
# 2. Emotions: weighted appraisals + a bias term
# ==========================================================================
# Each entry maps appraisal dimension -> weight.  The weights are the
# theoretical content: fear is high urgency, low coping, negative congruence;
# guilt is high self-agency, negative norm compatibility; etc.
EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "fear": dict(novelty=0.5, intrinsic_pleasantness=-0.5, goal_relevance=0.8,
                 goal_congruence=-0.8, agency_self=-0.4, coping_potential=-1.0,
                 certainty=-0.5, urgency=1.1),
    "anxiety": dict(novelty=0.3, intrinsic_pleasantness=-0.4, goal_relevance=0.6,
                    goal_congruence=-0.4, coping_potential=-0.7, certainty=-0.8,
                    urgency=0.6),
    "anger": dict(intrinsic_pleasantness=-0.5, goal_relevance=0.7,
                  goal_congruence=-0.7, agency_self=-0.7, coping_potential=0.7,
                  certainty=0.4, norm_compatibility=-0.2, urgency=0.9),
    "sadness": dict(intrinsic_pleasantness=-0.7, goal_relevance=0.6,
                    goal_congruence=-0.9, coping_potential=-0.8, certainty=0.4,
                    urgency=-0.5),
    "joy": dict(intrinsic_pleasantness=0.9, goal_relevance=0.5,
                goal_congruence=0.9, coping_potential=0.5, certainty=0.3),
    "surprise": dict(novelty=1.1, certainty=-1.0, urgency=0.3),
    "disgust": dict(intrinsic_pleasantness=-1.0, norm_compatibility=-0.7,
                    urgency=0.4),
    "contempt": dict(intrinsic_pleasantness=-0.4, norm_compatibility=-0.8,
                     social_evaluation=-0.6, coping_potential=0.6),
    "trust": dict(intrinsic_pleasantness=0.5, social_evaluation=0.7,
                  certainty=0.4, coping_potential=0.3),
    "anticipation": dict(novelty=0.4, goal_relevance=0.7, certainty=-0.3,
                         urgency=0.5),
    "love": dict(intrinsic_pleasantness=0.8, social_evaluation=0.8,
                 goal_relevance=0.5, goal_congruence=0.5),
    "guilt": dict(agency_self=0.9, norm_compatibility=-0.9,
                  intrinsic_pleasantness=-0.6, coping_potential=0.4),
    "shame": dict(agency_self=0.8, norm_compatibility=-1.0,
                  social_evaluation=-0.8, intrinsic_pleasantness=-0.6,
                  coping_potential=-0.4),
    "pride": dict(agency_self=0.9, norm_compatibility=0.6,
                  social_evaluation=0.8, goal_congruence=0.7,
                  intrinsic_pleasantness=0.6),
    "envy": dict(social_evaluation=-0.5, goal_congruence=-0.6,
                 intrinsic_pleasantness=-0.4, agency_self=-0.4),
    "jealousy": dict(social_evaluation=-0.4, goal_relevance=0.8,
                     goal_congruence=-0.6, intrinsic_pleasantness=-0.5,
                     urgency=0.5),
    "hope": dict(goal_relevance=0.7, goal_congruence=0.3, certainty=0.3,
                 coping_potential=0.6, intrinsic_pleasantness=0.4),
    "awe": dict(novelty=0.9, intrinsic_pleasantness=0.4,
                coping_potential=-0.6, certainty=-0.5),
    "gratitude": dict(intrinsic_pleasantness=0.8, social_evaluation=0.6,
                      goal_congruence=0.7, agency_self=-0.3),
    "embarrassment": dict(social_evaluation=-0.7, norm_compatibility=-0.6,
                          agency_self=0.7, intrinsic_pleasantness=-0.4),
    "curiosity": dict(novelty=0.9, goal_relevance=0.4, coping_potential=0.4,
                      certainty=-0.4),
    "boredom": dict(novelty=-1.0, goal_relevance=-0.3, urgency=-0.6),
    "calm": dict(intrinsic_pleasantness=0.5, certainty=0.5, coping_potential=0.5,
                 urgency=-0.8, goal_relevance=-0.4),
    "relief": dict(intrinsic_pleasantness=0.7, goal_congruence=0.6,
                   certainty=0.4, urgency=-0.6),
    "loneliness": dict(social_evaluation=-0.4, intrinsic_pleasantness=-0.6,
                       goal_relevance=0.5, goal_congruence=-0.6),
    "empathy": dict(intrinsic_pleasantness=-0.2, social_evaluation=0.4,
                    goal_relevance=0.5, agency_self=-0.5),
    "amusement": dict(novelty=0.7, intrinsic_pleasantness=0.8, urgency=-0.3),
    "satisfaction": dict(intrinsic_pleasantness=0.7, goal_congruence=0.8,
                         coping_potential=0.5, certainty=0.4),
}
EMOTIONS = tuple(EMOTION_PROFILES.keys())
N_EMOTIONS = len(EMOTIONS)
E = {n: i for i, n in enumerate(EMOTIONS)}

# Which emotions are positive / negative (for valence and mood)
EMOTION_VALENCE = np.array([
    -1.0, -0.7, -0.8, -0.9, 1.0, 0.1, -0.8, -0.5, 0.6, 0.3,
    0.9, -0.7, -0.9, 0.8, -0.5, -0.6, 0.6, 0.4, 0.8, -0.6,
    0.4, -0.4, 0.7, 0.8, -0.7, 0.0, 0.5, 0.8,
], float)
# Arousal contribution of each emotion
EMOTION_AROUSAL = np.array([
    0.9, 0.7, 0.9, 0.1, 0.7, 0.8, 0.4, 0.4, 0.2, 0.5,
    0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.3, 0.6, 0.4, 0.6,
    0.5, -0.3, -0.5, 0.2, 0.1, 0.4, 0.6, 0.2,
], float)
# How long each emotion lingers (seconds); fast emotions are reactive,
# slow ones are moods-like.
EMOTION_TAU = np.array([
    6.0, 60.0, 25.0, 120.0, 12.0, 3.0, 15.0, 40.0, 90.0, 20.0,
    300.0, 90.0, 180.0, 120.0, 60.0, 90.0, 45.0, 20.0, 60.0, 30.0,
    25.0, 45.0, 40.0, 10.0, 200.0, 30.0, 8.0, 35.0,
], float)

# appraisal weights matrix (N_EMOTIONS x N_APPRAISAL)
APPRAISAL_W = np.zeros((N_EMOTIONS, N_APPRAISAL))
for _i, _name in enumerate(EMOTIONS):
    for _dim, _w in EMOTION_PROFILES[_name].items():
        APPRAISAL_W[_i, A[_dim]] = _w
DELIBERATE_THRESHOLD = 0.35     # appraisal sum below this evokes nothing


# ==========================================================================
# 3. Neuromodulators
# ==========================================================================
NEUROMODULATORS = (
    "dopamine_phasic", "dopamine_tonic", "serotonin", "norepinephrine",
    "acetylcholine", "oxytocin", "vasopressin", "endorphin", "enkephalin",
    "endocannabinoid", "cortisol", "adrenaline", "noradrenaline_h",
    "testosterone", "estrogen", "progesterone", "prolactin", "melatonin",
    "adenosine", "histamine", "gaba", "glutamate", "substance_p", "bdnf",
    "orexin",
)
N_NEUROMODULATORS = len(NEUROMODULATORS)
NM = {n: i for i, n in enumerate(NEUROMODULATORS)}

NM_BASELINE = {
    "dopamine_phasic": 0.05, "dopamine_tonic": 0.35, "serotonin": 0.55,
    "norepinephrine": 0.30, "acetylcholine": 0.40, "oxytocin": 0.30,
    "vasopressin": 0.30, "endorphin": 0.20, "enkephalin": 0.25,
    "endocannabinoid": 0.25, "cortisol": 0.28, "adrenaline": 0.12,
    "noradrenaline_h": 0.30, "testosterone": 0.50, "estrogen": 0.45,
    "progesterone": 0.30, "prolactin": 0.25, "melatonin": 0.15,
    "adenosine": 0.25, "histamine": 0.20, "gaba": 0.45, "glutamate": 0.40,
    "substance_p": 0.15, "bdnf": 0.50, "orexin": 0.55,
}
NM_TAU = {
    "dopamine_phasic": 0.40, "dopamine_tonic": 45.0, "serotonin": 900.0,
    "norepinephrine": 1.2, "acetylcholine": 4.0, "oxytocin": 300.0,
    "vasopressin": 300.0, "endorphin": 45.0, "enkephalin": 90.0,
    "endocannabinoid": 200.0, "cortisol": 900.0, "adrenaline": 4.0,
    "noradrenaline_h": 12.0, "testosterone": 3600.0, "estrogen": 3600.0,
    "progesterone": 3600.0, "prolactin": 900.0, "melatonin": 3600.0,
    "adenosine": 5400.0, "histamine": 60.0, "gaba": 2.0, "glutamate": 0.5,
    "substance_p": 30.0, "bdnf": 7200.0, "orexin": 600.0,
}


@dataclass
class AffectInputs:
    """Everything the rest of the agent tells the affective system."""
    rpe: float = 0.0                  # reward prediction error
    reward: float = 0.0               # intrinsic reward this tick
    punishment: float = 0.0           # intrinsic cost this tick
    threat: float = 0.0               # proximity/severity of danger
    pain: float = 0.0
    itch: float = 0.0
    affective_touch: float = 0.0      # slow stroking -> oxytocin
    social_contact: float = 0.0
    novelty: float = 0.0              # prediction error magnitude
    control: float = 0.5              # sense of agency / coping
    safety: float = 0.7
    effort: float = 0.0
    balance_error: float = 0.0        # postural instability
    fell: float = 0.0
    thermal_discomfort: float = 0.0
    hunger: float = 0.0
    thirst: float = 0.0
    air_hunger: float = 0.0
    sleepiness: float = 0.0
    nausea: float = 0.0
    bladder: float = 0.0
    sleep_pressure: float = 0.0       # adenosine
    inflammation: float = 0.0
    goal_progress: float = 0.0        # progress towards preferred states
    predictability: float = 0.5       # 1 - normalised prediction error
    self_evaluation: float = 0.0
    motor_error: float = 0.0
    ocular_discomfort: float = 0.0    # dry / burning / gritty eyes (see ocular.py)
    # from the internal world (inner_world.py)
    inner_threat: float = 0.0         # neural-mass "amygdala" tone above baseline
    rumination: float = 0.0           # mind-wandering that has a threatened colour
    intero_surprise: float = 0.0      # the body is not doing what it predicted
    memory_valence: float = 0.0       # the mood the remembered past carries
    familiarity: float = 0.0          # how much like something remembered this is


@dataclass
class AffectFrame:
    t: float = 0.0
    appraisal: np.ndarray = field(default_factory=lambda: np.zeros(N_APPRAISAL))
    appraisal_raw: np.ndarray = field(default_factory=lambda: np.zeros(N_APPRAISAL))
    emotions: np.ndarray = field(default_factory=lambda: np.zeros(N_EMOTIONS))
    neuromodulators: np.ndarray = field(
        default_factory=lambda: np.zeros(N_NEUROMODULATORS))
    # core affect
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    tension: float = 0.0
    pleasantness: float = 0.0
    # slow states
    mood_valence: float = 0.0
    mood_arousal: float = 0.0
    mood_energy: float = 0.0
    temperament: np.ndarray = field(default_factory=lambda: np.zeros(5))
    stress: float = 0.0
    allostatic_load: float = 0.0
    # derived drives / action tendencies
    action_tendency: np.ndarray = field(default_factory=lambda: np.zeros(6))
    feeling_intensity: float = 0.0
    dominant_emotion: str = "neutral"
    dominant_intensity: float = 0.0

    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.appraisal, self.emotions, self.neuromodulators,
            [self.valence, self.arousal, self.dominance, self.tension,
             self.pleasantness, self.mood_valence, self.mood_arousal,
             self.mood_energy, self.stress, self.allostatic_load,
             self.feeling_intensity],
            self.temperament, self.action_tendency,
        ])

    def top_emotions(self, k: int = 5) -> list[tuple[str, float]]:
        order = np.argsort(-self.emotions)[:k]
        return [(EMOTIONS[i], float(self.emotions[i])) for i in order]

    def emotion(self, name: str) -> float:
        return float(self.emotions[E[name]])

    def neuromodulator(self, name: str) -> float:
        return float(self.neuromodulators[NM[name]])


TEMPERAMENT_TRAITS = ("neuroticism", "extraversion", "openness",
                      "agreeableness", "conscientiousness")

# Action tendencies: what the affect system is pressing the body to do
ACTION_TENDENCIES = ("approach", "avoid", "freeze", "attack", "withdraw",
                     "explore")


class AffectSystem:
    """The emotional core."""

    def __init__(self, cfg: SimConfig, temperament: np.ndarray | None = None):
        self.cfg = cfg
        self.A = cfg.affect
        self.rng = np.random.default_rng(cfg.seed + 53)

        self.emotions = np.zeros(N_EMOTIONS)
        self.nm = np.array([NM_BASELINE[n] for n in NEUROMODULATORS], float)
        self.appraisal = np.zeros(N_APPRAISAL)
        self.appraisal_target = np.zeros(N_APPRAISAL)

        self.valence = 0.05
        self.arousal = 0.25
        self.dominance = 0.55
        self.tension = 0.15
        self.mood_valence = 0.05
        self.mood_arousal = 0.28
        self.mood_energy = 0.60
        self.stress = 0.12
        self.allostatic_load = 0.05
        # temperament: stable dispositions that bias appraisal
        self.temperament = (np.array([0.45, 0.55, 0.62, 0.68, 0.58])
                            if temperament is None else np.asarray(temperament, float))
        self.action = np.zeros(len(ACTION_TENDENCIES))
        self.learning_rate_mod = 1.0
        self.history: list = []

    # ------------------------------------------------------------------
    def update(self, dt: float, inp: AffectInputs) -> AffectFrame:
        CFG = self.A          # AffectConfig (the module-level A is the appraisal index)
        T = self.temperament
        neuroticism, extraversion, openness, agreeableness, conscientiousness = T

        # ---------------- 1. appraisal ---------------------------------
        # Novelty and pleasantness come from prediction error and reward.
        raw = np.zeros(N_APPRAISAL)
        raw[A["novelty"]] = fclip(inp.novelty * 1.4, 0, 1.2)
        raw[A["intrinsic_pleasantness"]] = fclip(
            inp.reward - inp.punishment - 0.5 * inp.pain, -1.2, 1.2)
        raw[A["goal_relevance"]] = fclip(
            0.45 + 0.6 * abs(inp.rpe) + 0.5 * inp.threat + 0.4 * inp.pain
            + 0.35 * (inp.hunger + inp.thirst + inp.air_hunger) / 3.0
            + 0.3 * inp.goal_progress, 0, 1.3)
        raw[A["goal_congruence"]] = fclip(
            inp.reward - inp.punishment - 1.4 * inp.threat - 1.1 * inp.pain
            - 0.45 * (inp.hunger + inp.thirst) / 2.0
            + 0.9 * inp.goal_progress, -1.3, 1.3)
        raw[A["agency_self"]] = fclip(inp.control - 0.15 * inp.threat, -1, 1)
        raw[A["coping_potential"]] = fclip(
            inp.control + 0.5 * inp.safety - inp.effort * 0.4
            - 0.8 * inp.threat - 0.6 * inp.balance_error - 1.4 * inp.fell
            - 0.5 * inp.sleepiness, -1.2, 1.2)
        raw[A["certainty"]] = fclip(2.0 * inp.predictability - 1.0, -1, 1)
        raw[A["norm_compatibility"]] = fclip(
            0.8 * inp.self_evaluation - 0.6 * inp.nausea + 0.2 * agreeableness,
            -1.2, 1.2)
        raw[A["urgency"]] = fclip(
            0.9 * inp.threat + 1.1 * inp.pain + 0.7 * inp.fell
            + 0.5 * inp.balance_error + 0.4 * inp.air_hunger
            + 0.35 * inp.novelty, 0, 1.4)
        raw[A["social_evaluation"]] = fclip(
            0.9 * inp.social_contact + 0.6 * inp.affective_touch
            - 0.4 * inp.pain, -1, 1.2)

        # temperament biases appraisal
        raw[A["novelty"]] *= 0.7 + 0.6 * openness
        raw[A["urgency"]] *= 0.7 + 0.7 * neuroticism
        raw[A["coping_potential"]] *= 1.1 - 0.35 * neuroticism
        raw[A["social_evaluation"]] *= 0.7 + 0.6 * extraversion
        raw[A["norm_compatibility"]] *= 0.7 + 0.7 * conscientiousness

        # interoceptive colouring: drives press on appraisal directly
        drive_pressure = (0.5 * inp.hunger + 0.5 * inp.thirst + 0.6 * inp.air_hunger
                          + 0.7 * inp.pain + 0.5 * inp.nausea + 0.35 * inp.bladder
                          + 0.4 * inp.thermal_discomfort + 0.45 * inp.sleepiness
                          + 0.55 * inp.ocular_discomfort)
        raw[A["goal_congruence"]] -= 0.45 * drive_pressure
        raw[A["urgency"]] += 0.25 * drive_pressure
        # the internal world: a threatened inner tone, rumination, a body that
        # surprises its own predictions, and the mood that remembered situations carry
        raw[A["urgency"]] += 0.30 * inp.inner_threat + 0.25 * inp.intero_surprise
        raw[A["goal_congruence"]] += 0.35 * inp.memory_valence - 0.30 * inp.rumination
        raw[A["certainty"]] -= 0.30 * inp.intero_surprise
        raw[A["novelty"]] *= 1.0 - 0.35 * inp.familiarity

        self.appraisal_target = raw * self.A.appraisal_gain
        # appraisal itself is a fast but not instantaneous process
        self.appraisal += (dt / 0.35) * (self.appraisal_target - self.appraisal)

        # ---------------- 2. emotions ----------------------------------
        drive = APPRAISAL_W @ self.appraisal
        target = np.tanh(fclip(drive - DELIBERATE_THRESHOLD, -6, 6))
        target = fclip(target, 0.0, 1.0)

        # neuromodulator gain on emotional reactivity
        nor = self.nm[NM["norepinephrine"]]
        gaba = self.nm[NM["gaba"]]
        serotonin = self.nm[NM["serotonin"]]
        gain = (1.0 + 0.8 * nor) * (1.0 - 0.35 * gaba) * (1.0 + 0.4 * neuroticism)
        # serotonin damps negative emotions specifically
        neg = EMOTION_VALENCE < 0
        react = target * gain
        react[neg] *= (1.0 - 0.3 * fclip(serotonin, 0, 1.5))
        react = fclip(react, 0.0, 1.4)

        tau = EMOTION_TAU * (1.0 + 0.4 * neuroticism)
        self.emotions += (dt / tau) * (react - self.emotions)
        self.emotions = fclip(self.emotions, 0.0, 1.5)
        self.emotions[:] = np.nan_to_num(self.emotions)

        # ---------------- 3. neuromodulators ---------------------------
        rpe = fclip(inp.rpe, -2, 2)
        rew = fclip(inp.reward, -2, 2)
        threat = fclip(inp.threat, 0, 1.5)
        novelty = fclip(inp.novelty, 0, 1.5)
        fear = float(self.emotions[E["fear"]])
        anger = float(self.emotions[E["anger"]])
        joy = float(self.emotions[E["joy"]])
        love = float(self.emotions[E["love"]])
        sadness = float(self.emotions[E["sadness"]])
        relief = float(self.emotions[E["relief"]])
        curiosity = float(self.emotions[E["curiosity"]])
        calm = float(self.emotions[E["calm"]])

        nm_target = {
            "dopamine_phasic": 0.05 + 0.9 * max(rpe, 0.0) - 0.5 * max(-rpe, 0.0),
            "dopamine_tonic": 0.35 + 0.28 * rew + 0.2 * joy + 0.15 * curiosity
                              - 0.25 * sadness,
            "serotonin": 0.55 + 0.35 * joy + 0.3 * love + 0.2 * relief
                         - 0.35 * sadness - 0.3 * self.stress,
            "norepinephrine": 0.30 + 0.7 * threat + 0.5 * fear + 0.4 * arousal_now(self)
                              + 0.2 * inp.novelty + 0.3 * anger,
            "acetylcholine": 0.40 + 0.5 * inp.novelty + 0.4 * curiosity
                             + 0.3 * inp.control,
            "oxytocin": 0.30 + 0.9 * inp.affective_touch + 0.6 * inp.social_contact
                        + 0.3 * love - 0.3 * threat,
            "vasopressin": 0.30 + 0.4 * anger + 0.3 * inp.threat,
            "endorphin": 0.20 + 1.1 * inp.pain + 1.0 * inp.effort
                         + 0.5 * inp.threat,
            "enkephalin": 0.25 + 0.7 * inp.pain + 0.3 * inp.effort,
            "endocannabinoid": 0.25 + 0.5 * joy + 0.4 * calm
                               + 0.25 * inp.affective_touch,
            "cortisol": 0.28 + 0.8 * inp.threat + 0.7 * self.stress
                        + 0.5 * inp.pain + 0.3 * sadness - 0.3 * inp.safety,
            "adrenaline": 0.12 + 0.9 * inp.threat + 0.6 * inp.fell
                          + 0.35 * inp.effort + 0.3 * anger,
            "noradrenaline_h": 0.30 + 0.5 * self.arousal + 0.3 * inp.threat,
            "testosterone": 0.50 + 0.35 * inp.control + 0.3 * anger
                            - 0.2 * sadness,
            "estrogen": 0.45,
            "progesterone": 0.30,
            "prolactin": 0.25 + 0.4 * love + 0.3 * inp.affective_touch
                         + 0.2 * sadness,
            "melatonin": 0.15 + 0.9 * fclip(inp.sleepiness, 0, 1.2)
                         - 0.4 * self.arousal,
            "adenosine": 0.25 + 0.8 * inp.sleep_pressure,
            "histamine": 0.20 + 0.9 * inp.itch + 0.3 * inp.threat
                         + 0.25 * (1.0 - inp.sleepiness),
            "gaba": 0.45 + 0.4 * calm + 0.25 * inp.safety
                    + 0.2 * inp.affective_touch - 0.3 * threat,
            "glutamate": 0.40 + 0.4 * self.arousal + 0.3 * inp.novelty
                         + 0.3 * threat,
            "substance_p": 0.15 + 0.9 * inp.pain + 0.4 * inp.itch
                           + 0.2 * inp.inflammation,
            "bdnf": 0.50 + 0.3 * curiosity + 0.25 * inp.goal_progress
                    + 0.2 * joy - 0.2 * self.stress,
            "orexin": 0.55 + 0.5 * (1.0 - inp.sleepiness) + 0.3 * curiosity
                      + 0.3 * inp.goal_progress,
        }
        for name, tgt in nm_target.items():
            i = NM[name]
            tgt = float(fclip(tgt, 0.0, 2.0))
            tau_nm = NM_TAU[name]
            self.nm[i] += (dt / tau_nm) * (tgt - self.nm[i])
        self.nm[:] = fclip(np.nan_to_num(self.nm), 0.0, 2.5)

        # ---------------- 4. core affect -------------------------------
        pos = float((self.emotions * fclip(EMOTION_VALENCE, 0, None)).sum())
        negv = float((self.emotions * fclip(-EMOTION_VALENCE, 0, None)).sum())
        homeostatic = (0.6 * inp.reward - 0.6 * inp.punishment
                       - 0.5 * inp.pain - 0.35 * drive_pressure)
        val_target = np.tanh(1.1 * (pos - 1.35 * negv) + homeostatic
                             + 0.15 * (self.mood_valence - self.valence))
        self.valence += (dt / CFG.tau_valence) * (val_target - self.valence)

        arousal_target = fclip(
            0.18 + 0.45 * (self.nm[NM["norepinephrine"]] - NM_BASELINE["norepinephrine"])
            + 0.30 * (self.nm[NM["adrenaline"]] - NM_BASELINE["adrenaline"])
            + 0.22 * float((self.emotions * fclip(EMOTION_AROUSAL, 0, None)).sum())
            + 0.25 * inp.novelty + 0.25 * inp.threat + 0.2 * inp.effort,
            0.0, 1.5)
        self.arousal += (dt / CFG.tau_arousal) * (arousal_target - self.arousal)

        dom_target = fclip(
            0.35 + 0.5 * inp.control + 0.3 * inp.safety
            + 0.35 * (self.nm[NM["testosterone"]] - 0.5)
            - 0.5 * (self.nm[NM["cortisol"]] - 0.28)
            - 0.35 * inp.fell - 0.3 * inp.balance_error,
            0.0, 1.2)
        self.dominance += (dt / CFG.tau_dominance) * (dom_target - self.dominance)

        self.tension = float(fclip(
            0.25 * self.emotions[E["anxiety"]] + 0.3 * self.emotions[E["fear"]]
            + 0.2 * self.emotions[E["anger"]] + 0.4 * self.stress
            + 0.2 * (1.0 - inp.predictability), 0.0, 1.5))
        self.pleasantness = float(fclip(self.valence, -1.5, 1.5))

        # ---------------- 5. stress ------------------------------------
        stress_target = fclip(
            0.5 * inp.threat + 0.6 * inp.pain + 0.7 * inp.fell
            + 0.4 * inp.balance_error + 0.3 * inp.effort
            + 0.25 * drive_pressure - 0.45 * inp.control
            - 0.35 * inp.safety + 0.2 * (1 - inp.predictability),
            0.0, 1.5)
        self.stress += (dt / 12.0) * (stress_target - self.stress)
        self.stress = float(fclip(self.stress, 0.0, 1.5))
        self.allostatic_load = float(fclip(
            self.allostatic_load + dt * (0.002 * self.stress - 0.00035), 0, 1))

        # ---------------- 6. mood --------------------------------------
        self.mood_valence += (dt / CFG.tau_mood) * (self.valence - self.mood_valence)
        self.mood_arousal += (dt / CFG.tau_mood) * (self.arousal - self.mood_arousal)
        energy_target = fclip(
            0.8 - 0.5 * inp.sleepiness - 0.3 * inp.effort + 0.2 * self.valence,
            0.0, 1.2)
        self.mood_energy += (dt / (CFG.tau_mood * 0.7)) * (energy_target - self.mood_energy)

        # ---------------- 7. action tendencies -------------------------
        explore = (0.5 * self.emotions[E["curiosity"]] + 0.3 * openness
                   + 0.3 * self.nm[NM["dopamine_phasic"]]) * (1 - self.stress)
        act = np.array([
            # approach
            fclip(0.7 * self.valence + 0.5 * self.emotions[E["joy"]]
                    + 0.3 * self.dominance, 0, 1.5),
            # avoid
            fclip(0.8 * self.emotions[E["fear"]] + 0.5 * self.emotions[E["disgust"]]
                    + 0.6 * inp.threat, 0, 1.5),
            # freeze
            fclip(0.9 * self.emotions[E["fear"]] * (1.0 - self.dominance)
                    + 0.4 * self.emotions[E["anxiety"]], 0, 1.5),
            # attack
            fclip(0.9 * self.emotions[E["anger"]] * self.dominance
                    + 0.3 * (self.nm[NM["testosterone"]] - 0.5), 0, 1.5),
            # withdraw
            fclip(0.8 * self.emotions[E["sadness"]] + 0.5 * self.emotions[E["shame"]]
                    + 0.4 * self.emotions[E["loneliness"]], 0, 1.5),
            # explore
            fclip(explore, 0, 1.5),
        ])
        self.action += (dt / 1.5) * (act - self.action)

        # ---------------- 8. gain modulation ---------------------------
        # dopamine and acetylcholine set the effective learning rate;
        # noradrenaline and arousal set policy temperature and motor noise.
        self.learning_rate_mod = float(fclip(
            0.4 + 1.4 * self.nm[NM["dopamine_phasic"]]
            + 0.6 * self.nm[NM["acetylcholine"]], 0.2, 3.0))

        emo = self.emotions
        order = int(np.argmax(emo))
        dominant, dint = EMOTIONS[order], float(emo[order])
        intensity = float(fclip(np.abs(self.arousal) * 0.5
                                  + np.linalg.norm(emo) * 0.35
                                  + abs(self.valence) * 0.4, 0, 2.0))

        frame = AffectFrame(
            appraisal=self.appraisal.copy(),
            appraisal_raw=self.appraisal_target.copy(),
            emotions=self.emotions.copy(),
            neuromodulators=self.nm.copy(),
            valence=float(self.valence),
            arousal=float(self.arousal),
            dominance=float(self.dominance),
            tension=float(self.tension),
            pleasantness=float(self.pleasantness),
            mood_valence=float(self.mood_valence),
            mood_arousal=float(self.mood_arousal),
            mood_energy=float(self.mood_energy),
            temperament=self.temperament.copy(),
            stress=float(self.stress),
            allostatic_load=float(self.allostatic_load),
            action_tendency=self.action.copy(),
            feeling_intensity=intensity,
            dominant_emotion=dominant if dint > 0.08 else "neutral",
            dominant_intensity=dint,
        )
        return frame

    # ------------------------------------------------------------------
    def neuromodulator(self, name: str) -> float:
        return float(self.nm[NM[name]])

    def emotion(self, name: str) -> float:
        return float(self.emotions[E[name]])

    def reset(self) -> None:
        self.emotions[:] = 0.0
        self.appraisal[:] = 0.0
        self.valence = 0.05
        self.arousal = 0.25
        self.dominance = 0.55
        self.stress = 0.12

    def describe(self) -> dict:
        return {
            "appraisal_dimensions": list(APPRAISAL_DIMS),
            "n_emotions": N_EMOTIONS,
            "emotions": list(EMOTIONS),
            "n_neuromodulators": N_NEUROMODULATORS,
            "neuromodulators": list(NEUROMODULATORS),
            "temperament_traits": list(TEMPERAMENT_TRAITS),
            "action_tendencies": list(ACTION_TENDENCIES),
            "core_affect": ["valence", "arousal", "dominance", "tension",
                            "pleasantness"],
        }


def arousal_now(sys: AffectSystem) -> float:
    return float(sys.arousal)
