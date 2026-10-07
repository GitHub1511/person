"""
Shared state snapshot.

Every subsystem (receptors, interoception, affect, prediction, motor) reads
from this one immutable-ish view of the body, so no subsystem has to know how
to talk to MuJoCo.  It is rebuilt once per physics step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContactInfo:
    """One MuJoCo contact, resolved into body-level quantities."""
    geom1: int
    geom2: int
    body1: int
    body2: int
    name1: str
    name2: str
    pos: np.ndarray          # world contact point
    normal: np.ndarray       # world contact normal (from geom1 to geom2)
    force: np.ndarray        # (6,) contact force/torque in contact frame
    dist: float
    is_self: bool            # both sides belong to the human
    other_temp: float        # surface temperature of the thing being touched
    other_object: str        # scene-object name or ""
    friction: float
    temp1: float = 22.0      # surface temperature of geom1
    temp2: float = 22.0      # surface temperature of geom2

    def temp_for(self, body: int) -> float:
        """Temperature of the *other* surface from ``body``'s point of view."""
        return self.temp2 if body == self.body1 else self.temp1


@dataclass
class BodyState:
    """A complete snapshot of the body at one instant."""

    t: float = 0.0
    step: int = 0

    # ---- actuated joints (length nu, ordered as meta.joint_order) --------
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(0))      # measured
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros(0))     # commanded
    joint_limit_proximity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_min: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_max: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ---- floating base ---------------------------------------------------
    root_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    root_linvel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ---- whole-body mechanics -------------------------------------------
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    com_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    com_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    support_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cop: np.ndarray = field(default_factory=lambda: np.zeros(3))
    external_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    external_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    mechanical_power: float = 0.0
    actuator_power: float = 0.0
    torque_effort: float = 0.0
    com_over_support: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # ---- kinematics landmarks -------------------------------------------
    site_pos: dict = field(default_factory=dict)
    site_vel: dict = field(default_factory=dict)
    body_pos: dict = field(default_factory=dict)
    gaze_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gaze_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    head_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))

    # ---- contact ---------------------------------------------------------
    contacts: list = field(default_factory=list)
    n_contact: int = 0
    foot_contact: np.ndarray = field(default_factory=lambda: np.zeros(2, bool))
    self_touch: bool = False
    fallen: bool = False
    fall_energy: float = 0.0

    # ---- configuration metadata -----------------------------------------
    joint_names: list = field(default_factory=list)

    def qof(self, name: str) -> float:
        try:
            return float(self.q[self.joint_names.index(name)])
        except ValueError:
            return 0.0

    def qdof(self, name: str) -> float:
        try:
            return float(self.qd[self.joint_names.index(name)])
        except ValueError:
            return 0.0

    def tauof(self, name: str) -> float:
        try:
            return float(self.tau[self.joint_names.index(name)])
        except ValueError:
            return 0.0
