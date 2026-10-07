"""A deliberately tiny reference subsystem: copy its structure, not its content.

It models a bank of leaky integrators ("units") driven by a random projection of a few bus
inputs, reports their mean activity as a summary and pushes one standing output.  It exists so
that the framework (tools/ultra_bench.py, tests) has something to run, and so that a new
domain has a template for sizes-per-level, preallocation, budgets, outputs and counts().
"""
from __future__ import annotations

import numpy as np

from .ultra import Bus, Level, Subsystem, register


@register
class ExampleIntegrators(Subsystem):
    name = "example_integrators"
    domain = "example"
    rate_hz = 20.0
    summary_dim = 8
    reads = ("core_affect", "intero")
    writes = ("example_integrators.mean",)
    wall_budget_ms_per_sim_s = {"ultra": 20.0, "mega": 400.0}
    ram_budget_mb = {"ultra": 10.0, "mega": 100.0}

    def __init__(self, level: Level, seed: int):
        super().__init__(level, seed)
        self.n = level.pick(ultra=50_000, mega=1_000_000)          # sizes per level
        self.v = np.zeros(self.n, np.float32)                        # state: preallocated
        self.tau = (0.2 + 2.0 * self.rng.random(self.n)).astype(np.float32)
        self.w = self.rng.standard_normal((self.n, 4)).astype(np.float32) * 0.3
        self._u = np.zeros(4, np.float32)
        self._drive = np.zeros(self.n, np.float32)

    def step(self, dt: float, bus: Bus) -> None:
        core = bus.vec("core_affect", 10)
        intero = bus.vec("intero", 67)
        u = self._u
        u[0], u[1], u[2], u[3] = core[0], core[1], intero[0] * 0.01, np.tanh(intero[1] * 0.01)
        np.dot(self.w, u, out=self._drive)                             # no per-call allocation of size n
        self.v += (dt / self.tau) * (np.tanh(self._drive) - self.v)
        m = float(self.v.mean())
        self.summary[:] = (m, float(self.v.std()), float(np.abs(self.v).max()), 0, 0, 0, 0, 0)
        self.out["drive.curiosity"] = 0.02 * m                       # a standing level, not an event
        bus.put("example_integrators.mean", m)

    def n_state(self) -> int:
        return self.n

    def mem_bytes(self) -> int:
        return self.v.nbytes + self.tau.nbytes + self.w.nbytes + self._drive.nbytes

    def counts(self) -> dict:
        return {"units": self.n}

    def reset(self) -> None:
        self.v[:] = 0
