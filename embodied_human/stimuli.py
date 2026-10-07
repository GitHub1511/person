"""Static odorant / tastant sources coupled to body position."""
from __future__ import annotations

import numpy as np
from typing import List


class StimulusSource:
    def __init__(self, position, odor_rate: float = 0.0, tastant_rate: float = 0.0):
        self.position = np.array(position, dtype=np.float64)
        self.odor_rate = float(odor_rate)
        self.tastant_rate = float(tastant_rate)


class Stimuli:
    def __init__(self, sources: List[StimulusSource]):
        self.sources = sources

    def get_odorant_at(self, point: np.ndarray) -> float:
        p = np.asarray(point, dtype=np.float64)
        total = 0.0
        for s in self.sources:
            d = float(np.linalg.norm(p - s.position))
            total += s.odor_rate / (d + 1e-6)
        return float(total)

    def get_tastant_at(self, point: np.ndarray) -> float:
        p = np.asarray(point, dtype=np.float64)
        total = 0.0
        for s in self.sources:
            d = float(np.linalg.norm(p - s.position))
            total += s.tastant_rate / (d + 1e-6)
        return float(total)


def default_stimuli() -> Stimuli:
    sources = [
        StimulusSource(position=(1.0, 0.0, 0.5), odor_rate=2.0, tastant_rate=0.0),
        StimulusSource(position=(-1.0, 0.0, 0.5), odor_rate=0.0, tastant_rate=2.0),
    ]
    return Stimuli(sources)
