"""Cheap replacement for np.clip on the per-step hot path (the np.clip wrapper
costs several microseconds per call and it is called ~50 times per physics step)."""
import numpy as np

try:                                    # numpy >= 2
    _uclip = np._core.umath.clip
except AttributeError:                  # numpy 1.x
    _uclip = np.core.umath.clip


def fclip(a, lo, hi, out=None):
    if lo is None:
        return np.minimum(a, hi, out=out)
    if hi is None:
        return np.maximum(a, lo, out=out)
    return _uclip(a, lo, hi) if out is None else _uclip(a, lo, hi, out)
