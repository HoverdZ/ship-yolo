"""Gate-mask visualization helpers."""

from __future__ import annotations

import numpy as np


def normalize_array(array, shared_min=None, shared_max=None):
    values = np.asarray(array, dtype=np.float32)
    lo = float(np.min(values) if shared_min is None else shared_min)
    hi = float(np.max(values) if shared_max is None else shared_max)
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)
