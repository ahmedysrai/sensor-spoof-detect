"""GNSS position spoofing profiles applied to logged fixes.

These change the position values a receiver reports. They do not model the
RF side of spoofing.
"""

import numpy as np


def jump(fixes_m, t_s, t_start_s, offset_m):
    """Add a constant offset (east, north) to every fix from `t_start_s` on."""
    out = fixes_m.copy()
    out[t_s >= t_start_s] += np.asarray(offset_m, dtype=float)
    return out


def drag_off(fixes_m, t_s, t_start_s, rate_mps, direction=(1.0, 0.0)):
    """Pull the fixes away at `rate_mps` along `direction`, from `t_start_s` on.

    The offset grows linearly from zero, so no single fix looks out of place.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    elapsed = np.clip(t_s - t_start_s, 0.0, None)
    return fixes_m + np.outer(rate_mps * elapsed, d)
