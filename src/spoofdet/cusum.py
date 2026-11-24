"""Two-sided CUSUM on standardised innovations.

A per-step NIS gate compares each innovation with its own spread, so an
attack that keeps every innovation small passes it, even when they all lean
the same way. CUSUM adds up the lean. For samples y that are N(0, 1) when
nothing is wrong, the upper statistic is

    g_k = max(0, g_{k-1} + y_k - drift)

and the lower one is the same for -y. A mean shift of size m grows g by
about m - drift per step, so `drift` is usually half the smallest shift
worth detecting (0.5 for a shift of one standard deviation). The statistic
alarms when it passes `threshold` and then restarts from zero, so a
persistent attack keeps alarming and alarms can be counted.
"""

import numpy as np


def cusum(y, drift=0.5, threshold=10.0):
    """Two-sided CUSUM over each column of y, (n,) or (n, m).

    NaN samples (filter not started yet) leave the statistics unchanged.
    Returns the alarms, a bool array shaped like y that is True on the steps
    where a column's upper or lower statistic crossed `threshold`, and the
    signed statistic of each column: the upper one where it is larger,
    minus the lower one otherwise.

    `drift` has to be positive: at or below zero the statistic climbs on a
    sequence of zeros and the detector alarms on clean data. `threshold` has
    to be positive for the same reason, and `y` has to hold at least one
    sample, so a run too short to analyse is an error rather than a silence.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        raise ValueError("cusum: there is nothing to accumulate, y is empty")
    if y.ndim > 2:
        raise ValueError(f"cusum: y must be (n,) or (n, m), got shape {y.shape}")
    if not np.isfinite(drift) or drift <= 0.0:
        raise ValueError(f"cusum: drift must be positive, got {drift}")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"cusum: threshold must be positive, got {threshold}")
    flat = y.ndim == 1
    y = y.reshape(len(y), -1)
    g_hi = np.zeros(y.shape[1])
    g_lo = np.zeros(y.shape[1])
    alarms = np.zeros(y.shape, dtype=bool)
    stat = np.zeros(y.shape)
    for k, yk in enumerate(y):
        ok = ~np.isnan(yk)
        g_hi[ok] = np.maximum(0.0, g_hi[ok] + yk[ok] - drift)
        g_lo[ok] = np.maximum(0.0, g_lo[ok] - yk[ok] - drift)
        stat[k] = np.where(g_hi >= g_lo, g_hi, -g_lo)
        alarms[k] = (g_hi > threshold) | (g_lo > threshold)
        g_hi[alarms[k]] = 0.0
        g_lo[alarms[k]] = 0.0
    if flat:
        return alarms[:, 0], stat[:, 0]
    return alarms, stat
