"""GNSS position spoofing profiles applied to logged fixes.

These change the position values a receiver reports. They do not model the
RF side of spoofing.
"""

import numpy as np


def _fixes(fixes_m, name: str) -> np.ndarray:
    fixes = np.asarray(fixes_m, dtype=float)
    if fixes.ndim != 2 or fixes.shape[1] != 2:
        raise ValueError(
            f"{name}: fixes must be (n, 2) east and north, got shape {fixes.shape}"
        )
    return fixes


def _same_length(name: str, fixes: np.ndarray, **arrays) -> None:
    wrong = {key: len(value) for key, value in arrays.items() if len(value) != len(fixes)}
    if wrong:
        detail = ", ".join(f"{key} has {length}" for key, length in wrong.items())
        raise ValueError(
            f"{name}: {len(fixes)} fixes but {detail}; the attack needs one value per fix"
        )


def _finite(name: str, key: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name}: {key} must be a number, got {value}")
    return value


def jump(fixes_m, t_s, t_start_s, offset_m):
    """Add a constant offset (east, north) to every fix from `t_start_s` on."""
    fixes = _fixes(fixes_m, "jump")
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    _same_length("jump", fixes, t_s=t_s)
    offset = np.asarray(offset_m, dtype=float).reshape(-1)
    if offset.shape != (2,):
        raise ValueError(f"jump: the offset must be (east, north), got {offset_m!r}")
    if not np.isfinite(offset).all():
        raise ValueError(f"jump: the offset must be a number, got {offset_m!r}")
    t_start_s = _finite("jump", "t_start_s", t_start_s)
    out = fixes.copy()
    out[t_s >= t_start_s] += offset
    return out


def drag_off(fixes_m, t_s, t_start_s, rate_mps, direction=(1.0, 0.0)):
    """Pull the fixes away at `rate_mps` along `direction`, from `t_start_s` on.

    The offset grows linearly from zero, so no single fix looks out of place.
    """
    fixes = _fixes(fixes_m, "drag_off")
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    _same_length("drag_off", fixes, t_s=t_s)
    rate_mps = _finite("drag_off", "rate_mps", rate_mps)
    t_start_s = _finite("drag_off", "t_start_s", t_start_s)
    d = np.asarray(direction, dtype=float).reshape(-1)
    if d.shape != (2,):
        raise ValueError(f"drag_off: direction must be (east, north), got {direction!r}")
    norm = np.hypot(d[0], d[1])
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(
            f"drag_off: direction {direction!r} has no length, so there is no way to pull"
        )
    d = d / norm
    elapsed = np.clip(t_s - t_start_s, 0.0, None)
    return fixes + np.outer(rate_mps * elapsed, d)


def drag_across_track(fixes_m, t_s, t_start_s, rate_mps, vel_mps):
    """Pull the fixes sideways at `rate_mps`, to the left of the true track.

    Unlike drag_off, the direction turns with the vehicle: each step adds
    rate * dt along the left normal of `vel_mps[k]`, (n, 2) in m/s, so the
    pull stays across the track however long it runs. This is the attacker
    who knows the route. A negative rate pulls to the right.

    A sample with no usable direction (the vehicle stood still, or the
    velocity is not a number) keeps the direction of the last sample that has
    one, so a stop in the middle of a run does not turn the pull into NaNs;
    samples before the first usable one take that first direction too.
    """
    fixes = _fixes(fixes_m, "drag_across_track")
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    v = np.asarray(vel_mps, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(
            f"drag_across_track: velocities must be (n, 2), got shape {v.shape}"
        )
    _same_length("drag_across_track", fixes, t_s=t_s, vel_mps=v)
    rate_mps = _finite("drag_across_track", "rate_mps", rate_mps)
    t_start_s = _finite("drag_across_track", "t_start_s", t_start_s)

    speed = np.hypot(v[:, 0], v[:, 1])
    moving = np.isfinite(speed) & (speed > 0.0)
    if not moving.any():
        raise ValueError(
            "drag_across_track: no sample has a usable velocity direction, so there is "
            "nothing to pull across"
        )
    first = int(np.flatnonzero(moving)[0])
    left = np.zeros_like(v)
    left[moving] = np.column_stack([-v[moving, 1], v[moving, 0]]) / speed[moving, None]
    # carry the last usable direction forward, and the first one backward
    source = np.maximum.accumulate(np.where(moving, np.arange(len(fixes)), 0))
    source[:first] = first
    left = left[source]

    step_s = np.diff(np.clip(t_s - t_start_s, 0.0, None), prepend=0.0)
    return fixes + np.cumsum(rate_mps * step_s[:, None] * left, axis=0)
