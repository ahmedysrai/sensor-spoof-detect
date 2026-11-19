"""Planar vehicle trajectory and GNSS position fixes, ENU frame (east, north).

The motion model is a near-constant-velocity random walk: white acceleration
noise of `accel_std_mps2` per axis, held over each step. GNSS fixes are the
true position plus independent Gaussian noise of `gnss_std_m` per axis. Both
values are assumptions for the crude version; datasheet-based models replace
them later.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    t_s: np.ndarray  # (n,)
    pos_m: np.ndarray  # (n, 2) east, north
    vel_mps: np.ndarray  # (n, 2)


def simulate_trajectory(
    rng, n_steps, dt_s=0.1, accel_std_mps2=0.2, vel0_mps=(2.0, 0.5)
):
    t_s = np.arange(n_steps) * dt_s
    acc = rng.normal(0.0, accel_std_mps2, size=(n_steps, 2))
    acc[0] = 0.0
    vel = np.asarray(vel0_mps, dtype=float) + np.cumsum(acc * dt_s, axis=0)
    # position from the velocity at the start of each step plus the
    # acceleration term, the same discretisation the filter assumes
    pos = np.zeros((n_steps, 2))
    for k in range(1, n_steps):
        pos[k] = pos[k - 1] + vel[k - 1] * dt_s + 0.5 * acc[k] * dt_s**2
    return Trajectory(t_s, pos, vel)


def gnss_fixes(rng, traj, gnss_std_m=1.5):
    """Noisy GNSS position fixes, one per trajectory sample, (n, 2) in m."""
    return traj.pos_m + rng.normal(0.0, gnss_std_m, size=traj.pos_m.shape)


def wheel_speed(rng, traj, speed_std_mps=0.05):
    """Wheel odometry speed |v| per sample, (n,) in m/s.

    White Gaussian noise of `speed_std_mps` (assumed). Wheel slip and the
    scale error from a wrong wheel radius are not modelled yet.
    """
    speed = np.linalg.norm(traj.vel_mps, axis=1)
    return speed + rng.normal(0.0, speed_std_mps, size=speed.shape)
