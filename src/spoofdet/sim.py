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


def gyro_yaw_rate(rng, traj, gyro_std_radps=5.5e-4, bias_radps=0.0):
    """Gyro yaw rate per sample, (n,) in rad/s, counterclockwise positive.

    The heading is the direction of the velocity (a wheeled vehicle with no
    sideslip). Sample k is the mean rate over the step from k - 1 to k, so
    heading[k] = heading[k - 1] + rate[k] * dt; sample 0 is zero.

    The noise is white, `gyro_std_radps` per sample. The default is the
    Bosch BMI088 gyro noise density, 0.014 deg/s/sqrt(Hz) typ. (datasheet
    BST-BMI088-DS000-19, rev. 1.9, table 5), averaged over a 0.1 s sample:
    0.014 * sqrt(1 / (2 * 0.1 s)) = 0.031 deg/s = 5.5e-4 rad/s. That reads
    the density as one-sided, which matches the same table's 0.1 deg/s rms
    in 47 Hz (0.014 * sqrt(47) = 0.096).

    `bias_radps` is a constant zero-rate offset added to every sample. The
    same table gives +-1 deg/s typ. (0.017 rad/s) at 25 C with offset
    cancellation off, and a change over temperature of +-0.015 deg/s per K.
    The offset is held constant over a run here; temperature drift is not
    modelled.
    """
    psi = np.unwrap(np.arctan2(traj.vel_mps[:, 1], traj.vel_mps[:, 0]))
    dt_s = traj.t_s[1] - traj.t_s[0]
    rate = np.r_[0.0, np.diff(psi) / dt_s]
    return rate + bias_radps + rng.normal(0.0, gyro_std_radps, size=rate.shape)
