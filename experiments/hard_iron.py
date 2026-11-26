"""Hard-iron offset on the magnetometer: inflate its sd, or estimate it.

Follows magnetometer.py, whose last case put a 2 uT forward-axis offset
(the BMM150 zero-B offset after calibration, datasheet BST-BMM150-DS001-05
rev. 1.4, page 8) on the readings while the filter assumed white noise
only: 6065 false alarms an hour. Two ways to handle it are compared here
on the same data:

- inflated sd: the heading row keeps its form but its sd covers the
  offset, hypot(0.6, 2) / 25 = 0.084 rad instead of 0.024. A constant
  error is still treated as white noise, so this is a stopgap.
- offset states: two more states for the forward and left offset, prior
  sd 2 uT per axis, constant, heading row from mag_heading_model.

Everything else as magnetometer.py: 1 deg/s gyro offset estimated with
the assumed 1.1e-4 rad/s per sqrt(s) walk, BMM150 regular preset noise,
25 uT horizontal field (assumed), 600 s at 10 Hz, track-following
sideways drag from t = 60 s, 100 seeds, CUSUM drift 0.5, threshold 10.5
on all innovation columns. For rate 0 the table gives CUSUM false alarms
per hour, otherwise the fraction of seeds found and the median delay.

    python experiments/hard_iron.py
"""

import sys

import numpy as np

from spoofdet.attacks import drag_across_track
from spoofdet.cusum import cusum
from spoofdet.kf import mag_heading, run_gnss_odo_gyro_filter
from spoofdet.sim import (
    gnss_fixes,
    gyro_yaw_rate,
    magnetometer,
    simulate_trajectory,
    wheel_speed,
)

DT_S = 0.1
N_STEPS = 6000
ACC_STD_MPS2 = 0.2
GNSS_STD_M = 1.5
SPEED_STD_MPS = 0.05
GYRO_STD_RADPS = 5.5e-4  # BMI088, see sim.gyro_yaw_rate
OFFSET_RADPS = np.deg2rad(1.0)
WALK_RADPS = 1.1e-4  # per sqrt(s), assumed, see gyro_offset.py
FIELD_UT = 25.0  # assumed
MAG_NOISE_UT = 0.6  # BMM150 regular preset
MAG_STD_RAD = MAG_NOISE_UT / FIELD_UT
T_ATTACK_S = 60.0
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SETTLE_S = 5.0
STEPS_PER_HOUR = 3600 / DT_S
CUSUM_DRIFT = 0.5
CUSUM_THRESHOLD = 10.5
RATES_MPS = [0.0, 0.02, 0.05, 0.1]

HARD_IRON_STD_UT = 2.0  # BMM150 zero-B offset after calibration

# name, offset in the data (uT), heading sd told to the filter, offset states
CASES = [
    ("white noise only", (2.0, 0.0), MAG_STD_RAD, False),
    ("inflated sd", (2.0, 0.0), np.hypot(MAG_NOISE_UT, 2.0) / FIELD_UT, False),
    ("offset states", (2.0, 0.0), MAG_STD_RAD, True),
    ("states, no offset", (0.0, 0.0), MAG_STD_RAD, True),
]


def run(seed, rate_mps, hard_iron_ut, mag_std_rad, offset_states):
    rng = np.random.default_rng(seed)
    traj = simulate_trajectory(rng, N_STEPS, DT_S, ACC_STD_MPS2)
    speed = wheel_speed(rng, traj, SPEED_STD_MPS)
    z = gnss_fixes(rng, traj, GNSS_STD_M)
    z = drag_across_track(z, traj.t_s, T_ATTACK_S, rate_mps, traj.vel_mps)
    yaw = gyro_yaw_rate(rng, traj, GYRO_STD_RADPS, bias_radps=OFFSET_RADPS)
    kwargs = {"bias_std_radps": OFFSET_RADPS, "bias_rw_radps": WALK_RADPS}
    mag = magnetometer(rng, traj, FIELD_UT, MAG_NOISE_UT, hard_iron_ut)
    kwargs |= {"mag_psi_rad": mag_heading(mag), "mag_std_rad": mag_std_rad}
    if offset_states:
        kwargs |= {"hard_iron_std_ut": HARD_IRON_STD_UT, "field_ut": FIELD_UT}
    _, _, nu_track = run_gnss_odo_gyro_filter(
        z,
        speed,
        yaw,
        DT_S,
        ACC_STD_MPS2,
        GNSS_STD_M,
        SPEED_STD_MPS,
        GYRO_STD_RADPS,
        **kwargs,
    )
    alarms, _ = cusum(nu_track, CUSUM_DRIFT, CUSUM_THRESHOLD)
    return alarms.any(axis=1)


def main():
    t_s = np.arange(N_STEPS) * DT_S
    k0 = round(T_ATTACK_S / DT_S)
    settled = t_s >= SETTLE_S
    hours = N_SEEDS * settled.sum() / STEPS_PER_HOUR
    print(f"{'filter':<20}{'rate m/s':>9}{'FA/h':>8}{'found':>7}{'delay s':>9}")
    for name, *case in CASES:
        for rate in RATES_MPS:
            n_alarm, delays = 0, []
            for seed in range(N_SEEDS):
                alarm = run(seed, rate, *case)
                if rate == 0.0:
                    n_alarm += alarm[settled].sum()
                    continue
                after = np.flatnonzero(alarm[k0:])
                if len(after):
                    delays.append(after[0] * DT_S)
            if rate == 0.0:
                print(f"{name:<20}{'none':>9}{n_alarm / hours:8.1f}")
            else:
                d = f"{np.median(delays):9.1f}" if delays else f"{'-':>9}"
                print(f"{name:<20}{rate:9.2f}{'':8}{len(delays) / N_SEEDS:7.2f}{d}")


if __name__ == "__main__":
    main()
