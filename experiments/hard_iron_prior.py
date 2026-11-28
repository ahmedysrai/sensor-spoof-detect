"""How much a better hard-iron calibration buys back.

Follows hard_iron.py, where two offset states with a 2 uT prior per axis
fixed the false alarms from a 2 uT forward offset but let a slow sideways
drag hide partly in the offset estimate. Here the prior sd and the offset
in the data shrink together, as they would after a better calibration:
2, 1 and 0.5 uT. The last case keeps the 0.5 uT prior but puts 2 uT in
the data, a calibration that claims more than it delivers.

Everything else as hard_iron.py: 1 deg/s gyro offset estimated with the
assumed 1.1e-4 rad/s per sqrt(s) walk, BMM150 regular preset noise
(0.6 uT), 25 uT horizontal field (assumed), 600 s at 10 Hz,
track-following sideways drag from t = 60 s, 100 seeds, CUSUM drift 0.5,
threshold 10.5 on all innovation columns. For rate 0 the table gives
CUSUM false alarms per hour, otherwise the fraction of seeds found and
the median delay. The last column is the forward offset estimate minus
the true offset at the end of the run, mean +- sd over seeds.

    python experiments/hard_iron_prior.py
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

# name, prior sd per axis (uT), forward offset in the data (uT)
CASES = [
    ("prior 2, offset 2", 2.0, 2.0),
    ("prior 1, offset 1", 1.0, 1.0),
    ("prior 0.5, offset 0.5", 0.5, 0.5),
    ("prior 0.5, offset 2", 0.5, 2.0),
]


def run(seed, rate_mps, prior_ut, offset_ut):
    rng = np.random.default_rng(seed)
    traj = simulate_trajectory(rng, N_STEPS, DT_S, ACC_STD_MPS2)
    speed = wheel_speed(rng, traj, SPEED_STD_MPS)
    z = gnss_fixes(rng, traj, GNSS_STD_M)
    z = drag_across_track(z, traj.t_s, T_ATTACK_S, rate_mps, traj.vel_mps)
    yaw = gyro_yaw_rate(rng, traj, GYRO_STD_RADPS, bias_radps=OFFSET_RADPS)
    mag = magnetometer(rng, traj, FIELD_UT, MAG_NOISE_UT, (offset_ut, 0.0))
    xs, _, nu_track = run_gnss_odo_gyro_filter(
        z,
        speed,
        yaw,
        DT_S,
        ACC_STD_MPS2,
        GNSS_STD_M,
        SPEED_STD_MPS,
        GYRO_STD_RADPS,
        bias_std_radps=OFFSET_RADPS,
        bias_rw_radps=WALK_RADPS,
        mag_psi_rad=mag_heading(mag),
        mag_std_rad=MAG_STD_RAD,
        hard_iron_std_ut=prior_ut,
        field_ut=FIELD_UT,
    )
    alarms, _ = cusum(nu_track, CUSUM_DRIFT, CUSUM_THRESHOLD)
    return alarms.any(axis=1), xs[-1, -2] - offset_ut


def main():
    t_s = np.arange(N_STEPS) * DT_S
    k0 = round(T_ATTACK_S / DT_S)
    settled = t_s >= SETTLE_S
    hours = N_SEEDS * settled.sum() / STEPS_PER_HOUR
    print(
        f"{'case':<23}{'rate m/s':>9}{'FA/h':>8}{'found':>7}{'delay s':>9}"
        f"{'fwd err uT':>16}"
    )
    for name, *case in CASES:
        for rate in RATES_MPS:
            n_alarm, delays, errs = 0, [], []
            for seed in range(N_SEEDS):
                alarm, err = run(seed, rate, *case)
                errs.append(err)
                if rate == 0.0:
                    n_alarm += alarm[settled].sum()
                    continue
                after = np.flatnonzero(alarm[k0:])
                if len(after):
                    delays.append(after[0] * DT_S)
            e = f"{np.mean(errs):+8.2f} +- {np.std(errs):.2f}"
            if rate == 0.0:
                print(f"{name:<23}{'none':>9}{n_alarm / hours:8.1f}{'':16}{e}")
            else:
                d = f"{np.median(delays):9.1f}" if delays else f"{'-':>9}"
                found = len(delays) / N_SEEDS
                print(f"{name:<23}{rate:9.2f}{'':8}{found:7.2f}{d}{e}")


if __name__ == "__main__":
    main()
