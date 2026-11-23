"""What a gyro zero-rate offset does to the CUSUM, and the offset state.

The BMI088 gyro has a zero-rate offset of +-1 deg/s typ. with offset
cancellation off (datasheet BST-BMI088-DS000-19 rev. 1.9, table 5). Here
every run has a 1 deg/s offset, held constant, and the filter either
ignores it (four states) or estimates it (five states, prior sd 1 deg/s).
The five-state filter is run with no random walk on the offset and with
1.1e-4 rad/s per sqrt(s). That second value is an assumption: the same
table gives 0.015 deg/s per K of offset change over temperature, and a
10 K change in 10 min would move the offset by 0.15 deg/s, which is
0.15 / sqrt(600 s) = 0.0061 deg/s per sqrt(s) as a random walk. The
simulated offset itself does not drift, so the walk can only cost.

Same runs as slow_drag.py otherwise: 600 s at 10 Hz, track-following
sideways drag from t = 60 s, 100 seeds, CUSUM drift 0.5, threshold 10.5.
The first row is the old four-state filter with no offset in the data, the
baseline from slow_drag.py. For rate 0 the table gives CUSUM false alarms
per hour, otherwise the fraction of seeds found and the median delay. The
last two columns are the mean absolute error of the offset estimate at
60 s and at the end, from the clean runs.

    python experiments/gyro_offset.py
"""

import numpy as np

from spoofdet.attacks import drag_across_track
from spoofdet.cusum import cusum
from spoofdet.kf import run_gnss_odo_gyro_filter
from spoofdet.sim import gnss_fixes, gyro_yaw_rate, simulate_trajectory, wheel_speed

DT_S = 0.1
N_STEPS = 6000
ACC_STD_MPS2 = 0.2
GNSS_STD_M = 1.5
SPEED_STD_MPS = 0.05
GYRO_STD_RADPS = 5.5e-4  # BMI088, see sim.gyro_yaw_rate
OFFSET_RADPS = np.deg2rad(1.0)
WALK_RADPS = 1.1e-4  # per sqrt(s), assumed, see above
T_ATTACK_S = 60.0
N_SEEDS = 100
SETTLE_S = 5.0
STEPS_PER_HOUR = 3600 / DT_S
CUSUM_DRIFT = 0.5
CUSUM_THRESHOLD = 10.5
RATES_MPS = [0.0, 0.05, 0.1, 0.2]

# name, offset in the data, filter keyword arguments, rates to run
CASES = [
    ("4 states, no offset", 0.0, {}, RATES_MPS),
    ("4 states", OFFSET_RADPS, {}, [0.0]),
    ("5 states, no walk", OFFSET_RADPS, {"bias_std_radps": OFFSET_RADPS}, RATES_MPS),
    (
        "5 states, walk",
        OFFSET_RADPS,
        {"bias_std_radps": OFFSET_RADPS, "bias_rw_radps": WALK_RADPS},
        RATES_MPS,
    ),
]


def run(seed, rate_mps, offset_radps, kwargs):
    rng = np.random.default_rng(seed)
    traj = simulate_trajectory(rng, N_STEPS, DT_S, ACC_STD_MPS2)
    speed = wheel_speed(rng, traj, SPEED_STD_MPS)
    z = gnss_fixes(rng, traj, GNSS_STD_M)
    z = drag_across_track(z, traj.t_s, T_ATTACK_S, rate_mps, traj.vel_mps)
    yaw = gyro_yaw_rate(rng, traj, GYRO_STD_RADPS, bias_radps=offset_radps)
    xs, _, nu_track = run_gnss_odo_gyro_filter(
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
    return xs, alarms.any(axis=1)


def main():
    t_s = np.arange(N_STEPS) * DT_S
    k0 = round(T_ATTACK_S / DT_S)
    k60 = round(60.0 / DT_S)
    settled = t_s >= SETTLE_S
    hours = N_SEEDS * settled.sum() / STEPS_PER_HOUR
    print(
        f"{'filter':<20}{'rate m/s':>9}{'FA/h':>8}{'found':>7}{'delay s':>9}"
        f"{'b err 60 s':>12}{'b err end':>11}  (deg/s)"
    )
    for name, offset, kwargs, rates in CASES:
        for rate in rates:
            n_alarm, delays, b_err = 0, [], []
            for seed in range(N_SEEDS):
                xs, alarm = run(seed, rate, offset, kwargs)
                if rate == 0.0:
                    n_alarm += alarm[settled].sum()
                    if xs.shape[1] == 5:
                        b_err.append(np.abs(xs[[k60, -1], 4] - offset))
                    continue
                after = np.flatnonzero(alarm[k0:])
                if len(after):
                    delays.append(after[0] * DT_S)
            if rate == 0.0:
                b = ""
                if b_err:
                    e60, e_end = np.rad2deg(np.mean(b_err, axis=0))
                    b = f"{e60:12.3f}{e_end:11.4f}"
                print(f"{name:<20}{'none':>9}{n_alarm / hours:8.1f}{'':16}{b}")
            else:
                d = f"{np.median(delays):9.1f}" if delays else f"{'-':>9}"
                print(f"{name:<20}{rate:9.2f}{'':8}{len(delays) / N_SEEDS:7.2f}{d}")


if __name__ == "__main__":
    main()
