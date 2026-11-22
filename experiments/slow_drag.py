"""How slowly can a sideways GNSS drag go before the CUSUM stops seeing it?

The drag in cusum_vs_gate.py had a fixed direction, set across the track at
onset. Over a long run the vehicle turns and that drag ends up partly along
the track. Here the attacker knows the route and pulls to the left of the
true track at every step (attacks.drag_across_track), at several rates.

600 s at 10 Hz, attack from t = 60 s, 100 seeds. Filter, noise values and
detector settings are the ones in cusum_vs_gate.py: CUSUM drift 0.5,
threshold 10.5, and a NIS gate for the same false alarm budget. For each
rate: the fraction of seeds with an alarm before the end of the run, the
median delay and the median sideways offset at the first alarm (rate times
delay), and the mean across-track innovation over the attack (the lean the
CUSUM adds up), split into the first 60 s and the rest.

    python experiments/slow_drag.py
"""

import numpy as np

from spoofdet.attacks import drag_across_track
from spoofdet.cusum import cusum
from spoofdet.kf import run_gnss_odo_gyro_filter
from spoofdet.nis import nis_threshold
from spoofdet.sim import gnss_fixes, gyro_yaw_rate, simulate_trajectory, wheel_speed

DT_S = 0.1
N_STEPS = 6000
ACC_STD_MPS2 = 0.2
GNSS_STD_M = 1.5
SPEED_STD_MPS = 0.05
GYRO_STD_RADPS = 5.5e-4  # BMI088, see sim.gyro_yaw_rate
T_ATTACK_S = 60.0
N_SEEDS = 100
SETTLE_S = 5.0
STEPS_PER_HOUR = 3600 / DT_S
CUSUM_DRIFT = 0.5
CUSUM_THRESHOLD = 10.5
RATES_MPS = [0.0, 0.02, 0.05, 0.1, 0.2]
EARLY_S = 60.0


def main():
    t_s = np.arange(N_STEPS) * DT_S
    k0 = round(T_ATTACK_S / DT_S)
    k_early = k0 + round(EARLY_S / DT_S)
    settled = t_s >= SETTLE_S
    gate = nis_threshold(3, 1.0 / STEPS_PER_HOUR)
    print(
        f"{'rate m/s':>8}  {'detector':<9}{'FA/h':>6}{'found':>7}{'delay s':>9}"
        f"{'offset m':>10}{'lean 0-60 s':>13}{'lean later':>12}"
    )
    for rate in RATES_MPS:
        alarm_steps = {"nis gate": 0, "cusum": 0}
        delays = {"nis gate": [], "cusum": []}
        lean_early, lean_late = [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            traj = simulate_trajectory(rng, N_STEPS, DT_S, ACC_STD_MPS2)
            speed = wheel_speed(rng, traj, SPEED_STD_MPS)
            z = gnss_fixes(rng, traj, GNSS_STD_M)
            z = drag_across_track(z, traj.t_s, T_ATTACK_S, rate, traj.vel_mps)
            yaw = gyro_yaw_rate(rng, traj, GYRO_STD_RADPS)
            _, nis_vals, nu_track = run_gnss_odo_gyro_filter(
                z,
                speed,
                yaw,
                DT_S,
                ACC_STD_MPS2,
                GNSS_STD_M,
                SPEED_STD_MPS,
                GYRO_STD_RADPS,
            )
            lean_early.append(nu_track[k0:k_early, 1].mean())
            lean_late.append(nu_track[k_early:, 1].mean())
            cusum_alarms, _ = cusum(nu_track, CUSUM_DRIFT, CUSUM_THRESHOLD)
            alarms = {"nis gate": nis_vals > gate, "cusum": cusum_alarms.any(axis=1)}
            for det, alarm in alarms.items():
                alarm_steps[det] += alarm[settled].sum()
                after = np.flatnonzero(alarm[k0:])
                if len(after):
                    delays[det].append(after[0] * DT_S)
        hours = N_SEEDS * settled.sum() / STEPS_PER_HOUR
        lean = f"{np.mean(lean_early):13.2f}{np.mean(lean_late):12.2f}"
        for det, d in delays.items():
            if rate == 0.0:
                fa = alarm_steps[det] / hours
                print(f"{'none':>8}  {det:<9}{fa:6.1f}{'':26}{lean}")
                continue
            if d:
                cols = f"{np.median(d):9.1f}{rate * np.median(d):10.1f}"
            else:
                cols = f"{'-':>9}{'-':>10}"
            print(f"{rate:8.2f}  {det:<9}{'':6}{len(d) / N_SEEDS:7.2f}{cols}{lean}")


if __name__ == "__main__":
    main()
