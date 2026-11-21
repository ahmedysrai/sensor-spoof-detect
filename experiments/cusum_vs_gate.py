"""CUSUM on track-frame innovations against a NIS gate, same false alarm rate.

Same data as odo_cross_check.py (120 s at 10 Hz, attack from t = 60 s, 200
seeds, the same draws seed for seed), through the GNSS, wheel speed and gyro
filter only. Both detectors are set for about one false alarm per hour:

- the NIS gate at a per-step P_fa of 1 / 36000 (one step in an hour at
  10 Hz), chi-square with 3 dof
- a two-sided CUSUM on each of the along, across and speed innovations,
  drift 0.5 (tuned for a one standard deviation lean), threshold 10.5,
  which gave 0.83 alarms per hour over six statistics on 111 h of white
  N(0, 1) samples (NOTES.md, 2026-11-21)

False alarms per hour come from the attack-free runs (5 s to 120 s, 6.4 h
in all). For each attack: the fraction of seeds with an alarm within 60 s
of onset and the median delay to the first alarm at or after onset, over
the seeds that alarmed.

    python experiments/cusum_vs_gate.py
"""

import numpy as np

from spoofdet.attacks import drag_off, jump
from spoofdet.cusum import cusum
from spoofdet.kf import run_gnss_odo_gyro_filter
from spoofdet.nis import nis_threshold
from spoofdet.sim import gnss_fixes, gyro_yaw_rate, simulate_trajectory, wheel_speed

DT_S = 0.1
N_STEPS = 1200
ACC_STD_MPS2 = 0.2
GNSS_STD_M = 1.5
SPEED_STD_MPS = 0.05
GYRO_STD_RADPS = 5.5e-4  # BMI088, see sim.gyro_yaw_rate
T_ATTACK_S = 60.0
N_SEEDS = 200
SETTLE_S = 5.0
STEPS_PER_HOUR = 3600 / DT_S
CUSUM_DRIFT = 0.5
CUSUM_THRESHOLD = 10.5


def along(v):
    return v


def across(v):
    return np.array([-v[1], v[0]])


ATTACKS = {
    "none": lambda z, t, v: z,
    "jump 5 m": lambda z, t, v: jump(z, t, T_ATTACK_S, (5.0, 0.0)),
    "drag 1 m/s along": lambda z, t, v: drag_off(z, t, T_ATTACK_S, 1.0, along(v)),
    "drag 0.2 m/s along": lambda z, t, v: drag_off(z, t, T_ATTACK_S, 0.2, along(v)),
    "drag 1 m/s across": lambda z, t, v: drag_off(z, t, T_ATTACK_S, 1.0, across(v)),
    "drag 0.2 m/s across": lambda z, t, v: drag_off(z, t, T_ATTACK_S, 0.2, across(v)),
}


def detectors(nis_vals, nu_track):
    """Alarm per step, (n,) bool, for each detector."""
    gate = nis_threshold(3, 1.0 / STEPS_PER_HOUR)
    alarms, _ = cusum(nu_track, CUSUM_DRIFT, CUSUM_THRESHOLD)
    return {"nis gate": nis_vals > gate, "cusum": alarms.any(axis=1)}


def main():
    t_s = np.arange(N_STEPS) * DT_S
    k0 = round(T_ATTACK_S / DT_S)
    settled = t_s >= SETTLE_S
    print(f"{'attack':<22}{'detector':<10}{'FA/h':>6}{'found':>7}{'delay s':>9}")
    for name, attack in ATTACKS.items():
        alarm_steps = {"nis gate": 0, "cusum": 0}
        delays = {"nis gate": [], "cusum": []}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            traj = simulate_trajectory(rng, N_STEPS, DT_S, ACC_STD_MPS2)
            speed = wheel_speed(rng, traj, SPEED_STD_MPS)
            z = attack(gnss_fixes(rng, traj, GNSS_STD_M), traj.t_s, traj.vel_mps[k0])
            rate = gyro_yaw_rate(rng, traj, GYRO_STD_RADPS)
            _, nis_vals, nu_track = run_gnss_odo_gyro_filter(
                z,
                speed,
                rate,
                DT_S,
                ACC_STD_MPS2,
                GNSS_STD_M,
                SPEED_STD_MPS,
                GYRO_STD_RADPS,
            )
            for det, alarm in detectors(nis_vals, nu_track).items():
                alarm_steps[det] += alarm[settled].sum()
                after = np.flatnonzero(alarm[k0:])
                if len(after):
                    delays[det].append(after[0] * DT_S)
        hours = N_SEEDS * settled.sum() / STEPS_PER_HOUR
        for det, d in delays.items():
            if name == "none":
                print(f"{name:<22}{det:<10}{alarm_steps[det] / hours:6.1f}")
                continue
            med = f"{np.median(d):9.1f}" if d else f"{'-':>9}"
            print(f"{name:<22}{det:<10}{'':6}{len(d) / N_SEEDS:7.2f}{med}")


if __name__ == "__main__":
    main()
