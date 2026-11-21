"""Do wheel speed and a gyro expose the GNSS drag-off?

Same setup as crude_gnss_gate.py (120 s at 10 Hz, attack from t = 60 s,
per-step NIS gate at P_fa 0.01, 200 seeds), run through the GNSS-only
filter, the GNSS plus wheel speed filter and the unicycle filter that adds
the gyro yaw rate, all on the same data. Drag-offs are aimed along the true
direction of travel at onset, or across it, because wheel odometry measures
speed and not heading. Columns are the fraction of steps over the gate
before onset, in the first 2 s, from 2 s to the end, and in the last 10 s.

    python experiments/odo_cross_check.py
"""

import numpy as np

from spoofdet.attacks import drag_off, jump
from spoofdet.kf import run_gnss_filter, run_gnss_odo_filter, run_gnss_odo_gyro_filter
from spoofdet.nis import nis_threshold
from spoofdet.sim import gnss_fixes, gyro_yaw_rate, simulate_trajectory, wheel_speed

DT_S = 0.1
N_STEPS = 1200
ACC_STD_MPS2 = 0.2
GNSS_STD_M = 1.5
SPEED_STD_MPS = 0.05
GYRO_STD_RADPS = 5.5e-4  # BMI088, see sim.gyro_yaw_rate
T_ATTACK_S = 60.0
P_FA = 0.01
N_SEEDS = 200
SETTLE_S = 5.0


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


def gate_rates(nis_vals, t_s, gate):
    alarm = nis_vals > gate  # NaN steps (odometry skipped) count as no alarm
    t = t_s - T_ATTACK_S
    windows = [
        (t >= SETTLE_S - T_ATTACK_S) & (t < 0),
        (t >= 0) & (t < 2.0),
        t >= 2.0,
        t >= 50.0,
    ]
    hits = np.array([alarm[w].sum() for w in windows])
    counts = np.array([w.sum() for w in windows])
    return hits, counts


def main():
    gates = {
        "gnss": nis_threshold(2, P_FA),
        "gnss+odo": nis_threshold(3, P_FA),
        "+gyro": nis_threshold(3, P_FA),
    }
    print(f"gates {gates['gnss']:.2f} (2 dof), {gates['gnss+odo']:.2f} (3 dof)")
    print(
        f"{'attack':<22}{'filter':<10}{'before':>8}{'0-2 s':>8}{'2-60 s':>8}{'50-60 s':>9}"
    )
    k0 = round(T_ATTACK_S / DT_S)
    for name, attack in ATTACKS.items():
        hits = {f: np.zeros(4) for f in gates}
        counts = {f: np.zeros(4) for f in gates}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            traj = simulate_trajectory(rng, N_STEPS, DT_S, ACC_STD_MPS2)
            speed = wheel_speed(rng, traj, SPEED_STD_MPS)
            z = attack(gnss_fixes(rng, traj, GNSS_STD_M), traj.t_s, traj.vel_mps[k0])
            # drawn last, so the other sensors match the earlier runs seed for seed
            rate = gyro_yaw_rate(rng, traj, GYRO_STD_RADPS)
            _, nis_g = run_gnss_filter(z, DT_S, ACC_STD_MPS2, GNSS_STD_M)
            _, nis_o = run_gnss_odo_filter(
                z, speed, DT_S, ACC_STD_MPS2, GNSS_STD_M, SPEED_STD_MPS
            )
            _, nis_y, _ = run_gnss_odo_gyro_filter(
                z,
                speed,
                rate,
                DT_S,
                ACC_STD_MPS2,
                GNSS_STD_M,
                SPEED_STD_MPS,
                GYRO_STD_RADPS,
            )
            for f, vals in (("gnss", nis_g), ("gnss+odo", nis_o), ("+gyro", nis_y)):
                h, c = gate_rates(vals, traj.t_s, gates[f])
                hits[f] += h
                counts[f] += c
        for f in gates:
            rates = hits[f] / counts[f]
            print(
                f"{name:<22}{f:<10}"
                + "".join(f"{r:8.3f}" for r in rates[:3])
                + f"{rates[3]:9.3f}"
            )


if __name__ == "__main__":
    main()
