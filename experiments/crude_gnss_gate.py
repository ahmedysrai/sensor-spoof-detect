"""Crude end-to-end run: GNSS-only filter, per-step NIS gate, two kinds of attack.

Simulates 120 s of driving at 10 Hz and spoofs the fixes from t = 60 s.
Over 200 seeds it prints the fraction of steps whose NIS is over the gate:
before onset (the false alarm rate), in the first 2 s after onset, and from
2 s after onset to the end. A first-alarm delay would be misleading here:
at 1 % per step a false alarm turns up within a few seconds anyway.

    python experiments/crude_gnss_gate.py
"""

import numpy as np

from spoofdet.attacks import drag_off, jump
from spoofdet.kf import run_gnss_filter
from spoofdet.nis import nis_threshold
from spoofdet.sim import gnss_fixes, simulate_trajectory

DT_S = 0.1
N_STEPS = 1200
ACC_STD_MPS2 = 0.2
GNSS_STD_M = 1.5
T_ATTACK_S = 60.0
P_FA = 0.01
N_SEEDS = 200
SETTLE_S = 5.0  # filter start-up transient, excluded from the false alarm count

ATTACKS = {
    "none": lambda z, t: z,
    "jump 15 m": lambda z, t: jump(z, t, T_ATTACK_S, (15.0, 0.0)),
    "jump 5 m": lambda z, t: jump(z, t, T_ATTACK_S, (5.0, 0.0)),
    "drag 1 m/s": lambda z, t: drag_off(z, t, T_ATTACK_S, 1.0),
    "drag 0.2 m/s": lambda z, t: drag_off(z, t, T_ATTACK_S, 0.2),
}


def main():
    gate = nis_threshold(2, P_FA)
    print(f"gate {gate:.2f} (chi-square, 2 dof, P_fa {P_FA}), {N_SEEDS} seeds")
    print(f"{'attack':<14}{'before':>8}{'0-2 s':>8}{'2-60 s':>8}{'offset_m':>10}")
    for name, attack in ATTACKS.items():
        hits = np.zeros(3)
        counts = np.zeros(3)
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            traj = simulate_trajectory(rng, N_STEPS, DT_S, ACC_STD_MPS2)
            z = attack(gnss_fixes(rng, traj, GNSS_STD_M), traj.t_s)
            _, nis_vals = run_gnss_filter(z, DT_S, ACC_STD_MPS2, GNSS_STD_M)
            alarm = nis_vals > gate
            t = traj.t_s - T_ATTACK_S
            windows = [
                (t >= SETTLE_S - T_ATTACK_S) & (t < 0),
                (t >= 0) & (t < 2.0),
                t >= 2.0,
            ]
            for i, w in enumerate(windows):
                hits[i] += alarm[w].sum()
                counts[i] += w.sum()
        rates = hits / counts
        end_offset_m = attack(np.zeros((N_STEPS, 2)), traj.t_s)[-1, 0]
        print(
            f"{name:<14}"
            + "".join(f"{r:8.3f}" for r in rates)
            + f"{end_offset_m:10.1f}"
        )


if __name__ == "__main__":
    main()
