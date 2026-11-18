"""Constant-velocity Kalman filter with GNSS position updates, ENU frame.

State x = [east_m, north_m, ve_mps, vn_mps]. The usual tracking notation:
F, Q for the model, H, R for the measurement, nu = z - H x_pred and
S = H P_pred H^T + R for the innovation.
"""

import numpy as np

from spoofdet.nis import nis


def cv_model(dt_s, accel_std_mps2):
    """F and Q for white acceleration held constant over each step."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt_s
    G = np.array([[0.5 * dt_s**2, 0.0], [0.0, 0.5 * dt_s**2], [dt_s, 0.0], [0.0, dt_s]])
    Q = accel_std_mps2**2 * G @ G.T
    return F, Q


def run_gnss_filter(fixes_m, dt_s, accel_std_mps2, gnss_std_m):
    """Filter the fixes and return the state track and NIS per step.

    The first fix initialises the position with zero velocity and a loose
    velocity variance, so the first NIS value is NaN (no prediction yet).
    """
    F, Q = cv_model(dt_s, accel_std_mps2)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = gnss_std_m**2 * np.eye(2)

    n = len(fixes_m)
    x = np.r_[fixes_m[0], 0.0, 0.0]
    P = np.diag([gnss_std_m**2, gnss_std_m**2, 25.0, 25.0])
    xs = np.zeros((n, 4))
    nis_vals = np.full(n, np.nan)
    xs[0] = x
    for k in range(1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        nu = fixes_m[k] - H @ x
        S = H @ P @ H.T + R
        nis_vals[k] = nis(nu, S)
        K = np.linalg.solve(S, H @ P).T  # P H^T S^-1, S symmetric
        x = x + K @ nu
        # Joseph form keeps P symmetric and positive definite
        A = np.eye(4) - K @ H
        P = A @ P @ A.T + K @ R @ K.T
        xs[k] = x
    return xs, nis_vals
