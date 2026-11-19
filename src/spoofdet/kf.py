"""Constant-velocity Kalman filters with GNSS position updates, ENU frame.

State x = [east_m, north_m, ve_mps, vn_mps]. The usual tracking notation:
F, Q for the model, H, R for the measurement, nu = z - h(x_pred) and
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


def _update(x, P, nu, H, R):
    """Kalman update for innovation nu; returns x, P and the NIS of nu."""
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T  # P H^T S^-1, S symmetric
    x = x + K @ nu
    # Joseph form keeps P symmetric and positive definite
    A = np.eye(len(x)) - K @ H
    P = A @ P @ A.T + K @ R @ K.T
    return x, P, nis(nu, S)


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
        x, P, nis_vals[k] = _update(x, P, fixes_m[k] - H @ x, H, R)
        xs[k] = x
    return xs, nis_vals


def run_gnss_odo_filter(
    fixes_m,
    speed_mps,
    dt_s,
    accel_std_mps2,
    gnss_std_m,
    speed_std_mps,
    max_heading_std_rad=0.2,
):
    """GNSS filter with a wheel odometry speed measurement at every step.

    Wheel odometry gives the speed |v| but not its direction, so the
    measurement is z = [east_m, north_m, speed_mps] with h(x) = [e, n, |v|].
    The speed row is nonlinear and is linearised at the prediction (an EKF
    update), with d|v|/dv = v / |v|. That linearisation is only good when
    the direction of v is already known: the speed row is used once the
    velocity standard deviation (largest axis of P_vv) is under
    `max_heading_std_rad` times the predicted speed, which is the small-angle
    heading uncertainty. Before that (start-up, or a vehicle close to
    standing still) the step updates on GNSS alone and its NIS is NaN.
    Otherwise the NIS is the joint 3-dof value.
    """
    F, Q = cv_model(dt_s, accel_std_mps2)
    H_gnss = np.hstack([np.eye(2), np.zeros((2, 2))])
    R_gnss = gnss_std_m**2 * np.eye(2)
    R = np.diag([gnss_std_m**2, gnss_std_m**2, speed_std_mps**2])

    n = len(fixes_m)
    x = np.r_[fixes_m[0], 0.0, 0.0]
    P = np.diag([gnss_std_m**2, gnss_std_m**2, 25.0, 25.0])
    xs = np.zeros((n, 4))
    nis_vals = np.full(n, np.nan)
    xs[0] = x
    for k in range(1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        speed_pred = np.hypot(x[2], x[3])
        vel_std_mps = np.sqrt(np.linalg.eigvalsh(P[2:, 2:])[-1])
        if vel_std_mps > max_heading_std_rad * speed_pred:
            x, P, _ = _update(x, P, fixes_m[k] - H_gnss @ x, H_gnss, R_gnss)
        else:
            H = np.vstack([H_gnss, np.r_[0.0, 0.0, x[2:] / speed_pred]])
            nu = np.r_[fixes_m[k] - x[:2], speed_mps[k] - speed_pred]
            x, P, nis_vals[k] = _update(x, P, nu, H, R)
        xs[k] = x
    return xs, nis_vals
