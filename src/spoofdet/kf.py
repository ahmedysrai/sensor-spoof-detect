"""Kalman filters with GNSS position updates, ENU frame.

The constant-velocity filters use x = [east_m, north_m, ve_mps, vn_mps].
The gyro filter uses a unicycle state x = [east_m, north_m, psi_rad,
speed_mps], with heading psi counterclockwise from east, and optionally
the gyro zero-rate offset b_radps as a fifth entry. The usual tracking
notation: F, Q for the model, H, R for the measurement, nu = z - h(x_pred)
and S = H P_pred H^T + R for the innovation.
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


def unicycle_predict(
    x, P, yaw_rate_radps, dt_s, accel_std_mps2, gyro_std_radps, bias_rw_radps=0.0
):
    """Propagate the unicycle state over one step with the gyro as input.

    The heading turns by the measured yaw rate, so gyro noise is process
    noise on psi; white along-track acceleration is process noise on the
    speed. The vehicle moves along its heading (no sideslip), which is what
    lets the gyro hold the direction of travel against the GNSS fixes.

    With a fifth state, the gyro zero-rate offset b_radps, the heading turns
    by (rate - b) instead, and b is a random walk of `bias_rw_radps` per
    sqrt(s).
    """
    e, n, psi, s = x[:4]
    c, sn = np.cos(psi), np.sin(psi)
    m = len(x)
    rate = yaw_rate_radps - (x[4] if m == 5 else 0.0)
    x = x.copy()
    x[:4] = [e + s * dt_s * c, n + s * dt_s * sn, psi + rate * dt_s, s]
    F = np.eye(m)
    F[0, 2], F[0, 3] = -s * dt_s * sn, dt_s * c
    F[1, 2], F[1, 3] = s * dt_s * c, dt_s * sn
    # noise inputs [along-track accel, gyro noise]
    G = np.zeros((m, 2))
    G[:4] = [
        [0.5 * dt_s**2 * c, 0.0],
        [0.5 * dt_s**2 * sn, 0.0],
        [0.0, dt_s],
        [dt_s, 0.0],
    ]
    Q = G @ np.diag([accel_std_mps2**2, gyro_std_radps**2]) @ G.T
    if m == 5:
        F[2, 4] = -dt_s
        Q[4, 4] = bias_rw_radps**2 * dt_s
    return x, F @ P @ F.T + Q


def fit_start_heading(fixes_m, speed_mps, yaw_rate_radps, dt_s, gnss_std_m):
    """Position and heading at the last of `fixes_m`, from dead reckoning.

    Wheel speed and gyro give the path over the window up to a rotation by
    the unknown final heading psi. With c = cos(psi), s = sin(psi) each fix
    is linear in (east, north, c, s), so one least-squares solve gives them,
    and turns inside the window cost nothing, which a straight-line fit to
    the fixes would not allow. Returns pos_m (2,), psi_rad, its variance
    and |(c, s)|, which should be close to 1 if the wheel speed scale is
    right.
    """
    n = len(fixes_m)
    # heading of each step relative to the final heading, and the
    # dead-reckoned position of each fix relative to the final fix
    rel_psi = -np.r_[np.cumsum((yaw_rate_radps[1:] * dt_s)[::-1])[::-1], 0.0]
    step = (
        speed_mps[:-1, None]
        * dt_s
        * np.column_stack([np.cos(rel_psi[:-1]), np.sin(rel_psi[:-1])])
    )
    d = -np.r_[np.cumsum(step[::-1], axis=0)[::-1], np.zeros((1, 2))]
    A = np.zeros((2 * n, 4))
    A[0::2] = np.column_stack([np.ones(n), np.zeros(n), d[:, 0], -d[:, 1]])
    A[1::2] = np.column_stack([np.zeros(n), np.ones(n), d[:, 1], d[:, 0]])
    coef, *_ = np.linalg.lstsq(A, fixes_m.reshape(-1), rcond=None)
    cov = gnss_std_m**2 * np.linalg.inv(A.T @ A)
    c, sn = coef[2:]
    scale = np.hypot(c, sn)
    # heading variance: (c, s) covariance projected across (c, s)
    u = np.array([-sn, c]) / scale
    psi_var = u @ cov[2:, 2:] @ u / scale**2
    return coef[:2], float(np.arctan2(sn, c)), float(psi_var), float(scale)


def run_gnss_odo_gyro_filter(
    fixes_m,
    speed_mps,
    yaw_rate_radps,
    dt_s,
    accel_std_mps2,
    gnss_std_m,
    speed_std_mps,
    gyro_std_radps,
    n_init=50,
    max_heading_std_rad=0.2,
    bias_std_radps=None,
    bias_rw_radps=0.0,
):
    """GNSS, wheel speed and gyro yaw rate in a unicycle EKF.

    z = [east_m, north_m, speed_mps] is linear in this state, so only the
    prediction is linearised. The heading has to come from somewhere at the
    start: fit_start_heading over the first `n_init` samples gives position
    and heading at sample n_init - 1, where the filter starts. NIS is NaN up
    to there and 3-dof after it. Once started, the heading is carried by the
    gyro, so unlike run_gnss_odo_filter this needs no heading check. Raises
    ValueError if the start heading sd is over `max_heading_std_rad`, which
    happens when the vehicle barely moves in the start window.

    Returns the state track, the NIS per step and the innovations in the
    track frame, (n, 3): position along and across the predicted heading
    (across positive to the left) and speed, each divided by its standard
    deviation from S, so they are N(0, 1) when the filter matches the data.
    They are for the CUSUM in spoofdet.cusum, which needs a signed value per
    direction. The correlation between the three is ignored there.

    With `bias_std_radps` set, the state gets a fifth entry, the gyro
    zero-rate offset b_radps, with a zero-mean prior of that sd and a random
    walk of `bias_rw_radps` per sqrt(s) (see unicycle_predict). The start
    fit dead-reckons with the raw rates, so an offset b turns the fitted path
    and puts the start heading off by about b * T / 2 for a window of T
    seconds; the start covariance carries that term and its correlation
    with b.
    """
    n = len(fixes_m)
    k0 = n_init - 1
    pos0, psi0, psi_var, _ = fit_start_heading(
        fixes_m[:n_init], speed_mps[:n_init], yaw_rate_radps[:n_init], dt_s, gnss_std_m
    )
    if psi_var > max_heading_std_rad**2:
        raise ValueError(
            f"start heading sd is {np.sqrt(psi_var):.2f} rad after {n_init} "
            "samples; the vehicle has not moved far enough to fit a heading"
        )
    x = np.array([pos0[0], pos0[1], psi0, speed_mps[k0]])
    # position variance: about the fix noise averaged over the window. The
    # exact value matters little, the first GNSS updates settle it.
    P = np.diag([gnss_std_m**2, gnss_std_m**2, psi_var, speed_std_mps**2])
    if bias_std_radps is not None:
        half_window_s = 0.5 * k0 * dt_s
        x = np.r_[x, 0.0]
        P = np.pad(P, (0, 1))
        P[4, 4] = bias_std_radps**2
        P[2, 2] += (bias_std_radps * half_window_s) ** 2
        # heading error +b * T / 2 against offset error -b
        P[2, 4] = P[4, 2] = -(bias_std_radps**2) * half_window_s

    m = len(x)
    H = np.zeros((3, m))
    H[0, 0] = H[1, 1] = H[2, 3] = 1.0
    R = np.diag([gnss_std_m**2, gnss_std_m**2, speed_std_mps**2])

    xs = np.full((n, m), np.nan)
    nis_vals = np.full(n, np.nan)
    nu_track = np.full((n, 3), np.nan)
    xs[k0] = x
    for k in range(k0 + 1, n):
        x, P = unicycle_predict(
            x,
            P,
            yaw_rate_radps[k],
            dt_s,
            accel_std_mps2,
            gyro_std_radps,
            bias_rw_radps,
        )
        nu = np.r_[fixes_m[k], speed_mps[k]] - H @ x
        c, sn = np.cos(x[2]), np.sin(x[2])
        T = np.array([[c, sn, 0.0], [-sn, c, 0.0], [0.0, 0.0, 1.0]])
        S_track = T @ (H @ P @ H.T + R) @ T.T
        nu_track[k] = T @ nu / np.sqrt(np.diag(S_track))
        x, P, nis_vals[k] = _update(x, P, nu, H, R)
        xs[k] = x
    return xs, nis_vals, nu_track
