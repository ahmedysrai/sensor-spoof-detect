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


def _check_lengths(source: str, **arrays) -> int:
    """Return the number of samples, refusing arrays that disagree on it."""
    lengths = {name: len(value) for name, value in arrays.items() if value is not None}
    n = lengths.pop("fixes_m", None)
    if n is None:
        raise ValueError("fixes_m is required")
    wrong = {name: length for name, length in lengths.items() if length != n}
    if wrong:
        detail = ", ".join(f"{name} has {length}" for name, length in wrong.items())
        raise ValueError(
            f"{source}: {n} GNSS fixes but {detail}; the filters need one value per sample"
        )
    return n


def _non_negative(name: str, value, source: str) -> float:
    """Process-noise sds may be zero (a noise-free run), measurement ones not."""
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{source}: {name} must be zero or positive, got {value}")
    return value


def _positive(name: str, value, source: str) -> float:
    if value is None:
        raise ValueError(f"{source}: {name} is None; the filters need a positive sd")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{source}: {name} must be positive, got {value}")
    return value


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
    velocity variance, so the first NIS value is NaN (no prediction yet). A
    step whose fix is missing (NaN, a dropout) is predicted but not updated,
    and its NIS is NaN; one dropped sample therefore does not poison the
    state, which is what a log from a real receiver looks like.
    """
    fixes_m = np.asarray(fixes_m, dtype=float).reshape(-1, 2)
    accel_std_mps2 = _non_negative("accel_std_mps2", accel_std_mps2, "run_gnss_filter")
    gnss_std_m = _positive("gnss_std_m", gnss_std_m, "run_gnss_filter")
    n = len(fixes_m)
    if n < 2:
        raise ValueError(f"run_gnss_filter: {n} fixes, need at least two")
    if not np.isfinite(fixes_m[0]).all():
        raise ValueError("run_gnss_filter: the first fix is missing, nothing to start from")

    F, Q = cv_model(dt_s, accel_std_mps2)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = gnss_std_m**2 * np.eye(2)

    x = np.r_[fixes_m[0], 0.0, 0.0]
    P = np.diag([gnss_std_m**2, gnss_std_m**2, 25.0, 25.0])
    xs = np.zeros((n, 4))
    nis_vals = np.full(n, np.nan)
    xs[0] = x
    for k in range(1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        if np.isfinite(fixes_m[k]).all():
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

    A missing fix or a missing speed drops that measurement row for the step
    and the update runs on what is left, so the NIS dof can be lower on a
    step with a dropout. A step with nothing left is predicted only.
    """
    fixes_m = np.asarray(fixes_m, dtype=float).reshape(-1, 2)
    speed_mps = np.asarray(speed_mps, dtype=float).reshape(-1)
    n = _check_lengths("run_gnss_odo_filter", fixes_m=fixes_m, speed_mps=speed_mps)
    if n < 2:
        raise ValueError(f"run_gnss_odo_filter: {n} samples, need at least two")
    accel_std_mps2 = _non_negative("accel_std_mps2", accel_std_mps2, "run_gnss_odo_filter")
    gnss_std_m = _positive("gnss_std_m", gnss_std_m, "run_gnss_odo_filter")
    speed_std_mps = _positive("speed_std_mps", speed_std_mps, "run_gnss_odo_filter")
    if not np.isfinite(fixes_m[0]).all():
        raise ValueError(
            "run_gnss_odo_filter: the first fix is missing, nothing to start from"
        )

    F, Q = cv_model(dt_s, accel_std_mps2)
    H_gnss = np.hstack([np.eye(2), np.zeros((2, 2))])

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
        rows, values, variances = [], [], []
        if np.isfinite(fixes_m[k]).all():
            rows.extend([H_gnss[0], H_gnss[1]])
            values.extend(fixes_m[k])
            variances.extend([gnss_std_m**2] * 2)
        used_speed = bool(
            np.isfinite(speed_mps[k])
            and speed_pred > 0.0
            and vel_std_mps <= max_heading_std_rad * speed_pred
        )
        if used_speed:
            rows.append(np.r_[0.0, 0.0, x[2:] / speed_pred])
            values.append(speed_mps[k])
            variances.append(speed_std_mps**2)
        if rows:
            H = np.asarray(rows)
            nu = np.asarray(values) - H @ x
            x, P, step_nis = _update(x, P, nu, H, np.diag(variances))
            if used_speed:
                nis_vals[k] = step_nis
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


def mag_heading(mag_ut):
    """Heading psi in rad from horizontal magnetometer readings, (n, 2) in uT.

    Columns forward and left, as from spoofdet.sim.magnetometer: the field
    points north, so a vehicle heading psi from east reads it at
    [sin(psi), cos(psi)]. Declination and tilt are taken as zero.
    """
    return np.arctan2(mag_ut[:, 0], mag_ut[:, 1])


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
    mag_psi_rad=None,
    mag_std_rad=None,
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

    With `mag_psi_rad` (n,) set, a magnetometer heading (see mag_heading) is
    a fourth measurement row with sd `mag_std_rad`, its innovation wrapped
    to +-pi. NIS is then 4-dof and the track-frame innovations get a fourth
    column, the heading innovation over its sd. Unlike the gyro this says
    where the heading is, not how it changes, so the GNSS fixes cannot turn
    it onto a spoofed track without a heading innovation to match.

    A measurement that is missing on a step (NaN, a dropout) drops its row
    for that step only: the update runs on the rows that are there, the NIS
    is over those rows, and the dropped column of the track-frame
    innovations stays NaN, which the CUSUM holds rather than resets. A step
    with no usable measurement is predicted only.
    """
    fixes_m = np.asarray(fixes_m, dtype=float).reshape(-1, 2)
    speed_mps = np.asarray(speed_mps, dtype=float).reshape(-1)
    yaw_rate_radps = np.asarray(yaw_rate_radps, dtype=float).reshape(-1)
    n = _check_lengths(
        "run_gnss_odo_gyro_filter",
        fixes_m=fixes_m,
        speed_mps=speed_mps,
        yaw_rate_radps=yaw_rate_radps,
    )
    accel_std_mps2 = _non_negative(
        "accel_std_mps2", accel_std_mps2, "run_gnss_odo_gyro_filter"
    )
    gnss_std_m = _positive("gnss_std_m", gnss_std_m, "run_gnss_odo_gyro_filter")
    speed_std_mps = _positive("speed_std_mps", speed_std_mps, "run_gnss_odo_gyro_filter")
    gyro_std_radps = _positive("gyro_std_radps", gyro_std_radps, "run_gnss_odo_gyro_filter")
    if not isinstance(n_init, int) or isinstance(n_init, bool):
        raise TypeError(f"run_gnss_odo_gyro_filter: n_init must be an int, got {n_init!r}")
    if n_init < 2:
        raise ValueError(f"run_gnss_odo_gyro_filter: n_init must be at least 2, got {n_init}")
    if n_init > n:
        raise ValueError(
            f"run_gnss_odo_gyro_filter: n_init is {n_init} but the run has {n} samples; "
            "the start pose is fitted to the first n_init samples"
        )
    if (mag_psi_rad is None) != (mag_std_rad is None):
        raise ValueError(
            "run_gnss_odo_gyro_filter: mag_psi_rad and mag_std_rad go together; "
            f"got mag_psi_rad={'set' if mag_psi_rad is not None else 'None'}, "
            f"mag_std_rad={'set' if mag_std_rad is not None else 'None'}"
        )
    if bias_std_radps is None and bias_rw_radps:
        raise ValueError(
            "run_gnss_odo_gyro_filter: bias_rw_radps only applies with bias_std_radps set"
        )

    k0 = n_init - 1
    if not np.isfinite(fixes_m[:n_init]).all():
        bad = int(np.flatnonzero(~np.isfinite(fixes_m[:n_init]).all(axis=1))[0])
        raise ValueError(
            f"run_gnss_odo_gyro_filter: the fix of sample {bad} is missing, and the start "
            f"pose is fitted to the first {n_init} samples"
        )
    if not np.isfinite(speed_mps[:n_init]).all():
        bad = int(np.flatnonzero(~np.isfinite(speed_mps[:n_init]))[0])
        raise ValueError(
            f"run_gnss_odo_gyro_filter: the wheel speed of sample {bad} is missing, and the "
            f"start pose is fitted to the first {n_init} samples"
        )
    if not np.isfinite(yaw_rate_radps[:n_init]).all():
        bad = int(np.flatnonzero(~np.isfinite(yaw_rate_radps[:n_init]))[0])
        raise ValueError(
            f"run_gnss_odo_gyro_filter: the yaw rate of sample {bad} is missing, and the "
            f"start pose is fitted to the first {n_init} samples"
        )
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
    r = [gnss_std_m**2, gnss_std_m**2, speed_std_mps**2]
    if mag_psi_rad is not None:
        r.append(mag_std_rad**2)
    H = np.zeros((len(r), m))
    H[0, 0] = H[1, 1] = H[2, 3] = 1.0
    if mag_psi_rad is not None:
        H[3, 2] = 1.0
    R = np.diag(r)

    xs = np.full((n, m), np.nan)
    nis_vals = np.full(n, np.nan)
    nu_track = np.full((n, len(r)), np.nan)
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
        z = np.r_[fixes_m[k], speed_mps[k]]
        if mag_psi_rad is not None:
            z = np.r_[z, mag_psi_rad[k]]
        nu = z - H @ x
        if mag_psi_rad is not None:
            nu[3] = np.angle(np.exp(1j * nu[3]))
        c, sn = np.cos(x[2]), np.sin(x[2])
        T = np.eye(len(r))
        T[:2, :2] = [[c, sn], [-sn, c]]
        S_track = T @ (H @ P @ H.T + R) @ T.T
        present = np.isfinite(nu)
        # a plain T @ nu would turn the whole row into NaN, because 0 * NaN is
        # NaN in the rows the rotation does not touch: project zeros and put
        # the gaps back afterwards
        track = T @ np.where(present, nu, 0.0) / np.sqrt(np.diag(S_track))
        track[~present] = np.nan
        if len(present) > 2 and not present[:2].all():
            track[:2] = np.nan  # along and across need both position rows
        nu_track[k] = track
        if present.all():
            x, P, nis_vals[k] = _update(x, P, nu, H, R)
        elif present.any():
            rows = np.flatnonzero(present)
            x, P, nis_vals[k] = _update(
                x, P, nu[rows], H[rows], R[np.ix_(rows, rows)]
            )
        xs[k] = x
    return xs, nis_vals, nu_track
