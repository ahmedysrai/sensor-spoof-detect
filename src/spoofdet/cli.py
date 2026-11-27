"""Command line: run the gyro filter and CUSUM on a CSV sensor log.

    spoofdet detect log.csv [--out steps.csv]
    spoofdet simulate log.csv [--attack drag-across --rate 0.1]

A log has a header row and one row per sample at a fixed rate:

    t_s, east_m, north_m, speed_mps, yaw_rate_radps[, mag_fwd_ut, mag_left_ut]

Positions are GNSS fixes in a local ENU frame, speed is wheel odometry,
yaw rate is the gyro z axis (counterclockwise positive), and the optional
magnetometer columns are the horizontal body-frame reading, forward and
left. Every row needs every column; a sensor at a lower rate has to be
held or interpolated onto the common rate before this sees it.
"""

import argparse
import sys

import numpy as np

from spoofdet.attacks import drag_across_track, drag_off, jump
from spoofdet.cusum import cusum
from spoofdet.kf import mag_heading, run_gnss_odo_gyro_filter
from spoofdet.sim import (
    gnss_fixes,
    gyro_yaw_rate,
    magnetometer,
    simulate_trajectory,
    wheel_speed,
)

REQUIRED = ["t_s", "east_m", "north_m", "speed_mps", "yaw_rate_radps"]
MAG = ["mag_fwd_ut", "mag_left_ut"]
INNOVATION_NAMES = ["along", "across", "speed", "heading"]


def read_log(path):
    """Columns of a CSV log as a dict of float arrays, checked for use.

    Raises ValueError for a missing column, a non-numeric or empty cell, or
    sample times that are not evenly spaced (within 1 % of the median step).
    """
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(",")]
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
    has_mag = [c in header for c in MAG]
    if any(has_mag) and not all(has_mag):
        raise ValueError(f"{path}: need both {MAG[0]} and {MAG[1]}, or neither")
    # genfromtxt reads empty and non-numeric cells as NaN, caught below
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
    except ValueError as e:
        # genfromtxt lists every bad row; the first one is enough
        lines = [ln.strip() for ln in str(e).splitlines()]
        raise ValueError(f"{path}: {lines[1] if len(lines) > 1 else lines[0]}")
    if data.shape[1] != len(header):
        raise ValueError(
            f"{path}: {data.shape[1]} values per row, header has {len(header)}"
        )
    bad = np.flatnonzero(~np.isfinite(data).all(axis=1))
    if len(bad):
        # +2: one for the header, one for counting rows from 1
        raise ValueError(f"{path}: empty or non-finite value on line {bad[0] + 2}")
    log = {name: data[:, i] for i, name in enumerate(header)}
    steps = np.diff(log["t_s"])
    if len(steps) == 0:
        raise ValueError(f"{path}: need more than one sample")
    dt_s = float(np.median(steps))
    worst = np.argmax(np.abs(steps - dt_s))
    if dt_s <= 0 or abs(steps[worst] - dt_s) > 0.01 * dt_s:
        raise ValueError(
            f"{path}: sample step {steps[worst]:.4g} s at t = "
            f"{log['t_s'][worst]:.4g} s, median step {dt_s:.4g} s; the filter "
            "needs a fixed rate"
        )
    return log


def alarm_intervals(t_s, alarms, merge_gap_s):
    """Group alarm steps into (first_s, last_s, n_alarms) intervals.

    `alarms` is a bool array per step. Alarms less than `merge_gap_s` apart
    go in the same interval: a CUSUM restarts after each alarm, so a lasting
    attack gives a train of alarms, not one long run of True.
    """
    k = np.flatnonzero(alarms)
    if len(k) == 0:
        return []
    breaks = np.flatnonzero(np.diff(t_s[k]) >= merge_gap_s)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(k) - 1]
    return [(t_s[k[a]], t_s[k[b]], b - a + 1) for a, b in zip(starts, ends)]


def detect(log, args):
    """Filter the log; returns the NIS, track-frame innovations and alarms."""
    t_s = log["t_s"]
    dt_s = float(np.median(np.diff(t_s)))
    n_init = max(2, round(args.start_s / dt_s))
    if len(t_s) <= n_init:
        raise ValueError(
            f"{len(t_s)} samples, the {args.start_s:g} s start-up window alone "
            f"needs {n_init}"
        )
    kwargs = {}
    if args.gyro_offset_std_dps > 0:
        kwargs["bias_std_radps"] = np.deg2rad(args.gyro_offset_std_dps)
        kwargs["bias_rw_radps"] = args.gyro_offset_walk
    if MAG[0] in log:
        mag_ut = np.column_stack([log[MAG[0]], log[MAG[1]]])
        kwargs["mag_psi_rad"] = mag_heading(mag_ut)
        kwargs["mag_std_rad"] = args.mag_noise_ut / args.field_ut
        if args.hard_iron_std_ut > 0:
            kwargs["hard_iron_std_ut"] = args.hard_iron_std_ut
            kwargs["field_ut"] = args.field_ut
    _, nis_vals, nu_track = run_gnss_odo_gyro_filter(
        np.column_stack([log["east_m"], log["north_m"]]),
        log["speed_mps"],
        log["yaw_rate_radps"],
        dt_s,
        args.accel_std,
        args.gnss_std,
        args.speed_std,
        args.gyro_std,
        n_init=n_init,
        **kwargs,
    )
    alarms, _ = cusum(nu_track, args.drift, args.threshold)
    return nis_vals, nu_track, alarms


def cmd_detect(args):
    try:
        log = read_log(args.log)
        nis_vals, nu_track, alarms = detect(log, args)
    except (OSError, ValueError) as e:
        print(f"spoofdet: {e}", file=sys.stderr)
        return 2
    t_s = log["t_s"]
    names = INNOVATION_NAMES[: nu_track.shape[1]]
    if args.out:
        cols = np.column_stack([t_s, nis_vals, nu_track, alarms.any(axis=1)])
        header = ",".join(["t_s", "nis"] + [f"nu_{c}" for c in names] + ["alarm"])
        np.savetxt(
            args.out, cols, delimiter=",", header=header, comments="", fmt="%.6g"
        )

    any_alarm = alarms.any(axis=1)
    intervals = alarm_intervals(t_s, any_alarm, args.merge_gap_s)
    sensors = "GNSS, wheel speed, gyro" + (", magnetometer" if MAG[0] in log else "")
    print(f"{args.log}: {len(t_s)} samples, {t_s[-1] - t_s[0]:.1f} s, {sensors}")
    if not intervals:
        print("no alarms")
        return 0
    for first, last, count in intervals:
        inside = (t_s >= first) & (t_s <= last)
        per_col = alarms[inside].sum(axis=0)
        which = ", ".join(f"{c} {n}" for c, n in zip(names, per_col) if n)
        print(f"alarms {first:.1f} s to {last:.1f} s: {count} ({which})")
    return 1


def cmd_simulate(args):
    rng = np.random.default_rng(args.seed)
    dt_s = 0.1
    traj = simulate_trajectory(rng, round(args.duration_s / dt_s), dt_s)
    z = gnss_fixes(rng, traj)
    if args.attack == "jump":
        z = jump(z, traj.t_s, args.attack_start_s, (args.size, 0.0))
    elif args.attack == "drag":
        z = drag_off(z, traj.t_s, args.attack_start_s, args.size)
    elif args.attack == "drag-across":
        z = drag_across_track(z, traj.t_s, args.attack_start_s, args.size, traj.vel_mps)
    cols = [traj.t_s, z[:, 0], z[:, 1], wheel_speed(rng, traj)]
    cols.append(gyro_yaw_rate(rng, traj, bias_radps=np.deg2rad(args.gyro_offset_dps)))
    names = list(REQUIRED)
    if not args.no_mag:
        mag = magnetometer(rng, traj, offset_ut=(args.hard_iron_ut, 0.0))
        cols += [mag[:, 0], mag[:, 1]]
        names += MAG
    np.savetxt(
        args.log,
        np.column_stack(cols),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.6g",
    )
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="spoofdet", description="Find spoofed GNSS fixes in a sensor log."
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser(
        "detect",
        help="flag spoofed intervals in a CSV log",
        description="Columns: t_s, east_m, north_m, speed_mps, yaw_rate_radps, "
        "and optionally mag_fwd_ut, mag_left_ut. One row per sample at a fixed "
        "rate. Exit status 0 with no alarms, 1 with alarms, 2 on a bad log.",
    )
    d.add_argument("log", help="CSV log with a header row")
    d.add_argument("--out", help="write per-step NIS, innovations and alarms here")
    d.add_argument(
        "--gnss-std",
        type=float,
        default=1.5,
        metavar="M",
        help="GNSS position sd per axis in m (default 1.5)",
    )
    d.add_argument(
        "--speed-std",
        type=float,
        default=0.05,
        metavar="MPS",
        help="wheel speed sd in m/s (default 0.05)",
    )
    d.add_argument(
        "--gyro-std",
        type=float,
        default=5.5e-4,
        metavar="RADPS",
        help="gyro sd per sample in rad/s (default 5.5e-4, BMI088 at 10 Hz)",
    )
    d.add_argument(
        "--accel-std",
        type=float,
        default=0.2,
        metavar="MPS2",
        help="along-track acceleration sd in m/s^2 (default 0.2)",
    )
    d.add_argument(
        "--gyro-offset-std-dps",
        type=float,
        default=1.0,
        metavar="DPS",
        help="prior sd of the gyro zero-rate offset in deg/s, 0 to "
        "not estimate it (default 1.0)",
    )
    d.add_argument(
        "--gyro-offset-walk",
        type=float,
        default=1.1e-4,
        metavar="W",
        help="offset random walk in rad/s per sqrt(s) (default 1.1e-4)",
    )
    d.add_argument(
        "--mag-noise-ut",
        type=float,
        default=0.6,
        metavar="UT",
        help="magnetometer noise per axis in uT (default 0.6, BMM150)",
    )
    d.add_argument(
        "--field-ut",
        type=float,
        default=25.0,
        metavar="UT",
        help="horizontal field at the site in uT (default 25)",
    )
    d.add_argument(
        "--hard-iron-std-ut",
        type=float,
        default=2.0,
        metavar="UT",
        help="prior sd of the hard-iron offset in uT, 0 to not "
        "estimate it (default 2.0)",
    )
    d.add_argument(
        "--start-s",
        type=float,
        default=5.0,
        metavar="S",
        help="start-up window for the heading fit in s (default 5)",
    )
    d.add_argument(
        "--drift",
        type=float,
        default=0.5,
        help="CUSUM drift in innovation sd (default 0.5)",
    )
    d.add_argument(
        "--threshold",
        type=float,
        default=10.5,
        help="CUSUM threshold, 10.5 is about 1 false alarm an hour "
        "at 10 Hz (default 10.5)",
    )
    d.add_argument(
        "--merge-gap-s",
        type=float,
        default=30.0,
        metavar="S",
        help="alarms closer than this go in one interval (default 30)",
    )
    d.set_defaults(func=cmd_detect)

    s = sub.add_parser("simulate", help="write a simulated log, 10 Hz")
    s.add_argument("log", help="CSV file to write")
    s.add_argument(
        "--duration-s",
        type=float,
        default=600.0,
        metavar="S",
        help="length in s (default 600)",
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--attack",
        choices=["none", "jump", "drag", "drag-across"],
        default="none",
        help="GNSS spoofing profile (default none)",
    )
    s.add_argument(
        "--size",
        type=float,
        default=0.1,
        help="jump in m east, or drag rate in m/s (default 0.1)",
    )
    s.add_argument(
        "--attack-start-s",
        type=float,
        default=60.0,
        metavar="S",
        help="attack start in s (default 60)",
    )
    s.add_argument(
        "--gyro-offset-dps",
        type=float,
        default=1.0,
        metavar="DPS",
        help="gyro zero-rate offset in deg/s (default 1.0)",
    )
    s.add_argument(
        "--hard-iron-ut",
        type=float,
        default=0.0,
        metavar="UT",
        help="forward hard-iron offset in uT (default 0)",
    )
    s.add_argument("--no-mag", action="store_true", help="leave out the magnetometer")
    s.set_defaults(func=cmd_simulate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
