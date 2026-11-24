"""The spoofdet command line: detect on a recorded log, or simulate one.

    spoofdet detect run.csv                 # what the log's sensors support
    spoofdet detect run.jsonl --json        # machine-readable report
    spoofdet simulate --attack drag-across --out run.csv

`detect` exits 0 when nothing alarmed, 1 when the detector alarmed, and 2 on
bad input or arguments, so it can sit in a cron job or a systemd unit as it
is. The human-readable report goes to stdout, except under `--json`, where
the JSON object is the only thing on stdout and any note still goes to
stderr.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from spoofdet.detect import FILTERS, DetectConfig, Detection, detect
from spoofdet.logio import Log, LogError, format_log, parse_log, read_log, write_log
from spoofdet.logsim import ATTACKS, simulate_log

EXIT_CLEAN = 0
EXIT_ALARM = 1
EXIT_BAD_INPUT = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spoofdet",
        description="Find spoofed GNSS, IMU and magnetometer data in a recorded log.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    det = sub.add_parser(
        "detect",
        help="run the detector on a log",
        description=(
            "Run the Kalman filter the log's sensors support and report the alarms. "
            "Exit code 0 means nothing alarmed, 1 that the detector alarmed, 2 that "
            "the log or the arguments could not be used."
        ),
    )
    det.add_argument("log", help="CSV or JSON Lines log, or - for stdin")
    det.add_argument(
        "--filter",
        choices=FILTERS,
        default="auto",
        help="which filter to run (default: the best the log supports)",
    )
    det.add_argument("--json", action="store_true", help="print one JSON object instead")
    det.add_argument(
        "--drift",
        type=float,
        default=0.5,
        help="CUSUM drift in units of innovation sd (default 0.5)",
    )
    det.add_argument(
        "--threshold",
        type=float,
        default=10.5,
        help="CUSUM alarm threshold (default 10.5, about one false alarm an hour)",
    )
    det.add_argument(
        "--nis-pfa",
        type=float,
        default=1.0 / 36000.0,
        metavar="P",
        help=(
            "per-step false alarm probability of the NIS gate (default 1/36000, about "
            "one an hour at 10 Hz)"
        ),
    )
    det.add_argument(
        "--gnss-std", type=float, default=1.5, metavar="M", help="fix noise sd in m (default 1.5)"
    )
    det.add_argument(
        "--speed-std",
        type=float,
        default=0.05,
        metavar="MPS",
        help="wheel speed noise sd in m/s (default 0.05)",
    )
    det.add_argument(
        "--gyro-std",
        type=float,
        default=5.5e-4,
        metavar="RADPS",
        help="gyro noise sd in rad/s (default 5.5e-4, BMI088 at 10 Hz)",
    )
    det.add_argument(
        "--accel-std",
        type=float,
        default=0.2,
        metavar="MPS2",
        help="along-track acceleration noise sd in m/s^2 (default 0.2)",
    )
    det.add_argument(
        "--mag-std",
        type=float,
        default=0.6 / 25.0,
        metavar="RAD",
        help="magnetometer heading noise sd in rad (default 0.024)",
    )
    det.add_argument(
        "--no-mag", action="store_true", help="ignore the magnetometer even when the log has one"
    )
    det.add_argument(
        "--bias-std",
        type=float,
        default=None,
        metavar="RADPS",
        help="prior sd of the gyro zero-rate offset; setting it estimates the offset",
    )
    det.add_argument(
        "--bias-rw",
        type=float,
        default=0.0,
        metavar="RADPS",
        help="random walk of that offset in rad/s per sqrt(s) (default 0)",
    )
    det.add_argument(
        "--n-init",
        type=int,
        default=50,
        metavar="N",
        help="samples used to fit the start pose (default 50)",
    )
    det.add_argument(
        "--dt",
        type=float,
        default=None,
        metavar="S",
        help="sample interval, when the log has no time column",
    )
    det.add_argument(
        "--allow-jitter",
        action="store_true",
        help="accept timestamps that are not on a uniform grid instead of refusing",
    )
    det.add_argument(
        "--track", metavar="CSV", help="also write the filtered track and the statistic"
    )

    sim = sub.add_parser(
        "simulate",
        help="write a simulated scenario as a log",
        description=(
            "Simulate a drive with the sensors the filters use and write it as a log, "
            "optionally with a GNSS spoofing profile applied from --start on. Useful to "
            "see the expected column layout and to try the detector end to end."
        ),
    )
    sim.add_argument("--out", default="-", help="CSV or .jsonl path, or - for stdout")
    sim.add_argument("--duration", type=float, default=60.0, metavar="S", help="run length (60)")
    sim.add_argument("--hz", type=float, default=10.0, metavar="HZ", help="sample rate (10)")
    sim.add_argument("--seed", type=int, default=0, help="generator seed (0)")
    sim.add_argument(
        "--attack", choices=ATTACKS, default="none", help="spoofing profile (default none)"
    )
    sim.add_argument("--jump-m", type=float, default=15.0, metavar="M", help="jump offset (15)")
    sim.add_argument(
        "--rate", type=float, default=1.0, metavar="MPS", help="drag rate of either drag (1.0)"
    )
    sim.add_argument(
        "--start", type=float, default=30.0, metavar="S", help="attack onset (30)"
    )
    sim.add_argument(
        "--gyro-bias-deg",
        type=float,
        default=0.0,
        metavar="DPS",
        help="gyro zero-rate offset added to the data (0)",
    )
    sim.add_argument(
        "--mag-offset-ut",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("FWD", "LEFT"),
        help="hard-iron offset on the magnetometer in uT (0 0)",
    )
    sim.add_argument("--no-mag", action="store_true", help="leave the magnetometer out")
    sim.add_argument("--no-gyro", action="store_true", help="leave the gyro out")
    sim.add_argument("--no-speed", action="store_true", help="leave the wheel speed out")
    sim.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def _read_stdin_log(args) -> Log:
    return parse_log(
        sys.stdin.read(),
        source="-",
        dt_s=args.dt,
        jitter_tolerance=None if args.allow_jitter else 0.01,
    )


def _write_track(path: str, detection: Detection) -> None:
    """Per-step state and statistic, one row per sample, for plotting."""
    xs = detection.xs
    columns = ["t", "east", "north"]
    if detection.filter == "gyro":
        columns += ["psi", "speed"]
    columns += ["nis", "stat", "alarm"]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(columns) + "\n")
        for k, t in enumerate(detection.t_s):
            cells = [f"{t:.3f}", f"{xs[k, 0]:.4f}", f"{xs[k, 1]:.4f}"]
            if detection.filter == "gyro":
                cells += [f"{xs[k, 2]:.6f}", f"{xs[k, 3]:.4f}"]
            nis = detection.nis[k]
            row = detection.stat[k]
            stat = row[np.argmax(np.abs(row))] if np.isfinite(row).any() else np.nan
            cells += [
                "" if not np.isfinite(nis) else f"{nis:.4f}",
                "" if not np.isfinite(stat) else f"{stat:.4f}",
                "1" if detection.alarms[k].any() else "0",
            ]
            handle.write(",".join(cells) + "\n")


def _cmd_detect(args) -> int:
    try:
        log = _read_stdin_log(args) if args.log == "-" else read_log(
            args.log,
            dt_s=args.dt,
            jitter_tolerance=None if args.allow_jitter else 0.01,
        )
        config = DetectConfig(
            filter=args.filter,
            gnss_std_m=args.gnss_std,
            speed_std_mps=args.speed_std,
            gyro_std_radps=args.gyro_std,
            accel_std_mps2=args.accel_std,
            mag_std_rad=None if args.no_mag else args.mag_std,
            bias_std_radps=args.bias_std,
            bias_rw_radps=args.bias_rw,
            n_init=args.n_init,
            drift=args.drift,
            threshold=args.threshold,
            nis_p_false_alarm=args.nis_pfa,
        )
        detection = detect(log, config)
    except (LogError, ValueError, OSError) as exc:
        print(f"spoofdet: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if args.track:
        try:
            _write_track(args.track, detection)
        except OSError as exc:
            print(f"spoofdet: cannot write {args.track}: {exc.strerror or exc}", file=sys.stderr)
            return EXIT_BAD_INPUT

    if args.json:
        print(json.dumps(detection.to_dict(), indent=2))
        for note in detection.warnings:
            print(f"spoofdet: note: {note}", file=sys.stderr)
    else:
        print(detection.summary())
        if args.track:
            print(f"track written to {args.track}")
    return EXIT_ALARM if detection.alarm_steps else EXIT_CLEAN


def _cmd_simulate(args) -> int:
    try:
        log = simulate_log(
            duration_s=args.duration,
            hz=args.hz,
            seed=args.seed,
            attack=args.attack,
            jump_m=args.jump_m,
            rate_mps=args.rate,
            start_s=args.start,
            gyro_bias_radps=np.deg2rad(args.gyro_bias_deg),
            mag_offset_ut=tuple(args.mag_offset_ut),
            with_speed=not args.no_speed,
            with_gyro=not args.no_gyro,
            with_mag=not args.no_mag,
        )
    except (LogError, ValueError) as exc:
        print(f"spoofdet: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if args.out == "-":
        sys.stdout.write(format_log(log))
    else:
        try:
            write_log(args.out, log)
        except OSError as exc:
            print(f"spoofdet: cannot write {args.out}: {exc.strerror or exc}", file=sys.stderr)
            return EXIT_BAD_INPUT

    summary = log.telemetry()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        where = "stdout" if args.out == "-" else args.out
        print(
            f"{summary['samples']} samples of {summary['duration_s']:.1f} s at "
            f"{1.0 / summary['dt_s']:.1f} Hz written to {where}",
            file=sys.stderr if args.out == "-" else sys.stdout,
        )
    return EXIT_CLEAN


def main(argv=None) -> int:
    """Run the command line; returns the exit code instead of exiting."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "detect":
        return _cmd_detect(args)
    return _cmd_simulate(args)
