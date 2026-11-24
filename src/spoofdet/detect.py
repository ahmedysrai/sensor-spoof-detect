"""Run a detector over a log and report what it found.

Three filters are available, and the one that runs depends on the sensors the
log carries:

- "gyro": the unicycle EKF of spoofdet.kf.run_gnss_odo_gyro_filter, which
  needs the wheel speed and the gyro yaw rate, and uses a magnetometer
  heading when the log has one. Detection is a two-sided CUSUM on the
  innovations in the track frame (along, across, speed, and heading with a
  magnetometer), which is what finds a slow sideways drag that keeps every
  single innovation small.
- "gnss-odo": the constant-velocity EKF with the wheel speed, which catches a
  drag along the direction of travel but cannot see one across it.
- "gnss": the constant-velocity filter on the fixes alone, which catches a
  jump and nothing slower.

The constant-velocity filters return the NIS per step and no innovations, so
for them the detector is the per-step NIS gate. Both detectors report an
alarm rate per hour; on an attack-free log that rate is the false alarm rate,
and the parameters here (CUSUM drift 0.5, threshold 10.5, ~1 alarm/h on white
noise) are the ones the experiments in NOTES.md measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from spoofdet.cusum import cusum
from spoofdet.kf import (
    mag_heading,
    run_gnss_filter,
    run_gnss_odo_filter,
    run_gnss_odo_gyro_filter,
)
from spoofdet.logio import Log, LogError
from spoofdet.nis import nis_threshold

FILTERS = ("auto", "gyro", "gnss-odo", "gnss")

#: innovation columns of the gyro filter, in order
TRACK_COLUMNS = ("along", "across", "speed")


@dataclass(frozen=True)
class DetectConfig:
    """Sensor model and detector settings for one run.

    The standard deviations are the assumed noise of each sensor, not the
    measured one: the detector is only as good as these guesses, which is why
    they are all on the command line. `mag_std_rad` defaults to the Bosch
    BMM150 figure used in the experiments, 0.6 uT on a 25 uT horizontal
    field. `bias_std_radps` only applies to the gyro filter and turns on the
    zero-rate offset state.
    """

    filter: str = "auto"
    gnss_std_m: float = 1.5
    speed_std_mps: float = 0.05
    gyro_std_radps: float = 5.5e-4
    accel_std_mps2: float = 0.2
    mag_std_rad: float | None = 0.6 / 25.0
    bias_std_radps: float | None = None
    bias_rw_radps: float = 0.0
    n_init: int = 50
    drift: float = 0.5
    threshold: float = 10.5
    nis_p_false_alarm: float = 1.0 / 36000.0


@dataclass
class Detection:
    """What one detector run produced, with the numbers a report needs."""

    log: Log
    config: DetectConfig
    filter: str
    detector: str
    dof: int
    gate: float
    t_s: np.ndarray
    xs: np.ndarray
    nis: np.ndarray
    nu_track: np.ndarray | None
    alarms: np.ndarray
    stat: np.ndarray
    columns: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    @property
    def started_s(self) -> float:
        """Time of the first sample the filter produced a state for."""
        return float(self.t_s[0])

    @property
    def alarm_steps(self) -> int:
        """Number of steps on which any column alarmed."""
        return int(np.count_nonzero(self.alarms.any(axis=1)))

    @property
    def alarm_rate_per_hour(self) -> float:
        hours = self.log.duration_s / 3600.0
        if hours <= 0.0:
            return 0.0
        return self.alarm_steps / hours

    @property
    def first_alarm_s(self) -> float | None:
        steps = np.flatnonzero(self.alarms.any(axis=1))
        if not len(steps):
            return None
        return float(self.t_s[steps[0]])

    @property
    def first_alarm_by_column(self) -> dict[str, float]:
        out = {}
        for index, name in enumerate(self.columns):
            steps = np.flatnonzero(self.alarms[:, index])
            if len(steps):
                out[name] = float(self.t_s[steps[0]])
        return out

    @property
    def nis_over_gate(self) -> float:
        """Fraction of finite NIS values over the per-step gate."""
        finite = np.isfinite(self.nis)
        if not finite.any():
            return 0.0
        return float(np.count_nonzero(self.nis[finite] > self.gate) / finite.sum())

    @property
    def peak_stat(self) -> float:
        """Largest magnitude the statistic reached, NaN when it never ran."""
        finite = np.isfinite(self.stat)
        if not finite.any():
            return float("nan")
        return float(np.max(np.abs(self.stat[finite])))

    @property
    def expected_clean_alarms_per_hour(self) -> float:
        """What the NIS gate's per-step p works out to on clean data."""
        steps_per_hour = 3600.0 / float(self.log.dt_s)
        return self.config.nis_p_false_alarm * steps_per_hour

    @property
    def mean_nis(self) -> float:
        finite = np.isfinite(self.nis)
        if not finite.any():
            return float("nan")
        return float(np.mean(self.nis[finite]))

    def summary(self) -> str:
        """A few lines for a terminal, one fact per line."""
        samples = (
            f"{self.log.source}: {self.log.n_samples} samples, "
            f"{self.log.duration_s:.1f} s at {1.0 / self.log.dt_s:.1f} Hz"
        )
        columns = f" on {', '.join(self.columns)}" if self.columns else ""
        lines = [
            samples,
            (
                f"filter {self.filter}, detector {self.detector}{columns} "
                f"(gate {self.gate:.2f}, {self.dof} dof)"
            ),
        ]
        if self.first_alarm_s is None:
            lines.append(
                f"no alarm; {self.alarm_rate_per_hour:.2f} alarms/h over the run"
            )
        else:
            first = ", ".join(
                f"{name} at {t:.1f} s" for name, t in self.first_alarm_by_column.items()
            )
            lines.append(
                f"{self.alarm_steps} alarm steps, first at {self.first_alarm_s:.1f} s"
                + (f" ({first})" if first else "")
                + f"; {self.alarm_rate_per_hour:.2f} alarms/h"
            )
        lines.append(
            f"mean NIS {self.mean_nis:.2f}, {100 * self.nis_over_gate:.2f} % of steps "
            f"over the NIS gate (p {self.config.nis_p_false_alarm:g} is "
            f"{self.expected_clean_alarms_per_hour:.1f} alarms/h on clean data), "
            f"peak |stat| {self.peak_stat:.1f}"
        )
        lines.extend(f"note: {note}" for note in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """A JSON-ready report; no arrays, so it can be logged or shipped."""
        config = {
            key: (round(value, 9) if isinstance(value, float) else value)
            for key, value in asdict(self.config).items()
        }
        return {
            "detector": self.detector,
            "filter": self.filter,
            "config": config,
            "log": self.log.telemetry(),
            "nis": {
                "dof": self.dof,
                "gate": round(self.gate, 6),
                "p_false_alarm": self.config.nis_p_false_alarm,
                "expected_clean_alarms_per_hour": round(
                    self.expected_clean_alarms_per_hour, 4
                ),
                "mean": None if not np.isfinite(self.mean_nis) else round(self.mean_nis, 4),
                "over_gate_fraction": round(self.nis_over_gate, 6),
            },
            "statistic": {
                "drift": round(self.config.drift, 6),
                "threshold": round(self.config.threshold, 6),
                "peak": None if not np.isfinite(self.peak_stat) else round(self.peak_stat, 4),
            },
            "alarms": {
                "steps": self.alarm_steps,
                "rate_per_hour": round(self.alarm_rate_per_hour, 4),
                "first_s": None if self.first_alarm_s is None else round(self.first_alarm_s, 3),
                "by_column": {
                    name: round(value, 3) for name, value in self.first_alarm_by_column.items()
                },
            },
            "warnings": list(self.warnings),
        }


def _positive(name: str, value) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


def _resolve_filter(log: Log, wanted: str) -> str:
    if wanted not in FILTERS:
        raise ValueError(f"filter must be one of {', '.join(FILTERS)}, not {wanted!r}")
    if wanted == "auto":
        if log.has_gyro and log.has_speed:
            return "gyro"
        return "gnss-odo" if log.has_speed else "gnss"
    if wanted == "gyro" and not (log.has_gyro and log.has_speed):
        missing = "yaw_rate" if not log.has_gyro else "speed"
        raise ValueError(
            f"the gyro filter needs the wheel speed and the yaw rate; this log has no "
            f"{missing!r} column"
        )
    if wanted == "gnss-odo" and not log.has_speed:
        raise ValueError("the gnss-odo filter needs the wheel speed; this log has no 'speed' column")
    return wanted


def detect(log: Log, config: DetectConfig | None = None) -> Detection:
    """Run the detector the log supports and collect the alarms.

    Raises LogError when the log cannot be used (too short for the start
    window, no finite fix at the start), and ValueError for a configuration
    that makes no sense for it, for example a magnetometer sd with no
    magnetometer heading.
    """
    config = config or DetectConfig()
    log.check()

    kind = _resolve_filter(log, config.filter)
    n = log.n_samples
    if config.n_init > n:
        raise LogError(
            f"{log.source}: the filter starts after {config.n_init} samples and the log "
            f"has {n}; lower n_init or record a longer run"
        )
    if config.n_init < 2:
        raise ValueError(f"n_init must be at least 2, got {config.n_init}")
    start_fixes = log.fixes_m[: config.n_init]
    if not np.isfinite(start_fixes).all():
        bad = int(np.flatnonzero(~np.isfinite(start_fixes).all(axis=1))[0])
        raise LogError(
            f"{log.source}: the fix of sample {bad} is missing, and the filter fits its "
            f"start pose to the first {config.n_init} samples; start the log after the "
            "gap or fill it"
        )
    if kind == "gyro" and not np.isfinite(log.yaw_rate_radps[: config.n_init]).all():
        raise LogError(
            f"{log.source}: a yaw rate is missing inside the first {config.n_init} "
            "samples, where the start heading is fitted"
        )
    if kind == "gyro" and not np.isfinite(log.speed_mps[: config.n_init]).all():
        raise LogError(
            f"{log.source}: a wheel speed is missing inside the first {config.n_init} "
            "samples, where the start heading is fitted"
        )

    gnss_std = _positive("gnss_std_m", config.gnss_std_m)
    speed_std = _positive("speed_std_mps", config.speed_std_mps)
    gyro_std = _positive("gyro_std_radps", config.gyro_std_radps)
    accel_std = float(config.accel_std_mps2)
    if not np.isfinite(accel_std) or accel_std < 0.0:
        raise ValueError(
            f"accel_std_mps2 must be zero or positive, got {config.accel_std_mps2}"
        )

    dt_s = float(log.dt_s)
    warnings: list[str] = []

    if kind == "gnss":
        xs, nis_vals = run_gnss_filter(log.fixes_m, dt_s, accel_std, gnss_std)
        nu_track, columns, dof = None, (), 2
        warnings.append(
            "no wheel speed in the log, so only a jump can be seen: a drag-off looks "
            "like a slightly different velocity"
        )
    elif kind == "gnss-odo":
        xs, nis_vals = run_gnss_odo_filter(
            log.fixes_m,
            log.speed_mps,
            dt_s,
            accel_std,
            gnss_std,
            speed_std,
        )
        nu_track, columns, dof = None, (), 3
        warnings.append(
            "no gyro in the log, so a pull across the direction of travel is nearly "
            "invisible: the wheel speed says nothing about heading"
        )
    else:
        mag_std = None
        mag_psi = None
        if log.has_mag and config.mag_std_rad is not None:
            mag_std = _positive("mag_std_rad", config.mag_std_rad)
            mag_psi = mag_heading(log.mag_ut)
        elif log.has_mag:
            warnings.append(
                "the log has a magnetometer reading but mag_std_rad is off, so the "
                "heading is not measured"
            )
        bias_std = (
            None
            if config.bias_std_radps is None
            else _positive("bias_std_radps", config.bias_std_radps)
        )
        bias_rw = (
            _positive("bias_rw_radps", config.bias_rw_radps)
            if bias_std is not None and config.bias_rw_radps
            else 0.0
        )
        xs, nis_vals, nu_track = run_gnss_odo_gyro_filter(
            log.fixes_m,
            log.speed_mps,
            log.yaw_rate_radps,
            dt_s,
            accel_std,
            gnss_std,
            speed_std,
            gyro_std,
            n_init=config.n_init,
            bias_std_radps=bias_std,
            bias_rw_radps=bias_rw,
            mag_psi_rad=mag_psi,
            mag_std_rad=mag_std,
        )
        columns = TRACK_COLUMNS + (("heading",) if mag_psi is not None else ())
        dof = 3 + (1 if mag_psi is not None else 0)
        if bias_std is None:
            warnings.append(
                "the gyro zero-rate offset is not estimated; a 1 deg/s offset makes the "
                "CUSUM alarm on most steps (pass bias_std_radps to estimate it)"
            )
        if mag_psi is None:
            warnings.append(
                "no magnetometer in the log: the heading reference is the gyro alone, so "
                "a slow sideways pull is only visible while the lean lasts"
            )

    gate = float(nis_threshold(dof, config.nis_p_false_alarm))

    if nu_track is not None:
        alarms, stat = cusum(nu_track, drift=config.drift, threshold=config.threshold)
        detector = "cusum"
    else:
        alarms = (nis_vals > gate)[:, None]
        stat = nis_vals[:, None]
        detector = "nis gate"

    return Detection(
        log=log,
        config=config,
        filter=kind,
        detector=detector,
        dof=dof,
        gate=gate,
        t_s=log.t_s,
        xs=xs,
        nis=nis_vals,
        nu_track=nu_track,
        alarms=alarms,
        stat=stat,
        columns=columns,
        warnings=warnings,
    )
