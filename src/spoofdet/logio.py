"""Recorded sensor logs, read and written as CSV or JSON Lines.

A log holds what the detector needs, sample by sample: the GNSS fix (east and
north in m), and optionally the wheel speed |v| in m/s, the gyro yaw rate in
rad/s (counterclockwise positive) and the horizontal magnetometer reading in
uT, forward and left in the body frame. The filters in spoofdet.kf assume a
uniform step, so the timestamps are checked against it on the way in.

CSV needs a header row; columns are matched by name with case and spaces
ignored:

    t,east,north,speed,yaw_rate,mag_forward,mag_left
    0,12.4,3.1,2.0,0.01,1.2,24.9

Only `east` and `north` are required. An empty cell is a dropout (NaN) and
the filter skips that measurement on that step. A log written as JSON Lines
is the same data with one object per line, which is easier to append from a
robot:

    {"t": 0.0, "east": 12.4, "north": 3.1, "speed": 2.0}

Nothing here resamples or repairs data. A log that does not fit the model is
refused, because a filter fed the wrong step length produces confident
nonsense rather than an error.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: canonical column names, in the order write_log emits them
COLUMNS = ("t", "east", "north", "speed", "yaw_rate", "mag_forward", "mag_left")

#: accepted spellings per canonical column
ALIASES = {
    "t": ("t", "t_s", "time", "time_s", "timestamp", "sec"),
    "east": ("east", "east_m", "e"),
    "north": ("north", "north_m", "n"),
    "speed": ("speed", "speed_mps", "v", "odo_speed", "wheel_speed"),
    "yaw_rate": ("yaw_rate", "yaw_rate_radps", "yawrate", "gyro", "gyro_z"),
    "mag_forward": ("mag_forward", "mag_fwd", "mag_f", "mag_x"),
    "mag_left": ("mag_left", "mag_lat", "mag_l", "mag_y"),
}
_ALIAS = {alias: name for name, aliases in ALIASES.items() for alias in aliases}


class LogError(ValueError):
    """A log is unreadable, incomplete, or not sampled at a uniform step."""


@dataclass
class Log:
    """One recorded or simulated run, ready for spoofdet.detect.

    `t_s` is in seconds, `fixes_m` is (n, 2) east and north in m,
    `speed_mps` and `yaw_rate_radps` are (n,) or None, and `mag_ut` is
    (n, 2) forward and left in uT or None. `dt_s` is the step the filters
    use; it comes from the timestamps unless it is given.
    """

    t_s: np.ndarray
    fixes_m: np.ndarray
    speed_mps: np.ndarray | None = None
    yaw_rate_radps: np.ndarray | None = None
    mag_ut: np.ndarray | None = None
    source: str = "<memory>"
    dt_s: float | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.t_s = np.asarray(self.t_s, dtype=float).reshape(-1)
        self.fixes_m = np.asarray(self.fixes_m, dtype=float).reshape(-1, 2)
        if self.speed_mps is not None:
            self.speed_mps = np.asarray(self.speed_mps, dtype=float).reshape(-1)
        if self.yaw_rate_radps is not None:
            self.yaw_rate_radps = np.asarray(self.yaw_rate_radps, dtype=float).reshape(-1)
        if self.mag_ut is not None:
            self.mag_ut = np.asarray(self.mag_ut, dtype=float).reshape(-1, 2)
        if self.dt_s is None:
            self.dt_s = self._median_step()

    def _median_step(self) -> float:
        if len(self.t_s) < 2:
            raise LogError(
                f"{self.source}: a log needs at least two samples, got {len(self.t_s)}"
            )
        return float(np.median(np.diff(self.t_s)))

    @property
    def n_samples(self) -> int:
        return len(self.t_s)

    @property
    def duration_s(self) -> float:
        """Time from the first sample to the last."""
        return float(self.t_s[-1] - self.t_s[0])

    @property
    def has_speed(self) -> bool:
        return self.speed_mps is not None

    @property
    def has_gyro(self) -> bool:
        return self.yaw_rate_radps is not None

    @property
    def has_mag(self) -> bool:
        return self.mag_ut is not None

    def check(self) -> Log:
        """Raise LogError unless this is a log the filters can be fed."""
        n = self.n_samples
        if n < 2:
            raise LogError(f"{self.source}: a log needs at least two samples, got {n}")
        if self.fixes_m.shape[0] != n:
            raise LogError(
                f"{self.source}: east and north hold {self.fixes_m.shape[0]} rows "
                f"for {n} timestamps"
            )
        for name in ("speed_mps", "yaw_rate_radps"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != n:
                raise LogError(
                    f"{self.source}: {name} holds {value.shape[0]} values for {n} timestamps"
                )
        if self.mag_ut is not None and self.mag_ut.shape[0] != n:
            raise LogError(
                f"{self.source}: the magnetometer holds {self.mag_ut.shape[0]} rows "
                f"for {n} timestamps"
            )
        if not np.all(np.isfinite(self.t_s)):
            bad = int(np.flatnonzero(~np.isfinite(self.t_s))[0])
            raise LogError(f"{self.source}: the timestamp of sample {bad} is not a number")
        step = np.diff(self.t_s)
        if np.any(step <= 0.0):
            bad = int(np.flatnonzero(step <= 0.0)[0])
            raise LogError(
                f"{self.source}: the timestamp goes back at sample {bad + 1} "
                f"({self.t_s[bad]:.3f} s then {self.t_s[bad + 1]:.3f} s)"
            )
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise LogError(
                f"{self.source}: the step is {self.dt_s}, the filters need a positive one"
            )
        return self

    def telemetry(self) -> dict:
        """What the log carries, for a report."""
        return {
            "source": self.source,
            "samples": self.n_samples,
            "duration_s": round(self.duration_s, 3),
            "dt_s": round(float(self.dt_s), 6),
            "gnss_dropouts": int(np.sum(~np.isfinite(self.fixes_m).all(axis=1))),
            "has_speed": self.has_speed,
            "has_gyro": self.has_gyro,
            "has_mag": self.has_mag,
        }


def _column_of(name: str) -> str | None:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIAS.get(key)


def _float_or_nan(cell, where: str) -> float:
    if cell is None:
        return float("nan")
    text = str(cell).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        raise LogError(f"{where}: {text!r} is not a number") from None


def _check_step(t_s, dt_s, jitter_tolerance, source) -> float:
    """Return the step the filters will use, refusing timestamps that disagree."""
    if dt_s is None:
        dt = float(np.median(np.diff(t_s)))
    else:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise LogError(f"{source}: the step is {dt_s}, the filters need a positive one")
    if jitter_tolerance is None or len(t_s) < 3:
        return dt
    deviation = np.abs(np.diff(t_s) - dt)
    worst = int(np.argmax(deviation))
    if deviation[worst] > jitter_tolerance * dt:
        raise LogError(
            f"{source}: sample {worst + 1} is {np.diff(t_s)[worst]:.4f} s after the one "
            f"before, not {dt:.4f} s (tolerance {jitter_tolerance:.1%}). Resample the log "
            "to a uniform step, or pass jitter_tolerance=None (--allow-jitter) to use "
            "it anyway"
        )
    return dt


def _build(columns: dict, source: str, dt_s, jitter_tolerance) -> Log:
    if columns.get("east") is None or columns.get("north") is None:
        missing = "east" if columns.get("east") is None else "north"
        raise LogError(f"{source}: no {missing!r} column; the detector needs GNSS fixes")
    fixes = np.column_stack([columns["east"], columns["north"]])
    n = len(fixes)
    if n < 2:
        raise LogError(f"{source}: the log holds {n} sample(s), the filters need at least two")

    if columns.get("t") is not None:
        t_s = np.asarray(columns["t"], dtype=float)
        if np.all(np.isnan(t_s)):
            raise LogError(f"{source}: the 't' column holds no values; pass dt_s instead")
    elif dt_s is None:
        raise LogError(
            f"{source}: no 't' column, so the sample interval is unknown; pass dt_s "
            "(--dt on the command line)"
        )
    else:
        t_s = np.arange(n) * float(dt_s)

    mag = None
    if columns.get("mag_forward") is not None and columns.get("mag_left") is not None:
        mag = np.column_stack([columns["mag_forward"], columns["mag_left"]])

    log = Log(
        t_s=t_s,
        fixes_m=fixes,
        speed_mps=columns.get("speed"),
        yaw_rate_radps=columns.get("yaw_rate"),
        mag_ut=mag,
        source=source,
        dt_s=None,
    )
    log.check()
    log.dt_s = _check_step(log.t_s, dt_s, jitter_tolerance, source)
    return log


def _read_csv(text: str, source: str, dt_s, jitter_tolerance) -> Log:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise LogError(f"{source}: the CSV has no header row")
    mapping: dict[str, str] = {}
    for name in reader.fieldnames:
        canonical = _column_of(name)
        if canonical is None:
            continue
        if canonical in mapping.values():
            known = next(key for key, value in mapping.items() if value == canonical)
            raise LogError(
                f"{source}: {known!r} and {name!r} both name the {canonical} column"
            )
        mapping[name] = canonical
    if "east" not in mapping.values() or "north" not in mapping.values():
        missing = "east" if "east" not in mapping.values() else "north"
        raise LogError(
            f"{source}: no {missing!r} column; the header is {reader.fieldnames}"
        )

    values: dict[str, list] = {name: [] for name in set(mapping.values())}
    for row in reader:
        for header, canonical in mapping.items():
            where = f"{source}:{reader.line_num} column {header!r}"
            values[canonical].append(_float_or_nan(row.get(header), where))
    if not values.get("east"):
        raise LogError(f"{source}: the CSV has a header but no data rows")
    return _build(values, source, dt_s, jitter_tolerance)


def _read_jsonl(text: str, source: str, dt_s, jitter_tolerance) -> Log:
    values: dict[str, list] = {}
    seen = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogError(f"{source}:{number}: {exc.msg}") from None
        if not isinstance(record, dict):
            raise LogError(f"{source}:{number}: expected one object per line")
        found: dict[str, object] = {}
        for key, value in record.items():
            canonical = _column_of(key)
            if canonical is None:
                continue
            if canonical in found:
                raise LogError(f"{source}:{number}: two keys name the {canonical} column")
            found[canonical] = value
        for canonical in set(values) | set(found):
            values.setdefault(canonical, []).append(
                _float_or_nan(found.get(canonical), f"{source}:{number} column {canonical!r}")
            )
        seen += 1
    if seen == 0:
        raise LogError(f"{source}: the file holds no records")
    return _build(values, source, dt_s, jitter_tolerance)


def read_log(path, *, dt_s=None, jitter_tolerance=0.01) -> Log:
    """Read a CSV or JSON Lines log and check it against the filter's model.

    `dt_s` is the sample interval; it is read from the 't' column when there
    is one and required when there is not. `jitter_tolerance` is the largest
    relative deviation from that step that is accepted (1 % by default);
    None accepts any strictly increasing timestamps.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LogError(f"cannot read {p}: {exc.strerror or exc}") from exc
    return parse_log(text, source=str(p), dt_s=dt_s, jitter_tolerance=jitter_tolerance)


def parse_log(text: str, *, source="<text>", dt_s=None, jitter_tolerance=0.01) -> Log:
    """Parse log text that is already in memory (CSV or JSON Lines)."""
    stripped = text.lstrip()
    if not stripped:
        raise LogError(f"{source}: the log is empty")
    if stripped[0] in "{[":
        return _read_jsonl(stripped, source, dt_s, jitter_tolerance)
    return _read_csv(text, source, dt_s, jitter_tolerance)


def format_log(log: Log, *, digits: int = 9, jsonl: bool = False) -> str:
    """The log as CSV text, or as JSON Lines with `jsonl` set."""
    present = ["t", "east", "north"]
    if log.speed_mps is not None:
        present.append("speed")
    if log.yaw_rate_radps is not None:
        present.append("yaw_rate")
    if log.mag_ut is not None:
        present.extend(("mag_forward", "mag_left"))

    def record(index: int) -> dict[str, float]:
        out = {
            "t": float(log.t_s[index]),
            "east": float(log.fixes_m[index, 0]),
            "north": float(log.fixes_m[index, 1]),
        }
        if log.speed_mps is not None:
            out["speed"] = float(log.speed_mps[index])
        if log.yaw_rate_radps is not None:
            out["yaw_rate"] = float(log.yaw_rate_radps[index])
        if log.mag_ut is not None:
            out["mag_forward"] = float(log.mag_ut[index, 0])
            out["mag_left"] = float(log.mag_ut[index, 1])
        return out

    if jsonl:
        lines = []
        for index in range(log.n_samples):
            cells = {
                key: (value if np.isfinite(value) else None)
                for key, value in record(index).items()
            }
            lines.append(json.dumps(cells))
        return "\n".join(lines) + "\n"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=present, lineterminator="\n")
    writer.writeheader()
    for index in range(log.n_samples):
        cells = record(index)
        writer.writerow(
            {
                name: (f"{cells[name]:.{digits}g}" if np.isfinite(cells[name]) else "")
                for name in present
            }
        )
    return buffer.getvalue()


def write_log(path, log: Log, *, digits: int = 9) -> Path:
    """Write `log` as CSV, or as JSON Lines when the name ends in .jsonl/.ndjson.

    Only the columns the log carries are written, so a GNSS-only log stays a
    three-column log. Dropouts are written as empty cells (CSV) or null
    (JSON Lines).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    as_jsonl = p.suffix.lower() in (".jsonl", ".ndjson")
    p.write_text(format_log(log, digits=digits, jsonl=as_jsonl), encoding="utf-8")
    return p
