# sensor-spoof-detect

Find spoofed GNSS, IMU and LiDAR data on robots and drones by checking the
sensors against each other in a Kalman filter.

The project is simulation based. So far it has a planar vehicle simulator
with GNSS fixes, two GNSS spoofing profiles (a position jump and a slow
drag-off), wheel odometry speed, a gyro yaw rate with the Bosch BMI088
noise density and zero-rate offset, a magnetometer with Bosch BMM150
noise, a constant-velocity Kalman filter (an EKF once the speed is fused),
a unicycle EKF that lets the gyro carry the heading, a per-step normalised
innovation squared (NIS) gate and a two-sided CUSUM on the innovations.

With GNSS as the only sensor the gate catches a 15 m jump within about 2 s
but misses a 1 m/s drag-off completely, even after it has pulled the track
60 m away. Adding wheel speed exposes a 1 m/s drag-off along the direction
of travel (the gate trips on 94 % of steps after the first 2 s), but not a
pull sideways, because speed says nothing about heading. With the gyro a
1 m/s sideways pull trips the gate on 93 % of steps. A 0.2 m/s sideways
pull still trips it on only 8 %: the gyro measures turns, not heading, so
the filter's heading slowly follows the spoofed track. The innovations
stay small but lean to one side, and a CUSUM on them adds that up: at
about one false alarm an hour it finds the 0.2 m/s sideways pull on every
run after a median 8 s, where a NIS gate with the same false alarm rate
finds it on 54 % of runs after 38 s. An attacker who pulls more slowly
still gets through: at 0.02 m/s sideways the fixes end 11 m off after
9 minutes on most runs with no alarm, because the filter's heading turns
onto the spoofed track within a few minutes and the lean goes away.
A real gyro makes this worse. The BMI088's 1 deg/s zero-rate offset makes
the plain filter's CUSUM alarm on most steps, so the filter now estimates
the offset as a fifth state. That brings clean-run false alarms back to
0.4/h, but the offset estimate is one more way for the filter's heading to
turn onto the spoofed track: with a random walk on the offset for
temperature drift, a 0.1 m/s sideways pull is found on 22 % of runs,
against all of them before. A magnetometer heading (Bosch BMM150 noise)
closes that gap, since the filter's heading can no longer turn without the
magnetometer disagreeing: 0.1 m/s is found on every run after a median
13 s and 0.02 m/s on 97 % of runs. That holds only for a well calibrated
magnetometer. A 2 uT hard-iron offset, the residual the datasheet gives
after calibration, tilts the heading by about 4 degrees and the CUSUM then
alarms thousands of times an hour on clean runs. Estimating that offset as
two more filter states brings false alarms back to 0.5/h, at a price: the
filter can explain part of a slow pull as offset, so 0.05 m/s is found on
74 % of runs and 0.02 m/s on 25 %. A tighter calibration (0.5 uT prior
and offset) gives the same numbers, because the turns in the first minute
teach the filter the offset whatever the prior was. Raising the
magnetometer's noise figure instead still leaves 769 false alarms an hour. HOW_IT_WORKS.md explains
the filters, the detectors and where the simulation flatters them;
NOTES.md has the dated numbers.

## Run it

```sh
pip install -e ".[dev]"
pytest
python experiments/crude_gnss_gate.py   # about 35 s
python experiments/odo_cross_check.py   # about 150 s
python experiments/cusum_vs_gate.py     # about 140 s
python experiments/slow_drag.py         # about 4 min
python experiments/gyro_offset.py       # about 12 min
python experiments/magnetometer.py      # about 10 min
python experiments/hard_iron.py         # about 14 min
python experiments/hard_iron_prior.py   # about 18 min
```

## Check a log

`pip install -e .` also installs a `spoofdet` command. It reads a CSV log
with a header row and one row per sample at a fixed rate: `t_s`, `east_m`
and `north_m` (GNSS fixes in a local ENU frame), `speed_mps` (wheel
odometry), `yaw_rate_radps` (gyro, counterclockwise positive), and
optionally `mag_fwd_ut` and `mag_left_ut` (horizontal magnetometer
reading, body frame). It runs the five-state gyro filter, adds the
magnetometer and its hard-iron states when those columns are there, and
prints the CUSUM alarms grouped into intervals:

```sh
spoofdet simulate log.csv --attack drag-across --size 0.1   # 10 min, 10 Hz
spoofdet detect log.csv --out steps.csv                     # about 2 s
```

```
log.csv: 6000 samples, 599.9 s, GNSS, wheel speed, gyro, magnetometer
alarms 8.8 s to 8.8 s: 1 (across 1)
alarms 84.0 s to 162.7 s: 8 (across 5, speed 1, heading 2)
alarms 202.0 s to 305.9 s: 52 (across 51, heading 1)
alarms 342.2 s to 342.2 s: 1 (speed 1)
alarms 399.5 s to 597.9 s: 104 (across 104)
```

The pull starts at 60 s. The single alarms at 8.8 s and 342.2 s are false
alarms: the same seed without the attack alarms at 8.8 s, 155.8 s and
342.2 s. At the default threshold a clean log alarms about once an
hour (7 alarms in 9.9 h of clean simulated logs), so one isolated alarm
means little and a long train of them on the same column is the sign to
look for. `--out` writes the NIS, the track-frame innovations and the
alarm flag per step. The noise figures default to the parts the
simulator models (BMI088 gyro, BMM150 magnetometer) and the assumed GNSS
and wheel speed values; `spoofdet detect --help` lists them. Exit status
is 0 with no alarms, 1 with alarms and 2 for a log it cannot use.

A log from a GNSS receiver and an IMU running at different rates needs
resampling onto one rate first. The numbers above all come from simulated
logs.
