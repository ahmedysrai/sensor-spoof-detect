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
alarms thousands of times an hour on clean runs. NOTES.md has the numbers
and the reasons.

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
```

## Use it on a log

The experiments measure the detector on simulated data. The `spoofdet`
command runs it on a recording: one row per sample, the GNSS fix and
whichever sensors the vehicle has.

```
t,east,north,speed,yaw_rate,mag_forward,mag_left
0,12.400000,3.100000,2.000000,0.010000,1.200000,24.900000
```

`east` and `north` are required, in m, and so is `t` in s unless `--dt` gives
the sample interval. `speed` is the wheel speed in m/s, `yaw_rate` the gyro
rate in rad/s counterclockwise positive, and `mag_forward`/`mag_left` the
horizontal field in uT in the body frame. JSON Lines works too, one object
per line, which is easier to append from a robot. The filters assume a
uniform step, so timestamps off the grid are refused rather than quietly
integrated as if they were not (`--allow-jitter` overrides). An empty cell is
a dropout: that measurement is skipped on that step and the filter carries on
predicting, which is what a receiver that loses a fix looks like.

The detector runs the best filter the log supports (`--filter auto`): the
gyro filter with the CUSUM on the innovations when there is a wheel speed and
a gyro, the speed-only filter when there is no gyro, the fixes alone when
there is neither. The constant-velocity filters return no innovations, so for
them the detector is the per-step NIS gate, whose per-step false alarm
probability (`--nis-pfa`, one an hour at 10 Hz by default) is printed.

```sh
spoofdet simulate --duration 120 --attack drag-across --rate 0.1 --start 60 --out run.csv
spoofdet detect run.csv
spoofdet detect run.csv --json --track track.csv
spoofdet detect - < run.jsonl
```

```
run.csv: 1200 samples, 119.9 s at 10.0 Hz
filter gyro, detector cusum on along, across, speed, heading (gate 26.28, 4 dof)
41 alarm steps, first at 71.0 s (across at 71.0 s, heading at 113.6 s); 1231.03 alarms/h
mean NIS 5.01, 0.00 % of steps over the NIS gate (p 2.77778e-05 is 1.0
alarms/h on clean data), peak |stat| 14.3
note: the gyro zero-rate offset is not estimated; a 1 deg/s offset makes the
CUSUM alarm on most steps (pass bias_std_radps to estimate it)
```

That is the notes' 0.1 m/s sideways drag, found 11 s after it starts, on the
across column, with the heading column following a minute later as the pull
keeps going.

`detect` exits 0 when nothing alarmed, 1 when the detector alarmed and 2 when
the log or an argument could not be used, so it can be the body of a cron job
or a systemd unit as it is. Alarm steps per hour is the false alarm rate only
on a clean log; on an attack it is how loud the detector was. The sensor
noises (`--gnss-std`, `--speed-std`, `--gyro-std`, `--mag-std`) are what the
filter is told, not what it measures, and they are the parameters that decide
how sensitive it is.

The defaults are the ones the experiments established: CUSUM drift 0.5 and
threshold 10.5 for about one false alarm an hour, BMI088 gyro noise
(5.5e-4 rad/s at 10 Hz), BMM150 heading noise (0.024 rad on a 25 uT field),
0.05 m/s wheel speed. `--bias-std` turns on the gyro zero-rate offset state,
which the notes show is needed for a real gyro but also gives the filter a
new way to explain a spoofed turn; `--no-mag` leaves the magnetometer out.

