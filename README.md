# sensor-spoof-detect

Find spoofed GNSS, IMU and LiDAR data on robots and drones by checking the
sensors against each other in a Kalman filter.

The project is simulation based. So far it has a planar vehicle simulator
with GNSS fixes, two GNSS spoofing profiles (a position jump and a slow
drag-off), wheel odometry speed, a gyro yaw rate with the Bosch BMI088
noise density and zero-rate offset, a constant-velocity Kalman filter (an EKF once the speed is
fused), a unicycle EKF that lets the gyro carry the heading, a per-step
normalised innovation squared (NIS) gate and a two-sided CUSUM on the
innovations.

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
against all of them before. NOTES.md has the numbers and the reasons.

## Run it

```sh
pip install -e ".[dev]"
pytest
python experiments/crude_gnss_gate.py   # about 35 s
python experiments/odo_cross_check.py   # about 150 s
python experiments/cusum_vs_gate.py     # about 140 s
python experiments/slow_drag.py         # about 4 min
python experiments/gyro_offset.py       # about 12 min
```
