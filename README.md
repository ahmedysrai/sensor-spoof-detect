# sensor-spoof-detect

Find spoofed GNSS, IMU and LiDAR data on robots and drones by checking the
sensors against each other in a Kalman filter.

The project is simulation based. So far it has a planar vehicle simulator
with GNSS fixes, two GNSS spoofing profiles (a position jump and a slow
drag-off), wheel odometry speed, a constant-velocity Kalman filter (an EKF
once the speed is fused) and a per-step normalised innovation squared (NIS)
gate.

With GNSS as the only sensor the gate catches a 15 m jump within about 2 s
but misses a 1 m/s drag-off completely, even after it has pulled the track
60 m away. Adding wheel speed exposes a 1 m/s drag-off along the direction
of travel (the gate trips on 94 % of steps after the first 2 s), but a slow
pull sideways still gets through, because speed says nothing about heading.
NOTES.md has the numbers and the reasons.

## Run it

```sh
pip install -e ".[dev]"
pytest
python experiments/crude_gnss_gate.py   # about 35 s
python experiments/odo_cross_check.py   # about 100 s
```
