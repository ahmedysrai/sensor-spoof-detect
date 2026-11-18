# sensor-spoof-detect

Find spoofed GNSS, IMU and LiDAR data on robots and drones by checking the
sensors against each other in a Kalman filter.

The project is simulation based. So far it has a planar vehicle simulator
with GNSS fixes, two GNSS spoofing profiles (a position jump and a slow
drag-off), a constant-velocity Kalman filter and a per-step normalised
innovation squared (NIS) gate.

With GNSS as the only sensor the gate catches a 15 m jump within about 2 s
but misses a 1 m/s drag-off completely, even after it has pulled the track
60 m away. NOTES.md has the numbers and the reason.

## Run it

```sh
pip install -e ".[dev]"
pytest
python experiments/crude_gnss_gate.py   # about 35 s
```
