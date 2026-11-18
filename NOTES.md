# Notes

## 2026-11-18

First end-to-end run: GNSS-only constant-velocity filter, per-step NIS gate
at P_fa 0.01 (threshold 9.21, 2 dof), 200 seeds of 120 s at 10 Hz, attack
from t = 60 s. GNSS noise 1.5 m per axis and 0.2 m/s^2 acceleration noise
are assumed for now. Fraction of steps over the gate
(`python experiments/crude_gnss_gate.py`):

```
attack          before   0-2 s  2-60 s  offset_m
none             0.010   0.013   0.010       0.0
jump 15 m        0.010   0.863   0.026      15.0
jump 5 m         0.010   0.234   0.011       5.0
drag 1 m/s       0.010   0.019   0.011      59.9
drag 0.2 m/s     0.010   0.013   0.010      12.0
```

The false alarm rate before onset is 0.010, so the filter is consistent.
A 15 m jump is caught in the first couple of seconds, then the filter
accepts the new position and the gate goes quiet again. A 5 m jump is
caught on about a quarter of the early steps.

The drag-offs are not seen at all, even the one that ends 60 m away. A
linear ramp in position looks to this model like a slightly different
velocity, which the velocity state absorbs, so the innovations stay white.
With GNSS as the only sensor nothing in the filter disagrees with the
spoofed track. This is the case the IMU or odometry cross-check has to
cover: the attacker's implied velocity differs from what the wheels or the
accelerometers say.
