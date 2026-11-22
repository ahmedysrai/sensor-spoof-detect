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

## 2026-11-19

Added wheel odometry as a speed measurement (|v| only, 0.05 m/s noise,
assumed) to the filter, so the speed row is an EKF update linearised at
the prediction.

First version used the speed row whenever the predicted speed was over
0.3 m/s. The false alarm rate before onset came out at 0.088 instead of
0.010. Binning by time showed a start-up error, not a steady one: 0.27 of
steps over the gate in 0-5 s, 0.50 in 5-10 s, 0.21 in 10-20 s, back to
0.01 only after about 30 s. The filter starts with zero velocity and 5 m/s
sd, so the heading is unknown, and the speed row linearised along a wrong
heading makes the filter confident in a wrong velocity. Changed the
condition to a heading check: use the speed row only once the largest
velocity sd is under 0.2 times the predicted speed. The speed row then
starts at about 2.5 s and the false alarm rate is 0.010 from 5 s on.

Same data through both filters, drag-offs aimed along or across the true
heading at onset, 200 seeds (`python experiments/odo_cross_check.py`):

```
attack                filter      before   0-2 s  2-60 s
none                  gnss         0.010   0.006   0.010
none                  gnss+odo     0.010   0.012   0.010
jump 5 m              gnss         0.010   0.227   0.011
jump 5 m              gnss+odo     0.010   0.499   0.082
drag 1 m/s along      gnss         0.010   0.013   0.011
drag 1 m/s along      gnss+odo     0.010   0.020   0.944
drag 0.2 m/s along    gnss         0.010   0.006   0.010
drag 0.2 m/s along    gnss+odo     0.010   0.012   0.276
drag 1 m/s across     gnss         0.010   0.013   0.011
drag 1 m/s across     gnss+odo     0.010   0.016   0.330
drag 0.2 m/s across   gnss         0.010   0.006   0.010
drag 0.2 m/s across   gnss+odo     0.010   0.012   0.012
```

The along-track drag is now seen: 1 m/s on 0.94 of steps, 0.2 m/s on
0.28. Across the heading the attacker only changes the implied speed in
second order: at about 2 m/s, 1 m/s across gives sqrt(5) - 2 = 0.24 m/s
of extra speed (caught on 0.33 of steps), 0.2 m/s across gives 0.01 m/s
(not caught). Speed alone cannot see a sideways pull. That needs a heading
reference: yaw rate from the IMU gyro, or wheel speed per side.

The 5 m jump also gets better (0.50 early instead of 0.23), and keeps
tripping the gate on 0.08 of later steps. With the speed pinned, the filter
can only move onto the new position slowly, so the innovations stay large
for a while after the jump.

## 2026-11-20

Added the gyro yaw rate and a second filter with a unicycle state
[e, n, heading, speed]: the gyro turns the heading in the prediction, the
vehicle moves along its heading, and GNSS plus wheel speed are the
measurement. Gyro noise is the BMI088 noise density, 0.014 deg/s/sqrt(Hz)
(datasheet rev. 1.9, table 5), averaged over a 0.1 s sample: 5.5e-4 rad/s.
No gyro bias yet.

The start heading first came from a straight-line fit to the first 2 s of
fixes. On one of the 200 seeds the fitted speed came out at 0.41 m/s
(true speed about 2 m/s, fit sd 0.58 m/s per axis), too low to give a
heading, and the filter refused to start. A longer straight-line fit
would be biased by the turns inside it, so the start now dead-reckons the
first 5 s from wheel speed and gyro and fits only the final heading and
position to the fixes, which is linear in (e, n, cos, sin).

Same data as before, gyro drawn last so the other sensors match the
2026-11-19 run seed for seed (the gnss and gnss+odo rows reproduce it),
plus a column for the last 10 s (`python experiments/odo_cross_check.py`):

```
attack                filter      before   0-2 s  2-60 s  50-60 s
none                  +gyro        0.010   0.011   0.010    0.010
jump 5 m              +gyro        0.010   0.551   0.099    0.015
drag 1 m/s along      +gyro        0.010   0.022   0.946    1.000
drag 0.2 m/s along    +gyro        0.010   0.012   0.317    0.614
drag 1 m/s across     gnss+odo     0.010   0.016   0.330    0.571
drag 1 m/s across     +gyro        0.010   0.022   0.933    0.996
drag 0.2 m/s across   gnss+odo     0.010   0.012   0.012    0.015
drag 0.2 m/s across   +gyro        0.010   0.012   0.077    0.098
```

The gyro filter keeps the 0.010 false alarm rate. The 1 m/s sideways pull
goes from 0.33 to 0.93 of steps. The 0.2 m/s sideways pull is still mostly
missed: 0.077 of steps, about eight times the false alarm rate but nowhere
near a usable alarm per step.

The reason is that the gyro measures the change of heading, not the
heading. A constant sideways drag looks to this filter like a heading
error of 0.2 / 2 = 0.1 rad, and the GNSS updates slowly turn the heading
estimate towards it, at a rate set by how little gyro noise the filter
assumes. Each step's innovation stays small, but they all lean the same
way, which is what CUSUM on the innovations should pick up. That is the
next step.

## 2026-11-21

CUSUM on the innovations. The gyro filter now also returns its innovations
in the track frame (along and across the predicted heading, plus speed),
each divided by its sd from S. A scratch run over 40 seeds first checked
what the 0.2 m/s sideways drag does to them: before onset the across
column has mean 0.00 and sd 1.00 with lag-1 autocorrelation 0.002, so the
samples are close to white N(0, 1). After onset its mean is 0.64 in 2-10 s,
1.35 in 10-30 s and 1.49 in 30-60 s, while the along and speed columns stay
near zero. So the lean grows for the whole minute instead of fading as the
heading follows the spoof. One standard deviation of lean on one of three
dof raises the mean NIS from 3 to only about 4, which is why the per-step
gate missed it.

Two-sided CUSUM per column, drift 0.5. For the threshold I ran the CUSUM
on white N(0, 1) samples, 3 columns, 4 million steps (111 h at 10 Hz):
threshold 10 gave 1.42 alarms per hour, 10.5 gave 0.83, 11 gave 0.50.
Kept 10.5, about one an hour. The NIS gate it is compared with is set for
the same budget, a per-step P_fa of 1/36000.

`python experiments/cusum_vs_gate.py`, same 200 seeds as before, gyro
filter only (found: fraction of seeds with an alarm within 60 s of onset;
delay: median time to the first alarm at or after onset):

```
attack                detector    FA/h  found  delay s
none                  nis gate     0.9
none                  cusum        0.8
jump 5 m              nis gate           0.99      0.6
jump 5 m              cusum              1.00      0.3
drag 1 m/s along      nis gate           1.00      5.3
drag 1 m/s along      cusum              1.00      2.4
drag 0.2 m/s along    nis gate           1.00     26.8
drag 0.2 m/s along    cusum              1.00      7.4
drag 1 m/s across     nis gate           1.00      5.7
drag 1 m/s across     cusum              1.00      2.4
drag 0.2 m/s across   nis gate           0.54     38.2
drag 0.2 m/s across   cusum              1.00      8.0
```

The 0.2 m/s sideways drag, the case the gyro filter could not see per
step, is now found on every seed after a median 8.0 s, when the fixes are
pulled 1.6 m off, about one GNSS sd. The NIS gate at the same false alarm
rate finds it on 0.54 of seeds after 38 s. CUSUM is two to four times
faster on every drag-off.

The false alarm rates rest on only 6.4 h of attack-free runs (five or six
alarms each), so they only show that the filter innovations do not alarm
much more often than white noise does. A false alarm in the first 60 s
after onset would also count as found, under a 2 % chance per seed at
one an hour.

An attacker who knows the detector can still stay under it. With a lean
below the 0.5 drift the statistic still drifts down on average and alarms
stay rare. If the lean scales linearly with the drag rate (1.49 at
0.2 m/s), that is a sideways drag under about 0.07 m/s. Not tried yet.

## 2026-11-22

Tried the attacker from the last entry, who pulls slowly sideways to stay
under the CUSUM drift. First a scratch run with the old fixed-direction
drag over 600 s: the across-track lean changed sign late in the run,
because the vehicle turns over ten minutes and a drag set across the
track at onset ends up partly along it. So the attack now follows the
route (attacks.drag_across_track): each step pulls to the left of the
true velocity.

`python experiments/slow_drag.py` (600 s at 10 Hz, attack from 60 s, 100
seeds, about 4 min; found: an alarm at any time before the end of the run;
delay and offset are medians at the first alarm; lean is the mean
across-track innovation):

```
rate m/s  detector   FA/h  found  delay s  offset m  lean 0-60 s  lean later
    none  nis gate    1.1                                   0.01        0.00
    none  cusum       0.5                                   0.01        0.00
    0.02  nis gate          0.15    265.8       5.3         0.14        0.02
    0.02  cusum             0.21     89.6       1.8         0.14        0.02
    0.05  nis gate          0.16    269.2      13.5         0.34        0.04
    0.05  cusum             0.84     35.0       1.8         0.34        0.04
    0.10  nis gate          0.26    265.4      26.5         0.67        0.08
    0.10  cusum             1.00     15.6       1.6         0.67        0.08
    0.20  nis gate          0.81     46.2       9.2         1.34        0.16
    0.20  cusum             1.00      7.6       1.5         1.34        0.16
```

The lean in the first minute does scale linearly with the rate, about 6.7
per m/s, which puts the 0.5 drift at about 0.075 m/s, close to the guess
from the 11-21 run. But 0.05 m/s is still found on 84 % of seeds, because
the lean is not steady: it only lasts about a minute and then falls to a
tenth of that. Whenever the CUSUM finds the drag, it is when the fixes are
1.5 to 1.8 m off, about one GNSS sd.

Why the lean fades: a sideways pull at a constant rate makes the spoofed
track run at a constant angle atan(rate / speed) to the true one. The gyro
only measures turns, so the GNSS updates rotate the filter's heading onto
the spoofed track and the innovations go back to zero mean. A scratch
check on five seeds at 0.2 m/s: the filter's heading error is under 0.5
deg 10 s after onset and ends within about 1 deg of atan(0.2 / speed)
(3.87 deg against 3.87 on seed 0, 2.56 against 2.50 on seed 4).

So the attacker does not need to stay under the drift forever, only for
the first minute or so. At 0.02 m/s the CUSUM finds 0.21 of seeds, and at
0.5 false alarms an hour about 0.07 of seeds would alarm in 540 s anyway,
so most of those are probably false alarms. The rest end 10.8 m off after
9 minutes with nothing seen, and the pull can go on. Ramping the rate up
slowly would probably let the attacker go faster still; not tried.

The NIS gate is no use here: its late alarms (median 265 s) at the low
rates are about what its 1.1/h false alarm rate gives over 540 s.

The gap is the heading. Something that measures heading itself and not
its change (a magnetometer, a second GNSS antenna) or a map of the route
would see the constant angle. The +-1 deg/s gyro zero-rate offset, still
not modelled, works the other way: it is a heading drift of the same kind
the attacker is hiding in.
