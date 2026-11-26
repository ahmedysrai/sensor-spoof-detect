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

## 2026-11-23

Gyro zero-rate offset. The BMI088 datasheet (BST-BMI088-DS000-19 rev. 1.9,
table 5) gives +-1 deg/s typ. with offset cancellation off, and +-0.015
deg/s per K over temperature. The simulator now takes a constant offset,
and run_gnss_odo_gyro_filter can carry it as a fifth state
(bias_std_radps, bias_rw_radps).

Scratch check first, 20 seeds, 600 s clean: with 1 deg/s in the data the
four-state filter's CUSUM alarms about 32000 times an hour, nearly every
step. The heading runs off at 1 deg/s and the fixes keep pulling it back,
so the across-track innovation leans all the time. With the offset state
false alarms go back to 0.3/h.

The start fit dead-reckons with the raw rates, so the fitted start heading
is off by b * T / 2 (T = 4.9 s window, checked on a noise-free straight
run). I put that in the start covariance with its correlation to the
offset. Over 200 runs of 10 s with a 1 deg/s offset, the estimate is still
short by 0.57 of the offset on average with the correlation, 0.72 without
it, and 0.97 with its sign the wrong way round (I had the sign as a switch
in a scratch run to check the derivation).

Then the slow-drag runs again (experiments/gyro_offset.py, 100 seeds,
600 s, track-following drag from 60 s, CUSUM 0.5 / 10.5). The walk value
1.1e-4 rad/s per sqrt(s) is an assumption: 10 K in 10 min at 0.015 deg/s
per K. The simulated offset does not drift.

```
filter               rate m/s    FA/h  found  delay s  b err 60 s  b err end  (deg/s)
4 states, no offset      none     0.5
4 states, no offset      0.05           0.84     35.0
4 states, no offset      0.10           1.00     15.6
4 states, no offset      0.20           1.00      7.6
4 states                 none 32462.1
5 states, no walk        none     0.4                       0.011     0.0003
5 states, no walk        0.05           0.68    141.6
5 states, no walk        0.10           1.00     79.5
5 states, no walk        0.20           1.00     12.0
5 states, walk           none     0.4                       0.012     0.0099
5 states, walk           0.05           0.11    104.7
5 states, walk           0.10           0.22    190.8
5 states, walk           0.20           0.88     14.4
```

The offset state hurts detection. A sideways drag needs the filter's
heading to turn by atan(rate / speed) to follow the spoofed track (notes
11-22). Without an offset state only the GNSS updates on psi can do that.
With one, the filter can also explain the turn as a change in the offset.
Scratch check on 5 seeds at 0.1 m/s: the offset estimate moves by -0.04
to -0.11 deg/s in the first 30 s of the attack compared with the same
seed run clean, which turns the heading left by a degree or two over that
time, towards the drag. With no walk the offset variance keeps shrinking,
so the estimate settles back and the drag is still found, only later
(15.6 s -> 79.5 s at 0.1 m/s). With the walk the offset stays free to move
and 0.1 m/s gets through on 78 % of runs.

So the walk value matters a lot and I have no measurement behind mine.
The attack here starts at 60 s, when the offset is known to 0.011 deg/s;
a later start with no walk would probably look like the four-state case,
not tried. Temperature (or a model of offset against temperature) as an
input would let the walk be much smaller. A heading reference
(magnetometer) is still the real fix for both problems.

## 2026-11-24

Magnetometer heading as a fourth measurement in the gyro filter
(experiments/magnetometer.py). Same set-up as the walk case of 11-23:
1 deg/s gyro offset in the data, five-state filter with the assumed
1.1e-4 rad/s per sqrt(s) walk, track-following sideways drag from 60 s,
600 s runs, 100 seeds, CUSUM 0.5 / 10.5 on every innovation column.

Magnetometer noise from the BMM150 datasheet (BST-BMM150-DS001-05
rev. 1.4, page 8): 0.6 uT rms per axis in the regular preset, which runs
at 10 Hz, the filter rate. The horizontal field of 25 uT is my
assumption. Heading sd 0.6 / 25 = 0.024 rad (1.4 deg) per sample.

```
filter               rate m/s    FA/h  found  delay s
gyro only                none     0.4
gyro only                0.02           0.06     61.7
gyro only                0.05           0.11    104.7
gyro only                0.10           0.22    190.8
+ magnetometer           none     0.7
+ magnetometer           0.02           0.97     78.5
+ magnetometer           0.05           1.00     25.2
+ magnetometer           0.10           1.00     13.1
+ mag, 2 uT offset       none  6065.2
+ mag, 2 uT offset       0.02           1.00      0.3
+ mag, 2 uT offset       0.05           1.00      0.3
+ mag, 2 uT offset       0.10           1.00      0.3
```

The gyro-only rows match 11-23, so the baseline reproduces. With a clean
magnetometer the gap closes: 0.1 m/s goes from 0.22 to all runs, and even
0.02 m/s, which ended 11 m off unnoticed on 11-22, is found on 97 % of
runs after a median 79 s. The heading can no longer turn onto the
spoofed track, because every degree it turns shows up in the heading
innovation. False alarms 0.7/h against 0.4/h; 1.6 h of clean data each,
so I would not read much into the difference.

The last case is the catch. The same datasheet page gives +-2 uT zero-B
offset after calibration, and a 2 uT body-frame offset on the forward
axis puts 3 to 4.6 deg of heading error on these runs (they mostly head
east-north-east, where that offset does the most; scratch check on 5
seeds). The filter is told the readings only have white noise, so it
follows the biased heading (filter heading 3.2 to 4.0 deg off, 3 seeds)
and at 2 m/s that is about 0.14 m/s of apparent sideways motion against
the fixes: the across-track column alarms over a thousand times a run
and the heading column almost never. 6065 alarms an hour, and the
"found" rows with 0.3 s delay are just those false alarms.

So an uncalibrated-looking residual of the size Bosch quotes after their
own calibration looks exactly like the attack. The magnetometer is only
useful if the remaining offset is either much smaller than 2 uT or in the
filter. A hard-iron offset as two more states is the obvious next step,
but it is observable only while the vehicle turns, and on these mostly
straight runs a slow drag could probably be absorbed into it the same way
it was absorbed into the gyro offset. Not tried yet.

## 2026-11-26

Two ways to deal with the 2 uT hard-iron offset from 11-24
(experiments/hard_iron.py). Same data and settings as magnetometer.py:
1 deg/s gyro offset estimated with the assumed walk, 2 uT on the forward
axis, 600 s runs, 100 seeds, CUSUM 0.5 / 10.5 on every column.

- inflated sd: the heading row sd raised from 0.024 rad to
  hypot(0.6, 2) / 25 = 0.084 rad, nothing else changed.
- offset states: forward and left offset as two constant states, prior sd
  2 uT each, heading row through mag_heading_model (the exact atan2 of the
  offset reading, linearised at the prediction).
- states, no offset: the same filter on data without an offset, to see
  what carrying the states costs by itself.

```
filter               rate m/s    FA/h  found  delay s
white noise only         none  6065.2
white noise only         0.02           1.00      0.3
white noise only         0.05           1.00      0.3
white noise only         0.10           1.00      0.3
inflated sd              none   768.9
inflated sd              0.02           1.00      2.2
inflated sd              0.05           1.00      2.3
inflated sd              0.10           0.92      2.2
offset states            none     0.5
offset states            0.02           0.25    337.1
offset states            0.05           0.74     58.7
offset states            0.10           0.99     18.0
states, no offset        none     0.5
states, no offset        0.02           0.28    332.1
states, no offset        0.05           0.75     61.2
states, no offset        0.10           0.99     17.8
```

The first block reproduces 11-24. Inflating the sd is not enough: 769
false alarms an hour, and the "found" rows are those false alarms again
(delays of about 2 s). The offset is constant for a given heading, so a
larger white-noise sd only lowers its weight; the filter still follows a
biased heading. I did not try a larger sd, since it would throw away the
reason for having the magnetometer.

The offset states fix the false alarms (0.5/h, back to the clean
magnetometer's 0.7/h within what 1.6 h of data can tell). The offset is
learned well on clean runs: scratch check on 20 seeds, error at 600 s
0.00 +- 0.03 uT forward and -0.01 +- 0.05 uT left. On these runs the
heading wanders 36 to 106 deg over the 10 minutes (5 seeds), so there are
enough turns.

The cost is what I expected on 11-24: the drag hides partly in the new
states. Found fraction and delay at 0.02 / 0.05 / 0.1 m/s go from
0.97 (79 s) / 1.00 (25 s) / 1.00 (13 s) with a perfectly calibrated
magnetometer to 0.25 (337 s) / 0.74 (59 s) / 0.99 (18 s). The last block
shows that is all from carrying the states, not from the offset in the
data. Same 20-seed scratch check under the attack: the forward offset
estimate ends 0.17 +- 0.08 uT low at 0.02 m/s and 0.42 +- 0.21 uT low at
0.05 m/s, so the filter explains part of the sideways pull as offset. That
is still far better than the gyro alone (0.06 / 0.11 / 0.22).

So the offset states are the right default when the calibration is not
trusted to well under 2 uT. A tighter prior (a better calibration) should
move the result back towards the clean case; not run yet. Run time about
14 min.
