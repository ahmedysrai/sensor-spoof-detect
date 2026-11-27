# How it works

A ground robot or small drone usually trusts its GNSS position. Someone
who can make the receiver report false but plausible positions can pull
the vehicle off its route without any single fix looking wrong. The
vehicle also carries sensors the spoofer does not control (wheel
odometry, a gyro, a magnetometer), so this project fuses them with the
fixes in a Kalman filter and watches the innovations for a disagreement
that should not be there.

The project is simulation based. The attacks change the position values
a receiver reports; the RF side of spoofing is out of scope, and there is
no code that generates spoofing signals. NOTES.md is the dated log behind
every number here.

## From sensors to an alarm

1. `sim.simulate_trajectory`: a planar vehicle in ENU at 10 Hz, white
   acceleration noise on a constant-velocity model, starting at
   (2.0, 0.5) m/s.
2. `sim.gnss_fixes`, `sim.wheel_speed`, `sim.gyro_yaw_rate` and
   `sim.magnetometer` turn the true track into sensor readings.
3. `attacks.jump`, `attacks.drag_off` and `attacks.drag_across_track`
   change the fixes from t = 60 s on.
4. `kf.run_gnss_odo_gyro_filter` fuses everything and returns the state
   track, the NIS per step and the innovations in the track frame.
5. `nis.nis_threshold` (per-step chi-square gate) and `cusum.cusum`
   (two-sided CUSUM per innovation column) turn those into alarms.
6. The scripts in `experiments/` run 100 or 200 seeds per case and print
   the fraction of runs found, the median delay to the first alarm and
   the false alarms per hour on clean runs.

## The filters

There are three, and each is kept as the baseline for the next.

The first is a constant-velocity Kalman filter on GNSS alone, state
$x = [e, n, v_e, v_n]$. It catches a 15 m jump but misses a drag-off
entirely: a position ramp looks like a slightly different velocity, the
velocity state absorbs it and the innovations stay white. With GNSS as
the only sensor nothing in the filter can disagree with the spoofed
track.

The second adds wheel speed $|v|$ as a third measurement row, linearised
at the prediction with $\partial |v| / \partial v = v / |v|$. The row is
only used once the heading is known, meaning the largest velocity sd is
under 0.2 of the predicted speed. Without that check the filter started
with an unknown heading, linearised the speed row along a wrong one and
raised the false alarm rate from 0.010 to 0.088 per step, nearly all of
it in the first 30 s.
Speed catches a pull along the track but says almost nothing about a pull
sideways, which only changes the implied speed in second order.

The third is a unicycle EKF, state $x = [e, n, \psi, s]$ with heading
$\psi$ counterclockwise from east, and the gyro as the input of the
prediction:

$$
e_k = e_{k-1} + s\,\Delta t \cos\psi, \quad
n_k = n_{k-1} + s\,\Delta t \sin\psi, \quad
\psi_k = \psi_{k-1} + (\omega_k - b)\,\Delta t
$$

Gyro noise is process noise on $\psi$ and along-track acceleration is
process noise on $s$. The vehicle moves along its heading with no
sideslip, which is what lets the gyro hold the direction of travel
against the fixes. GNSS position and wheel speed are linear measurement
rows.

This filter needs a heading to start from. A straight-line fit to the
first 2 s of fixes failed on one seed in 200 (fitted speed 0.41 m/s, too
low to give a heading), and a longer straight fit would be biased by the
turns inside it. So `fit_start_heading` dead-reckons the first 5 s from
wheel speed and gyro and solves for the final position and heading: with
$c = \cos\psi$ and $s = \sin\psi$ every fix is linear in $(e, n, c, s)$, so
it is one least-squares solve and the turns cost nothing.

Two optional parts sit on top. The gyro zero-rate offset $b$ can be a
fifth state with a random walk. Because the start fit dead-reckons with
the raw rates, an offset turns the fitted path and puts the start heading
off by about $b T / 2$ for a window of $T$ seconds, so the start
covariance carries that term and its correlation $-\sigma_b^2 T / 2$ with
$b$. And a magnetometer heading can be a fourth measurement row on
$\psi$, its innovation wrapped to $\pm\pi$.

The filter returns its position innovations rotated into the track frame
(along and across the predicted heading, left positive), plus the speed
and magnetometer innovations, each divided by its sd from $S$. When the
filter matches the data they are close to white $N(0, 1)$: before an
attack the across column had mean 0.00, sd 1.00 and lag-1
autocorrelation 0.002 over 40 seeds.

## The detectors

The NIS gate tests each step on its own:
$\nu^T S^{-1} \nu$ against the chi-square quantile for the measurement
dimension. A slow drag keeps every innovation small. One sd of lean on
one of three dimensions raises the mean NIS from 3 to only about 4, which
the gate cannot tell from noise.

CUSUM adds the lean up. For each column $y$, with drift $d$ and threshold
$h$,

$$
g_k = \max(0,\ g_{k-1} + y_k - d),
$$

the same for $-y$, alarm when either passes $h$, then restart from zero
so a persistent attack keeps alarming. I kept $d = 0.5$, the usual choice
for a shift of one sd. The threshold came from running the CUSUM on 111 h
of white $N(0, 1)$ samples in three columns: 10 gave 1.42 alarms an hour,
10.5 gave 0.83 and 11 gave 0.50. I kept 10.5, about one an hour. The NIS
gate it is compared with is set to the same budget, a per-step false
alarm probability of 1/36000. The CUSUM treats the columns as
independent and ignores their correlation in $S$.

## What is simulated, and where it flatters the method

Everything is simulated, and in several places the generator is kinder
than a real vehicle would be.

- GNSS noise is white, 1.5 m per axis, independent every 0.1 s. That
  number is an assumption, not yet from a receiver datasheet. Real GNSS
  error is correlated over seconds to minutes (multipath, the
  atmosphere), and correlated error looks much more like a slow drag than
  white noise does. I think this is the assumption that flatters the
  detector most.
- The filter is told the exact noise model the generator used, for every
  sensor. There is no model mismatch, apart from the deliberate cases
  below.
- The trajectory is a smooth random walk in velocity at about 2 m/s. It
  never stops, never reverses and mostly heads east-north-east. It is
  planar and level, and the vehicle never slips sideways.
- Wheel speed has white noise of 0.05 m/s (assumed) and no slip or
  wheel-radius scale error.
- Gyro noise is the Bosch BMI088 noise density, 0.014 deg/s/sqrt(Hz)
  (datasheet BST-BMI088-DS000-19 rev. 1.9, table 5), which is 5.5e-4
  rad/s per 0.1 s sample. The 1 deg/s zero-rate offset is the same
  table's typical value with offset cancellation off. In the data it is
  constant. The filter's walk on it, 1.1e-4 rad/s per sqrt(s), is my
  assumption from the table's +-0.015 deg/s per K and 10 K of warming in
  10 minutes; nothing has measured it.
- The magnetometer noise is the Bosch BMM150 regular preset, 0.6 uT rms
  per axis at 10 Hz (datasheet BST-BMM150-DS001-05 rev. 1.4, page 8).
  The horizontal field of 25 uT is assumed (it is site dependent), the
  field points north with no declination, and there is no tilt, no soft
  iron and no disturbance from motors or currents. Hard iron appears only
  in the cases that test it.
- The attacker starts at 60 s, once the filter has settled.

## Reproducing the numbers

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
```

Seeds are fixed, so each script prints the same table every time. The
script for each result below is named with it.

## Results

GNSS only, per-step gate at a 0.01 false alarm rate, 200 seeds
(`crude_gnss_gate.py`): a 15 m jump trips the gate on 0.86 of steps in
the first 2 s. A 1 m/s drag-off trips it on 0.011 of steps against 0.010
before the attack, after pulling the track 60 m away.

Adding wheel speed (`odo_cross_check.py`): a 1 m/s drag along the track
trips the gate on 0.94 of steps after the first 2 s, the same drag across
the track on 0.33. With the gyro filter the sideways 1 m/s goes to 0.93,
but a sideways 0.2 m/s only trips it on 0.077 of steps.

CUSUM against the gate at about one false alarm an hour, 200 seeds of
120 s (`cusum_vs_gate.py`): CUSUM finds the 0.2 m/s sideways drag on
every run after a median 8.0 s, when the fixes are 1.6 m off; the gate
finds it on 0.54 of runs after 38 s. CUSUM is two to four times faster on
every drag-off.

A slower attacker over 600 s with a drag that follows the route, 100
seeds (`slow_drag.py`): CUSUM finds 0.05 m/s on 0.84 of runs, 0.02 m/s on
0.21, and about 0.07 of runs would alarm in that time from false alarms
alone. The undetected 0.02 m/s runs end 10.8 m off. The reason is that
the gyro measures turns, not heading: the fixes rotate the filter's
heading onto the spoofed track, which runs at a constant angle
$\arctan(\text{rate} / \text{speed})$ to the true one, and after about a
minute the lean is a tenth of what it was.

With the BMI088's 1 deg/s offset in the data (`gyro_offset.py`), the
four-state filter alarms 32462 times an hour on clean runs. The offset
state brings that back to 0.4/h, but gives the heading a second way to
turn onto the spoofed track. With the assumed walk, 0.1 m/s is found on
0.22 of runs and 0.2 m/s on 0.88.

With a magnetometer heading on top (`magnetometer.py`): 0.02, 0.05 and
0.1 m/s are found on 0.97, 1.00 and 1.00 of runs after a median 79, 25
and 13 s, at 0.7 false alarms an hour. With a 2 uT hard-iron offset on
the forward axis, the residual the BMM150 datasheet quotes after
calibration, the heading is off by 3 to 4.6 degrees, the filter follows
it, and the across-track column alarms 6065 times an hour on clean runs.

Two fixes for that offset on the same data (`hard_iron.py`). Raising the
magnetometer heading sd to cover it, from 0.024 to 0.084 rad, still gives
769 false alarms an hour: the error is constant for a given heading, and
a white-noise sd only lowers its weight. Two more states for the forward
and left offset (prior 2 uT per axis) bring it to 0.5/h and learn the
offset to a few hundredths of a uT on clean runs, which turn enough over
10 minutes. But the drag can now hide in the offset, as it did in the
gyro offset: 0.02, 0.05 and 0.1 m/s are found on 0.25, 0.74 and 0.99 of
runs after a median 337, 59 and 18 s, and the forward offset estimate
ends 0.42 uT low under the 0.05 m/s drag. The same filter on data with
no offset gives the same numbers, so the cost is in carrying the states.

## Limits and what comes next

- The hard-iron states rely on the turns in these runs (36 to 106 deg
  of heading change in 10 minutes on the seeds checked). A vehicle that
  drives straight would leave the offset and a heading error
  indistinguishable. How much a tighter offset prior (a better
  calibration) buys back is not measured yet.
- The gyro offset walk sets how much a drag can hide in the offset, and
  my value is a guess. A temperature input would let it be much smaller.
- The false alarm rates rest on 6.4 h of clean runs for the CUSUM
  comparison and 16.5 h per magnetometer case, a handful of alarms each,
  so they show the order of magnitude, not the exact rate.
- Not tried yet: an attacker who ramps the rate up slowly, and an attack
  that starts long after the offset has settled.
- Still to come from the scope: cited u-blox M8 GNSS noise in place of
  the assumed 1.5 m, wheel slip, the learned detector on innovation
  sequences and LiDAR. The `spoofdet` command (README) runs the
  filter on a CSV log, but needs every sensor on one fixed rate.
