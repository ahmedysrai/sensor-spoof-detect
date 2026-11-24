"""A simulated scenario written as a log, for testing the detector end to end.

This is the same data the experiment scripts put through the filters, packed
into a spoofdet.logio.Log: a trajectory from spoofdet.sim, the sensors that
were asked for, and optionally one of the GNSS spoofing profiles from
spoofdet.attacks applied to the fixes from `start_s` on. Everything is drawn
from one seeded generator in a fixed order (trajectory, fixes, speed, yaw
rate, magnetometer), so the same seed gives the same log.
"""

import numpy as np

from spoofdet.attacks import drag_across_track, drag_off, jump
from spoofdet.logio import Log, LogError
from spoofdet.sim import (
    gnss_fixes,
    gyro_yaw_rate,
    magnetometer,
    simulate_trajectory,
    wheel_speed,
)

ATTACKS = ("none", "jump", "drag", "drag-across")


def simulate_log(
    duration_s=60.0,
    hz=10.0,
    seed=0,
    attack="none",
    jump_m=15.0,
    rate_mps=1.0,
    start_s=30.0,
    gyro_bias_radps=0.0,
    mag_offset_ut=(0.0, 0.0),
    gnss_std_m=1.5,
    accel_std_mps2=0.2,
    speed_std_mps=0.05,
    gyro_std_radps=5.5e-4,
    mag_noise_ut=0.6,
    field_ut=25.0,
    with_speed=True,
    with_gyro=True,
    with_mag=True,
) -> Log:
    """Simulate a run and return it as a Log.

    `attack` is one of "none", "jump" (`jump_m` metres north), "drag"
    (`rate_mps` along the initial heading) or "drag-across" (`rate_mps` to the
    left of the track, turning with the vehicle). `gyro_bias_radps` is the
    zero-rate offset added to the gyro and `mag_offset_ut` the hard-iron
    offset on the magnetometer, both constants over the run.
    """
    if attack not in ATTACKS:
        raise LogError(f"attack must be one of {', '.join(ATTACKS)}, not {attack!r}")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise LogError(f"duration_s must be positive, got {duration_s}")
    if not np.isfinite(hz) or hz <= 0.0:
        raise LogError(f"hz must be positive, got {hz}")
    n_steps = round(duration_s * hz)
    if n_steps < 2:
        raise LogError(f"{duration_s} s at {hz} Hz is {n_steps} sample(s), need at least two")
    dt_s = 1.0 / hz
    if attack != "none" and not 0.0 <= start_s < duration_s:
        raise LogError(
            f"the attack starts at {start_s} s, outside the {duration_s} s run; "
            "nothing would be spoofed"
        )

    rng = np.random.default_rng(seed)
    traj = simulate_trajectory(rng, n_steps, dt_s, accel_std_mps2)
    fixes = gnss_fixes(rng, traj, gnss_std_m)
    speed = wheel_speed(rng, traj, speed_std_mps) if with_speed else None
    rate = gyro_yaw_rate(rng, traj, gyro_std_radps, gyro_bias_radps) if with_gyro else None
    mag = (
        magnetometer(rng, traj, field_ut, mag_noise_ut, mag_offset_ut) if with_mag else None
    )

    if attack == "jump":
        fixes = jump(fixes, traj.t_s, start_s, (0.0, jump_m))
    elif attack == "drag":
        heading = traj.vel_mps[-1] / np.linalg.norm(traj.vel_mps[-1])
        fixes = drag_off(fixes, traj.t_s, start_s, rate_mps, heading)
    elif attack == "drag-across":
        fixes = drag_across_track(fixes, traj.t_s, start_s, rate_mps, traj.vel_mps)

    return Log(
        t_s=traj.t_s,
        fixes_m=fixes,
        speed_mps=speed,
        yaw_rate_radps=rate,
        mag_ut=mag,
        source=f"simulated(seed={seed}, attack={attack})",
        dt_s=dt_s,
    )
