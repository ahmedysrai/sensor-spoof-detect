import numpy as np
import pytest

from spoofdet.attacks import drag_off, jump
from spoofdet.kf import (
    fit_start_heading,
    run_gnss_filter,
    run_gnss_odo_filter,
    run_gnss_odo_gyro_filter,
)
from spoofdet.nis import nis_threshold
from spoofdet.sim import gnss_fixes, gyro_yaw_rate, simulate_trajectory, wheel_speed

DT_S, ACC_STD, GNSS_STD = 0.1, 0.2, 1.5
SPEED_STD = 0.05
GYRO_STD = 5.5e-4


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_mean_nis_of_matched_filter_is_measurement_dim(rng):
    traj = simulate_trajectory(rng, 5000, DT_S, ACC_STD)
    z = gnss_fixes(rng, traj, GNSS_STD)
    _, nis_vals = run_gnss_filter(z, DT_S, ACC_STD, GNSS_STD)
    # skip the start-up transient. A matched filter's innovations are white,
    # so 4800 chi-square(2) values (sd 2) give a standard error of about 0.03
    assert np.mean(nis_vals[200:]) == pytest.approx(2.0, abs=0.12)


def test_filter_tracks_position_better_than_raw_fixes(rng):
    traj = simulate_trajectory(rng, 3000, DT_S, ACC_STD)
    z = gnss_fixes(rng, traj, GNSS_STD)
    xs, _ = run_gnss_filter(z, DT_S, ACC_STD, GNSS_STD)
    err_kf = np.sqrt(np.mean((xs[200:, :2] - traj.pos_m[200:]) ** 2))
    err_raw = np.sqrt(np.mean((z[200:] - traj.pos_m[200:]) ** 2))
    assert err_kf < 0.6 * err_raw


def test_position_jump_trips_the_gate_on_the_step_it_lands(rng):
    traj = simulate_trajectory(rng, 600, DT_S, ACC_STD)
    z = jump(gnss_fixes(rng, traj, GNSS_STD), traj.t_s, 30.0, (15.0, 0.0))
    _, nis_vals = run_gnss_filter(z, DT_S, ACC_STD, GNSS_STD)
    k = int(np.argmax(traj.t_s >= 30.0))
    assert nis_vals[k] > nis_threshold(2, 0.01)


def test_drag_off_is_zero_before_start_and_linear_after():
    t_s = np.arange(0.0, 10.0, 1.0)
    z = drag_off(np.zeros((10, 2)), t_s, 4.0, 0.5, direction=(0.0, 2.0))
    np.testing.assert_allclose(z[:5], 0.0)
    np.testing.assert_allclose(z[:, 1], 0.5 * np.clip(t_s - 4.0, 0, None))
    np.testing.assert_allclose(z[:, 0], 0.0)


def test_trajectory_without_accel_noise_is_a_straight_line(rng):
    traj = simulate_trajectory(rng, 101, 0.1, 0.0, vel0_mps=(2.0, -1.0))
    np.testing.assert_allclose(traj.pos_m[-1], [20.0, -10.0], atol=1e-9)


def test_mean_nis_of_matched_gnss_odo_filter_is_three(rng):
    traj = simulate_trajectory(rng, 5000, DT_S, ACC_STD)
    speed = wheel_speed(rng, traj, SPEED_STD)
    z = gnss_fixes(rng, traj, GNSS_STD)
    _, nis_vals = run_gnss_odo_filter(z, speed, DT_S, ACC_STD, GNSS_STD, SPEED_STD)
    # from 5 s on, the same settle time the experiments use. Without the
    # heading check the start-up error lasts about 30 s and this fails.
    # 4950 chi-square(3) values (sd 2.45): standard error about 0.035
    assert np.mean(nis_vals[50:]) == pytest.approx(3.0, abs=0.15)


def test_odometry_waits_until_heading_is_known(rng):
    traj = simulate_trajectory(rng, 100, DT_S, ACC_STD)
    speed = wheel_speed(rng, traj, SPEED_STD)
    z = gnss_fixes(rng, traj, GNSS_STD)
    _, nis_vals = run_gnss_odo_filter(z, speed, DT_S, ACC_STD, GNSS_STD, SPEED_STD)
    # the filter starts with zero velocity and 5 m/s sd, so the first steps
    # cannot have a heading
    assert np.all(np.isnan(nis_vals[:5]))
    assert not np.any(np.isnan(nis_vals[60:]))


def test_wheel_speed_exposes_drag_off_that_gnss_alone_misses(rng):
    traj = simulate_trajectory(rng, 900, DT_S, ACC_STD)
    speed = wheel_speed(rng, traj, SPEED_STD)
    k0 = 300
    z = drag_off(gnss_fixes(rng, traj, GNSS_STD), traj.t_s, 30.0, 1.0, traj.vel_mps[k0])
    _, nis_g = run_gnss_filter(z, DT_S, ACC_STD, GNSS_STD)
    _, nis_o = run_gnss_odo_filter(z, speed, DT_S, ACC_STD, GNSS_STD, SPEED_STD)
    late = slice(k0 + 100, None)  # from 10 s after onset
    assert np.mean(nis_g[late] > nis_threshold(2, 0.01)) < 0.05
    assert np.mean(nis_o[late] > nis_threshold(3, 0.01)) > 0.5


def test_noise_free_gyro_integrates_to_the_heading(rng):
    traj = simulate_trajectory(rng, 500, DT_S, ACC_STD)
    rate = gyro_yaw_rate(rng, traj, 0.0)
    psi = np.unwrap(np.arctan2(traj.vel_mps[:, 1], traj.vel_mps[:, 0]))
    np.testing.assert_allclose(psi[0] + np.cumsum(rate * DT_S), psi, atol=1e-9)


def test_start_heading_fit_is_exact_on_a_noise_free_turn():
    # 5 s at 1.5 m/s while turning 0.4 rad/s: two radians of turn, where a
    # straight-line fit to the fixes would be far off
    n, speed, rate = 50, 1.5, 0.4
    psi = 0.3 + rate * DT_S * np.arange(n)
    step = speed * DT_S * np.column_stack([np.cos(psi), np.sin(psi)])
    fixes = np.r_[np.zeros((1, 2)), np.cumsum(step[:-1], axis=0)] + (10.0, -4.0)
    pos, psi_end, _, scale = fit_start_heading(
        fixes, np.full(n, speed), np.r_[0.0, np.full(n - 1, rate)], DT_S, GNSS_STD
    )
    np.testing.assert_allclose(pos, fixes[-1], atol=1e-9)
    assert psi_end == pytest.approx(psi[-1], abs=1e-9)
    assert scale == pytest.approx(1.0, abs=1e-9)


def gnss_odo_gyro(rng, n_steps, attack=None):
    traj = simulate_trajectory(rng, n_steps, DT_S, ACC_STD)
    speed = wheel_speed(rng, traj, SPEED_STD)
    rate = gyro_yaw_rate(rng, traj, GYRO_STD)
    z = gnss_fixes(rng, traj, GNSS_STD)
    if attack is not None:
        z = attack(z, traj)
    return traj, z, speed, rate


def test_mean_nis_of_matched_gnss_odo_gyro_filter_is_three(rng):
    _, z, speed, rate = gnss_odo_gyro(rng, 5000)
    _, nis_vals, _ = run_gnss_odo_gyro_filter(
        z, speed, rate, DT_S, ACC_STD, GNSS_STD, SPEED_STD, GYRO_STD
    )
    assert np.all(np.isnan(nis_vals[:50]))
    # 4950 chi-square(3) values from 5 s on: standard error about 0.035
    assert np.mean(nis_vals[50:]) == pytest.approx(3.0, abs=0.15)


def test_gyro_exposes_sideways_drag_off_that_speed_alone_misses(rng):
    k0 = 300

    def sideways(z, traj):
        v = traj.vel_mps[k0]
        return drag_off(z, traj.t_s, 30.0, 1.0, (-v[1], v[0]))

    _, z, speed, rate = gnss_odo_gyro(rng, 900, sideways)
    _, nis_o = run_gnss_odo_filter(z, speed, DT_S, ACC_STD, GNSS_STD, SPEED_STD)
    _, nis_y, _ = run_gnss_odo_gyro_filter(
        z, speed, rate, DT_S, ACC_STD, GNSS_STD, SPEED_STD, GYRO_STD
    )
    late = slice(k0 + 100, None)
    gate = nis_threshold(3, 0.01)
    assert np.mean(nis_o[late] > gate) < 0.5
    assert np.mean(nis_y[late] > gate) > 0.5


def test_gyro_filter_refuses_to_start_without_a_heading(rng):
    traj = simulate_trajectory(rng, 100, DT_S, 0.0, vel0_mps=(0.1, 0.0))
    z = gnss_fixes(rng, traj, GNSS_STD)
    speed = wheel_speed(rng, traj, SPEED_STD)
    rate = gyro_yaw_rate(rng, traj, GYRO_STD)
    with pytest.raises(ValueError, match="not moved far enough"):
        run_gnss_odo_gyro_filter(
            z, speed, rate, DT_S, 0.0, GNSS_STD, SPEED_STD, GYRO_STD
        )


def test_track_frame_innovations_are_standard_normal_when_matched(rng):
    _, z, speed, rate = gnss_odo_gyro(rng, 5000)
    _, _, y = run_gnss_odo_gyro_filter(
        z, speed, rate, DT_S, ACC_STD, GNSS_STD, SPEED_STD, GYRO_STD
    )
    assert np.all(np.isnan(y[:50]))
    # 4950 samples per column: standard error of the mean about 0.014
    np.testing.assert_allclose(np.mean(y[50:], axis=0), 0.0, atol=0.06)
    np.testing.assert_allclose(np.std(y[50:], axis=0), 1.0, atol=0.06)


def test_sideways_drag_off_leans_the_across_track_innovation(rng):
    k0 = 300

    def left(z, traj):
        v = traj.vel_mps[k0]
        return drag_off(z, traj.t_s, 30.0, 0.2, (-v[1], v[0]))

    _, z, speed, rate = gnss_odo_gyro(rng, 900, left)
    _, _, y = run_gnss_odo_gyro_filter(
        z, speed, rate, DT_S, ACC_STD, GNSS_STD, SPEED_STD, GYRO_STD
    )
    late = slice(k0 + 100, None)
    # fixes pulled to the left of travel give positive across-track values
    assert np.mean(y[late, 1]) > 0.5
    assert abs(np.mean(y[late, 0])) < 0.3
