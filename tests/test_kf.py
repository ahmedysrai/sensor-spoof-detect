import numpy as np
import pytest

from spoofdet.attacks import drag_off, jump
from spoofdet.kf import run_gnss_filter
from spoofdet.nis import nis_threshold
from spoofdet.sim import gnss_fixes, simulate_trajectory

DT_S, ACC_STD, GNSS_STD = 0.1, 0.2, 1.5


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
