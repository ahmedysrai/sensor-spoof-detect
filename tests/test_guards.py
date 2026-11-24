"""The refusals: what each entry point does with input it cannot model."""

import numpy as np
import pytest

from spoofdet.attacks import drag_off
from spoofdet.cusum import cusum
from spoofdet.kf import run_gnss_filter, run_gnss_odo_filter, run_gnss_odo_gyro_filter
from spoofdet.logsim import simulate_log
from spoofdet.nis import nis_threshold
from spoofdet.sim import Trajectory, gyro_yaw_rate, simulate_trajectory

DT, ACC, GNSS, SPD, GYR = 0.1, 0.2, 1.5, 0.05, 5.5e-4


def test_a_gnss_run_with_a_dropout_keeps_its_state():
    log = simulate_log(duration_s=60.0, seed=1, with_gyro=False, with_speed=False)
    broken = simulate_log(duration_s=60.0, seed=1, with_gyro=False, with_speed=False)
    broken.fixes_m[300] = np.nan
    xs, nis_vals = run_gnss_filter(broken.fixes_m, DT, ACC, GNSS)
    clean, _ = run_gnss_filter(log.fixes_m, DT, ACC, GNSS)
    assert np.isnan(nis_vals[300])
    assert np.isfinite(xs).all()
    assert np.linalg.norm(xs[400, :2] - clean[400, :2]) < 2.0


def test_a_gnss_run_needs_a_first_fix_to_start_from():
    fixes = np.array([[np.nan, 0.0], [0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="the first fix is missing"):
        run_gnss_filter(fixes, DT, ACC, GNSS)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda z: run_gnss_filter(z[:1], DT, ACC, GNSS), "1 fixes, need at least two"),
        (lambda z: run_gnss_filter(z, DT, ACC, 0.0), "gnss_std_m must be positive"),
        (lambda z: run_gnss_filter(z, DT, -0.1, GNSS), "accel_std_mps2 must be zero or positive"),
        (
            lambda z: run_gnss_odo_filter(z, np.zeros(9), DT, ACC, GNSS, SPD),
            "10 GNSS fixes but speed_mps has 9",
        ),
        (
            lambda z: run_gnss_odo_filter(z, np.zeros(10), DT, ACC, GNSS, -1.0),
            "speed_std_mps must be positive",
        ),
    ],
)
def test_the_constant_velocity_filters_name_the_bad_argument(call, message):
    z = np.zeros((10, 2))
    with pytest.raises(ValueError, match=message):
        call(z)


def test_a_missing_speed_drops_only_that_row_of_the_update():
    log = simulate_log(duration_s=60.0, seed=1, with_gyro=False)
    log.speed_mps[300] = np.nan
    xs, nis_vals = run_gnss_odo_filter(log.fixes_m, log.speed_mps, DT, ACC, GNSS, SPD)
    assert np.isfinite(xs).all()
    # the NIS is only reported on steps where the speed row took part
    assert np.isnan(nis_vals[300])


def test_the_gyro_filter_needs_a_start_window_that_fits():
    log = simulate_log(duration_s=10.0, seed=1)
    args = (log.fixes_m, log.speed_mps, log.yaw_rate_radps, DT, ACC, GNSS, SPD, GYR)
    with pytest.raises(ValueError, match="n_init is 101 but the run has 100 samples"):
        run_gnss_odo_gyro_filter(*args, n_init=101)
    with pytest.raises(ValueError, match="n_init must be at least 2"):
        run_gnss_odo_gyro_filter(*args, n_init=1)
    with pytest.raises(TypeError, match="n_init must be an int"):
        run_gnss_odo_gyro_filter(*args, n_init=50.0)


def test_the_gyro_filter_refuses_a_heading_without_its_noise():
    log = simulate_log(duration_s=20.0, seed=1)
    with pytest.raises(ValueError, match="mag_psi_rad and mag_std_rad go together"):
        run_gnss_odo_gyro_filter(
            log.fixes_m,
            log.speed_mps,
            log.yaw_rate_radps,
            DT,
            ACC,
            GNSS,
            SPD,
            GYR,
            mag_psi_rad=np.zeros(len(log.fixes_m)),
        )


def test_the_gyro_filter_refuses_a_random_walk_without_an_offset_state():
    log = simulate_log(duration_s=20.0, seed=1)
    with pytest.raises(ValueError, match="bias_rw_radps only applies with bias_std_radps"):
        run_gnss_odo_gyro_filter(
            log.fixes_m, log.speed_mps, log.yaw_rate_radps, DT, ACC, GNSS, SPD, GYR,
            bias_rw_radps=1e-4,
        )


def test_the_gyro_filter_names_the_sample_missing_from_the_start_window():
    log = simulate_log(duration_s=20.0, seed=1)
    log.yaw_rate_radps[7] = np.nan
    with pytest.raises(ValueError, match="the yaw rate of sample 7 is missing"):
        run_gnss_odo_gyro_filter(
            log.fixes_m, log.speed_mps, log.yaw_rate_radps, DT, ACC, GNSS, SPD, GYR
        )


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: cusum(np.array([])), "y is empty"),
        (lambda: cusum(np.zeros((4, 2, 2))), "y must be"),
        (lambda: cusum(np.zeros(4), drift=0.0), "drift must be positive"),
        (lambda: cusum(np.zeros(4), drift=-0.5), "drift must be positive"),
        (lambda: cusum(np.zeros(4), threshold=0.0), "threshold must be positive"),
        (lambda: cusum(np.zeros(4), threshold=np.inf), "threshold must be positive"),
    ],
)
def test_the_cusum_refuses_settings_that_would_alarm_on_noise(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_a_cusum_at_a_zero_drift_would_alarm_on_zeros():
    # this is the reason the drift is validated: it climbs on a flat zero run
    with pytest.raises(ValueError):
        cusum(np.zeros(100), drift=0.0, threshold=10.0)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: simulate_trajectory(np.random.default_rng(0), 1), "1 steps, need at least two"),
        (
            lambda: simulate_trajectory(np.random.default_rng(0), 10, dt_s=0.0),
            "dt_s must be positive",
        ),
        (
            lambda: simulate_trajectory(np.random.default_rng(0), 10, accel_std_mps2=-1.0),
            "accel_std_mps2 must be zero or positive",
        ),
    ],
)
def test_the_simulator_refuses_a_run_it_cannot_make(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_a_single_sample_trajectory_cannot_be_differenced():
    traj = Trajectory(t_s=np.array([0.0]), pos_m=np.zeros((1, 2)), vel_mps=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="needs two samples"):
        gyro_yaw_rate(np.random.default_rng(0), traj, GYR)


def test_a_negative_drag_rate_still_means_pull_backwards():
    t_s = np.arange(5) * 0.1
    z = drag_off(np.zeros((5, 2)), t_s, 0.0, -0.5, (0.0, 1.0))
    np.testing.assert_allclose(z[:, 1], -0.5 * t_s)


def test_the_gate_moves_with_the_degrees_of_freedom():
    assert nis_threshold(2, 0.01) < nis_threshold(3, 0.01) < nis_threshold(4, 0.01)
