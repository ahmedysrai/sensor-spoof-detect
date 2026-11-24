import numpy as np
import pytest

from spoofdet.logio import LogError
from spoofdet.logsim import simulate_log


def test_simulate_log_carries_the_sensors_asked_for():
    log = simulate_log(duration_s=10.0, hz=5.0, with_mag=False)
    assert log.n_samples == 50
    assert log.dt_s == pytest.approx(0.2)
    assert log.has_speed and log.has_gyro and not log.has_mag

    bare = simulate_log(duration_s=10.0, with_speed=False, with_gyro=False, with_mag=False)
    assert not (bare.has_speed or bare.has_gyro or bare.has_mag)


def test_same_seed_same_log():
    a = simulate_log(duration_s=10.0, seed=7, attack="drag-across", rate_mps=0.1, start_s=5.0)
    b = simulate_log(duration_s=10.0, seed=7, attack="drag-across", rate_mps=0.1, start_s=5.0)
    np.testing.assert_array_equal(a.fixes_m, b.fixes_m)
    np.testing.assert_array_equal(a.mag_ut, b.mag_ut)

    c = simulate_log(duration_s=10.0, seed=8, attack="drag-across", rate_mps=0.1, start_s=5.0)
    assert not np.array_equal(a.fixes_m, c.fixes_m)


def test_a_jump_moves_the_fixes_from_onset_by_the_offset():
    log = simulate_log(duration_s=30.0, seed=2, attack="jump", jump_m=15.0, start_s=10.0)
    clean = simulate_log(duration_s=30.0, seed=2)
    shift = log.fixes_m - clean.fixes_m
    onset = round(10.0 / log.dt_s)
    np.testing.assert_allclose(shift[:onset], 0.0)
    np.testing.assert_allclose(shift[onset:, 0], 0.0)
    np.testing.assert_allclose(shift[onset:, 1], 15.0)


def test_a_drag_grows_from_zero_at_the_rate():
    log = simulate_log(duration_s=30.0, seed=3, attack="drag", rate_mps=0.5, start_s=20.0)
    clean = simulate_log(duration_s=30.0, seed=3)
    offset = np.linalg.norm(log.fixes_m - clean.fixes_m, axis=1)
    elapsed = np.clip(log.t_s - 20.0, 0.0, None)
    np.testing.assert_allclose(offset, 0.5 * elapsed, atol=1e-9)


def test_the_gyro_bias_and_the_magnetometer_offset_reach_the_data():
    plain = simulate_log(duration_s=10.0, seed=5)
    biased = simulate_log(duration_s=10.0, seed=5, gyro_bias_radps=np.deg2rad(1.0))
    np.testing.assert_allclose(biased.yaw_rate_radps - plain.yaw_rate_radps, np.deg2rad(1.0))

    shifted = simulate_log(duration_s=10.0, seed=5, mag_offset_ut=(2.0, 0.0))
    np.testing.assert_allclose(shifted.mag_ut[:, 0] - plain.mag_ut[:, 0], 2.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"attack": "teleport"}, "attack must be one of"),
        ({"duration_s": 0.0}, "duration_s must be positive"),
        ({"hz": 0.0}, "hz must be positive"),
        ({"duration_s": 0.01, "hz": 10.0}, "need at least two"),
        (
            {"duration_s": 10.0, "attack": "jump", "start_s": 10.0},
            "the attack starts at 10.0 s, outside the 10.0 s run",
        ),
    ],
)
def test_a_scenario_that_makes_no_sense_is_refused(kwargs, message):
    with pytest.raises(LogError, match=message):
        simulate_log(**kwargs)
