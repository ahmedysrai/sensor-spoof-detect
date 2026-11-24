import numpy as np
import pytest

from spoofdet.attacks import drag_across_track, drag_off, jump


def test_across_track_drag_on_a_straight_track_matches_drag_off():
    t_s = np.arange(50) * 0.1
    vel = np.tile([2.0, 0.5], (50, 1))
    z = np.zeros((50, 2))
    got = drag_across_track(z, t_s, 2.0, 0.3, vel)
    np.testing.assert_allclose(got, drag_off(z, t_s, 2.0, 0.3, (-0.5, 2.0)))


def test_across_track_drag_turns_with_the_vehicle():
    # a full circle at 1 rad/s: each step's pull is perpendicular to the
    # velocity, to its left, and rate * dt long
    t_s = np.arange(63) * 0.1
    vel = np.column_stack([np.cos(t_s), np.sin(t_s)])
    step = np.diff(drag_across_track(np.zeros((63, 2)), t_s, 0.0, 0.2, vel), axis=0)
    v = vel[1:]
    np.testing.assert_allclose(np.sum(step * v, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(v[:, 0] * step[:, 1] - v[:, 1] * step[:, 0], 0.02)


def test_a_sideways_drag_survives_a_stop_in_the_track():
    # a vehicle that stands still at a light: the velocity is zero for a while
    t_s = np.arange(20) * 0.1
    vel = np.tile([2.0, 0.0], (20, 1))
    vel[8:12] = 0.0
    fixes = drag_across_track(np.zeros((20, 2)), t_s, 0.3, 0.5, vel)
    assert np.isfinite(fixes).all()
    step = np.diff(fixes, axis=0)
    # while stopped (samples 8 to 11) the pull keeps the last direction, +north
    np.testing.assert_allclose(step[9:11, 1], 0.05, atol=1e-12)
    np.testing.assert_allclose(step[9:11, 0], 0.0, atol=1e-12)


def test_a_track_that_never_moves_cannot_be_pulled_across():
    with pytest.raises(ValueError, match="no sample has a usable velocity direction"):
        drag_across_track(np.zeros((5, 2)), np.arange(5) * 0.1, 0.0, 0.2, np.zeros((5, 2)))


def test_attacks_refuse_arrays_that_do_not_line_up():
    fixes = np.zeros((5, 2))
    with pytest.raises(ValueError, match="5 fixes but t_s has 4"):
        jump(fixes, np.arange(4) * 0.1, 0.0, (1.0, 0.0))
    with pytest.raises(ValueError, match="fixes must be"):
        jump(np.zeros((5, 3)), np.arange(5) * 0.1, 0.0, (1.0, 0.0))
    with pytest.raises(ValueError, match="has no length"):
        drag_off(fixes, np.arange(5) * 0.1, 0.0, 1.0, (0.0, 0.0))
