import numpy as np

from spoofdet.attacks import drag_across_track, drag_off


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
