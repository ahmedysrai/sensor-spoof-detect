import numpy as np

from spoofdet.cusum import cusum


def test_constant_shift_alarms_when_the_sum_passes_the_threshold():
    # g grows by 1 - 0.5 per step, so it is 10.5 on the 21st step
    alarms, stat = cusum(np.ones(30), drift=0.5, threshold=10.0)
    assert np.flatnonzero(alarms).tolist() == [20]
    assert stat[19] == 10.0
    assert stat[20] == 10.5


def test_negative_shift_alarms_on_the_lower_side():
    alarms, stat = cusum(np.full(30, -1.0), drift=0.5, threshold=10.0)
    assert np.flatnonzero(alarms).tolist() == [20]
    assert stat[19] == -10.0


def test_statistic_restarts_after_an_alarm():
    alarms, _ = cusum(np.ones(100), drift=0.5, threshold=10.0)
    assert np.flatnonzero(alarms).tolist() == [20, 41, 62, 83]


def test_nan_samples_hold_the_statistic():
    y = np.r_[np.ones(4), np.nan, np.nan, np.ones(2)]
    _, stat = cusum(y, drift=0.5, threshold=10.0)
    np.testing.assert_allclose(stat, [0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0])


def test_columns_are_independent():
    y = np.column_stack([np.ones(30), np.zeros(30), -np.ones(30)])
    alarms, _ = cusum(y, drift=0.5, threshold=10.0)
    assert alarms[:, 0].sum() == 1
    assert not alarms[:, 1].any()
    assert alarms[:, 2].sum() == 1
