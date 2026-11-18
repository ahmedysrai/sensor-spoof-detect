import numpy as np
import pytest

from spoofdet.nis import nis, nis_threshold


def test_nis_matches_mahalanobis():
    S = np.diag([4.0, 1.0])
    assert nis([2.0, 1.0], S) == pytest.approx(2.0)


def test_consistent_filter_passes_gate_at_expected_rate():
    rng = np.random.default_rng(0)
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(S)
    vals = [nis(L @ rng.normal(size=2), S) for _ in range(20_000)]
    rate = np.mean(np.array(vals) > nis_threshold(2, 0.01))
    assert rate == pytest.approx(0.01, abs=0.003)
