import numpy as np
from scipy.stats import chi2


def nis(nu, S):
    """Normalised innovation squared, nu^T S^-1 nu."""
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    return float(nu @ np.linalg.solve(S, nu))


def nis_threshold(dim, p_false_alarm=0.01):
    """Chi-square gate for one NIS value with `dim` degrees of freedom."""
    return chi2.ppf(1.0 - p_false_alarm, dim)
