"""Miscellaneous statistical functions."""

import scipy.stats as ss
from scipy.optimize import root

from .estimators import validate_input


@validate_input
def q_profile(y, v, X, alpha=0.05):
    """Get tau^2 CIs via the Q-Profile method (Viechtbauer, 2007)."""
    k, p = X.shape
    df = k - p
    l_crit = ss.chi2.ppf(1 - alpha / 2, df)
    u_crit = ss.chi2.ppf(alpha / 2, df)
    args = (y, v, X)
    lb = root(lambda x: (q_gen(*args, x) - l_crit)**2, 0).x[0]
    ub = root(lambda x: (q_gen(*args, x) - u_crit)**2, 100).x[0]
    return {'ci_l': lb, 'ci_u': ub}


@validate_input
def q_gen(y, v, X, tau2):
    from .estimators import weighted_least_squares
    beta = weighted_least_squares(y, v, X, tau2=tau2)['beta'][:, None]
    w = 1. / (v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum()
