"""Miscellaneous statistical functions."""

import scipy.stats as ss
from scipy.optimize import root

from .estimators import validate_input


@validate_input
def q_profile(y, v, X, alpha=0.05):
    """Get the CI for tau^2 via the Q-Profile method (Viechtbauer, 2007).

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray): 1d or 2d array containing study-level predictors
            (including intercept); has dimensions K x P, where K is the number
            of studies and P is the number of predictor variables.
        alpha (float, optional): alpha value defining the coverage of the CIs,
            where width(CI) = 1 - alpha. Defaults to 0.05.

    Returns:
        A dictionary with keys 'ci_l' and 'ci_u', corresponding to the lower
        and upper bounds of the tau^2 confidence interval, respectively.

    Notes:
        Following the Viechtbauer implementation, this method returns the
        interval that gives an equal probability mass at both tails (i.e.,
        P(tau^2 <= lower_bound)  == P(tau^2 >= upper_bound) == alpha/2), and
        *not* the smallest possible range of tau^2 values that provides the
        desired coverage.
    """
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
    """Cochran's Q-statistic.

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray): 1d or 2d array containing study-level predictors
            (including intercept); has dimensions K x P, where K is the number
            of studies and P is the number of predictor variables.
        tau2 (float): Between-study variance. Must be >= 0.

    Returns:
        A float giving the value of Cochran's Q-statistic.
    """
    if tau2 < 0:
        raise ValueError("Value of tau^2 must be >= 0.")
    from .estimators import weighted_least_squares
    beta = weighted_least_squares(y, v, X, tau2=tau2)['beta'][:, None]
    w = 1. / (v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum()
