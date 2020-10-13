"""Miscellaneous statistical functions."""

import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize, Bounds


def weighted_least_squares(y, v, X, tau2=0., return_cov=False):
    """2-D weighted least squares.

    Args:
        y (NDArray): 2-d array of estimates (studies x parallel datasets)
        v (NDArray): 2-d array of sampling variances
        X (NDArray): Fixed effect design matrix
        tau2 (float): tau^2 estimate to use for weights
        return_cov (bool): Whether or not to return the inverse cov matrix

    Returns:
        If return_cov is True, returns both fixed parameter estimates and the
        inverse covariance matrix; if False, only the parameter estimates.
    """

    w = 1. / (v + tau2)

    # Einsum indices: k = studies, p = predictors, i = parallel iterates
    wX = np.einsum('kp,ki->ipk', X, w)
    cov = wX.dot(X)

    # numpy >= 1.8 inverts stacked matrices along the first N - 2 dims, so we
    # can vectorize computation along the second dimension (parallel datasets)
    precision = np.linalg.pinv(cov).T

    pwX = np.einsum('ipk,qpi->iqk', wX, precision)
    beta = np.einsum('ipk,ik->ip', pwX, y.T).T

    return (beta, precision) if return_cov else beta


def ensure_2d(arr):
    """Ensure the passed array has 2 dimensions."""
    if arr is None:
        return arr
    try:
        arr = np.array(arr)
    except:
        return arr
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


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

    References:
        Viechtbauer, W. (2007). Confidence intervals for the amount of
        heterogeneity in meta-analysis. Statistics in Medicine, 26(1), 37-52.
    """
    k, p = X.shape
    df = k - p
    l_crit = ss.chi2.ppf(1 - alpha / 2, df)
    u_crit = ss.chi2.ppf(alpha / 2, df)
    args = (ensure_2d(y), ensure_2d(v), X)
    bds = Bounds([0], [np.inf], keep_feasible=True)

    # Use the D-L estimate of tau^2 as a starting point; when using a fixed
    # value, minimize() sometimes fails to stay in bounds.
    from .estimators import DerSimonianLaird
    ub_start = 2 * DerSimonianLaird().fit(y, v, X).params_['tau2']

    lb = minimize(lambda x: (q_gen(*args, x) - l_crit)**2, [0],
                  bounds=bds).x[0]
    ub = minimize(lambda x: (q_gen(*args, x) - u_crit)**2, [ub_start],
                  bounds=bds).x[0]
    return {'ci_l': lb, 'ci_u': ub}


def q_gen(y, v, X, tau2):
    """Generalized form of Cochran's Q-statistic.

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray): 1d or 2d array containing study-level predictors
            (including intercept); has dimensions K x P, where K is the number
            of studies and P is the number of predictor variables.
        tau2 (float): Between-study variance. Must be >= 0.

    Returns:
        A float giving the value of Cochran's Q-statistic.

    References:
    Veroniki, A. A., Jackson, D., Viechtbauer, W., Bender, R., Bowden, J.,
    Knapp, G., Kuss, O., Higgins, J. P., Langan, D., & Salanti, G. (2016).
    Methods to estimate the between-study variance and its uncertainty in
    meta-analysis. Research synthesis methods, 7(1), 55–79.
    https://doi.org/10.1002/jrsm.1164
    """
    if np.any(tau2 < 0):
        raise ValueError("Value of tau^2 must be >= 0.")
    beta = weighted_least_squares(y, v, X, tau2)
    w = 1. / (v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum(0)
