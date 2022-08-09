"""Miscellaneous statistical functions."""

import numpy as np
import scipy.stats as ss
from scipy.optimize import Bounds, minimize


def weighted_least_squares(y, v, X, tau2=0.0, return_cov=False):
    """Perform 2-D weighted least squares.

    Parameters
    ----------
    y : :obj:`numpy.ndarray`
        2-d array of estimates (studies x parallel datasets)
    v : :obj:`numpy.ndarray`
        2-d array of sampling variances
    X : :obj:`numpy.ndarray`
        Fixed effect design matrix
    tau2 : :obj:`float`, optional
        tau^2 estimate to use for weights.
        Default = 0.
    return_cov : :obj:`bool`, optional
        Whether or not to return the inverse cov matrix.
        Default = False.

    Returns
    -------
    params[, cov]
        If return_cov is True, returns both fixed parameter estimates and the
        inverse covariance matrix; if False, only the parameter estimates.
    """
    w = 1.0 / (v + tau2)

    # Einsum indices: k = studies, p = predictors, i = parallel iterates
    wX = np.einsum("kp,ki->ipk", X, w)
    cov = wX.dot(X)

    # numpy >= 1.8 inverts stacked matrices along the first N - 2 dims, so we
    # can vectorize computation along the second dimension (parallel datasets)
    precision = np.linalg.pinv(cov).T

    pwX = np.einsum("ipk,qpi->iqk", wX, precision)
    beta = np.einsum("ipk,ik->ip", pwX, y.T).T

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
    """Get the CI for tau^2 via the Q-Profile method.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K,)
        1d array of study-level estimates
    v : :obj:`numpy.ndarray` of shape (K,)
        1d array of study-level variances
    X : :obj:`numpy.ndarray` of shape (K[, P])
        1d or 2d array containing study-level predictors
        (including intercept); has dimensions K x P, where K is the number
        of studies and P is the number of predictor variables.
    alpha : :obj:`float`, optional
        alpha value defining the coverage of the CIs,
        where width(CI) = 1 - alpha. Default = 0.05.

    Returns
    -------
    :obj:`dict`
        A dictionary with keys 'ci_l' and 'ci_u', corresponding to the lower
        and upper bounds of the tau^2 confidence interval, respectively.

    Notes
    -----
    Following the :footcite:t:`viechtbauer2007confidence` implementation,
    this method returns the interval that gives an equal probability mass at both tails
    (i.e., ``P(tau^2 <= lower_bound)  == P(tau^2 >= upper_bound) == alpha/2``),
    and *not* the smallest possible range of tau^2 values that provides the desired coverage.

    References
    ----------
    .. footbibliography::
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

    ub_start = 2 * DerSimonianLaird().fit(y, v, X).params_["tau2"]

    lb = minimize(lambda x: (q_gen(*args, x) - l_crit) ** 2, [0], bounds=bds).x[0]
    ub = minimize(lambda x: (q_gen(*args, x) - u_crit) ** 2, ub_start, bounds=bds).x[0]
    return {"ci_l": lb, "ci_u": ub}


def q_gen(y, v, X, tau2):
    """Calculate a generalized form of Cochran's Q-statistic.

    This version of the Q statistic is described in :footcite:t:`veroniki2016methods`.

    Parameters
    ----------
    y : :obj:`numpy.ndarray`
        1d array of study-level estimates
    v : :obj:`numpy.ndarray`
        1d array of study-level variances
    X : :obj:`numpy.ndarray`
        1d or 2d array containing study-level predictors
        (including intercept); has dimensions K x P, where K is the number
        of studies and P is the number of predictor variables.
    tau2 : :obj:`float`
        Between-study variance. Must be >= 0.

    Returns
    -------
    :obj:`float`
        A float giving the value of Cochran's Q-statistic.

    References
    ----------
    .. footbibliography::
    """
    if np.any(tau2 < 0):
        raise ValueError("Value of tau^2 must be >= 0.")

    beta = weighted_least_squares(y, v, X, tau2)
    w = 1.0 / (v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum(0)


def bonferroni(p_values):
    """Perform Bonferroni correction on p values.

    This correction is based on the one described in :footcite:t:`bonferroni1936teoria` and
    :footcite:t:`shaffer1995multiple`.

    .. versionadded:: 0.0.4

    Parameters
    ----------
    p_values : :obj:`numpy.ndarray`
        Uncorrected p values.

    Returns
    -------
    p_corr : :obj:`numpy.ndarray`
        Corrected p values.

    References
    ----------
    .. footbibliography::
    """
    p_corr = p_values * p_values.size
    p_corr[p_corr > 1] = 1
    return p_corr


def fdr(p_values, q=0.05, method="bh"):
    """Perform FDR correction on p values.

    .. versionadded:: 0.0.4

    Parameters
    ----------
    p_values : :obj:`numpy.ndarray`
        Array of p values.
    q : :obj:`float`, optional
        Alpha value. Default is 0.05.
    method : {"bh", "by"}, optional
        Method to use for correction.
        Either "bh" (Benjamini-Hochberg :footcite:p:`benjamini1995controlling`) or
        "by" (Benjamini-Yekutieli :footcite:p:`benjamini2001control`).
        Default is "bh".

    Returns
    -------
    p_adjusted : :obj:`numpy.ndarray`
        Array of adjusted p values.

    Notes
    -----
    This function is adapted from ``statsmodels``, which is licensed under a BSD-3 license.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    statsmodels.stats.multitest.fdrcorrection
    """
    sort_idx = np.argsort(p_values)
    revert_idx = np.argsort(sort_idx)
    p_sorted = p_values[sort_idx]

    n_tests = p_values.size

    # empirical cumulative density function
    ecdf = np.linspace(0, 1, n_tests + 1)[1:]
    if method == "by":
        # NOTE: I don't know what cm stands for
        cm = np.sum(1 / np.arange(1, n_tests + 1))
        ecdffactor = ecdf / cm
    else:
        ecdffactor = ecdf

    p_adjusted = p_sorted / ecdffactor
    p_adjusted = np.minimum.accumulate(p_adjusted[::-1])[::-1]
    # NOTE: Why not this?
    # p_adjusted = np.maximum.accumulate(p_adjusted)

    p_adjusted[p_adjusted > 1] = 1
    p_adjusted = p_adjusted[revert_idx]

    return p_adjusted


def var_to_ci(y, v, n):
    """Convert sampling variance to 95% CI."""
    term = 1.96 * np.sqrt(v) / np.sqrt(n)
    return y - term, y + term
