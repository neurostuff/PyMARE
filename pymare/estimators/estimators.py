"""Meta-regression estimator classes."""

from functools import wraps
from inspect import getfullargspec

import numpy as np
from scipy.optimize import minimize


def validate_input(func):
    """Validates input to estimators.

    If a Dataset is passed as the first argument, maps its attributes onto
    estimator input args. Also inspects y, v, and X arguments and ensures
    they're always 2d.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):

        from ..core import Dataset

        func_args = getfullargspec(func).args

        # if Dataset is the first arg, map its attributes onto target func args
        if isinstance(args[0], Dataset):
            dataset = args[0]
            args = args[1:]
            valid_args = set(['y', 'v', 'X']) | set(dataset.kwargs.keys())
            arg_names = set(func_args) & valid_args
            dset_args = {name: getattr(dataset, name) for name in arg_names}
            # Directly passed arguments take precedence over Dataset contents
            kwargs = dict(dset_args, **kwargs)

        # move all args into kwargs
        for i, val in enumerate(args):
            kwargs[func_args[i]] = val

        # Ensure key arguments are 2d arrays
        for var in ('y', 'v', 'X'):
            if var in kwargs and kwargs[var] is not None:
                vals = np.array(kwargs[var])
                if len(vals.shape) == 1:
                    vals = vals[:, None]
                kwargs[var] = vals

        return func(**kwargs)
    return wrapped


@validate_input
def weighted_least_squares(y, v, X, tau2=0.):
    """ Weighted least-squares meta-regression.

    Provides the weighted least-squares estimate of the fixed effects given
    known/assumed between-study variance tau^2. When tau^2 = 0 (default), the
    model is the standard inverse-weighted fixed-effects meta-regression.

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray, optional): 1d or 2d array containing study-level
            predictors (or covariates); has dimensions K x P, where K is the
            number of studies and P is the number of predictor variables.
        tau2 (float, optional): Assumed/known value of tau^2. Must be >= 0.
            Defaults to 0.

    Returns:
        A dictionary with key 'beta' that maps to a 1d array of beta estimates
            (length = P).
    """
    w = 1. / (v + tau2)
    precision = np.linalg.pinv((X * w).T.dot(X))
    beta = (precision.dot(X.T) * w.T).dot(y).ravel()
    return {'beta': beta, 'tau2': tau2}


@validate_input
def dersimonian_laird(y, v, X):
    """ DerSimonian-Laird meta-regression estimator.

    Estimates the between-subject variance tau^2 using the DerSimonian-Laird
    (1986) method-of-moments approach.

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray, optional): 1d or 2d array containing study-level
            predictors (or covariates); has dimensions K x P, where K is the
            number of studies and P is the number of predictor variables.

    Returns:
        A dictionary with keys 'beta' and 'tau2' that map respectively to a
            1d array of beta estimates (length = P) and a float for the tau^2
            estimate.

    References:
        DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
        Controlled clinical trials, 7(3), 177-188.
    """
    k, p = X.shape
    beta_wls = weighted_least_squares(y, v, X, 0)['beta'][:, None]
    # Cochrane's Q
    w = 1. / v
    w_sum = w.sum()
    Q = (w * (y - X.dot(beta_wls)) ** 2)
    Q = Q.sum()
    # D-L estimate of tau^2
    precision = np.linalg.pinv((X * w).T.dot(X))
    A = w_sum - np.trace(X.dot(precision.dot(X.T) * (w**2).T))
    tau_dl = np.max([0., (Q - k + p) / A])
    # Re-estimate beta with tau^2 estimate
    beta_dl = weighted_least_squares(y, v, X, tau_dl)['beta'].ravel()
    return {'beta': beta_dl, 'tau2': tau_dl}


@validate_input
def likelihood_based(y, v, X, method='ml', beta=None, tau2=None, **kwargs):
    """ Likelihood-based meta-regression estimator.

    Iteratively estimates the between-subject variance tau^2 and fixed effects
    betas using the specified likelihood-based estimator (ML or REML).

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray, optional): 1d or 2d array containing study-level
            predictors (or covariates); has dimensions K x P, where K is the
            number of studies and P is the number of predictor variables.
        method (str, optional): The estimation method to use. Either 'ML' (for
            maximum-likelihood) or 'REML' (restricted maximum-likelihood).
            Defaults to 'ML'.
        beta (array, optional): Initial beta values to use in optimization. If
            None (default), the DerSimonian-Laird estimate is used.
        tau^2 (float, optional): Initial tau^2 value to use in optimization. If
            None (default), the DerSimonian-Laird estimate is used.
        kwargs (dict, optional): Keyword arguments to pass to the SciPy
            minimizer.

    Returns:
        A dictionary with keys 'beta' and 'tau2' that map respectively to a
            1d array of beta estimates (length = P) and a float for the tau^2
            estimate.

    Notes:
        The ML and REML solutions are obtained via SciPy's scalar function
        minimizer (scipy.optimize.minimize). Parameters to minimize() can be
        passed in as keyword arguments to the present function.
    """
    # use D-L estimate for initial values if none provided
    if tau2 is None or beta is None:
        est_DL = dersimonian_laird(y, v, X)
        beta = est_DL['beta'] if beta is None else beta
        tau2 = est_DL['tau2'] if tau2 is None else tau2

    ll_func = globals().get('_{}_nll'.format(method.lower()))
    if ll_func is None:
        raise ValueError("No log-likelihood function defined for method '{}'."
                         .format(method))

    theta_init = np.r_[beta.ravel(), tau2]
    res = minimize(ll_func, theta_init, (y, v, X), **kwargs).x
    beta, tau = res[:-1], float(res[-1])
    tau = np.max([tau, 0])
    return {'beta': beta, 'tau2': tau}


def _ml_nll(theta, y, v, X):
    """ ML negative log-likelihood for meta-regression model. """
    beta, tau2 = theta[:-1, None], theta[-1]
    if tau2 < 0:
        tau2 = 0
    w = 1. / (v + tau2)
    R = y - X.dot(beta)
    ll = 0.5 * (np.log(w).sum() - (R * w * R).sum())
    return -ll


def _reml_nll(theta, y, v, X):
    """ REML negative log-likelihood for meta-regression model. """
    ll_ = _ml_nll(theta, y, v, X)
    tau2 = theta[-1]
    w = 1. / (v + tau2)
    F = (X * w).T.dot(X)
    return ll_ + 0.5 * np.log(np.linalg.det(F))
