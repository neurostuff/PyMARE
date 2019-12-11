"""Meta-regression estimator classes."""

import numpy as np
from scipy.optimize import minimize


def weighted_least_squares(y, v, X, tau2=0):
    """ Weighted least-squares estimation of fixed effects. """
    w = 1. / (v + tau2)
    beta = (np.linalg.pinv((X.T * w).dot(X)).dot(X.T) * w).dot(y)
    return {'beta': beta}


def dersimonian_laird(y, v, X):
    k, p = X.shape
    beta_wls = weighted_least_squares(y, v, X, 0)['beta']
    # Cochrane's Q
    w = 1. / v
    w_sum = w.sum()
    Q = (w * (y - X.dot(beta_wls)) ** 2).sum()
    # D-L estimate of tau^2
    precision = np.linalg.pinv((X.T * w).dot(X))
    A = w_sum - np.trace((precision.dot(X.T) * w**2).dot(X))
    tau_dl = np.max([0., (Q - k + p) / A])
    # Re-estimate beta with tau^2 estimate
    beta_dl = weighted_least_squares(y, v, X, tau_dl)['beta']
    return {'beta': beta_dl, 'tau2': tau_dl}


def likelihood_based(y, v, X, method='ml', beta=None, tau2=None, **kwargs):
    # use D-L estimate for initial values if none provided
    if tau2 is None or beta is None:
        est_DL = dersimonian_laird(y, v, X)
        beta = est_DL['beta'] if beta is None else beta
        tau2 = est_DL['tau2'] if tau2 is None else tau2

    ll_func = globals().get('_{}_nll'.format(method.lower()))
    if ll_func is None:
        raise ValueError("No log-likelihood function defined for method '{}'."
                         .format(method))

    theta_init = np.r_[beta, tau2]
    res = minimize(ll_func, theta_init, (y, v, X), **kwargs).x
    beta, tau = res[:-1], float(res[-1])
    tau = np.max([tau, 0])
    return {'beta': beta, 'tau2': tau}


def _ml_nll(theta, y, v, X):
    """ ML negative log-likelihood for meta-regression model. """
    beta, tau2 = theta[:-1], theta[-1]
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
    F = (X.T * w).dot(X)
    return ll_ + 0.5 * np.log(np.linalg.det(F))
