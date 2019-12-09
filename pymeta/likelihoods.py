""" Log-likelihood functions for meta-regression/meta-analysis models. """

import numpy as np


def meta_regression_ml_nll(theta, y, v, X, sigma):
    """ ML negative log-likelihood for meta-regression model. """
    k = len(y)
    beta, tau = theta[:-1], theta[-1]
    if tau < 0:
        tau = 0
    W = np.linalg.inv(sigma + tau * np.eye(k))
    R = y - X.dot(beta)
    ll = 0.5 * np.log(np.linalg.det(W)) - 0.5 * R.T.dot(W).dot(R)
    return -ll


def meta_regression_reml_nll(theta, y, v, X, sigma):
    """ REML negative log-likelihood for meta-regression model. """
    ll_ = meta_regression_ml_nll(theta, y, v, X, sigma)
    k = len(y)
    W = np.linalg.inv(sigma + theta[-1] * np.eye(k))
    F = X.T.dot(W).dot(X)
    return ll_ + 0.5 * np.log(np.linalg.det(F))
