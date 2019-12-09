from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize

from .likelihoods import meta_regression_ml_nll, meta_regression_reml_nll


class Estimator(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, dataset):
        pass

    def set_params(self, **params):
        for (key, value) in params.items():
            setattr(self, key, value)


class WeightedLeastSquares(Estimator):
    """ Weighted least-squares estimation of fixed effects. """

    def __init__(self, tau2=0):
        self.tau2 = tau2

    def fit(self, dataset):
        v, tau2, X, y = dataset.v, self.tau2, dataset.X, dataset.y
        w = 1. / (v + tau2)
        beta = (np.linalg.pinv((X.T * w).dot(X)).dot(X.T) * w).dot(y)
        return beta, None


class DerSimonianLaird(Estimator):
    """ DerSimonian-Laird meta-regression estimator. """

    def fit(self, dataset):
        y, X, v, k, p = dataset.y, dataset.X, dataset.v, dataset.k, dataset.p
        # WLS estimate of beta with tau^2 = 0
        beta_wls, _ = WeightedLeastSquares(0).fit(dataset)
        # Cochrane's Q
        w = 1. / v
        w_sum = w.sum()
        Q = (w * (y - X.dot(beta_wls)) ** 2).sum()
        # D-L estimate of tau^2
        precision = np.linalg.pinv((X.T * w).dot(X))
        A = w_sum - np.trace((precision.dot(X.T) * w**2).dot(X))
        tau_dl = np.max([0., (Q - k + p) / A])
        # Re-estimate beta with tau^2 estimate
        beta_dl, _ = WeightedLeastSquares(tau_dl).fit(dataset)
        return beta_dl, tau_dl


class LikelihoodEstimator(Estimator):

    def __init__(self, method='DL', beta=None, tau2=None, **kwargs):
        self.method = method
        self.beta = beta
        self.tau2 = tau2
        self.kwargs = kwargs
        self.ll_func = {
            'ml': meta_regression_ml_nll,
            'reml': meta_regression_reml_nll
        }[self.method]

    def fit(self, dataset):
        # use D-L estimate for initial values
        if self.tau2 is None or self.beta is None:
            _beta_dl, _tau2_dl = DerSimonianLaird().fit(dataset)
            beta = _beta_dl if self.beta is None else self.beta
            tau2 = _tau2_dl if self.beta is None else self.tau2

        theta_init = np.r_[beta, tau2]

        res = minimize(self.ll_func, theta_init, dataset, **self.kwargs).x
        beta, tau = res[:-1], float(res[-1])
        tau = np.max([tau, 0])
        return beta, tau
