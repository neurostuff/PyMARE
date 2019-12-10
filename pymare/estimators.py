from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import minimize

from .results import MetaRegressionResults


class Estimator(metaclass=ABCMeta):

    def fit(self, dataset):
        beta, tau2 = self._fit(dataset)
        return MetaRegressionResults(self, dataset, beta, tau2)

    @abstractmethod
    def _fit(self, dataset):
        pass

    def set_params(self, **params):
        for (key, value) in params.items():
            setattr(self, key, value)


class WeightedLeastSquares(Estimator):
    """ Weighted least-squares estimation of fixed effects. """

    def __init__(self, tau2=0):
        self.tau2 = tau2

    def _fit(self, dataset):
        v, tau2, X, y = dataset.v, self.tau2, dataset.X, dataset.y
        w = 1. / (v + tau2)
        beta = (np.linalg.pinv((X.T * w).dot(X)).dot(X.T) * w).dot(y)
        return beta, self.tau2


class DerSimonianLaird(Estimator):
    """ DerSimonian-Laird meta-regression estimator. """

    def _fit(self, dataset):
        y, X, v, k, p = dataset.y, dataset.X, dataset.v, dataset.k, dataset.p
        # WLS estimate of beta with tau^2 = 0
        beta_wls = WeightedLeastSquares(0).fit(dataset).beta['est']
        # Cochrane's Q
        w = 1. / v
        w_sum = w.sum()
        Q = (w * (y - X.dot(beta_wls)) ** 2).sum()
        # D-L estimate of tau^2
        precision = np.linalg.pinv((X.T * w).dot(X))
        A = w_sum - np.trace((precision.dot(X.T) * w**2).dot(X))
        tau_dl = np.max([0., (Q - k + p) / A])
        # Re-estimate beta with tau^2 estimate
        beta_dl = WeightedLeastSquares(tau_dl).fit(dataset).beta['est']
        return beta_dl, tau_dl


class LikelihoodEstimator(Estimator):

    def __init__(self, **kwargs):
        self.beta = None
        self.tau2 = None
        self.kwargs = kwargs

    @staticmethod
    @abstractmethod
    def nll(theta, dataset):
        """Negative log-likelihood function."""
        pass

    def _fit(self, dataset):
        # use D-L estimate for initial values
        if self.tau2 is None or self.beta is None:
            results = DerSimonianLaird().fit(dataset)
            beta = results.beta['est'] if self.beta is None else self.beta
            tau2 = results.tau2['est'] if self.beta is None else self.tau2

        theta_init = np.r_[beta, tau2]

        res = minimize(self.nll, theta_init, dataset, **self.kwargs).x
        beta, tau = res[:-1], float(res[-1])
        tau = np.max([tau, 0])
        return beta, tau


class MLMetaRegression(LikelihoodEstimator):

    @staticmethod
    def nll(theta, dataset):
        """ ML negative log-likelihood for meta-regression model. """
        y, v, X = dataset.y, dataset.v, dataset.X
        beta, tau = theta[:-1], theta[-1]
        if tau < 0:
            tau = 0
        w = 1. / (v + tau)
        R = y - X.dot(beta)
        ll = 0.5 * np.log(w).sum() - 0.5 *(R * w * R).sum()
        return -ll


class REMLMetaRegression(LikelihoodEstimator):

    @staticmethod
    def nll(theta, dataset):
        """ REML negative log-likelihood for meta-regression model. """
        v, X, tau2 = dataset.v, dataset.X, theta[-1]
        ll_ = MLMetaRegression.nll(theta, dataset)
        w = 1. / (v + tau2)
        F = (X.T * w).dot(X)
        return ll_ + 0.5 * np.log(np.linalg.det(F))
