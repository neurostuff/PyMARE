from abc import ABCMeta, abstractmethod

import numpy as np


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
        return (np.linalg.pinv((X.T * w).dot(X)).dot(X.T) * w).dot(y)


class DersimonianLaird(Estimator):
    """ DerSimonian-Laird meta-regression estimator. """

    def fit(self, dataset):
        y, X, v, k, p = dataset.y, dataset.X, dataset.v, dataset.k, dataset.p
        # WLS estimate of beta with tau^2 = 0
        beta_wls = WeightedLeastSquares(0).fit(dataset)
        # Cochrane's Q
        w = 1. / v
        w_sum = w.sum()
        Q = (w * (y - X.dot(beta_wls))**2).sum()
        # D-L estimate of tau^2
        precision = np.linalg.pinv((X.T * w).dot(X))
        A = w_sum - np.trace((precision.dot(X.T) * w**2).dot(X))
        tau_dl = np.max([0., (Q - k + p) / A])
        # Re-estimate beta with tau^2 estimate
        beta_dl = WeightedLeastSquares(tau_dl).fit(dataset)
        return beta_dl, tau_dl
