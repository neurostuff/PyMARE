"""Tools for representing and manipulating meta-regression results."""

import numpy as np
from scipy.optimize import root
import scipy.stats as ss


class MetaRegressionResults:

    def __init__(self, estimator, dataset, beta, tau2, ci=0.95):
        self.estimator = estimator
        self.dataset = dataset
        self.beta = beta
        self.tau2 = tau2
        self.ci = ci

    def summary(self):
        pass

    def plot(self):
        pass


def q_gen(tau2, dataset):
    from .estimators import WeightedLeastSquares
    beta = WeightedLeastSquares(tau2).fit(dataset).beta
    v, y, X = dataset.v, dataset.y, dataset.X
    w = 1. / (dataset.v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum()


def q_profile(results, alpha=0.05):
    """Get CIs for tau^2 via the Q-Profile method (e.g., Viechtbauer, 2007)."""
    dataset = results.dataset
    df = dataset.k - dataset.p
    l_crit = ss.chi2.ppf(1 - alpha / 2, df)
    u_crit = ss.chi2.ppf(alpha / 2, df)
    lb = root(lambda x: (q_gen(x, dataset) - l_crit)**2, 1).x[0]
    ub = root(lambda x: (q_gen(x, dataset) - u_crit)**2, 100).x[0]
    return lb, ub
