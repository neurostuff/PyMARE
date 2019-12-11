"""Core classes and functions."""

from functools import partial

import numpy as np
import pandas as pd

from .estimators import (weighted_least_squares, dersimonian_laird,
                         likelihood_based)
from .results import MetaRegressionResults


class Dataset:

    def __init__(self, estimates, variances=None, predictors=None, names=None,
                 add_intercept=True, **kwargs):
        self.estimates = estimates
        self.variances = variances
        self.kwargs = kwargs
        X, n = self._setup_predictors(predictors, names, add_intercept)
        self.predictors = X
        self.names = n

    def __getattr__(self, key):
        if key in self.kwargs:
            return self.kwargs[key]
        raise AttributeError

    def _setup_predictors(self, X, names, add_intercept):
        if X is None and not add_intercept:
            raise ValueError("No fixed predictors found. If no X matrix is "
                             "provided, add_intercept must be True!")
        X = pd.DataFrame(X)
        if names is not None:
            X.columns = names
        if add_intercept:
            intercept = pd.DataFrame({'intercept': np.ones(len(self.y))})
            X = pd.concat([intercept, X], axis=1)
        return X.values, X.columns.tolist()

    @property
    def y(self):
        return self.estimates

    @property
    def v(self):
        return self.variances

    @property
    def X(self):
        return self.predictors


def meta_regression(estimates, variances=None, predictors=None, names=None,
                    add_intercept=True, method='ML', ci_method='QP',
                    alpha=0.05, **kwargs):

    dataset = Dataset(estimates, variances, predictors, names, add_intercept,
                      **kwargs)

    method = method.lower()

    estimator = {
        'ml': partial(likelihood_based, method=method),
        'reml': partial(likelihood_based, method=method),
        'dl': dersimonian_laird,
        'wls': weighted_least_squares,
        'fe': weighted_least_squares,
    }[method]

    # Get estimates
    estimates = estimator(dataset, )

    # Return results object with computed stats
    results = MetaRegressionResults(estimates, dataset, ci_method, alpha)
    results.compute_stats(method=ci_method, alpha=alpha)
    return results
