"""Core classes and functions."""

from inspect import getfullargspec

import numpy as np
import pandas as pd

from .estimators import (weighted_least_squares, dersimonian_laird,
                         likelihood_based)
from .results import MetaRegressionResults


def _setup_X(X, y, add_intercept):
    if X is None and not self.add_intercept:
        raise ValueError("No fixed predictors found. If no X matrix is "
                         "provided, add_intercept must be True!")
    X = pd.DataFrame(X)
    if add_intercept:
        intercept = pd.DataFrame({'intercept': np.ones(len(y))})
        X = pd.concat([intercept, X], axis=1)
    return X.values, X.columns.tolist()


class Dataset:

    def __init__(self, y, v=None, X=None, add_intercept=True):
        self.y = y
        self.v = v
        self.add_intercept = add_intercept
        self.X, self.X_names = _setup_X(X, y, add_intercept)


def meta_regression(y, v=None, X=None, method='ML', add_intercept=True,
                    tau2=None, alpha=0.05, ci_method='QP'):

    # add intercept
    X, X_names = _setup_X(X, y, add_intercept)

    def _get_X(self, X):
        if X is None and not self.add_intercept:
            raise ValueError("No fixed predictors found. If no X matrix is "
                             "provided, add_intercept must be True!")
        X = pd.DataFrame(X)
        if self.add_intercept:
            intcpt = pd.DataFrame({'intercept': np.ones(len(self.y))})
            X = pd.concat([intcpt, X], axis=1)
        return X

    method = method.lower()

    estimator = {
        'ml': likelihood_based,
        'reml': likelihood_based,
        'dl': dersimonian_laird,
        'wls': weighted_least_squares,
        'fe': weighted_least_squares,
    }[method]

    # Map available arguments onto target function's arguments
    valid_args = set(['y', 'v', 'X', 'tau2', 'method'])
    arg_names = set(getfullargspec(estimator).args) & valid_args
    local_vars = locals()
    kwargs = {name:local_vars[name] for name in arg_names}

    estimates = estimator(**kwargs)
    dataset = Dataset(y, v, X, add_intercept=False)
    results = MetaRegressionResults(dataset, **estimates)
    results.compute_stats(method=ci_method, alpha=alpha)
    return results
