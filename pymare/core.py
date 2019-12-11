"""Core classes and functions."""

from inspect import getfullargspec
from collections import namedtuple

import numpy as np
import pandas as pd

from .estimators import (weighted_least_squares, dersimonian_laird,
                         likelihood_based)
from .results import MetaRegressionResults


def _setup_X(X, y, add_intercept):
    if X is None and not add_intercept:
        raise ValueError("No fixed predictors found. If no X matrix is "
                         "provided, add_intercept must be True!")
    X = pd.DataFrame(X)
    if add_intercept:
        intercept = pd.DataFrame({'intercept': np.ones(len(y))})
        X = pd.concat([intercept, X], axis=1)
    return X.values, X.columns.tolist()


# Container for input arguments we want to store in results
InputArgs = namedtuple('InputArgs', ['estimates', 'variances', 'predictors',
                                     'names', 'method'])


def meta_regression(estimates, variances=None, predictors=None,
                    add_intercept=True, method='ML', tau2=None, alpha=0.05,
                    ci_method='QP'):

    # add intercept
    predictors, names = _setup_X(predictors, estimates, add_intercept)

    method = method.lower()

    estimator = {
        'ml': likelihood_based,
        'reml': likelihood_based,
        'dl': dersimonian_laird,
        'wls': weighted_least_squares,
        'fe': weighted_least_squares,
    }[method]

    # Rename arguments to match more compact internal estimator args
    y, v, X = estimates, variances, predictors

    # Map available arguments onto target function's arguments
    valid_args = set(['y', 'v', 'X', 'tau2', 'method'])
    arg_names = set(getfullargspec(estimator).args) & valid_args
    local_vars = locals()
    kwargs = {name:local_vars[name] for name in arg_names}

    # Get estimates
    estimates = estimator(**kwargs)

    # Store key input values/arguments in results
    input_args = InputArgs(y, v, X, names, method)

    # Return results object with computed stats
    results = MetaRegressionResults(estimates, input_args, ci_method, alpha)
    results.compute_stats(method=ci_method, alpha=alpha)
    return results
