"""Core classes and functions."""

from functools import partial

import numpy as np
import pandas as pd

from .estimators import (WeightedLeastSquares, DerSimonianLaird,
                         VarianceBasedLikelihoodEstimator,
                         SampleSizeBasedLikelihoodEstimator,
                         StanMetaRegression, Hedges)
from .stats import ensure_2d


class Dataset:
    """Container for input data and arguments to estimators.

    Args:
        y (array-like): 1d array of study-level estimates with length K
        v (array-like, optional): 1d array of study-level variances
            with length K
        X (array-like, optional): 1d or 2d array containing
            study-level predictors (or covariates); has dimensions K x P
        n (array-like, optional): 1d array of study-level sample
            sizes (length K)
        X_names ([str], optional): List of length P containing the names of the
            predictors
        add_intercept (bool, optional): If True, an intercept column is
            automatically added to the predictor matrix. If False, the
            predictors matrix is passed as-is to estimators.
        kwargs (dict, optional): Keyword arguments to pass onto estimators
    """
    def __init__(self, y, v=None, X=None, n=None, X_names=None,
                 add_intercept=True, **kwargs):
        self.y = ensure_2d(y)
        self.v = ensure_2d(v)
        self.n = ensure_2d(n)
        self.kwargs = kwargs
        X, names = self._get_predictors(X, X_names, add_intercept)
        self.X = X
        self.X_names = names

    def __getattr__(self, key):
        # Provide convenient access to stored kwargs.
        if key in self.kwargs:
            return self.kwargs[key]
        raise AttributeError("{} object has no attribute {}".format(
                                self.__class__.__name__, key))

    def _get_predictors(self, X, names, add_intercept):
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


def meta_regression(y, v=None, X=None, n=None, X_names=None, add_intercept=True,
                    method='ML', ci_method='QP', alpha=0.05, **kwargs):
    """Fits the standard meta-regression/meta-analysis model to provided data.

    Args:
        y (array-like): 1d array of study-level estimates with length K
        v (array-like, optional): 1d array of study-level variances
            with length K
        X (array-like, optional): 1d or 2d array containing
            study-level predictors (or covariates); has dimensions K x P. If
            omitted, add_intercept must be True.
        n (array-like, optional): 1d array of study-level sample
            sizes (length K)
        X_names ([str], optional): List of length P containing the names of the
            predictors (i.e., columns of X)
        add_intercept (bool, optional): If True, an intercept column is
            automatically added to the predictor matrix. If False, the
            predictors matrix is passed as-is to estimators.
        method (str, optional): Name of estimation method. Defaults to 'ML'.
            Supported estimators include:
                * 'ML': Maximum-likelihood estimator
                * 'REML': Restricted maximum-likelihood estimator
                * 'DL': DerSimonian-Laird estimator
                * 'HE': Hedges estimator
                * 'WLS' or 'FE': Weighted least squares (fixed effects only)
                * 'Stan': Full Bayesian MCMC estimation via Stan
        ci_method (str, optional): Estimation method to use when computing
            uncertainty estimates. Currently only 'QP' is supported. Defaults
            to 'QP'. Ignored if method == 'Stan'.
        alpha (float, optional): Desired alpha level (CIs will have 1 - alpha
            coverage). Defaults to 0.05.
        kwargs: Optional keyword arguments to pass onto the chosen estimator.

    Returns:
        A MetaRegressionResults or BayesianMetaRegressionResults instance,
        depending on the specified method ('Stan' will return the latter; all
        other methods return the former).
    """
    dataset = Dataset(y, v, X, n, X_names, add_intercept)

    method = method.lower()

    if method in ['ml', 'reml']:
        if v is not None:
            est_cls = partial(VarianceBasedLikelihoodEstimator, method=method)
        elif n is not None:
            est_cls = partial(SampleSizeBasedLikelihoodEstimator, method=method)
        else:
            raise ValueError("If method is ML or REML, one of `variances` or "
                             "`sample sizes` must be passed!")
    else:
        est_cls = {
            'dl': DerSimonianLaird,
            'wls': WeightedLeastSquares,
            'fe': WeightedLeastSquares,
            'stan': StanMetaRegression,
            'he': Hedges
        }[method]

    # Get estimates
    est = est_cls(**kwargs)
    results = est.fit(dataset)
    if hasattr(results, 'compute_stats'):
        results.compute_stats(ci_method=ci_method, alpha=alpha)
    return results
