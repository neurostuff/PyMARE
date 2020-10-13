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
        y (array-like, str): 1d array of study-level estimates with length K,
            or the name of the column in data containing the y values.
        v (array-like, str, optional): 1d array of study-level variances with
            length K, or the name of the column in data containing v values.
        X (array-like, list, optional): 1d or 2d array containing study-level
            predictors (dimensions K x P), or a list of strings giving the
            names of the columns in data containing the X values.
        n (array-like, str, optional): 1d array of study-level sample sizes
            (length K), or the name of the corresponding column in data.
        data (pandas.DataFrame, optional): A pandas DataFrame containing y, v,
            X, and/or n values. By default, columns are expected to have the
            same names as arguments (e.g., the y values will be expected in the
            'y' column). This can be modified by passing strings giving column
            names to any of the y, v, X, or n arguments.
        X_names ([str], optional): List of length P containing the names of the
            predictors. Ignored if data is provided (use X to specify columns).
        add_intercept (bool, optional): If True, an intercept column is
            automatically added to the predictor matrix. If False, the
            predictors matrix is passed as-is to estimators.
    """
    def __init__(self, y=None, v=None, X=None, n=None, data=None, X_names=None,
                 add_intercept=True):

        if y is None and data is None:
            raise ValueError("If no y values are provided, a pandas DataFrame "
                             "containing a 'y' column must be passed to the "
                             "data argument.")

        # Extract columns from DataFrame
        if data is not None:
            y = data.loc[:, y or 'y'].values
            v = data.loc[:, v or 'v'].values
            X_names = X or 'X'
            X = data.loc[:, X_names].values

        self.y = ensure_2d(y)
        self.v = ensure_2d(v)
        self.n = ensure_2d(n)
        X, names = self._get_predictors(X, X_names, add_intercept)
        self.X = X
        self.X_names = names

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


def meta_regression(y=None, v=None, X=None, n=None, data=None, X_names=None,
                    add_intercept=True, method='ML', ci_method='QP',
                    alpha=0.05, **kwargs):
    """Fits the standard meta-regression/meta-analysis model to provided data.

    Args:
        y (array-like, str): 1d array of study-level estimates with length K,
            or the name of the column in data containing the y values.
        v (array-like, str, optional): 1d array of study-level variances with
            length K, or the name of the column in data containing v values.
        X (array-like, list, optional): 1d or 2d array containing study-level
            predictors (dimensions K x P), or a list of strings giving the
            names of the columns in data containing the X values.
        n (array-like, str, optional): 1d array of study-level sample sizes
            (length K), or the name of the corresponding column in data.
        data (pandas.DataFrame, pymare.Dataset, optional): If a Dataset
            instance is passed, the y, v, X, n and associated arguments are
            ignored, and data is passed directly to the selected estimator.
            If a pandas DataFrame, y, v, X and/or n values are taken from the
            DF columns. By default, columns are expected to have the same names
            as arguments (e.g., the y values will be expected in the 'y'
            column). This can be modified by passing strings giving column
            names to any of the y, v, X, or n arguments.
        X_names ([str], optional): List of length P containing the names of the
            predictors. Ignored if data is provided (use X to specify columns).
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
    # if data is None or not isinstance(data, Dataset):
    if data is None or not data.__class__.__name__ == 'Dataset':
        data = Dataset(y, v, X, n, data, X_names, add_intercept)

    method = method.lower()

    if method in ['ml', 'reml']:
        if v is not None:
            est_cls = partial(VarianceBasedLikelihoodEstimator, method=method)
        elif n is not None:
            est_cls = partial(SampleSizeBasedLikelihoodEstimator, method=method)
        else:
            raise ValueError("If method is ML or REML, one of `v` or `n` must "
                             "be passed!")
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
    est.fit_dataset(data)
    return est.summary()
