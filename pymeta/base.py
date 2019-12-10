import numpy as np
import scipy as sp
import pandas as pd

from .estimators import (WeightedLeastSquares, DerSimonianLaird,
                         MLMetaRegression, REMLMetaRegression)


class Dataset:

    def __init__(self, y, v, X=None, add_intercept=True):
        self.y = y
        self.v = v
        self.add_intercept = add_intercept

        X = self._get_X(X)
        self.X = X.values
        self.X_names = X.columns.tolist()

        # Precompute commonly used quantities
        self.k = len(y)
        self.p = self.X.shape[1]

    def _get_X(self, X):
        if X is None and not self.add_intercept:
            raise ValueError("No fixed predictors found. If no X matrix is "
                             "provided, add_intercept must be True!")
        X = pd.DataFrame(X)
        if self.add_intercept:
            intcpt = pd.DataFrame({'intercept': np.ones(len(self.y))})
            X = pd.concat([intcpt, X], axis=1)
        return X


def meta_regression(y, v, X=None, method='ML', beta=None, tau2=None,
                    add_intercept=True, alpha=0.05, ci_method='QP',
                    **optim_kwargs):

    dataset = Dataset(y, v, X, add_intercept)

    method = method.lower()

    # Optimization-based estimation methods
    if method in ['ml', 'reml']:
        EstimatorClass = {
            'ml': MLMetaRegression,
            'reml': REMLMetaRegression
        }[method]
        est = EstimatorClass(beta, tau2, **optim_kwargs)
    # Analytical estimation methods
    else:
        EstimatorClass = {
            'dl': DerSimonianLaird
        }[method]
        est = EstimatorClass()

    return est.fit(dataset)
