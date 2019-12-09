import numpy as np
import scipy as sp

from .estimators import WeightedLeastSquares, DerSimonianLaird
from .likelihoods import meta_regression_ml_nll, meta_regression_reml_nll


class Dataset:

    def __init__(self, y, v, X=None, add_intercept=True):
        self.y = y
        self.v = v
        self.add_intercept = add_intercept
        self.X = self._get_X(X)

        # Precompute commonly used quantities
        self.k = len(y)
        self.p = self.X.shape[1]

    def _get_X(self, X):
        if self.add_intercept:
            intercept = np.ones((len(y), 1))
            X = intercept if X is None else np.c_[intercept, X]
        if X is None:
            raise ValueError("No fixed predictors found. If no X matrix is "
                            "provided, add_intercept must be True!")
        return X


def meta_regression(y, v, X=None, method='ML', beta=None, tau2=None,
                    add_intercept=True, **optim_kwargs):

    dataset = Dataset(y, v, X, add_intercept)

    method = method.lower()

    # Optimization-based estimation methods
    if method in ['ml', 'reml']:

        # use D-L estimate for initial values
        if tau2 is None or beta is None:
            _beta_dl, _tau2_dl = DerSimonianLaird().fit(dataset)
            if beta is None:
                beta = _beta_dl
            if tau2 is None:
                tau2 = _tau2_dl

        theta_init = np.r_[beta, tau2]

        ll_func = {
            'ml': meta_regression_ml_nll,
            'reml': meta_regression_reml_nll
        }[method]

        res = sp.optimize.minimize(ll_func, theta_init, dataset,
                                   **optim_kwargs).x
        beta, tau = res[:-1], float(res[-1])
        tau = np.max([tau, 0])

    # Analytical estimation methods
    else:
        Estimator = {
            'dl': DerSimonianLaird
        }[method]
        est = Estimator()
        beta, tau = est.fit(dataset)

    return beta, tau
