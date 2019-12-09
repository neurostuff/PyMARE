import numpy as np
import scipy as sp

from .estimators import fixed_effects, dersimonian_laird
from .likelihoods import meta_regression_ml_nll, meta_regression_reml_nll


def meta_regression(y, v, X=None, method='ML', beta=None, tau2=None,
                    **optim_kwargs):

    intercept = np.ones((len(y), 1))
    if X is None:
        X = intercept
    else:
        X = np.c_[intercept, X]

    method = method.lower()

    # Analytical estimation methods
    if method in ['dl']:
        pass

    # Optimization-based estimation methods
    else:

        # use D-L estimate for initial values
        if tau2 is None or beta is None:
            _beta_dl, _tau2_dl = dersimonian_laird(y, v, X)
            if beta is None:
                beta = _beta_dl
            if tau2 is None:
                tau2 = _tau2_dl

        theta_init = np.r_[beta, tau2]

        ll_func = {
            'ml': meta_regression_ml_nll,
            'reml': meta_regression_reml_nll
        }[method]

        sigma = np.diag(v)
        args = (y, v, X, sigma)
        res = sp.optimize.minimize(ll_func, theta_init, args, **optim_kwargs).x
        beta, tau = res[:-1], float(res[-1])
        tau = np.max([tau, 0])

    return beta, tau
