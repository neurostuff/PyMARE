"""Meta-regression estimator classes."""

from abc import ABCMeta, abstractmethod
from inspect import getfullargspec

import numpy as np
from scipy.optimize import minimize

from ..results import MetaRegressionResults, BayesianMetaRegressionResults


class BaseEstimator(metaclass=ABCMeta):

    # default results container
    _result_cls = MetaRegressionResults

    @abstractmethod
    def _fit(self):
        # Subclasses must implement _fit() method that directly takes arrays.
        # The following named arguments are allowed, and will be automatically
        # extracted from the Dataset instance:
        # * y (estimates)
        # * v (variances)
        # * n (sample_sizes)
        # * X (predictors)
        pass

    def accepts_dataset(self, dataset):
        """ Returns whether current class can fit the passed Dataset.

        Args:
            dataset (Dataset): A Dataset instance

        Returns:
            A boolean.
        """
        args = getfullargspec(self._fit)[0][1:]
        for name in args:
            if getattr(dataset, name) is None:
                return False
        return True

    def fit(self, dataset):
        kwargs = {}
        spec = getfullargspec(self._fit)
        n_kw = len(spec.defaults) if spec.defaults else 0
        n_args = len(spec.args) - n_kw - 1
        for i, name in enumerate(spec.args[1:]):
            if i >= n_args:
                kwargs[name] = getattr(dataset, name, spec.defaults[i-n_args])
            else:
                kwargs[name] = getattr(dataset, name)

        results = self._fit(**kwargs)
        return self._result_cls(results, dataset, self)


class WeightedLeastSquares(BaseEstimator):
    """ Weighted least-squares meta-regression.

    Provides the weighted least-squares estimate of the fixed effects given
    known/assumed between-study variance tau^2. When tau^2 = 0 (default), the
    model is the standard inverse-weighted fixed-effects meta-regression.

    Args:
        tau2 (float, optional): Assumed/known value of tau^2. Must be >= 0.
            Defaults to 0.
    """
    def __init__(self, tau2=0.):
        self.tau2 = tau2

    def _fit(self, y, v, X):
        w = 1. / (v + self.tau2)
        precision = np.linalg.pinv((X * w).T.dot(X))
        beta = (precision.dot(X.T) * w.T).dot(y).ravel()
        return {'beta': beta, 'tau2': self.tau2}


class DerSimonianLaird(BaseEstimator):
    """ DerSimonian-Laird meta-regression estimator.

    Estimates the between-subject variance tau^2 using the DerSimonian-Laird
    (1986) method-of-moments approach.

    References:
        DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
        Controlled clinical trials, 7(3), 177-188.
    """
    def _fit(self, y, v, X):
        k, p = X.shape
        beta_wls = WeightedLeastSquares(0.)._fit(y, v, X)['beta'][:, None]
        # Cochrane's Q
        w = 1. / v
        w_sum = w.sum()
        Q = (w * (y - X.dot(beta_wls)) ** 2)
        Q = Q.sum()
        # D-L estimate of tau^2
        precision = np.linalg.pinv((X * w).T.dot(X))
        A = w_sum - np.trace(X.dot(precision.dot(X.T) * (w**2).T))
        tau_dl = np.max([0., (Q - k + p) / A])
        # Re-estimate beta with tau^2 estimate
        beta_dl = WeightedLeastSquares(tau_dl)._fit(y, v, X)['beta'].ravel()
        return {'beta': beta_dl, 'tau2': tau_dl}


class LikelihoodBased(BaseEstimator):
    """ Likelihood-based meta-regression estimator.

    Iteratively estimates the between-subject variance tau^2 and fixed effects
    betas using the specified likelihood-based estimator (ML or REML).

    Args:
        method (str, optional): The estimation method to use. Either 'ML' (for
            maximum-likelihood) or 'REML' (restricted maximum-likelihood).
            Defaults to 'ML'.
        beta (array, optional): Initial beta values to use in optimization. If
            None (default), the DerSimonian-Laird estimate is used.
        tau2 (float, optional): Initial tau^2 value to use in optimization. If
            None (default), the DerSimonian-Laird estimate is used.
        kwargs (dict, optional): Keyword arguments to pass to the SciPy
            minimizer.

    Notes:
        The ML and REML solutions are obtained via SciPy's scalar function
        minimizer (scipy.optimize.minimize). Parameters to minimize() can be
        passed in as keyword arguments.
    """

    def __init__(self, method='ml', beta=None, tau2=None, **kwargs):
        self.method = method
        self.beta = beta
        self.tau2 = tau2
        self.kwargs = kwargs

    def _fit(self, y, v, X):
        # use D-L estimate for initial values if none provided
        if self.tau2 is None or self.beta is None:
            est_DL = DerSimonianLaird()._fit(y, v, X)
            beta = est_DL['beta'] if self.beta is None else self.beta
            tau2 = est_DL['tau2'] if self.tau2 is None else self.tau2

        ll_func = getattr(self, '_{}_nll'.format(self.method.lower()))
        if ll_func is None:
            raise ValueError("No log-likelihood function defined for method "
                             "'{}'.".format(self.method))

        theta_init = np.r_[beta.ravel(), tau2]
        res = minimize(ll_func, theta_init, (y, v, X), **self.kwargs).x
        beta, tau = res[:-1], float(res[-1])
        tau = np.max([tau, 0])
        return {'beta': beta, 'tau2': tau}

    def _ml_nll(self, theta, y, v, X):
        """ ML negative log-likelihood for meta-regression model. """
        beta, tau2 = theta[:-1, None], theta[-1]
        if tau2 < 0:
            tau2 = 0
        w = 1. / (v + tau2)
        R = y - X.dot(beta)
        return -0.5 * (np.log(w).sum() - (R * w * R).sum())

    def _reml_nll(self, theta, y, v, X):
        """ REML negative log-likelihood for meta-regression model. """
        ll_ = self._ml_nll(theta, y, v, X)
        tau2 = theta[-1]
        w = 1. / (v + tau2)
        F = (X * w).T.dot(X)
        return ll_ + 0.5 * np.log(np.linalg.det(F))


class StanMetaRegression(BaseEstimator):
    """Bayesian meta-regression estimator using Stan.

    Args:
        sampling_kwargs: Optional keyword arguments to pass on to the MCMC
            sampler (e.g., `iter` for number of iterations).

    Notes:
        For most uses, this class should be ignored in favor of the functional
        stan() estimator. The object-oriented interface is useful primarily
        when fitting the meta-regression model repeatedly to different data;
        the separation of .compile() and .fit() steps allows one to compile
        the model only once.
    """

    _result_cls = BayesianMetaRegressionResults

    def __init__(self, **sampling_kwargs):
        self.sampling_kwargs = sampling_kwargs
        self.model = None
        self.result_ = None

    def compile(self):
        """Compile the Stan model."""
        # Note: we deliberately use a centered parameterization for the
        # thetas at the moment. This is sub-optimal in terms of estimation,
        # but allows us to avoid having to add extra logic to detect and
        # handle intercepts in X.
        spec = f"""
        data {{
            int<lower=1> N;
            int<lower=1> K;
            vector[N] y;
            int<lower=1,upper=K> id[N];
            int<lower=1> C;
            matrix[K, C] X;
            vector[N] sigma;
        }}
        parameters {{
            vector[C] beta;
            vector[K] theta;
            real<lower=0> tau2;
        }}
        transformed parameters {{
            vector[N] mu;
            mu = theta[id] + X * beta;
        }}
        model {{
            y ~ normal(mu, sigma);
            theta ~ normal(0, tau2);
        }}
        """
        from pystan import StanModel
        self.model = StanModel(model_code=spec)

    def _fit(self, y, v, X, groups=None):
        """Run the Stan sampler and return results.

        Args:
            y (ndarray): 1d array of study-level estimates
            v (ndarray): 1d array of study-level variances
            X (ndarray): 1d or 2d array containing study-level predictors
                (including intercept); has dimensions K x P, where K is the
                number of studies and P is the number of predictor variables.
            groups ([int], optional): 1d array of integers identifying
                groups/clusters of observations in the y/v/X inputs. If
                provided, values must consist of integers in the range of 1..k
                (inclusive), where k is the number of distinct groups. When
                None (default), it is assumed that each observation in the
                inputs is a separate group.

        Returns:
            A StanFit4Model object (see PyStan documentation for details).

        Notes:
            This estimator supports (simple) hierarchical models. When multiple
            estimates are available for at least one of the studies in `y`, the
            `groups` argument can be used to specify the nesting structure
            (i.e., which rows in `y`, `v`, and `X` belong to each study).
        """
        if self.model is None:
            self.compile()

        N = y.shape[0]
        groups = groups or np.arange(1, N + 1, dtype=int)
        K = len(np.unique(groups))

        data = {
            "K": K,
            "N": N,
            'id': groups,
            'C': X.shape[1],
            'X': X,
            'y': y.ravel(),
            'sigma': v.ravel()
        }

        self.result_ = self.model.sampling(data=data, **self.sampling_kwargs)
        return self.result_
