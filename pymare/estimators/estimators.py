"""Meta-regression estimator classes."""

from abc import ABCMeta, abstractmethod
from inspect import getfullargspec

import numpy as np
from scipy.optimize import minimize, Bounds

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
                kwargs[name] = getattr(dataset, name, spec.defaults[i - n_args])
            else:
                kwargs[name] = getattr(dataset, name)
        self.params_ = self._fit(**kwargs)
        self.dataset_ = dataset
        return self

    def summary(self):
        if not hasattr(self, 'params_'):
            name = self.__class__.__name__
            raise ValueError("This {} instance hasn't been fitted yet. Please "
                             "call fit() before summary().".format(name))
        return self._result_cls(self.params_, self.dataset_, self)


class WeightedLeastSquares(BaseEstimator):
    """ Weighted least-squares meta-regression.

    Provides the weighted least-squares estimate of the fixed effects given
    known/assumed between-study variance tau^2. When tau^2 = 0 (default), the
    model is the standard inverse-weighted fixed-effects meta-regression.

    References:
        Brockwell, S. E., & Gordon, I. R. (2001). A comparison of statistical
        methods for meta-analysis. Statistics in Medicine, 20(6), 825–840.
        https://doi.org/10.1002/sim.650

    Args:
        tau2 (float or 1-D array, optional): Assumed/known value of tau^2. Must
            be >= 0. Defaults to 0.

    Notes:
        This estimator accepts 2-D inputs for y and v--i.e., it can produce
        estimates simultaneously for multiple independent sets of y/v values
        (use the 2nd dimension for the parallel iterates). The X matrix must be
        identical for all iterates.
    """
    def __init__(self, tau2=0.):
        self.tau2 = tau2

    def _fit(self, y, v, X):
        w = 1. / (v + self.tau2)
        # Einsum indices: k = studies, p = predictors, i = parallel iterates
        wX = np.einsum('kp,ki->ipk', X, w)
        wX_cov = wX.dot(X)
        # numpy >= 1.8 inverts stacked matrices along the first N - 2 dims
        precision = np.linalg.pinv(wX_cov)
        pWX = np.einsum('ipk,ipq->iqk', wX, precision)
        beta = np.einsum('ipk,ik->ip', pWX, y.T).T
        return {'beta': beta, 'tau2': self.tau2}


class DerSimonianLaird(BaseEstimator):
    """ DerSimonian-Laird meta-regression estimator.

    Estimates the between-subject variance tau^2 using the DerSimonian-Laird
    (1986) method-of-moments approach.

    References:
        DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
        Controlled clinical trials, 7(3), 177-188.
        Kosmidis, I., Guolo, A., & Varin, C. (2017). Improving the accuracy of
        likelihood-based inference in meta-analysis and meta-regression.
        Biometrika, 104(2), 489–496. https://doi.org/10.1093/biomet/asx001

    Notes:
        This estimator accepts 2-D inputs for y and v--i.e., it can produce
        estimates simultaneously for multiple independent sets of y/v values
        (use the 2nd dimension for the parallel iterates). The X matrix must be
        identical for all iterates.
    """
    def _fit(self, y, v, X):
        k, p = X.shape

        # Estimate initial betas with WLS
        w = 1. / v
        
        # Einsum indices: k = studies, p = predictors, i = parallel iterates
        wX = np.einsum('kp,ki->ipk', X, w)
        wX_cov = wX.dot(X)
        # numpy >= 1.8 inverts stacked matrices along the first N - 2 dims
        precision = np.linalg.pinv(wX_cov)
        pwX = np.einsum('ipk,ipq->iqk', wX, precision)
        beta_wls = np.einsum('ipk,ik->ip', pwX, y.T).T

        # Cochrane's Q
        w_sum = w.sum(0)
        Q = (w * (y - X.dot(beta_wls)) ** 2).sum(0)

        # D-L estimate of tau^2
        Xw2 = np.einsum('kp,ki->ipk', X, w**2)
        pXw2 = np.einsum('ipk,ipq->iqk', Xw2, precision)
        A = w_sum - np.trace(pXw2.dot(X), axis1=1, axis2=2)
        tau_dl = (Q - (k - p)) / A
        tau_dl = np.maximum(0., tau_dl)

        # Re-estimate beta with tau^2 estimate
        beta_dl = WeightedLeastSquares(tau_dl)._fit(y, v, X)['beta'].ravel()
        return {'beta': beta_dl, 'tau2': tau_dl}


class Hedges(BaseEstimator):
    """ Hedges meta-regression estimator.

    Estimates the between-subject variance tau^2 using the Hedges & Olkin
    (1985) approach.

    References:
        Hedges LV, Olkin I. 1985. Statistical Methods for Meta‐Analysis.
    """
    def _fit(self, y, v, X):
        k, p = X.shape
        precision = np.linalg.pinv(X.T.dot(X))
        beta = precision.dot(X.T).dot(y).ravel()
        mse = ((y.ravel() - X.dot(beta)) ** 2).sum() / (k - p)
        tau_ho = mse - v.sum() / k
        tau_ho = max([0, tau_ho])
        # Estimate beta with tau^2 estimate
        beta_ho = WeightedLeastSquares(tau_ho)._fit(y, v, X)['beta'].ravel()
        return {'beta': beta_ho, 'tau2': tau_ho}


class VarianceBasedLikelihoodEstimator(BaseEstimator):
    """ Likelihood-based estimator for estimates with known variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Args:
        method (str, optional): The estimation method to use. Either 'ML' (for
            maximum-likelihood) or 'REML' (restricted maximum-likelihood).
            Defaults to 'ML'.
        kwargs (dict, optional): Keyword arguments to pass to the SciPy
            minimizer.

    Notes:
        The ML and REML solutions are obtained via SciPy's scalar function
        minimizer (scipy.optimize.minimize). Parameters to minimize() can be
        passed in as keyword arguments.

    References:
        DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
        Controlled clinical trials, 7(3), 177-188.
        Kosmidis, I., Guolo, A., & Varin, C. (2017). Improving the accuracy of
        likelihood-based inference in meta-analysis and meta-regression.
        Biometrika, 104(2), 489–496. https://doi.org/10.1093/biomet/asx001
    """

    def __init__(self, method='ml', **kwargs):
        nll_func = getattr(self, '_{}_nll'.format(method.lower()))
        if nll_func is None:
            raise ValueError("No log-likelihood function defined for method "
                             "'{}'.".format(method))
        self._nll_func = nll_func
        self.kwargs = kwargs

    def _fit(self, y, v, X):
        # use D-L estimate for initial values
        est_DL = DerSimonianLaird()._fit(y, v, X)
        beta = est_DL['beta']
        tau2 = est_DL['tau2']

        theta_init = np.r_[beta.ravel(), tau2]

        lb = np.ones(len(theta_init)) * -np.inf
        ub = -lb
        lb[-1] = 0.  # bound only the variance
        bds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(self._nll_func, theta_init, (y, v, X), bounds=bds,
                       **self.kwargs).x
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


class SampleSizeBasedLikelihoodEstimator(BaseEstimator):
    """ Likelihood-based estimator for estimates with known sample sizes but
    unknown variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Args:
        method (str, optional): The estimation method to use. Either 'ML' (for
            maximum-likelihood) or 'REML' (restricted maximum-likelihood).
            Defaults to 'ML'.
        beta (array, optional): Initial beta values to use in optimization. If
            None (default), uses the weighted least squares estimate.
        tau2 (float, optional): Initial tau^2 value to use in optimization.
            Defaults to 0.
        kwargs (dict, optional): Keyword arguments to pass to the SciPy
            minimizer.

    Notes:
        The ML and REML solutions are obtained via SciPy's scalar function
        minimizer (scipy.optimize.minimize). Parameters to minimize() can be
        passed in as keyword arguments.

    References:
        Sangnawakij, P., Böhning, D., Niwitpong, S. A., Adams, S., Stanton, M., 
        & Holling, H. (2019). Meta-analysis without study-specific variance 
        information: Heterogeneity case. Statistical Methods in Medical Research, 
        28(1), 196–210. https://doi.org/10.1177/0962280217718867    
    """

    def __init__(self, method='ml', **kwargs):
        nll_func = getattr(self, '_{}_nll'.format(method.lower()))
        if nll_func is None:
            raise ValueError("No log-likelihood function defined for method "
                             "'{}'.".format(method))
        self._nll_func = nll_func
        self.kwargs = kwargs

    def _fit(self, y, n, X):
        if n.std() < np.sqrt(np.finfo(float).eps):
            raise ValueError("Sample size-based likelihood estimator cannot "
                             "work with all-equal sample sizes.")
        if n.std() < n.mean() / 10:
            raise Warning("Sample sizes are too close, sample size-based "
                          "likelihood estimator may fail.")
        # set tau^2 to 0 and compute starting values
        tau2 = 0.
        k, p = X.shape
        beta = WeightedLeastSquares(tau2=tau2)._fit(y, n, X)['beta'][:, None]
        sigma = ((y - X.dot(beta))**2 * n).sum() / (k - p)
        theta_init = np.r_[beta.ravel(), sigma, tau2]

        lb = np.ones(len(theta_init)) * -np.inf
        ub = -lb
        lb[-2:] = 0.  # bound only the variances
        bds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(self._nll_func, theta_init, (y, n, X), bounds=bds,
                       **self.kwargs).x
        beta, sigma, tau = res[:-2], float(res[-2]), float(res[-1])
        tau = np.max([tau, 0])
        return {'beta': beta, 'sigma2': sigma, 'tau2': tau}

    def _ml_nll(self, theta, y, n, X):
        """ ML negative log-likelihood for meta-regression model. """
        beta, sigma2, tau2 = theta[:-2, None], theta[-2], theta[-1]
        if tau2 < 0:
            tau2 = 0
        if sigma2 < 0:
            sigma2 = 0
        w = 1 / (tau2 + sigma2 / n)
        R = y - X.dot(beta)
        return -0.5 * (np.log(w).sum() - (R * w * R).sum())

    def _reml_nll(self, theta, y, n, X):
        """ REML negative log-likelihood for meta-regression model. """
        ll_ = self._ml_nll(theta, y, n, X)
        sigma2, tau2 = theta[-2:]
        w = 1 / (tau2 + sigma2 / n)
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
