"""Meta-regression estimator classes."""

from abc import ABCMeta, abstractmethod
from inspect import getfullargspec
from warnings import warn

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import stats as ss
import wrapt

from ..stats import weighted_least_squares, ensure_2d
from ..results import (MetaRegressionResults, BayesianMetaRegressionResults,
                       CombinationTestResults)


@wrapt.decorator
def _loopable(wrapped, instance, args, kwargs):
    # Decorator for fit() method of Estimator classes to handle naive looping
    # over the 2nd dimension of y/v/n inputs, and reconstruction of outputs.
    n_iter = kwargs['y'].shape[1]
    if n_iter > 10:
        warn("Input contains {} parallel datasets (in 2nd dim of y and"
                " v). The selected estimator will loop over datasets"
                " naively, and this may be slow for large numbers of "
                "datasets. Consider using the DL, HE, or WLS estimators, "
                "which handle parallel datasets more efficiently."
                .format(n_iter))

    param_dicts = []
    for i in range(n_iter):
        iter_kwargs = {'X': kwargs['X']}
        iter_kwargs['y'] = kwargs['y'][:, i, None]
        if 'v' in kwargs:
            iter_kwargs['v'] = kwargs['v'][:, i, None]
        if 'n' in kwargs:
            n = kwargs['n'][:, i, None] if kwargs['n'].shape[1] > 1 else kwargs['n']
            iter_kwargs['n'] = n
        wrapped(**iter_kwargs)
        param_dicts.append(instance.params_.copy())

    params = {}
    for k in param_dicts[0]:
        concat = np.stack([pd[k].squeeze() for pd in param_dicts], axis=-1)
        params[k] = np.atleast_2d(concat)

    instance.params_ = params
    return instance


class BaseEstimator(metaclass=ABCMeta):

    # A class-level mapping from Dataset attributes to fit() arguments. Used by
    # fit_dataset() for estimators that take non-standard arguments (e.g., 'z'
    # instead of 'y'). Keys are default Dataset attribute names (e.g., 'y') and
    # values are the target arg names in the estimator class's fit() method
    # (e.g., 'z').
    _dataset_attr_map = {}

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def fit_dataset(self, dataset, *args, **kwargs):
        """ Applies the current estimator to the passed Dataset container.

        A convenience interface that wraps fit() and automatically aligns the
        variables held in a Dataset with the required arguments.

        Args:
            dataset (Dataset): A PyMARE Dataset instance holding the data.
            args, kwargs: optional positional and keyword arguments to pass
                onto the fit() method.
        """
        all_kwargs = {}
        spec = getfullargspec(self.fit)
        n_kw = len(spec.defaults) if spec.defaults else 0
        n_args = len(spec.args) - n_kw - 1

        for i, name in enumerate(spec.args[1:]):
            # Check for remapped name
            attr_name = self._dataset_attr_map.get(name, name)
            if i >= n_args:
                all_kwargs[name] = getattr(dataset, attr_name,
                                           spec.defaults[i - n_args])
            else:
                all_kwargs[name] = getattr(dataset, attr_name)

        all_kwargs.update(kwargs)
        self.fit(*args, **all_kwargs)
        self.dataset_ = dataset

        return self

    def get_v(self, dataset):
        """Get the variances, or an estimate thereof, from the given Dataset.

        Args:
            dataset (Dataset): The dataset to use to retrieve/estimate v.

        Returns:
            A 2-d NDArray.

        Notes:
            This is equivalent to directly accessing `dataset.v` when variances
            are present, but affords a way of estimating v from sample size (n)
            for any estimator that implicitly estimates a sigma^2 parameter.
        """
        if dataset.v is not None:
            return dataset.v
        # Estimate sampling variances from sigma^2 and n if available.
        if dataset.n is None:
            raise ValueError("Dataset does not contain sampling variances (v),"
                             " and no estimate of v is possible without sample"
                             " sizes (n).")
        if 'sigma2' not in self.params_:
            raise ValueError("Dataset does not contain sampling variances (v),"
                             " and no estimate of v is possible because no "
                             "sigma^2 parameter was found.")
        return self.params_['sigma2'] / dataset.n

    def summary(self):
        if not hasattr(self, 'params_'):
            name = self.__class__.__name__
            raise ValueError("This {} instance hasn't been fitted yet. Please "
                             "call fit() before summary().".format(name))
        p = self.params_
        return MetaRegressionResults(self, self.dataset_, p['fe_params'],
                                     p['inv_cov'], p['tau2'])


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
        identical for all iterates. If no v argument is passed to fit(), unit
        weights will be used, resulting in the ordinary least-squares (OLS)
        solution.
    """

    def __init__(self, tau2=0.):
        self.tau2 = tau2

    def fit(self, y, X, v=None):
        if v is None:
            v = np.ones_like(y)
        beta, inv_cov = weighted_least_squares(y, v, X, self.tau2,
                                               return_cov=True)
        self.params_ = {'fe_params': beta, 'tau2': self.tau2, 'inv_cov': inv_cov}
        return self


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

    def fit(self, y, v, X):

        y = ensure_2d(y)
        v = ensure_2d(v)

        k, p = X.shape

        # Estimate initial betas with WLS, assuming tau^2=0
        beta_wls, inv_cov = weighted_least_squares(y, v, X, return_cov=True)

        # Cochrane's Q
        w = 1. / v
        w_sum = w.sum(0)
        Q = (w * (y - X.dot(beta_wls)) ** 2).sum(0)

        # Einsum indices: k = studies, p = predictors, i = parallel iterates.
        # q is a dummy for 2nd p when p x p covariance matrix is passed.
        Xw2 = np.einsum('kp,ki->ipk', X, w**2)
        pXw2 = np.einsum('ipk,qpi->iqk', Xw2, inv_cov)
        A = w_sum - np.trace(pXw2.dot(X), axis1=1, axis2=2)
        tau_dl = (Q - (k - p)) / A
        tau_dl = np.maximum(0., tau_dl)

        # Re-estimate beta with tau^2 estimate
        beta_dl, inv_cov = weighted_least_squares(y, v, X, tau2=tau_dl,
                                                  return_cov=True)
        self.params_ = {'fe_params': beta_dl, 'tau2': tau_dl, 'inv_cov': inv_cov}
        return self


class Hedges(BaseEstimator):
    """ Hedges meta-regression estimator.

    Estimates the between-subject variance tau^2 using the Hedges & Olkin
    (1985) approach.

    References:
        Hedges LV, Olkin I. 1985. Statistical Methods for Meta‐Analysis.

    Notes:
        This estimator accepts 2-D inputs for y and v--i.e., it can produce
        estimates simultaneously for multiple independent sets of y/v values
        (use the 2nd dimension for the parallel iterates). The X matrix must be
        identical for all iterates.
    """

    def fit(self, y, v, X):
        k, p = X.shape[:2]
        _unit_v = np.ones_like(y)
        beta, inv_cov = weighted_least_squares(y, _unit_v, X, return_cov=True)
        mse = ((y - X.dot(beta)) ** 2).sum(0) / (k - p)
        tau_ho = mse - v.sum(0) / k
        tau_ho = np.maximum(0, tau_ho)
        # Estimate beta with tau^2 estimate
        beta_ho = weighted_least_squares(y, v, X, tau2=tau_ho)
        self.params_ = {'fe_params': beta_ho, 'tau2': tau_ho, 'inv_cov': inv_cov}
        return self


class VarianceBasedLikelihoodEstimator(BaseEstimator):
    """ Likelihood-based estimator for estimates with known variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    coefficients using the specified likelihood-based estimator (ML or REML).

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

    @_loopable
    def fit(self, y, v, X):
        # use D-L estimate for initial values
        est_DL = DerSimonianLaird().fit(y, v, X).params_
        beta = est_DL['fe_params']
        tau2 = est_DL['tau2']

        theta_init = np.r_[beta.ravel(), tau2]

        lb = np.ones(len(theta_init)) * -np.inf
        ub = -lb
        lb[-1] = 0.  # bound only the variance
        bds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(self._nll_func, theta_init, (y, v, X), bounds=bds,
                       **self.kwargs)
        beta, tau = res.x[:-1], float(res.x[-1])
        tau = np.max([tau, 0])
        _, inv_cov = weighted_least_squares(y, v, X, tau, True)
        self.params_ = {'fe_params': beta[:, None], 'tau2': tau, 'inv_cov': inv_cov}
        return self

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
    unknown sampling variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Args:
        method (str, optional): The estimation method to use. Either 'ML' (for
            maximum-likelihood) or 'REML' (restricted maximum-likelihood).
            Defaults to 'ML'.
        kwargs (dict, optional): Keyword arguments to pass to the SciPy
            minimizer.

    Notes:
        Homogeneity of sigma^2 across studies is assumed. The ML and REML
        solutions are obtained via SciPy's scalar function minimizer
        (scipy.optimize.minimize). Parameters to minimize() can be passed in as
        keyword arguments.

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

    @_loopable
    def fit(self, y, n, X):
        if n.std() < np.sqrt(np.finfo(float).eps):
            raise ValueError("Sample size-based likelihood estimator cannot "
                             "work with all-equal sample sizes.")
        if n.std() < n.mean() / 10:
            raise Warning("Sample sizes are too close, sample size-based "
                          "likelihood estimator may fail.")
        # set tau^2 to 0 and compute starting values
        tau2 = 0.
        k, p = X.shape
        beta = weighted_least_squares(y, n, X, tau2)
        sigma = ((y - X.dot(beta))**2 * n).sum() / (k - p)
        theta_init = np.r_[beta.ravel(), sigma, tau2]

        lb = np.ones(len(theta_init)) * -np.inf
        ub = -lb
        lb[-2:] = 0.  # bound only the variances
        bds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(self._nll_func, theta_init, (y, n, X), bounds=bds,
                       **self.kwargs)
        beta, sigma, tau = res.x[:-2], float(res.x[-2]), float(res.x[-1])
        tau = np.max([tau, 0])
        _, inv_cov = weighted_least_squares(y, sigma / n, X, tau, True)
        self.params_ = {
            'fe_params': beta[:, None],
            'sigma2': np.array(sigma),
            'tau2': tau,
            'inv_cov': inv_cov
        }
        return self

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
        spec = """
        data {
            int<lower=1> N;
            int<lower=1> K;
            vector[N] y;
            int<lower=1,upper=K> id[N];
            int<lower=1> C;
            matrix[K, C] X;
            vector[N] sigma;
        }
        parameters {
            vector[C] beta;
            vector[K] theta;
            real<lower=0> tau2;
        }
        transformed parameters {
            vector[N] mu;
            mu = theta[id] + X * beta;
        }
        model {
            y ~ normal(mu, sigma);
            theta ~ normal(0, tau2);
        }
        """
        from pystan import StanModel
        self.model = StanModel(model_code=spec)

    def fit(self, y, v, X, groups=None):
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
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError("The StanMetaRegression estimator currently does "
                             "not support 2-dimensional inputs. Passed y has "
                             "shape {}.".format(y.shape))

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
        return self

    def summary(self, ci=95):
        if self.result_ is None:
            name = self.__class__.__name__
            raise ValueError("This {} instance hasn't been fitted yet. Please "
                             "call fit() before summary().".format(name))
        return BayesianMetaRegressionResults(self.result_, self.dataset_, ci)
