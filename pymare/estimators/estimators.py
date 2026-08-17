"""Meta-regression estimator classes."""

import sys
from abc import ABCMeta, abstractmethod
from inspect import getfullargspec
from warnings import warn

import numpy as np
import wrapt
from scipy.optimize import Bounds, minimize

from ..results import BayesianMetaRegressionResults, MetaRegressionResults
from ..stats import (
    DEFAULT_CLUSTER_RHO,
    cluster_robust_cov,
    cluster_weights,
    collapse_clusters,
    collapse_clusters_by_n,
    ensure_2d,
    weighted_least_squares,
)

WEIGHT_SCHEMES = ("individual", "cluster")


def _check_weight_scheme(weight_scheme):
    """Validate the ``weight_scheme`` argument shared by the WLS-family estimators."""
    if weight_scheme not in WEIGHT_SCHEMES:
        raise ValueError(
            f"Invalid weight_scheme '{weight_scheme}'; must be one of {list(WEIGHT_SCHEMES)}."
        )


def _resolve_weights(v, groups, tau2, weight_scheme):
    """Return WLS weights, or None to fall back to ``1 / (v + tau2)``.

    ``"cluster"`` divides each estimate's weight by the size of its cluster, so
    a study contributing many estimates no longer outvotes one contributing a
    single estimate. It is a no-op when no group labels are supplied.
    """
    if weight_scheme == "individual" or groups is None:
        return None

    return cluster_weights(v, groups, tau2=tau2)


def _tau2_inputs(y, v, X, groups, weight_scheme, rho, by_sample_size=False):
    """Return the (y, v, X) that tau^2 should be estimated from.

    Moment-based tau^2 estimators treat every row as an independent study, so a
    study contributing several images makes the observed dispersion look
    smaller than the row count implies and biases tau^2 downward. Collapsing to
    one effect per cluster first removes that pseudo-replication. Falls back to
    the raw inputs when there are too few clusters to fit the design.
    """
    if weight_scheme != "cluster" or groups is None:
        return y, v, X

    n_groups = np.unique(np.asarray(groups).ravel()).size
    if n_groups <= X.shape[1]:
        return y, v, X

    collapse = collapse_clusters_by_n if by_sample_size else collapse_clusters
    return collapse(y, v, X, groups, rho=rho)


def _dersimonian_laird_tau2(y, v, X):
    """Method-of-moments tau^2 from Cochran's Q."""
    k, p = X.shape

    # Estimate initial betas with WLS, assuming tau^2=0
    beta_wls, inv_cov = weighted_least_squares(y, v, X, return_cov=True)

    # Cochran's Q
    w = 1.0 / v
    w_sum = w.sum(0)
    Q = (w * (y - X.dot(beta_wls)) ** 2).sum(0)

    # Einsum indices: k = studies, p = predictors, i = parallel iterates.
    # q is a dummy for 2nd p when p x p covariance matrix is passed.
    Xw2 = np.einsum("kp,ki->ipk", X, w**2)
    pXw2 = np.einsum("ipk,qpi->iqk", Xw2, inv_cov)
    A = w_sum - np.trace(pXw2.dot(X), axis1=1, axis2=2)
    return np.maximum(0.0, (Q - (k - p)) / A)


def _cluster_robust_inv_cov(y, v, X, beta, groups, tau2=0.0, inv_cov=None, w=None):
    """Compute the cluster-robust covariance for a fitted model.

    Returns ``(robust_cov, n_clusters)``, or ``(None, None)`` when ``groups``
    is None so that the model-based covariance is left untouched. Pass the
    model-based ``inv_cov`` when available to skip a redundant pseudo-inverse,
    and the same ``w`` that produced ``beta`` so the residuals match the fit.

    The count is returned separately rather than folded into ``params_``
    because :func:`_loopable` stacks every entry of ``params_`` across
    parallel datasets, which only works for arrays.
    """
    if groups is None:
        return None, None

    groups = np.asarray(groups).ravel()
    robust_cov = cluster_robust_cov(y, v, X, beta, groups, tau2=tau2, inv_cov=inv_cov, w=w)
    return robust_cov, int(np.unique(groups).size)


@wrapt.decorator
def _loopable(wrapped, instance, args, kwargs):
    """Decorate fit() method of Estimator classes.

    Designed to handle naive looping over the 2nd dimension of y/v/n inputs, and reconstruction of
    outputs.
    """
    n_iter = kwargs["y"].shape[1]
    if n_iter > 10:
        warn(
            "Input contains {} parallel datasets (in 2nd dim of y and"
            " v). The selected estimator will loop over datasets"
            " naively, and this may be slow for large numbers of "
            "datasets. Consider using the DL, HE, or WLS estimators, "
            "which handle parallel datasets more efficiently.".format(n_iter)
        )

    param_dicts = []
    for i in range(n_iter):
        iter_kwargs = {"X": kwargs["X"]}
        iter_kwargs["y"] = kwargs["y"][:, i, None]
        if "v" in kwargs:
            iter_kwargs["v"] = kwargs["v"][:, i, None]

        if "n" in kwargs:
            n = kwargs["n"][:, i, None] if kwargs["n"].shape[1] > 1 else kwargs["n"]
            iter_kwargs["n"] = n

        # Group labels are per-study, not per-dataset, so they are shared
        # across iterates rather than sliced.
        if kwargs.get("g") is not None:
            iter_kwargs["g"] = kwargs["g"]

        wrapped(**iter_kwargs)
        param_dicts.append(instance.params_.copy())

    params = {}
    for k in param_dicts[0]:
        concat = np.stack([pd[k].squeeze() for pd in param_dicts], axis=-1)
        params[k] = np.atleast_2d(concat)

    instance.params_ = params
    return instance


class BaseEstimator(metaclass=ABCMeta):
    """A base class for Estimators."""

    # A class-level mapping from Dataset attributes to fit() arguments. Used by
    # fit_dataset() for estimators that take non-standard arguments (e.g., 'z'
    # instead of 'y'). Keys are default Dataset attribute names (e.g., 'y') and
    # values are the target arg names in the estimator class's fit() method
    # (e.g., 'z').
    _dataset_attr_map = {}

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the estimator to data."""
        pass

    def fit_dataset(self, dataset, *args, **kwargs):
        """Apply the current estimator to the passed Dataset container.

        A convenience interface that wraps fit() and automatically aligns the
        variables held in a Dataset with the required arguments.

        Parameters
        ----------
        dataset : :obj:`~pymare.core.Dataset`
            A PyMARE Dataset instance holding the data.
        *args
            Optional positional arguments to pass onto the :meth:`~pymare.core.Dataset.fit` method.
        **kwargs
            Optional keyword arguments to pass onto the :meth:`~pymare.core.Dataset.fit` method.
        """
        all_kwargs = {}
        spec = getfullargspec(self.fit)
        n_kw = len(spec.defaults) if spec.defaults else 0
        n_args = len(spec.args) - n_kw - 1

        for i, name in enumerate(spec.args[1:]):
            # Check for remapped name
            attr_name = self._dataset_attr_map.get(name, name)
            if i >= n_args:
                all_kwargs[name] = getattr(dataset, attr_name, spec.defaults[i - n_args])
            else:
                all_kwargs[name] = getattr(dataset, attr_name)

        all_kwargs.update(kwargs)
        self.fit(*args, **all_kwargs)
        self.dataset_ = dataset

        return self

    def get_v(self, dataset):
        """Get the variances, or an estimate thereof, from the given Dataset.

        Parameters
        ----------
        dataset : :obj:`~pymare.core.Dataset`
            The dataset to use to retrieve/estimate v.

        Returns
        -------
        :obj:`numpy.ndarray`
            2-dimensional array of variances/variance estimates.

        Notes
        -----
        This is equivalent to directly accessing ``dataset.v`` when variances are present,
        but affords a way of estimating v from sample size (n) for any estimator that implicitly
        estimates a sigma^2 parameter.
        """
        if dataset.v is not None:
            return dataset.v

        # Estimate sampling variances from sigma^2 and n if available.
        if dataset.n is None:
            raise ValueError(
                "Dataset does not contain sampling variances (v),"
                " and no estimate of v is possible without sample"
                " sizes (n)."
            )

        if "sigma2" not in self.params_:
            raise ValueError(
                "Dataset does not contain sampling variances (v),"
                " and no estimate of v is possible because no "
                "sigma^2 parameter was found."
            )

        return self.params_["sigma2"] / dataset.n

    def summary(self):
        """Generate a MetaRegressionResults object for the fitted estimator.

        Returns
        -------
        :obj:`~pymare.results.MetaRegressionResults`
        """
        if not hasattr(self, "params_"):
            name = self.__class__.__name__
            raise ValueError(
                "This {} instance hasn't been fitted yet. Please "
                "call fit() before summary().".format(name)
            )

        p = self.params_
        return MetaRegressionResults(self, self.dataset_, p["fe_params"], p["inv_cov"], p["tau2"])


class WeightedLeastSquares(BaseEstimator):
    """Weighted least-squares meta-regression.

    Provides the weighted least-squares estimate of the fixed effects given known/assumed
    between-study variance tau^2, as described in :footcite:t:`brockwell2001comparison`.
    When tau^2 = 0 (default), the model is the standard inverse-weighted fixed-effects
    meta-regression.

    Parameters
    ----------
    tau2 : :obj:`float` or :obj:`numpy.ndarray` of shape (d), optional
        Assumed/known value of tau^2. Must be >= 0.
        If an array, must have ``d`` elements, where ``d`` refers to the number of datasets.
        Default = 0.

    Notes
    -----
    This estimator accepts 2-D inputs for ``y`` and ``v``--i.e., it can produce estimates
    simultaneously for multiple independent sets of ``y``/``v`` values
    (use the 2nd dimension for the parallel iterates).
    The ``X`` matrix must be identical for all iterates.
    If no ``v`` argument is passed to :meth:`fit`, unit weights will be used, resulting in the
    ordinary least-squares (OLS) solution.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, tau2=0.0, weight_scheme="individual"):
        _check_weight_scheme(weight_scheme)
        self.tau2 = tau2
        self.weight_scheme = weight_scheme

    def fit(self, y, X, v=None, g=None):
        """Fit the estimator to data.

        Parameters
        ----------
        y : :obj:`numpy.ndarray` of shape (n, d)
            The dependent variable(s) (y).
        X : :obj:`numpy.ndarray` of shape (n, p)
            The independent variable(s) (X).
        v : :obj:`numpy.ndarray` of shape (n, d), optional
            Sampling variances. If not provided, unit weights will be used.
        g : :obj:`numpy.ndarray` of shape (n,), optional
            Group (cluster) labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            Point estimates are unaffected unless ``weight_scheme='cluster'``.

        Returns
        -------
        :obj:`~pymare.estimators.WeightedLeastSquares`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        if v is None:
            v = np.ones_like(y)

        w = _resolve_weights(v, g, self.tau2, self.weight_scheme)
        beta, inv_cov = weighted_least_squares(y, v, X, self.tau2, return_cov=True, w=w)
        robust_cov, self.n_clusters_ = _cluster_robust_inv_cov(
            y, v, X, beta, g, tau2=self.tau2, inv_cov=inv_cov, w=w
        )
        self.params_ = {
            "fe_params": beta,
            "tau2": self.tau2,
            "inv_cov": inv_cov if robust_cov is None else robust_cov,
        }
        return self


class DerSimonianLaird(BaseEstimator):
    """DerSimonian-Laird meta-regression estimator.

    Estimates the between-subject variance tau^2 using the :footcite:t:`dersimonian1986meta`
    method-of-moments approach.

    Notes
    -----
    This estimator accepts 2-D inputs for ``y`` and ``v``--i.e., it can produce estimates
    simultaneously for multiple independent sets of ``y``/``v`` values
    (use the 2nd dimension for the parallel iterates).
    The ``X`` matrix must be identical for all iterates.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, weight_scheme="individual", cluster_rho=DEFAULT_CLUSTER_RHO):
        _check_weight_scheme(weight_scheme)
        self.weight_scheme = weight_scheme
        self.cluster_rho = cluster_rho

    def fit(self, y, v, X, g=None):
        """Fit the estimator to data.

        Parameters
        ----------
        y : :obj:`numpy.ndarray` of shape (n, d)
            The dependent variable(s) (y).
        v : :obj:`numpy.ndarray` of shape (n, d)
            Sampling variances.
        X : :obj:`numpy.ndarray` of shape (n, p)
            The independent variable(s) (X).
        g : :obj:`numpy.ndarray` of shape (n,), optional
            Group (cluster) labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme='cluster'`` tau^2 is estimated from one
            effect per group.

        Returns
        -------
        :obj:`~pymare.estimators.DerSimonianLaird`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        y = ensure_2d(y)
        v = ensure_2d(v)

        tau_dl = _dersimonian_laird_tau2(
            *_tau2_inputs(y, v, X, g, self.weight_scheme, self.cluster_rho)
        )

        # Re-estimate beta with tau^2 estimate
        w = _resolve_weights(v, g, tau_dl, self.weight_scheme)
        beta_dl, inv_cov = weighted_least_squares(y, v, X, tau2=tau_dl, return_cov=True, w=w)
        robust_cov, self.n_clusters_ = _cluster_robust_inv_cov(
            y, v, X, beta_dl, g, tau2=tau_dl, inv_cov=inv_cov, w=w
        )
        self.params_ = {
            "fe_params": beta_dl,
            "tau2": tau_dl,
            "inv_cov": inv_cov if robust_cov is None else robust_cov,
        }
        return self


class Hedges(BaseEstimator):
    """Hedges meta-regression estimator.

    Estimates the between-subject variance tau^2 using the :footcite:t:`hedges2014statistical`
    approach.

    Notes
    -----
    This estimator accepts 2-D inputs for ``y`` and ``v``--i.e., it can produce estimates
    simultaneously for multiple independent sets of ``y``/``v`` values
    (use the 2nd dimension for the parallel iterates).
    The ``X`` matrix must be identical for all iterates.

    .. warning::
        The model-based ``inv_cov`` this estimator reports is derived from *unit* weights,
        while the coefficients themselves are estimated with ``1 / (v + tau^2)`` weights.
        The two are therefore on different scales, and the reported standard errors do not
        correspond to the reported coefficients. This is longstanding behaviour and is left
        as-is, but it means the cluster-robust errors obtained by passing ``g`` -- which do
        use the same weights as the coefficients -- are not directly comparable to the
        model-based ones for this estimator.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, weight_scheme="individual", cluster_rho=DEFAULT_CLUSTER_RHO):
        _check_weight_scheme(weight_scheme)
        self.weight_scheme = weight_scheme
        self.cluster_rho = cluster_rho

    def fit(self, y, v, X, g=None):
        """Fit the estimator to data.

        Parameters
        ----------
        y : :obj:`numpy.ndarray` of shape (n, d)
            The dependent variable(s) (y).
        v : :obj:`numpy.ndarray` of shape (n, d)
            Sampling variances.
        X : :obj:`numpy.ndarray` of shape (n, p)
            The independent variable(s) (X).
        g : :obj:`numpy.ndarray` of shape (n,), optional
            Group (cluster) labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme='cluster'`` tau^2 is estimated from one
            effect per group.

        Returns
        -------
        :obj:`~pymare.estimators.Hedges`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        _unit_v = np.ones_like(y)
        beta, inv_cov = weighted_least_squares(y, _unit_v, X, return_cov=True)

        tau_y, tau_v, tau_X = _tau2_inputs(y, v, X, g, self.weight_scheme, self.cluster_rho)
        tau_k, tau_p = tau_X.shape[:2]
        tau_beta = weighted_least_squares(tau_y, np.ones_like(tau_y), tau_X)
        mse = ((tau_y - tau_X.dot(tau_beta)) ** 2).sum(0) / (tau_k - tau_p)
        tau_ho = np.maximum(0, mse - tau_v.sum(0) / tau_k)
        # Estimate beta with tau^2 estimate
        w = _resolve_weights(v, g, tau_ho, self.weight_scheme)
        beta_ho = weighted_least_squares(y, v, X, tau2=tau_ho, w=w)
        # Unlike the model-based inv_cov above, which this estimator derives
        # from unit weights, the sandwich has to use the same weights that
        # produced beta_ho for the residuals to be the right ones.
        robust_cov, self.n_clusters_ = _cluster_robust_inv_cov(
            y, v, X, beta_ho, g, tau2=tau_ho, w=w
        )
        self.params_ = {
            "fe_params": beta_ho,
            "tau2": tau_ho,
            "inv_cov": inv_cov if robust_cov is None else robust_cov,
        }
        return self


class VarianceBasedLikelihoodEstimator(BaseEstimator):
    """Likelihood-based estimator for estimates with known variances.

    Initially estimates the between-subject variance tau^2 and fixed effect coefficients
    using :footcite:t:`dersimonian1986meta` method-of-moments approach, and then
    iteratively estimates them using the specified likelihood-based estimator (ML or REML)
    :footcite:p:`kosmidis2017improving`.

    Parameters
    ----------
    method : {"ML", "REML"}, optional
        The estimation method to use.
        Either 'ML' (for maximum-likelihood) or 'REML' (restricted maximum-likelihood).
        Default = 'ML'.
    **kwargs
        Keyword arguments to pass to the SciPy minimizer.

    Notes
    -----
    The ML and REML solutions are obtained via SciPy's scalar function minimizer
    (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self, method="ml", weight_scheme="individual", cluster_rho=DEFAULT_CLUSTER_RHO, **kwargs
    ):
        _check_weight_scheme(weight_scheme)
        self.weight_scheme = weight_scheme
        self.cluster_rho = cluster_rho
        nll_func = getattr(self, "_{}_nll".format(method.lower()))
        if nll_func is None:
            raise ValueError("No log-likelihood function defined for method '{}'.".format(method))

        self._nll_func = nll_func
        self.kwargs = kwargs

    @_loopable
    def fit(self, y, v, X, g=None):
        """Fit the estimator to data.

        Parameters
        ----------
        y : :obj:`numpy.ndarray` of shape (n, d)
            The dependent variable(s) (y).
        v : :obj:`numpy.ndarray` of shape (n, d)
            Sampling variances.
        X : :obj:`numpy.ndarray` of shape (n, p)
            The independent variable(s) (X).
        g : :obj:`numpy.ndarray` of shape (n,), optional
            Group (cluster) labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme='cluster'`` tau^2 is estimated from one
            effect per group.

        Returns
        -------
        :obj:`~pymare.estimators.VarianceBasedLikelihoodEstimator`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        # The likelihood treats every row as an independent study, so tau^2 is
        # fitted on one effect per cluster to avoid counting a study's repeated
        # estimates as independent evidence.
        fit_y, fit_v, fit_X = _tau2_inputs(y, v, X, g, self.weight_scheme, self.cluster_rho)

        # use D-L estimate for initial values
        est_DL = DerSimonianLaird().fit(fit_y, fit_v, fit_X).params_
        beta = est_DL["fe_params"]
        tau2 = est_DL["tau2"]

        theta_init = np.r_[beta.ravel(), tau2]

        lb = np.ones(len(theta_init)) * -np.inf
        ub = -lb
        lb[-1] = 0.0  # bound only the variance
        bds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(
            self._nll_func, theta_init, (fit_y, fit_v, fit_X), bounds=bds, **self.kwargs
        )
        beta, tau = res.x[:-1], float(res.x[-1])
        tau = np.max([tau, 0])
        beta = beta[:, None]
        w = _resolve_weights(v, g, tau, self.weight_scheme)
        if w is None:
            _, inv_cov = weighted_least_squares(y, v, X, tau, True)
        else:
            # Cluster weighting changes the estimand, so beta has to be
            # recomputed under those weights rather than kept from the
            # likelihood, whose working model assumes independence.
            beta, inv_cov = weighted_least_squares(y, v, X, tau, True, w=w)
        robust_cov, self.n_clusters_ = _cluster_robust_inv_cov(
            y, v, X, beta, g, tau2=tau, inv_cov=inv_cov, w=w
        )
        self.params_ = {
            "fe_params": beta,
            "tau2": tau,
            "inv_cov": inv_cov if robust_cov is None else robust_cov,
        }
        return self

    def _ml_nll(self, theta, y, v, X):
        """ML negative log-likelihood for meta-regression model."""
        beta, tau2 = theta[:-1, None], theta[-1]
        if tau2 < 0:
            tau2 = 0
        w = 1.0 / (v + tau2)
        R = y - X.dot(beta)
        return -0.5 * (np.log(w).sum() - (R * w * R).sum())

    def _reml_nll(self, theta, y, v, X):
        """REML negative log-likelihood for meta-regression model."""
        ll_ = self._ml_nll(theta, y, v, X)
        tau2 = theta[-1]
        w = 1.0 / (v + tau2)
        F = (X * w).T.dot(X)
        return ll_ + 0.5 * np.log(np.linalg.det(F))


class SampleSizeBasedLikelihoodEstimator(BaseEstimator):
    """Likelihood-based estimator for data with known sample sizes but unknown sampling variances.

    Iteratively estimates the between-subject variance tau^2 and fixed effect betas using the
    specified likelihood-based estimator (ML or REML) :footcite:p:`sangnawakij2019meta`.

    Parameters
    ----------
    method : {"ML", "REML"}, optional
        The estimation method to use.
        Either 'ML' (for maximum-likelihood) or 'REML' (restricted maximum-likelihood).
        Default = 'ML'.
    **kwargs
        Keyword arguments to pass to the SciPy minimizer.

    Notes
    -----
    Homogeneity of sigma^2 across studies is assumed.

    The ML and REML solutions are obtained via SciPy's scalar function minimizer
    (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self, method="ml", weight_scheme="individual", cluster_rho=DEFAULT_CLUSTER_RHO, **kwargs
    ):
        _check_weight_scheme(weight_scheme)
        self.weight_scheme = weight_scheme
        self.cluster_rho = cluster_rho
        nll_func = getattr(self, "_{}_nll".format(method.lower()))
        if nll_func is None:
            raise ValueError("No log-likelihood function defined for method '{}'.".format(method))

        self._nll_func = nll_func
        self.kwargs = kwargs

    @_loopable
    def fit(self, y, n, X, g=None):
        """Fit the estimator to data.

        Parameters
        ----------
        y : :obj:`numpy.ndarray` of shape (n, d)
            The dependent variable(s) (y).
        n : :obj:`numpy.ndarray` of shape (n, d)
            Sample sizes.
        X : :obj:`numpy.ndarray` of shape (n, p)
            The independent variable(s) (X).
        g : :obj:`numpy.ndarray` of shape (n,), optional
            Group (cluster) labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme='cluster'`` the variance components are
            estimated from one effect per group.

        Returns
        -------
        :obj:`~pymare.estimators.SampleSizeBasedLikelihoodEstimator`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        if n.std() < np.sqrt(np.finfo(float).eps):
            raise ValueError(
                "Sample size-based likelihood estimator cannot "
                "work with all-equal sample sizes."
            )

        if n.std() < n.mean() / 10:
            raise Warning(
                "Sample sizes are too close, sample size-based likelihood estimator may fail."
            )

        # Both variance components are fitted on one effect per cluster; a
        # study's repeated estimates agree with each other by construction, and
        # counting them as independent shrinks sigma^2 toward zero.
        fit_y, fit_n, fit_X = _tau2_inputs(
            y, n, X, g, self.weight_scheme, self.cluster_rho, by_sample_size=True
        )

        # set tau^2 to 0 and compute starting values
        tau2 = 0.0
        k, p = fit_X.shape
        beta = weighted_least_squares(fit_y, fit_n, fit_X, tau2)
        sigma = ((fit_y - fit_X.dot(beta)) ** 2 * fit_n).sum() / (k - p)
        theta_init = np.r_[beta.ravel(), sigma, tau2]

        lb = np.ones(len(theta_init)) * -np.inf
        ub = -lb
        lb[-2:] = 0.0  # bound only the variances
        bds = Bounds(lb, ub, keep_feasible=True)

        res = minimize(
            self._nll_func, theta_init, (fit_y, fit_n, fit_X), bounds=bds, **self.kwargs
        )
        beta, sigma, tau = res.x[:-2], float(res.x[-2]), float(res.x[-1])
        tau = np.max([tau, 0])
        beta = beta[:, None]
        v = sigma / n
        w = _resolve_weights(v, g, tau, self.weight_scheme)
        if w is None:
            _, inv_cov = weighted_least_squares(y, v, X, tau, True)
        else:
            # Cluster weighting changes the estimand, so beta has to be
            # recomputed under those weights rather than kept from the
            # likelihood, whose working model assumes independence.
            beta, inv_cov = weighted_least_squares(y, v, X, tau, True, w=w)
        robust_cov, self.n_clusters_ = _cluster_robust_inv_cov(
            y, v, X, beta, g, tau2=tau, inv_cov=inv_cov, w=w
        )
        self.params_ = {
            "fe_params": beta,
            "sigma2": np.array(sigma),
            "tau2": tau,
            "inv_cov": inv_cov if robust_cov is None else robust_cov,
        }
        return self

    def _ml_nll(self, theta, y, n, X):
        """ML negative log-likelihood for meta-regression model."""
        beta, sigma2, tau2 = theta[:-2, None], theta[-2], theta[-1]
        if tau2 < 0:
            tau2 = 0
        if sigma2 < 0:
            sigma2 = 0
        w = 1 / (tau2 + sigma2 / n)
        R = y - X.dot(beta)
        return -0.5 * (np.log(w).sum() - (R * w * R).sum())

    def _reml_nll(self, theta, y, n, X):
        """REML negative log-likelihood for meta-regression model."""
        ll_ = self._ml_nll(theta, y, n, X)
        sigma2, tau2 = theta[-2:]
        w = 1 / (tau2 + sigma2 / n)
        F = (X * w).T.dot(X)
        return ll_ + 0.5 * np.log(np.linalg.det(F))


class StanMetaRegression(BaseEstimator):
    """Bayesian meta-regression estimator using Stan.

    Parameters
    ----------
    **sampling_kwargs
        Optional keyword arguments to pass on to the MCMC sampler
        (e.g., `iter` for number of iterations).

    Notes
    -----
    For most uses, this class should be ignored in favor of the functional
    stan() estimator. The object-oriented interface is useful primarily
    when fitting the meta-regression model repeatedly to different data;
    the separation of .compile() and .fit() steps allows one to compile
    the model only once.

    Warning
    -------
    :obj:`~pymare.estimators.StanMetaRegression` uses Pystan 3, which requires Python 3.7.
    Pystan 3 should not be used with PyMARE and Python 3.6 or earlier.
    """

    _result_cls = BayesianMetaRegressionResults

    def __init__(self, **sampling_kwargs):
        self.sampling_kwargs = sampling_kwargs
        self.model = None
        self.result_ = None

        if sys.version_info < (3, 7):
            raise RuntimeError(
                "StanMetaRegression uses Pystan 3, which requires python 3.7 or higher. "
                f"You are running Python {sys.version_info.major}.{sys.version_info.minor}. "
                "Pystan 3 should not be used with PyMARE and Python 3.6 or earlier."
            )

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
            array[N] int<lower=1,upper=K> id;
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
        try:
            import stan
        except ImportError:
            raise ImportError("Please install pystan.")

        self.model = stan.build(spec, data=self.data)

    def fit(self, y, v, X, groups=None):
        """Run the Stan sampler and return results.

        Parameters
        ----------
        y : :obj:`numpy.ndarray` of shape (K,)
            1d array of study-level estimates
        v : :obj:`numpy.ndarray` of shape (K,)
            1d array of study-level variances
        X : :obj:`numpy.ndarray` of shape (K[, P])
            1d or 2d array containing study-level predictors
            (including intercept); has dimensions K x P, where K is the
            number of studies and P is the number of predictor variables.
        groups : :obj:`list` of :obj:`int`, optional
            1d array of integers identifying
            groups/clusters of observations in the y/v/X inputs. If
            provided, values must consist of integers in the range of 1..k
            (inclusive), where k is the number of distinct groups. When
            None (default), it is assumed that each observation in the
            inputs is a separate group.

        Returns
        -------
        A StanFit4Model object (see PyStan documentation for details).

        Notes
        -----
        This estimator supports (simple) hierarchical models. When multiple
        estimates are available for at least one of the studies in `y`, the
        `groups` argument can be used to specify the nesting structure
        (i.e., which rows in `y`, `v`, and `X` belong to each study).
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(
                "The StanMetaRegression estimator currently does "
                "not support 2-dimensional inputs. Passed y has "
                "shape {}.".format(y.shape)
            )

        N = y.shape[0]
        groups = groups or np.arange(1, N + 1, dtype=int)
        K = len(np.unique(groups))

        data = {
            "K": K,
            "N": N,
            "id": groups,
            "C": X.shape[1],
            "X": X,
            "y": y.ravel(),
            "sigma": v.ravel(),
        }

        self.data = data

        if self.model is None:
            self.compile()

        self.result_ = self.model.sample(**self.sampling_kwargs)
        return self

    def summary(self, ci=95):
        """Generate a BayesianMetaRegressionResults object from the fitted estimator."""
        if self.result_ is None:
            name = self.__class__.__name__
            raise ValueError(
                "This {} instance hasn't been fitted yet. Please "
                "call fit() before summary().".format(name)
            )
        return BayesianMetaRegressionResults(self.result_, self.dataset_, ci)
