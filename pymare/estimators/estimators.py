"""Meta-regression estimator classes."""

import sys
from abc import ABCMeta, abstractmethod
from inspect import getfullargspec
from warnings import warn

import numpy as np

from ..results import BayesianMetaRegressionResults, MetaRegressionResults
from ..stats import (
    DEFAULT_RHO,
    TAU2_AGGREGATE,
    TAU2_CORRELATED,
    TAU2_INDEPENDENT,
    bounded_scalar_min,
    broadcast_columns,
    cluster_robust_cov,
    collapse_groups,
    collapse_groups_by_n,
    correlated_effects_tau2,
    correlated_effects_weights,
    encode_groups,
    ensure_2d,
    satterthwaite_dof,
    weighted_least_squares,
)

WEIGHT_SCHEMES = ("individual", "rescale", "collapse")

#: Upper end of the bounded search variable that stands in for tau^2. The
#: variable is a fraction, and one just short of 1 maps to a very large tau^2
#: rather than to an infinite one; see :func:`_tau2_from_search`.
_SEARCH_MAX = 1.0 - 1e-9


class Options:
    """Constrain a parameter to a fixed set of values."""

    def __init__(self, allowed):
        self.allowed = tuple(allowed)

    def check(self, name, value):
        """Raise if ``value`` is not one of the allowed options."""
        if value not in self.allowed:
            raise ValueError(f"Invalid {name} '{value}'; must be one of {list(self.allowed)}.")


class Interval:
    """Constrain a parameter to a closed numeric interval."""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def check(self, name, value):
        """Raise if ``value`` is not a real number inside the interval."""
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"Invalid {name} {value!r}; must be a number.")
        if not self.low <= float(value) <= self.high:
            raise ValueError(f"Invalid {name} {value!r}; must lie in [{self.low}, {self.high}].")


#: Constraints shared by the estimators that accept group labels. Assigned to
#: each class's ``_parameter_constraints`` so that both parameters are checked
#: by the same mechanism, in the same place.
WEIGHTING_CONSTRAINTS = {
    "weight_scheme": Options(WEIGHT_SCHEMES),
    "rho": Interval(0.0, 1.0),
}


def _resolve_rho(rho, weight_scheme):
    """Return the assumed within-group correlation, warning when it cannot apply.

    Parameters
    ----------
    rho : None or :obj:`float`
        The value supplied by the caller, or None for the default.
    weight_scheme : :obj:`str`
        The scheme supplied alongside it.

    Returns
    -------
    :obj:`float`
        ``rho``, or :data:`~pymare.stats.DEFAULT_RHO` when none was supplied.

    Warns
    -----
    UserWarning
        If ``rho`` was set explicitly under ``weight_scheme="individual"``, which
        models no within-group correlation and therefore never reads it.

    Notes
    -----
    The default is None rather than the number itself so that "the user asked
    for 0.8" can be told apart from "nobody chose". Without that distinction a
    deliberate ``rho`` under the default scheme is silently inert, which is the
    one case where the caller clearly expected it to matter.
    """
    if rho is None:
        return DEFAULT_RHO
    if weight_scheme == "individual":
        warn(
            "rho was supplied but weight_scheme='individual', which models no "
            "within-group correlation and so ignores it. Use "
            "weight_scheme='rescale' or 'collapse' for rho to take effect.",
            stacklevel=3,
        )
    return rho


def _tau2_from_search(u, scale):
    """Map the bounded search variable back to tau^2.

    Parameters
    ----------
    u : :obj:`numpy.ndarray` of shape (D,)
        The search variable, in ``[0, 1)``.
    scale : :obj:`numpy.ndarray` of shape (D,)
        The tau^2 value that ``u = 0.5`` stands for, per parallel dataset.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (D,)
        ``scale * u / (1 - u)``.

    Notes
    -----
    tau^2 has no upper bound, so searching it directly means choosing one and
    risking an optimum outside it. This mapping is monotone and sends ``[0, 1)``
    onto ``[0, inf)``, so a bounded search over ``u`` cannot truncate the
    parameter space; ``scale`` only decides where in it the scan looks hardest.
    Monotonicity is what keeps the objective as unimodal in ``u`` as it is in
    tau^2, which is what the refinement in
    :func:`~pymare.stats.bounded_scalar_min` needs.
    """
    return scale * u / (1.0 - u)


def _search_scale(scale):
    """Return a positive search scale, falling back to 1 where the estimate is not.

    Parameters
    ----------
    scale : :obj:`numpy.ndarray` of shape (D,)
        Candidate scale per parallel dataset, from the data.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (D,)
        The candidate where it is positive and finite, else 1.

    Notes
    -----
    A degenerate dataset -- every variance zero, say -- can produce a scale of
    zero or a non-finite one, which would collapse the search to a single point.
    Falling back to 1 still spans all of ``[0, inf)``, so the substitution costs
    resolution where the scan looks first and nothing else.
    """
    scale = np.asarray(scale, dtype=float)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


def _resolve_weights(v, groups, tau2, weight_scheme):
    """Return WLS weights, or None to fall back to ``1 / (v + tau2)``.

    Parameters
    ----------
    v : :obj:`numpy.ndarray` of shape (K, D)
        Sampling variances.
    groups : None or :obj:`numpy.ndarray` of shape (K,)
        Group labels, or None when no dependence was declared.
    tau2 : :obj:`float` or :obj:`numpy.ndarray`
        The tau^2 estimate to fold into the weights.
    weight_scheme : {"individual", "rescale", "collapse"}
        The scheme requested by the caller.

    Returns
    -------
    None or :obj:`numpy.ndarray` of shape (K, D)
        Rescaled weights for ``"rescale"``, else None, meaning the caller should
        use its own default.

    Notes
    -----
    ``"rescale"`` gives every row of a group the same weight, built from the
    group's mean variance, so row multiplicity does not buy group influence.
    These are the weights the correlated-effects model of
    :footcite:t:`hedges2010robust` uses and the ones its tau^2 estimator is
    derived under; see :func:`~pymare.stats.correlated_effects_weights`.
    ``"collapse"`` returns None because the aggregation has already happened
    upstream, leaving one row per group for which the ordinary weight is
    correct. It is a no-op when no group labels are supplied.

    References
    ----------
    .. footbibliography::
    """
    if weight_scheme in ("individual", "collapse") or groups is None:
        return None

    return correlated_effects_weights(v, groups, tau2=tau2)


def _tau2_model(groups, X, weight_scheme, correlated_effects=False):
    """Name the reduction tau^2 will be estimated under.

    Parameters
    ----------
    groups : None or :obj:`numpy.ndarray` of shape (K,)
        Group labels, or None when no dependence was declared.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    weight_scheme : {"individual", "rescale", "collapse"}
        The scheme requested by the caller.
    correlated_effects : :obj:`bool`, optional
        Whether the calling estimator has a published correlated-effects
        counterpart of its tau^2 estimator. Only DerSimonian-Laird does; see
        :func:`~pymare.stats.correlated_effects_tau2`. Default = False.

    Returns
    -------
    :obj:`str`
        One of ``TAU2_INDEPENDENT``, ``TAU2_CORRELATED`` or ``TAU2_AGGREGATE``.

    Notes
    -----
    Single source of truth for a decision three places have to agree on: which
    arrays tau^2 comes from, whether the design has to be constant within a
    group, and which reduction the interval and Q in :mod:`pymare.results`
    describe. Each of those used to re-derive it from ``weight_scheme``.
    """
    if weight_scheme not in ("rescale", "collapse") or groups is None:
        return TAU2_INDEPENDENT
    if encode_groups(np.asarray(groups).ravel())[1].size <= X.shape[1]:
        # Too few groups to fit the reduced design; fall back to the raw rows
        # rather than refusing to run. See _check_collapsed_design.
        return TAU2_INDEPENDENT
    if correlated_effects and weight_scheme == "rescale":
        return TAU2_CORRELATED
    return TAU2_AGGREGATE


def _validate_group_design(X, groups, weight_scheme="collapse"):
    """Reject group aggregation when predictors vary within a group.

    Parameters
    ----------
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per observation.

    weight_scheme : {"collapse", "rescale"}, optional
        The scheme that requested the reduction, named in the error message.
        Default = "collapse".

    Raises
    ------
    ValueError
        If any predictor takes more than one value within a group.

    Notes
    -----
    Collapsing a group to one row replaces its design values with their mean, so
    a predictor that varied within the group would silently change meaning: the
    coefficient would answer a between-group question instead.

    ``"rescale"`` keeps every row when *fitting*, but every estimator except
    DerSimonian-Laird still estimates tau^2 from the same one-row-per-group
    reduction, so the restriction applies there too. DerSimonian-Laird is exempt
    because it has a published correlated-effects estimator that works from the
    observation-level design; see
    :func:`~pymare.stats.correlated_effects_tau2`.

    The comparison uses a relative tolerance rather than bit-equality. A
    group-level predictor built by arithmetic -- a ratio, a centered score -- is
    constant in intent but can differ in the last bits across a group's rows, and
    rejecting those designs for floating-point reasons would be wrong.
    """
    groups = np.asarray(groups).ravel()
    # Against the design's row count: passing groups.shape[0] made the check
    # vacuous, and a mismatched label array then failed further down with an
    # opaque boolean-indexing IndexError instead of naming the real problem.
    codes, labels = encode_groups(groups, n_observations=X.shape[0])
    for group in range(labels.size):
        member_X = X[codes == group]
        # A relative tolerance, not bit-equality: a group-level predictor
        # built by arithmetic (a ratio, a centered score) is constant in intent
        # but can differ in the last bits across a group's rows.
        if not np.allclose(member_X, member_X[[0]], rtol=1e-10, atol=0.0):
            reduction = (
                "reduces the data to one row per group"
                if weight_scheme == "collapse"
                else "estimates tau^2 from one aggregate per group for this estimator"
            )
            raise ValueError(
                f"weight_scheme='{weight_scheme}' {reduction}, which requires design "
                "values to be constant within each group. DerSimonianLaird with "
                "weight_scheme='rescale' estimates tau^2 from the correlated-effects "
                "model instead, and has no such restriction."
            )


def _check_collapsed_design(collapsed_y, collapsed_X):
    """Reject a group reduction that leaves fewer rows than predictors.

    Parameters
    ----------
    collapsed_y : :obj:`numpy.ndarray` of shape (m, D)
        Estimates after aggregation, one row per group.
    collapsed_X : :obj:`numpy.ndarray` of shape (m, P)
        Design matrix after aggregation.

    Raises
    ------
    ValueError
        If the number of groups does not exceed the number of predictors.

    Notes
    -----
    ``weight_scheme="collapse"`` replaces K rows with m, so a dataset that was
    comfortably identified before collapsing can be saturated after it. Left
    unchecked the moment estimators divide by ``m - p == 0`` and report
    ``tau^2 = inf`` with zero standard errors, from input the user had no reason
    to think was degenerate.

    :func:`_tau2_inputs` meets the same condition and instead falls back to the
    raw rows. Both are correct, because they guard different schemes: under
    ``"collapse"`` the reduction *is* the model and a saturated collapse is fatal,
    whereas under ``"rescale"`` every row is kept for the coefficients and only
    tau^2 uses the reduction, so degrading to a biased-but-computable estimate
    beats refusing to run.
    """
    n_rows, n_preds = collapsed_X.shape
    if n_rows <= n_preds:
        raise ValueError(
            f"weight_scheme='collapse' reduces the data to {n_rows} row(s) for "
            f"{n_preds} predictor(s); the number of groups must exceed the "
            "number of predictors. Use weight_scheme='rescale' to keep every "
            "row while still accounting for dependence."
        )


def _collapse_inputs(y, v, X, groups, weight_scheme, rho):
    """Return one effect and variance per independent group when requested.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        Estimates.
    v : :obj:`numpy.ndarray` of shape (K, D)
        Sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : None or :obj:`numpy.ndarray` of shape (K,)
        Group labels, or None when no dependence was declared.
    weight_scheme : {"individual", "rescale", "collapse"}
        Only ``"collapse"`` triggers aggregation; the others pass through.
    rho : :obj:`float`
        Assumed within-group correlation used by the aggregation.

    Returns
    -------
    :obj:`tuple`
        ``(y, v, X, groups)``, aggregated to one row per group and relabelled
        ``arange(m)`` when ``weight_scheme="collapse"``, else the inputs unchanged.

    See Also
    --------
    pymare.stats.collapse_groups : Performs the aggregation.

    Notes
    -----
    Groups are relabelled because after aggregation each row *is* one independent
    unit, so the post-collapse invariant is one distinct label per row. Keeping
    the original labels would leave stale group structure implied by the labels.
    """
    if weight_scheme != "collapse" or groups is None:
        return y, v, X, groups
    _validate_group_design(X, groups)
    collapsed_y, collapsed_v, collapsed_X = collapse_groups(y, v, X, groups, rho=rho)
    _check_collapsed_design(collapsed_y, collapsed_X)
    collapsed_groups = np.arange(collapsed_y.shape[0])
    return collapsed_y, collapsed_v, collapsed_X, collapsed_groups


def _collapse_n_inputs(y, n, X, groups, weight_scheme, rho):
    r"""Return one effect and an effective ``n`` per independent group.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        Estimates.
    n : :obj:`numpy.ndarray` of shape (K, D)
        Sample sizes.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : None or :obj:`numpy.ndarray` of shape (K,)
        Group labels, or None when no dependence was declared.
    weight_scheme : {"individual", "rescale", "collapse"}
        Only ``"collapse"`` triggers aggregation; the others pass through.
    rho : :obj:`float`
        Assumed within-group correlation used by the aggregation.

    Returns
    -------
    :obj:`tuple`
        ``(y, n, X, groups)``, aggregated to one row per group and relabelled
        ``arange(m)`` when ``weight_scheme="collapse"``, else the inputs unchanged.

    See Also
    --------
    pymare.stats.collapse_groups_by_n : Performs the aggregation.
    _collapse_inputs : The counterpart for variance-parameterized models.

    Notes
    -----
    The effective sample size, not the raw one. Rows in a group share subjects,
    so ``n`` must not be counted once per row -- but they are also not perfect
    duplicates, and treating them as such is its own error. For a group of ``s``
    estimates from the same ``n`` subjects correlated at ``rho``,

    .. math::
        \operatorname{Var}(\bar{y}) = \frac{\sigma^2}{n}
            \cdot \frac{1 + \rho(s-1)}{s}
            = \frac{\sigma^2}{n^{\text{eff}}}, \qquad
        n^{\text{eff}} = \frac{sn}{1 + \rho(s-1)},

    which runs from ``n`` at rho=1 up to ``s*n`` at rho=0. Fixing it at ``n`` --
    as counting each group's sample size once does -- is only right for perfectly
    correlated repeats, and biases sigma^2 low by ``(1 + rho(s-1)) / s``
    otherwise: a factor of 4 for four uncorrelated estimates per group. In
    image-based meta-analysis a group is a study and its rows are separate
    contrast images, which share subjects but measure different things, so rho is
    well below one and the bias is real.
    """
    if weight_scheme != "collapse" or groups is None:
        return y, n, X, groups
    _validate_group_design(X, groups)
    collapsed_y, collapsed_n, collapsed_X = collapse_groups_by_n(y, n, X, groups, rho=rho)
    _check_collapsed_design(collapsed_y, collapsed_X)
    collapsed_groups = np.arange(collapsed_y.shape[0])
    return collapsed_y, collapsed_n, collapsed_X, collapsed_groups


def _tau2_inputs(y, v, X, groups, weight_scheme, rho, by_n=False, model=None):
    """Return the (y, v, X) that tau^2 should be estimated from.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        Estimates.
    v : :obj:`numpy.ndarray` of shape (K, D)
        Sampling variances, or sample sizes when ``by_n`` is True.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : None or :obj:`numpy.ndarray` of shape (K,)
        Group labels, or None when no dependence was declared.
    weight_scheme : {"individual", "rescale", "collapse"}
        ``"individual"`` returns the inputs untouched.
    rho : :obj:`float`
        Assumed within-group correlation used by the aggregation.
    by_n : :obj:`bool`, optional
        Whether the second array holds sample sizes rather than variances.
        Default = False.
    model : None or :obj:`str`, optional
        The reduction resolved by :func:`_tau2_model`. Passed in by callers that
        have already recorded it, so that the decision is made once.
        Default = None, meaning resolve it here.

    Returns
    -------
    :obj:`tuple`
        ``(y, v, X)`` aggregated to one row per group, or the inputs unchanged.

    Notes
    -----
    Moment-based tau^2 estimators treat every row as independent, so repeated
    observations from a group distort the observed dispersion relative to what the
    row count implies. Collapsing to one effect per group first removes that
    pseudo-replication.

    Falls back to the raw inputs when there are too few groups to fit the
    collapsed design, rather than raising as :func:`_check_collapsed_design` does;
    the two conditions are the same but only one of them is load-bearing. See that
    function's notes.
    """
    if model is None:
        model = _tau2_model(groups, X, weight_scheme)
    if model != TAU2_AGGREGATE:
        return y, v, X

    # The aggregate replaces each group's predictors with their mean, which is
    # only meaningful when they do not vary within the group.
    _validate_group_design(X, groups, weight_scheme=weight_scheme)
    collapse = collapse_groups_by_n if by_n else collapse_groups
    return collapse(y, v, X, groups, rho=rho)


def _dersimonian_laird_tau2(y, v, X):
    """Estimate tau^2 by the method of moments, from Cochran's Q.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        Estimates.
    v : :obj:`numpy.ndarray` of shape (K, D)
        Sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (D,)
        The tau^2 estimate per parallel dataset, floored at zero.

    Notes
    -----
    ``Q`` is the observed weighted dispersion, whose expectation is ``K - P``
    under tau^2 = 0, so ``Q - (K - P)`` is the excess and ``A`` converts it to the
    tau^2 scale. Because ``K`` counts *rows*, dependent rows make the subtracted
    term grow faster than ``Q`` does; see :func:`_tau2_inputs` for the reduction
    that removes the pseudo-replication before this is called.
    """
    k, p = X.shape

    # Estimate initial betas with WLS, assuming tau^2=0
    beta_wls, model_cov = weighted_least_squares(y, v, X, return_cov=True)

    # Cochran's Q
    w = 1.0 / v
    w_sum = w.sum(0)
    Q = (w * (y - X.dot(beta_wls)) ** 2).sum(0)

    # Einsum indices: k = observations, p = predictors, i = parallel iterates.
    # q is a dummy for 2nd p when p x p covariance matrix is passed.
    Xw2 = np.einsum("kp,ki->ipk", X, w**2)
    pXw2 = np.einsum("ipk,qpi->iqk", Xw2, model_cov)
    A = w_sum - np.trace(pXw2.dot(X), axis1=1, axis2=2)
    return np.maximum(0.0, (Q - (k - p)) / A)


def _robust_cov_and_dof(y, v, X, beta, groups, tau2=0.0, model_cov=None, w=None):
    """Compute the cluster-robust covariance and its degrees of freedom together.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        Estimates.
    v : :obj:`numpy.ndarray` of shape (K, D)
        Sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    beta : :obj:`numpy.ndarray` of shape (P, D)
        The fitted coefficients whose covariance is wanted.
    groups : None or :obj:`numpy.ndarray` of shape (K,)
        Group labels, or None to leave the model-based covariance untouched.
    tau2 : :obj:`float` or :obj:`numpy.ndarray`, optional
        The tau^2 used for the weights, matching the value that produced ``beta``.
        Default = 0.
    model_cov : None or :obj:`numpy.ndarray` of shape (P, P, D), optional
        The model-based covariance, passed through to avoid a redundant
        pseudo-inverse. Default = None.
    w : None or :obj:`numpy.ndarray` of shape (K, D), optional
        The weights that produced ``beta``, so the residuals match the fit.
        Default = None.

    Returns
    -------
    robust_cov : None or :obj:`numpy.ndarray` of shape (P, P, D)
        The sandwich covariance, or None when ``groups`` is None.
    n_groups : None or :obj:`int`
        The number of groups, or None when ``groups`` is None.
    dof : None or :obj:`numpy.ndarray` of shape (P, D)
        Satterthwaite degrees of freedom, or None when ``groups`` is None.

    See Also
    --------
    pymare.stats.cluster_robust_cov
    pymare.stats.satterthwaite_dof

    Notes
    -----
    The degrees of freedom are computed here, alongside the sandwich, so that they
    are guaranteed to reflect the same weights and the same group structure.
    Computing them in ``results`` instead would mean reconstructing both, and any
    drift would silently change the reference distribution rather than raising.

    ``model_cov`` is only reusable when it is ``(X'WX)^-1`` under these same
    weights; callers that cannot promise that must pass None, in which case
    :func:`~pymare.stats.satterthwaite_dof` rebuilds it.

    The group count is returned separately rather than folded into ``params_``,
    which holds only per-dataset arrays. ``dof`` is an array of shape ``(P, D)``,
    so it does travel in ``params_``.
    """
    if groups is None:
        return None, None, None

    groups = np.asarray(groups).ravel()
    n_groups = encode_groups(groups, n_observations=y.shape[0])[1].size
    robust_cov = cluster_robust_cov(y, v, X, beta, groups, tau2=tau2, model_cov=model_cov, w=w)
    # v may be a single shared column while y has many. cluster_robust_cov
    # broadcasts on entry; do the same here or the dof come back with v's
    # column count and no longer line up with fe_params.
    weights = broadcast_columns(1.0 / (v + tau2) if w is None else w, y.shape[1])
    # model_cov is only reusable when it is (X'WX)^-1 under these same weights.
    # Callers that cannot promise that must pass None, in which case
    # satterthwaite_dof rebuilds the bread from ``weights`` itself.
    dof = satterthwaite_dof(X, weights, groups, model_cov=model_cov)
    return robust_cov, n_groups, dof


class BaseEstimator(metaclass=ABCMeta):
    """A base class for Estimators."""

    #: Declarative constraints on the constructor arguments, checked together by
    #: :meth:`_validate_params`. Declaring them means a new parameter is
    #: validated by adding a line here rather than by remembering to write a
    #: check; ``weight_scheme`` was validated this way and ``rho`` was not.
    _parameter_constraints = {}

    def _validate_params(self):
        """Check every declared constraint against the values just assigned.

        Notes
        -----
        Called from ``__init__``, so a typo surfaces at construction rather than
        after the caller has assembled a Dataset. That is the opposite of
        scikit-learn's convention, which defers validation to ``fit`` because
        ``set_params`` would otherwise bypass it -- a rationale that does not
        apply here, since PyMARE estimators expose no ``set_params``.
        """
        for name, constraint in self._parameter_constraints.items():
            constraint.check(name, getattr(self, name))

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
    between-unit variance tau^2, as described in :footcite:t:`brockwell2001comparison`.
    When tau^2 = 0 (default), the model is the standard inverse-weighted fixed-effects
    meta-regression.

    Parameters
    ----------
    tau2 : :obj:`float` or :obj:`numpy.ndarray` of shape (d), optional
        Assumed/known value of tau^2. Must be >= 0.
        If an array, must have ``d`` elements, where ``d`` refers to the number of datasets.
        Default = 0.
    weight_scheme : {"individual", "rescale", "collapse"}, optional
        ``"individual"`` uses one inverse-variance weight per row.
        ``"rescale"`` retains all rows but divides their weights by group
        size for correlated-effects robust estimation. ``"collapse"`` first
        reduces every group to one equal-weight mean and its correlated-mean
        variance. Default is ``"individual"``.
    rho : None or :obj:`float`, optional
        Assumed within-group correlation, used by the schemes that model a
        group. Must lie in [0, 1]. Setting it under ``"individual"``, which
        models no correlation, warns. Default is None, meaning 0.8.

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

    _parameter_constraints = WEIGHTING_CONSTRAINTS

    def __init__(
        self,
        tau2=0.0,
        weight_scheme="individual",
        rho=None,
    ):
        self.tau2 = tau2
        self.weight_scheme = weight_scheme
        self.rho = _resolve_rho(rho, weight_scheme)
        self._validate_params()

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
            Group labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            Point estimates are also grouped when ``weight_scheme='collapse'`` or
            reweighted when ``weight_scheme='rescale'``.

        Returns
        -------
        :obj:`~pymare.estimators.WeightedLeastSquares`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        if v is None:
            v = np.ones_like(y)

        y, v, X, fit_groups = _collapse_inputs(
            ensure_2d(y), ensure_2d(v), X, g, self.weight_scheme, self.rho
        )
        self.tau2_model_ = _tau2_model(fit_groups, X, self.weight_scheme, correlated_effects=True)
        w = _resolve_weights(v, fit_groups, self.tau2, self.weight_scheme)
        beta, model_cov = weighted_least_squares(y, v, X, self.tau2, return_cov=True, w=w)
        robust_cov, self.n_groups_, dof = _robust_cov_and_dof(
            y, v, X, beta, fit_groups, tau2=self.tau2, model_cov=model_cov, w=w
        )
        self.params_ = {
            "fe_params": beta,
            "tau2": self.tau2,
            # NB: the key is a legacy misnomer; the value is a covariance.
            "inv_cov": model_cov if robust_cov is None else robust_cov,
        }
        if dof is not None:
            self.params_["dof"] = dof
        return self


class DerSimonianLaird(BaseEstimator):
    """DerSimonian-Laird meta-regression estimator.

    Estimates the between-unit variance tau^2 using the :footcite:t:`dersimonian1986meta`
    method-of-moments approach.

    Parameters
    ----------
    weight_scheme : {"individual", "rescale", "collapse"}, optional
        Row-level weighting, per-row weights divided by group size, or one
        aggregate per group. Default is ``"individual"``.
    rho : None or :obj:`float`, optional
        Assumed within-group correlation, used by the schemes that model a
        group. Must lie in [0, 1]. Setting it under ``"individual"``, which
        models no correlation, warns. Default is None, meaning 0.8.

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

    _parameter_constraints = WEIGHTING_CONSTRAINTS

    def __init__(self, weight_scheme="individual", rho=None):
        self.weight_scheme = weight_scheme
        self.rho = _resolve_rho(rho, weight_scheme)
        self._validate_params()

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
            Group labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme="rescale"`` tau^2 comes from the
            correlated-effects estimator of
            :func:`~pymare.stats.correlated_effects_tau2`, which reads the
            observation-level design. ``"collapse"`` instead reduces every group
            to one row and fits both tau^2 and the coefficients to those.

        Returns
        -------
        :obj:`~pymare.estimators.DerSimonianLaird`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        y = ensure_2d(y)
        v = ensure_2d(v)

        model_y, model_v, model_X, model_groups = _collapse_inputs(
            y, v, X, g, self.weight_scheme, self.rho
        )

        self.tau2_model_ = _tau2_model(
            model_groups, model_X, self.weight_scheme, correlated_effects=True
        )
        if self.tau2_model_ == TAU2_CORRELATED:
            # The correlated-effects generalization of this estimator's own
            # method of moments, from the observation-level design.
            tau_dl = correlated_effects_tau2(model_y, model_v, model_X, model_groups, rho=self.rho)
        else:
            tau_dl = _dersimonian_laird_tau2(
                *_tau2_inputs(
                    model_y,
                    model_v,
                    model_X,
                    model_groups,
                    self.weight_scheme,
                    self.rho,
                    model=self.tau2_model_,
                )
            )

        # Re-estimate beta with tau^2 estimate
        w = _resolve_weights(model_v, model_groups, tau_dl, self.weight_scheme)
        beta_dl, model_cov = weighted_least_squares(
            model_y, model_v, model_X, tau2=tau_dl, return_cov=True, w=w
        )
        robust_cov, self.n_groups_, dof = _robust_cov_and_dof(
            model_y,
            model_v,
            model_X,
            beta_dl,
            model_groups,
            tau2=tau_dl,
            model_cov=model_cov,
            w=w,
        )
        self.params_ = {
            "fe_params": beta_dl,
            "tau2": tau_dl,
            # NB: the key is a legacy misnomer; the value is a covariance.
            "inv_cov": model_cov if robust_cov is None else robust_cov,
        }
        if dof is not None:
            self.params_["dof"] = dof
        return self


class Hedges(BaseEstimator):
    """Hedges meta-regression estimator.

    Estimates the between-unit variance tau^2 using the :footcite:t:`hedges2014statistical`
    approach.

    Parameters
    ----------
    weight_scheme : {"individual", "rescale", "collapse"}, optional
        Row-level weighting, per-row weights divided by group size, or one
        aggregate per group. Default is ``"individual"``.
    rho : None or :obj:`float`, optional
        Assumed within-group correlation, used by the schemes that model a
        group. Must lie in [0, 1]. Setting it under ``"individual"``, which
        models no correlation, warns. Default is None, meaning 0.8.

    Notes
    -----
    This estimator accepts 2-D inputs for ``y`` and ``v``--i.e., it can produce estimates
    simultaneously for multiple independent sets of ``y``/``v`` values
    (use the 2nd dimension for the parallel iterates).
    The ``X`` matrix must be identical for all iterates.

    Unlike the coefficients, tau^2 is derived from an *unweighted* fit: it is the excess of
    the ordinary mean squared error over the mean sampling variance. The coefficients are
    then refitted with ``1 / (v + tau^2)`` weights, and the reported covariance comes from
    that second fit.

    .. versionchanged:: 0.0.11

        The reported model-based covariance was previously taken from the unweighted fit
        used to obtain tau^2, so the standard errors did not correspond to the reported
        coefficients and were substantially too small. They are now ``(X'WX)^-1`` under the
        same ``1 / (v + tau^2)`` weights that produce the coefficients, matching
        :obj:`~pymare.estimators.DerSimonianLaird` and the ``metafor`` R package. Point
        estimates and tau^2 are unaffected.

    References
    ----------
    .. footbibliography::
    """

    _parameter_constraints = WEIGHTING_CONSTRAINTS

    def __init__(self, weight_scheme="individual", rho=None):
        self.weight_scheme = weight_scheme
        self.rho = _resolve_rho(rho, weight_scheme)
        self._validate_params()

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
            Group labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme`` set to ``"rescale"`` or ``"collapse"``,
            tau^2 is estimated from one aggregate per group, which requires the
            design to be constant within a group. ``"collapse"`` also fits the
            fixed effects to those aggregates. Use
            :class:`~pymare.estimators.DerSimonianLaird` for a design that varies
            within a group.

        Returns
        -------
        :obj:`~pymare.estimators.Hedges`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        y = ensure_2d(y)
        v = ensure_2d(v)
        model_y, model_v, model_X, model_groups = _collapse_inputs(
            y, v, X, g, self.weight_scheme, self.rho
        )

        self.tau2_model_ = _tau2_model(model_groups, model_X, self.weight_scheme)
        tau_y, tau_v, tau_X = _tau2_inputs(
            model_y,
            model_v,
            model_X,
            model_groups,
            self.weight_scheme,
            self.rho,
            model=self.tau2_model_,
        )
        tau_k, tau_p = tau_X.shape[:2]
        # tau^2 is the excess of the *unweighted* mean squared error over the
        # mean sampling variance, so this initial fit is deliberately OLS. It
        # feeds the variance component only; the coefficients are refitted with
        # inverse-variance weights below.
        tau_beta = weighted_least_squares(tau_y, np.ones_like(tau_y), tau_X)
        mse = ((tau_y - tau_X.dot(tau_beta)) ** 2).sum(0) / (tau_k - tau_p)
        tau_ho = np.maximum(0, mse - tau_v.sum(0) / tau_k)

        # Estimate beta with tau^2 estimate. The covariance has to come from
        # this fit rather than the OLS one above: (X'WX)^-1 is only the
        # covariance of the coefficients computed under those same weights.
        w = _resolve_weights(model_v, model_groups, tau_ho, self.weight_scheme)
        beta_ho, model_cov = weighted_least_squares(
            model_y, model_v, model_X, tau2=tau_ho, return_cov=True, w=w
        )
        robust_cov, self.n_groups_, dof = _robust_cov_and_dof(
            model_y,
            model_v,
            model_X,
            beta_ho,
            model_groups,
            tau2=tau_ho,
            model_cov=model_cov,
            w=w,
        )
        self.params_ = {
            "fe_params": beta_ho,
            "tau2": tau_ho,
            # NB: the key is a legacy misnomer; the value is a covariance.
            "inv_cov": model_cov if robust_cov is None else robust_cov,
        }
        if dof is not None:
            self.params_["dof"] = dof
        return self


class VarianceBasedLikelihoodEstimator(BaseEstimator):
    """Likelihood-based estimator for estimates with known variances.

    Initially estimates the between-unit variance tau^2 and fixed effect coefficients
    using :footcite:t:`dersimonian1986meta` method-of-moments approach, and then
    iteratively estimates them using the specified likelihood-based estimator (ML or REML)
    :footcite:p:`kosmidis2017improving`.

    Parameters
    ----------
    method : {"ML", "REML"}, optional
        The estimation method to use.
        Either 'ML' (for maximum-likelihood) or 'REML' (restricted maximum-likelihood).
        Default = 'ML'.
    weight_scheme : {"individual", "rescale", "collapse"}, optional
        Row-level weighting, per-row weights divided by group size, or one
        aggregate per group. Default is ``"individual"``.
    rho : None or :obj:`float`, optional
        Assumed within-group correlation, used by the schemes that model a
        group. Must lie in [0, 1]. Setting it under ``"individual"``, which
        models no correlation, warns. Default is None, meaning 0.8.
    **kwargs
        Keyword arguments to pass to :func:`~pymare.stats.bounded_scalar_min`,
        which searches for the variance components (e.g., ``xtol``).

    Notes
    -----
    The coefficients are profiled out of the likelihood, leaving tau^2 as the
    only free parameter, and that one-dimensional search is run for every
    parallel dataset at once by :func:`~pymare.stats.bounded_scalar_min`. The
    coefficients reported are the weighted least-squares solution at the fitted
    tau^2, which is what profiling them out makes them.

    References
    ----------
    .. footbibliography::
    """

    _parameter_constraints = WEIGHTING_CONSTRAINTS

    def __init__(self, method="ml", weight_scheme="individual", rho=None, **kwargs):
        self.method = method
        self.weight_scheme = weight_scheme
        self.rho = _resolve_rho(rho, weight_scheme)
        self._validate_params()
        nll_func = getattr(self, "_{}_profile_nll".format(method.lower()), None)
        if nll_func is None:
            raise ValueError("No log-likelihood function defined for method '{}'.".format(method))

        self._nll_func = nll_func
        self.kwargs = kwargs

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
            Group labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme`` set to ``"rescale"`` or ``"collapse"``, the
            likelihood is evaluated on one aggregate per group. ``"collapse"``
            also fits the fixed effects to those aggregates.

        Returns
        -------
        :obj:`~pymare.estimators.VarianceBasedLikelihoodEstimator`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        y = ensure_2d(y)
        # A single column of v applies to every parallel dataset; expand it once
        # here so everything downstream sees one shape.
        v = broadcast_columns(ensure_2d(v), y.shape[1])
        model_y, model_v, model_X, model_groups = _collapse_inputs(
            y, v, X, g, self.weight_scheme, self.rho
        )

        # The likelihood treats every row as independent, so tau^2 is fitted
        # on one effect per group to avoid counting a group's repeated
        # estimates as independent evidence.
        self.tau2_model_ = _tau2_model(model_groups, model_X, self.weight_scheme)
        fit_y, fit_v, fit_X = _tau2_inputs(
            model_y,
            model_v,
            model_X,
            model_groups,
            self.weight_scheme,
            self.rho,
            model=self.tau2_model_,
        )

        # The D-L moment estimate sets the scale of the search rather than a
        # starting value: it places the scan where tau^2 plausibly lies, which is
        # the vectorized counterpart of warm-starting a per-dataset optimizer.
        scale = _search_scale(_dersimonian_laird_tau2(fit_y, fit_v, fit_X) + fit_v.mean(axis=0))
        u, _ = bounded_scalar_min(
            lambda t: self._nll_func(_tau2_from_search(t, scale), fit_y, fit_v, fit_X),
            np.zeros(fit_y.shape[1]),
            np.full(fit_y.shape[1], _SEARCH_MAX),
            **self.kwargs,
        )
        tau2 = _tau2_from_search(u, scale)

        w = _resolve_weights(model_v, model_groups, tau2, self.weight_scheme)
        # The coefficients are the weighted least-squares solution at the fitted
        # tau^2 -- profiling them out of the likelihood is what makes them that --
        # computed here on the row-level model rather than on the inputs tau^2 was
        # fitted to, which may have been aggregated. Under cluster weighting they
        # have to come from these weights in any case, because that weighting
        # changes the estimand.
        beta, model_cov = weighted_least_squares(model_y, model_v, model_X, tau2, True, w=w)
        robust_cov, self.n_groups_, dof = _robust_cov_and_dof(
            model_y,
            model_v,
            model_X,
            beta,
            model_groups,
            tau2=tau2,
            model_cov=model_cov,
            w=w,
        )
        self.params_ = {
            "fe_params": beta,
            # Kept 2d, as the per-dataset loop this replaced used to leave it.
            "tau2": np.atleast_2d(tau2),
            # NB: the key is a legacy misnomer; the value is a covariance.
            "inv_cov": model_cov if robust_cov is None else robust_cov,
        }
        if dof is not None:
            self.params_["dof"] = dof
        return self

    @staticmethod
    def _profile_nll(w, resid):
        """Compute the ML negative log-likelihood from the weights and residuals.

        Parameters
        ----------
        w : :obj:`numpy.ndarray` of shape (K, D)
            The weights ``1 / (v + tau^2)``.
        resid : :obj:`numpy.ndarray` of shape (K, D)
            Residuals from the weighted least-squares fit under those weights.

        Returns
        -------
        :obj:`numpy.ndarray` of shape (D,)
            The negative log-likelihood per dataset, up to an additive constant.
        """
        return -0.5 * (np.log(w).sum(axis=0) - (resid * w * resid).sum(axis=0))

    def _profile_fit(self, tau2, y, v, X):
        """Fit the coefficients at a fixed tau^2 and report what the likelihood needs.

        Parameters
        ----------
        tau2 : :obj:`numpy.ndarray` of shape (D,)
            One candidate tau^2 per parallel dataset.
        y : :obj:`numpy.ndarray` of shape (K, D)
            Estimates.
        v : :obj:`numpy.ndarray` of shape (K, D)
            Sampling variances.
        X : :obj:`numpy.ndarray` of shape (K, P)
            Fixed effect design matrix.

        Returns
        -------
        w : :obj:`numpy.ndarray` of shape (K, D)
            The weights.
        resid : :obj:`numpy.ndarray` of shape (K, D)
            Residuals from the weighted least-squares fit.
        cov_beta : :obj:`numpy.ndarray` of shape (P, P, D)
            ``(X'WX)^-1``, which REML needs and ML ignores.

        Notes
        -----
        The weights are computed once here and handed to
        :func:`~pymare.stats.weighted_least_squares` rather than recomputed
        inside it. This runs once per evaluation of the objective and the
        objective is evaluated dozens of times per fit, so the arrays it touches
        are worth touching once.
        """
        w = 1.0 / (v + tau2)
        beta, cov_beta = weighted_least_squares(y, v, X, return_cov=True, w=w)
        return w, y - X.dot(beta), cov_beta

    def _ml_profile_nll(self, tau2, y, v, X):
        """Compute the ML negative log-likelihood, profiled over the coefficients.

        Parameters
        ----------
        tau2 : :obj:`numpy.ndarray` of shape (D,)
            One candidate tau^2 per parallel dataset.
        y : :obj:`numpy.ndarray` of shape (K, D)
            Estimates.
        v : :obj:`numpy.ndarray` of shape (K, D)
            Sampling variances.
        X : :obj:`numpy.ndarray` of shape (K, P)
            Fixed effect design matrix.

        Returns
        -------
        :obj:`numpy.ndarray` of shape (D,)
            The negative log-likelihood per dataset, up to an additive constant.

        Notes
        -----
        For a fixed tau^2 the likelihood is maximized over the coefficients by
        their weighted least-squares solution, so substituting it leaves a
        function of tau^2 alone. Minimizing that is equivalent to minimizing the
        joint likelihood over the coefficients and tau^2 together, and it turns
        the fit into a one-dimensional search that can be run for every parallel
        dataset at once.
        """
        w, resid, _ = self._profile_fit(tau2, y, v, X)
        return self._profile_nll(w, resid)

    def _reml_profile_nll(self, tau2, y, v, X):
        """Compute the REML negative log-likelihood, profiled over the coefficients.

        Parameters
        ----------
        tau2 : :obj:`numpy.ndarray` of shape (D,)
            One candidate tau^2 per parallel dataset.
        y : :obj:`numpy.ndarray` of shape (K, D)
            Estimates.
        v : :obj:`numpy.ndarray` of shape (K, D)
            Sampling variances.
        X : :obj:`numpy.ndarray` of shape (K, P)
            Fixed effect design matrix.

        Returns
        -------
        :obj:`numpy.ndarray` of shape (D,)
            The negative restricted log-likelihood per dataset, up to an additive
            constant.

        Notes
        -----
        The restriction term ``0.5 * log|X'WX|`` does not involve the
        coefficients, so profiling them out of the ML part is unaffected by it.
        It is read off the covariance the fit already produced --
        ``log|X'WX| = -log|(X'WX)^-1|`` -- rather than by forming ``X'WX`` a
        second time. ``slogdet`` is the stable form of ``log(det(...))``.
        """
        w, resid, cov_beta = self._profile_fit(tau2, y, v, X)
        return self._profile_nll(w, resid) - 0.5 * np.linalg.slogdet(cov_beta.T)[1]


class SampleSizeBasedLikelihoodEstimator(BaseEstimator):
    """Likelihood-based estimator for data with known sample sizes but unknown sampling variances.

    Iteratively estimates the between-unit variance tau^2 and fixed effect betas using the
    specified likelihood-based estimator (ML or REML) :footcite:p:`sangnawakij2019meta`.

    Parameters
    ----------
    method : {"ML", "REML"}, optional
        The estimation method to use.
        Either 'ML' (for maximum-likelihood) or 'REML' (restricted maximum-likelihood).
        Default = 'ML'.
    weight_scheme : {"individual", "rescale", "collapse"}, optional
        Row-level, correlated-effects, or one-aggregate-per-group weighting.
        In ``"collapse"`` mode a group's rows are replaced by one row carrying
        the effective sample size ``s*n / (1 + rho(s-1))``, so their ``n`` may
        differ. Default is ``"individual"``.
    rho : None or :obj:`float`, optional
        Assumed within-group correlation, used by the schemes that model a
        group. Must lie in [0, 1]. Setting it under ``"individual"``, which
        models no correlation, warns. Default is None, meaning 0.8.
    **kwargs
        Keyword arguments to pass to :func:`~pymare.stats.bounded_scalar_min`,
        which searches for the variance components (e.g., ``xtol``).

    Notes
    -----
    Homogeneity of sigma^2 across input units is assumed.

    The coefficients and the overall scale of the two variance components are
    profiled out of the likelihood, leaving their ratio
    ``tau^2 / (tau^2 + sigma^2)`` as the only free parameter, and that
    one-dimensional search is run for every parallel dataset at once by
    :func:`~pymare.stats.bounded_scalar_min`.

    References
    ----------
    .. footbibliography::
    """

    _parameter_constraints = WEIGHTING_CONSTRAINTS

    def __init__(self, method="ml", weight_scheme="individual", rho=None, **kwargs):
        self.method = method
        self.weight_scheme = weight_scheme
        self.rho = _resolve_rho(rho, weight_scheme)
        self._validate_params()
        nll_func = getattr(self, "_{}_profile_nll".format(method.lower()), None)
        if nll_func is None:
            raise ValueError("No log-likelihood function defined for method '{}'.".format(method))

        self._nll_func = nll_func
        self.kwargs = kwargs

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
            Group labels marking dependent estimates. If provided,
            standard errors are computed with the cluster-robust estimator of
            :footcite:t:`hedges2010robust` instead of the model-based one.
            With ``weight_scheme`` set to ``"rescale"`` or ``"collapse"``, the
            variance components are estimated from one aggregate per group.
            ``"collapse"`` preserves one unchanged ``n`` value per group and
            also fits the fixed effects to those aggregates.

        Returns
        -------
        :obj:`~pymare.estimators.SampleSizeBasedLikelihoodEstimator`
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        y = ensure_2d(y)
        # A single column of n applies to every parallel dataset; expand it once
        # here so everything downstream sees one shape.
        n = broadcast_columns(ensure_2d(n), y.shape[1])
        model_y, model_n, model_X, model_groups = _collapse_n_inputs(
            y, n, X, g, self.weight_scheme, self.rho
        )

        # Both variance components are fitted on one effect per group; a
        # a group's repeated estimates agree with each other by construction, and
        # counting them as independent shrinks sigma^2 toward zero.
        self.tau2_model_ = _tau2_model(model_groups, model_X, self.weight_scheme)
        fit_y, fit_n, fit_X = _tau2_inputs(
            model_y,
            model_n,
            model_X,
            model_groups,
            self.weight_scheme,
            self.rho,
            by_n=True,
            model=self.tau2_model_,
        )

        # sigma^2 and tau^2 are separately identified only when the sample
        # sizes vary, so the check belongs on ``fit_n`` -- the values the
        # likelihood actually sees. Under weight_scheme='rescale' those are
        # effective sample sizes, which can vary sharply even when every raw
        # ``n`` is identical (and, less often, the reverse).
        if fit_n.std() < np.sqrt(np.finfo(float).eps):
            raise ValueError(
                "Sample size-based likelihood estimator cannot "
                "work with all-equal sample sizes."
            )

        if fit_n.std() < fit_n.mean() / 10:
            # ``raise Warning`` aborts the fit instead of warning about it.
            warn(
                "Sample sizes are too close, sample size-based likelihood estimator may fail.",
                UserWarning,
            )

        # Only the ratio of the two variance components is searched for; the
        # scale they share and the coefficients both have closed forms at a fixed
        # ratio, so there is one bounded parameter left rather than three.
        ratio, _ = bounded_scalar_min(
            lambda r: self._nll_func(r, fit_y, fit_n, fit_X),
            np.zeros(fit_y.shape[1]),
            np.ones(fit_y.shape[1]),
            **self.kwargs,
        )
        v_unit = self._unit_variance(ratio, fit_n)
        scale, _ = self._profile_at_ratio(v_unit, fit_y, fit_X, ddof=self._profile_ddof(fit_X))
        sigma2 = scale * (1.0 - ratio)
        tau2 = scale * ratio

        v = sigma2 / model_n
        w = _resolve_weights(v, model_groups, tau2, self.weight_scheme)
        # The coefficients are the weighted least-squares solution at the fitted
        # variance components -- profiling them out of the likelihood is what makes
        # them that -- computed here on the row-level model rather than on the
        # inputs the components were fitted to, which may have been aggregated.
        # Under cluster weighting they have to come from these weights in any
        # case, because that weighting changes the estimand.
        beta, model_cov = weighted_least_squares(model_y, v, model_X, tau2, True, w=w)
        robust_cov, self.n_groups_, dof = _robust_cov_and_dof(
            model_y,
            v,
            model_X,
            beta,
            model_groups,
            tau2=tau2,
            model_cov=model_cov,
            w=w,
        )
        self.params_ = {
            "fe_params": beta,
            # Kept 2d, as the per-dataset loop these replaced used to leave them.
            "sigma2": np.atleast_2d(sigma2),
            "tau2": np.atleast_2d(tau2),
            # NB: the key is a legacy misnomer; the value is a covariance.
            "inv_cov": model_cov if robust_cov is None else robust_cov,
        }
        if dof is not None:
            self.params_["dof"] = dof
        return self

    def _profile_ddof(self, X):
        """Return the degrees of freedom the profiled scale loses under this method.

        Parameters
        ----------
        X : :obj:`numpy.ndarray` of shape (K, P)
            Fixed effect design matrix.

        Returns
        -------
        :obj:`int`
            ``P`` under REML, whose restriction term contributes
            ``-P log(scale)``, and 0 under ML.
        """
        return X.shape[1] if self.method.lower() == "reml" else 0

    @staticmethod
    def _unit_variance(ratio, n):
        """Return the observation variances at a fixed variance ratio, up to their scale.

        Parameters
        ----------
        ratio : :obj:`numpy.ndarray` of shape (D,)
            ``tau^2 / (tau^2 + sigma^2)`` per parallel dataset.
        n : :obj:`numpy.ndarray` of shape (K, D)
            Sample sizes.

        Returns
        -------
        :obj:`numpy.ndarray` of shape (K, D)
            ``(tau^2 + sigma^2 / n) / (tau^2 + sigma^2)``.

        Notes
        -----
        Writing the variances this way separates the shape of the weights, which
        the ratio fixes, from their overall scale, which drops out of the
        weighted least-squares fit and has a closed form in the likelihood. See
        :meth:`_profile_at_ratio`.
        """
        return ratio + (1.0 - ratio) / n

    def _profile_at_ratio(self, v_unit, y, X, ddof=0):
        """Return the total variance that maximizes the likelihood, and the covariance.

        Parameters
        ----------
        v_unit : :obj:`numpy.ndarray` of shape (K, D)
            Observation variances up to their scale, from :meth:`_unit_variance`.
        y : :obj:`numpy.ndarray` of shape (K, D)
            Estimates.
        X : :obj:`numpy.ndarray` of shape (K, P)
            Fixed effect design matrix.
        ddof : :obj:`int`, optional
            Degrees of freedom to subtract from ``K``: 0 for ML, ``P`` for REML.
            Default = 0.

        Returns
        -------
        scale : :obj:`numpy.ndarray` of shape (D,)
            ``tau^2 + sigma^2`` per dataset.
        cov_beta : :obj:`numpy.ndarray` of shape (P, P, D)
            ``(X'WX)^-1`` under these weights, which REML needs and ML ignores.
            Returned alongside the scale because both come out of the one
            weighted least-squares fit, and the objective is evaluated dozens of
            times per fit.

        Notes
        -----
        Scaling both variance components by the same factor divides every weight
        by that factor, so the weighted dispersion the likelihood penalizes and
        the log-determinant term both depend on the scale in closed form. Solving
        for it gives the weighted residual sum of squares over ``K - ddof``, the
        familiar variance estimate, and leaves the ratio as the only parameter
        that has to be searched for. The ``ddof`` term is what distinguishes REML
        from ML here: the restriction term contributes ``-P log(scale)``.

        Floored at the smallest positive double, so that a saturated design whose
        residuals vanish gives a degenerate fit rather than ``log(0)``.
        """
        beta, cov_beta = weighted_least_squares(y, v_unit, X, return_cov=True)
        resid = y - X.dot(beta)
        dispersion = np.maximum((resid**2 / v_unit).sum(axis=0), np.finfo(float).tiny)
        return dispersion / (X.shape[0] - ddof), cov_beta

    def _ml_profile_nll(self, ratio, y, n, X):
        """Compute the ML negative log-likelihood, profiled over everything but the ratio.

        Parameters
        ----------
        ratio : :obj:`numpy.ndarray` of shape (D,)
            One candidate ``tau^2 / (tau^2 + sigma^2)`` per parallel dataset.
        y : :obj:`numpy.ndarray` of shape (K, D)
            Estimates.
        n : :obj:`numpy.ndarray` of shape (K, D)
            Sample sizes.
        X : :obj:`numpy.ndarray` of shape (K, P)
            Fixed effect design matrix.

        Returns
        -------
        :obj:`numpy.ndarray` of shape (D,)
            The negative log-likelihood per dataset, up to an additive constant.

        Notes
        -----
        The coefficients and the scale of the variance components are both
        replaced by their maximizing values at this ratio, which is equivalent to
        minimizing the joint likelihood over all three and leaves a bounded
        one-dimensional search. See :meth:`_profile_at_ratio`.
        """
        v_unit = self._unit_variance(ratio, n)
        k = X.shape[0]
        scale, _ = self._profile_at_ratio(v_unit, y, X)
        return 0.5 * (np.log(v_unit).sum(axis=0) + k * np.log(scale) + k)

    def _reml_profile_nll(self, ratio, y, n, X):
        """Compute the REML negative log-likelihood, profiled over everything but the ratio.

        Parameters
        ----------
        ratio : :obj:`numpy.ndarray` of shape (D,)
            One candidate ``tau^2 / (tau^2 + sigma^2)`` per parallel dataset.
        y : :obj:`numpy.ndarray` of shape (K, D)
            Estimates.
        n : :obj:`numpy.ndarray` of shape (K, D)
            Sample sizes.
        X : :obj:`numpy.ndarray` of shape (K, P)
            Fixed effect design matrix.

        Returns
        -------
        :obj:`numpy.ndarray` of shape (D,)
            The negative restricted log-likelihood per dataset, up to an additive
            constant.

        Notes
        -----
        The restriction term ``0.5 * log|X'WX|`` contributes ``-P log(scale)``,
        which is why the scale is the residual sum of squares over ``K - P`` here
        and over ``K`` for ML. It does not involve the coefficients, so profiling
        those out is unaffected by it. It is read off the covariance the fit
        already produced -- ``log|X'WX| = -log|(X'WX)^-1|`` -- rather than by
        forming ``X'WX`` a second time. ``slogdet`` is the stable form of
        ``log(det(...))``.
        """
        v_unit = self._unit_variance(ratio, n)
        k, p = X.shape
        scale, cov_beta = self._profile_at_ratio(v_unit, y, X, ddof=p)
        return 0.5 * (
            np.log(v_unit).sum(axis=0)
            + (k - p) * np.log(scale)
            + (k - p)
            - np.linalg.slogdet(cov_beta.T)[1]
        )


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
            1d array of observation-level estimates
        v : :obj:`numpy.ndarray` of shape (K,)
            1d array of observation-level variances
        X : :obj:`numpy.ndarray` of shape (K[, P])
            1d or 2d array containing observation-level predictors
            (including intercept); has dimensions K x P, where K is the
            number of observations and P is the number of predictor variables.
        groups : :obj:`list` of :obj:`int`, optional
            1d array of integers identifying
            groups of observations in the y/v/X inputs. If
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
        observations belong to at least one common sampling unit, the `groups`
        argument can specify the nesting structure (i.e., which rows in `y`,
        `v`, and `X` belong to each group).
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
        K = encode_groups(np.asarray(groups).ravel())[1].size

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
