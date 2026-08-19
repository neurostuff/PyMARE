"""Meta-regression estimator classes."""

import os
import os.path as op
import shutil
from abc import ABCMeta, abstractmethod
from inspect import getfullargspec, signature
from warnings import warn

import numpy as np
import wrapt
from scipy.optimize import Bounds, minimize

from ..results import BayesianMetaRegressionResults, MetaRegressionResults
from ..stats import (
    DEFAULT_RHO,
    TAU2_AGGREGATE,
    TAU2_CORRELATED,
    TAU2_INDEPENDENT,
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


class Options:
    """Constrain a parameter to a fixed set of values."""

    def __init__(self, allowed):
        self.allowed = tuple(allowed)

    def check(self, name, value):
        """Raise if ``value`` is not one of the allowed options."""
        if value not in self.allowed:
            raise ValueError(f"Invalid {name} '{value}'; must be one of {list(self.allowed)}.")


class Interval:
    """Constrain a parameter to a numeric interval.

    Parameters
    ----------
    low, high : :obj:`float`
        The endpoints. Use ``np.inf`` for an unbounded side.
    closed : {"both", "left", "right", "neither"}, optional
        Which endpoints are themselves allowed. Default = "both".
    allow_none : :obj:`bool`, optional
        Whether ``None`` passes the check, for a parameter whose default is
        resolved from the data at fit time. Default = False.
    """

    #: Bracket characters per ``closed`` value, so the error message shows the
    #: same interval notation the constraint was declared with.
    _BRACKETS = {
        "both": ("[", "]"),
        "left": ("[", ")"),
        "right": ("(", "]"),
        "neither": ("(", ")"),
    }

    def __init__(self, low, high, closed="both", allow_none=False):
        if closed not in self._BRACKETS:
            raise ValueError(f"Invalid closed {closed!r}; must be one of {list(self._BRACKETS)}.")
        self.low = low
        self.high = high
        self.closed = closed
        self.allow_none = allow_none

    def check(self, name, value):
        """Raise if ``value`` is not a real number inside the interval."""
        if value is None:
            if self.allow_none:
                return
            raise ValueError(f"Invalid {name} None; must be a number.")
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"Invalid {name} {value!r}; must be a number.")
        value = float(value)
        low_ok = self.low <= value if self.closed in ("both", "left") else self.low < value
        high_ok = value <= self.high if self.closed in ("both", "right") else value < self.high
        if not (low_ok and high_ok):
            left, right = self._BRACKETS[self.closed]
            raise ValueError(
                f"Invalid {name} {value!r}; must lie in {left}{self.low}, {self.high}{right}."
            )


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

    The group count is returned separately rather than folded into ``params_``
    because :func:`_loopable` stacks every entry of ``params_`` across parallel
    datasets, which only works for arrays. ``dof`` is an array of shape ``(P, D)``,
    so it stacks correctly and does travel in ``params_``.
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


@wrapt.decorator
def _loopable(wrapped, instance, args, kwargs):
    """Decorate fit() method of Estimator classes.

    Designed to handle naive looping over the 2nd dimension of y/v/n inputs, and reconstruction of
    outputs.
    """
    # fit() is routinely called positionally; binding against the wrapped
    # signature makes both forms work rather than raising KeyError('y').
    if args:
        bound = signature(wrapped).bind(*args, **kwargs)
        bound.apply_defaults()
        kwargs = dict(bound.arguments)
        args = ()

    n_iter = kwargs["y"].shape[1]
    # A single column of v or n applies to every parallel dataset. Expand it
    # once here so the loop below has one shape to slice, rather than each
    # argument carrying its own convention.
    for name in ("v", "n"):
        if kwargs.get(name) is not None:
            kwargs[name] = broadcast_columns(kwargs[name], n_iter)
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
            iter_kwargs["n"] = kwargs["n"][:, i, None]

        # Group labels are per-observation, not per-dataset, so they are shared
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

    # A class-level mapping from fit() arguments to Dataset attributes. Used by
    # fit_dataset() for estimators that take non-standard arguments (e.g., 'z'
    # instead of 'y'). Keys are the argument names in the estimator class's
    # fit() method and values are the Dataset attributes they are filled from,
    # so {'z': 'y'} reads "fit()'s z argument takes dataset.y". An argument
    # absent from the mapping is filled from the attribute of the same name.
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

    _parameter_constraints = WEIGHTING_CONSTRAINTS

    def __init__(self, method="ml", weight_scheme="individual", rho=None, **kwargs):
        self.weight_scheme = weight_scheme
        self.rho = _resolve_rho(rho, weight_scheme)
        self._validate_params()
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
        v = ensure_2d(v)
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
        w = _resolve_weights(model_v, model_groups, tau, self.weight_scheme)
        if w is None:
            _, model_cov = weighted_least_squares(model_y, model_v, model_X, tau, True)
        else:
            # Cluster weighting changes the estimand, so beta has to be
            # recomputed under those weights rather than kept from the
            # likelihood, whose working model assumes independence.
            beta, model_cov = weighted_least_squares(model_y, model_v, model_X, tau, True, w=w)
        robust_cov, self.n_groups_, dof = _robust_cov_and_dof(
            model_y,
            model_v,
            model_X,
            beta,
            model_groups,
            tau2=tau,
            model_cov=model_cov,
            w=w,
        )
        self.params_ = {
            "fe_params": beta,
            "tau2": tau,
            # NB: the key is a legacy misnomer; the value is a covariance.
            "inv_cov": model_cov if robust_cov is None else robust_cov,
        }
        if dof is not None:
            self.params_["dof"] = dof
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
        Keyword arguments to pass to the SciPy minimizer.

    Notes
    -----
    Homogeneity of sigma^2 across input units is assumed.

    The ML and REML solutions are obtained via SciPy's scalar function minimizer
    (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    References
    ----------
    .. footbibliography::
    """

    _parameter_constraints = WEIGHTING_CONSTRAINTS

    def __init__(self, method="ml", weight_scheme="individual", rho=None, **kwargs):
        self.weight_scheme = weight_scheme
        self.rho = _resolve_rho(rho, weight_scheme)
        self._validate_params()
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
        n = ensure_2d(n)
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
        v = sigma / model_n
        w = _resolve_weights(v, model_groups, tau, self.weight_scheme)
        if w is None:
            _, model_cov = weighted_least_squares(model_y, v, model_X, tau, True)
        else:
            # Cluster weighting changes the estimand, so beta has to be
            # recomputed under those weights rather than kept from the
            # likelihood, whose working model assumes independence.
            beta, model_cov = weighted_least_squares(model_y, v, model_X, tau, True, w=w)
        robust_cov, self.n_groups_, dof = _robust_cov_and_dof(
            model_y,
            v,
            model_X,
            beta,
            model_groups,
            tau2=tau,
            model_cov=model_cov,
            w=w,
        )
        self.params_ = {
            "fe_params": beta,
            "sigma2": np.array(sigma),
            "tau2": tau,
            # NB: the key is a legacy misnomer; the value is a covariance.
            "inv_cov": model_cov if robust_cov is None else robust_cov,
        }
        if dof is not None:
            self.params_["dof"] = dof
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
        # Clamp as _ml_nll does; the Bounds keep these non-negative today, but
        # the two halves of the objective should not disagree about that.
        sigma2, tau2 = np.maximum(theta[-2:], 0.0)
        w = 1 / (tau2 + sigma2 / n)
        F = (X * w).T.dot(X)
        return ll_ + 0.5 * np.log(np.linalg.det(F))


#: Location of the Stan program compiled by :obj:`~pymare.estimators.StanMetaRegression`.
#: CmdStanPy needs a real filesystem path -- it hands the file to ``make`` -- so this
#: is a plain join rather than ``importlib.resources``, which would need ``as_file``
#: to materialize a path PyMARE never needs because it is not zip-imported.
STAN_MODEL_PATH = op.join(op.dirname(__file__), "stan", "meta_regression.stan")

#: Sampler arguments that PyStan named differently from CmdStanPy. Mapped rather
#: than silently ignored: passing ``num_samples`` to CmdStanPy's ``sample()``
#: raises a bare ``TypeError`` naming no alternative, and every PyMARE example
#: and test written against the old backend used these names.
PYSTAN_SAMPLING_KWARGS = {
    "num_samples": "iter_sampling",
    "num_warmup": "iter_warmup",
    "num_chains": "chains",
    "num_thin": "thin",
}


def _import_cmdstanpy():
    """Return the ``cmdstanpy`` module, or raise naming the step that is missing.

    Returns
    -------
    module
        The imported ``cmdstanpy`` module.

    Raises
    ------
    ImportError
        If ``cmdstanpy`` is not installed, or if it is installed but no CmdStan
        installation can be found.

    Notes
    -----
    The two failures are reported separately because their fixes are different
    and neither implies the other: ``pip install cmdstanpy`` succeeds without
    installing CmdStan itself, which is a C++ toolchain build rather than a
    Python package.
    """
    try:
        import cmdstanpy
    except ImportError:
        raise ImportError(
            "StanMetaRegression requires cmdstanpy, which is an optional dependency. "
            "Install it with `pip install pymare[stan]`."
        )

    try:
        cmdstanpy.cmdstan_path()
    except Exception as exc:
        raise ImportError(
            "cmdstanpy is installed, but no CmdStan installation was found. Install one "
            "with `python -m cmdstanpy.install_cmdstan` (this downloads and builds CmdStan, "
            "and needs a C++ toolchain), or point the CMDSTAN environment variable at an "
            f"existing installation. cmdstanpy reported: {exc}"
        )

    return cmdstanpy


def _build_stan_data(y, v, X, groups=None, tau_prior_scale=None):
    """Canonicalize the estimator's inputs into the data block of the Stan program.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Observation-level estimates.
    v : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Observation-level sampling *variances*.
    X : :obj:`numpy.ndarray` of shape (K,) or (K, P)
        Observation-level predictors, including the intercept.
    groups : None or array-like of shape (K,) or (K, 1), optional
        One scalar label per observation -- a string, integer or any other
        hashable that numpy stores as a single element. Composite labels such
        as tuples are not accepted, because numpy reads a sequence of them as a
        second dimension. When None (default), every observation is its own
        group.
    tau_prior_scale : None or :obj:`float`, optional
        Scale of the half-normal prior on tau. When None (default), it is set to
        ``max(std(y), sqrt(mean(v)))``: the larger of the observed spread of the
        estimates and the typical sampling standard deviation.

    Returns
    -------
    :obj:`dict`
        The data block, with every array in the shape and units the Stan program
        declares.

    Raises
    ------
    ValueError
        If ``y`` is 2-dimensional with more than one column, if ``v`` or ``X``
        disagree with ``y`` about the number of observations, or if any sampling
        variance is not positive.

    Notes
    -----
    Every shape and unit decision the Stan program depends on is made here and
    nowhere else, so ``fit`` carries no downstream conditionals and the
    translation is testable without a CmdStan installation.

    Two of those decisions are conversions, not conveniences. ``sigma`` is
    ``sqrt(v)``, because Stan's ``normal`` takes a standard deviation and PyMARE
    stores variances. ``id`` is 1-based consecutive codes from
    :func:`~pymare.stats.encode_groups`, because the Stan program declares it
    ``int<lower=1, upper=K>`` -- which is why arbitrary scalar labels are
    accepted here.
    """
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] > 1:
        raise ValueError(
            "The StanMetaRegression estimator currently does "
            "not support 2-dimensional inputs. Passed y has "
            "shape {}.".format(y.shape)
        )
    y = np.asarray(y, dtype=float).reshape(-1)
    n_observations = y.shape[0]
    if not np.all(np.isfinite(y)):
        raise ValueError("Estimates (y) must all be finite.")

    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape[0] != n_observations:
        raise ValueError(
            f"v must contain one sampling variance per observation: expected "
            f"{n_observations}, got {v.shape[0]}."
        )
    # Order matters: NaN fails every comparison, so `v <= 0` alone would pass it
    # through to sqrt() and on to CmdStan, which rejects it while reading the
    # data -- a long way from the input that caused it.
    if not np.all(np.isfinite(v)):
        raise ValueError("Sampling variances (v) must all be finite.")
    if np.any(v <= 0):
        raise ValueError("Sampling variances (v) must all be positive.")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != n_observations:
        raise ValueError(
            f"X must contain one row per observation: expected {n_observations}, "
            f"got {X.shape[0]}."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("Predictors (X) must all be finite.")

    codes, labels = encode_groups(groups, n_observations=n_observations)

    if tau_prior_scale is None:
        tau_prior_scale = max(np.std(y), np.sqrt(np.mean(v)))

    return {
        "N": n_observations,
        "C": X.shape[1],
        "K": int(labels.size),
        "y": y,
        "sigma": np.sqrt(v),
        "X": X,
        "id": (codes + 1).astype(int),
        "tau_prior_scale": float(tau_prior_scale),
    }


class StanMetaRegression(BaseEstimator):
    r"""Bayesian meta-regression estimator using Stan.

    Parameters
    ----------
    tau_prior_scale : None or :obj:`float`, optional
        Scale of the half-normal prior on tau, the between-group standard
        deviation. When None (default), it is set to
        ``max(std(y), sqrt(mean(v)))``, the larger of the observed spread of the
        estimates and the typical sampling standard deviation.
    **sampling_kwargs
        Optional keyword arguments to pass on to CmdStanPy's sampler
        (e.g., ``iter_sampling`` for the number of post-warmup draws per chain,
        ``chains``, ``seed``, ``adapt_delta``).

    Notes
    -----
    The model is

    .. math::

        y_i &\sim \mathcal{N}(x_i' \beta + \theta_{g(i)}, \sigma_i) \\
        \theta_g &\sim \mathcal{N}(0, \tau)

    where :math:`\sigma_i = \sqrt{v_i}` is the known sampling standard deviation
    of observation :math:`i` and :math:`g(i)` is its group. This is the random-effects
    meta-analysis model of the Stan User's Guide [1]_ with that guide's stated
    extension to observation-level predictors. The reported ``tau2`` is
    :math:`\tau^2`, the between-group *variance*, matching what every other
    PyMARE estimator reports under that name.

    ``theta`` is given a non-centered parameterization (``theta = tau *
    theta_raw`` with ``theta_raw`` standard normal). The centered form produces
    the funnel geometry that dominates divergences in hierarchical models with
    few groups, which is this estimator's principal use case.

    :math:`\tau` gets a half-normal prior, weakly informative per Stan's
    recommendations [2]_ for models with few groups. Its scale is derived from the
    data rather than fixed, since a fixed scale would be crushingly informative on
    data measured in thousands and vacuous on data measured in thousandths.

    The default scale is ``max(std(y), sqrt(mean(v)))``. :math:`\tau` cannot
    plausibly exceed the spread of the estimates, and should not be presumed
    smaller than a typical standard error, so the larger of the two never asserts
    that :math:`\tau` is small when either quantity says otherwise. Erring large
    is deliberate: too small a scale costs coverage, too large costs only
    precision in :math:`\tau^2`. ``validation/stan`` records the measurements
    behind that choice. Pass ``tau_prior_scale`` to override it.

    :math:`\beta` keeps Stan's implicit improper uniform prior, so under a
    diffuse prior on :math:`\tau` the posterior means agree with
    :obj:`~pymare.estimators.VarianceBasedLikelihoodEstimator` at ``method="ML"``.

    The Stan program is compiled on first use and cached beside the installed
    source, so the cost is paid once per installation rather than per fit.

    References
    ----------
    .. [1] Stan Development Team. Stan User's Guide, "Measurement Error and
           Meta-Analysis", section "Meta-Analysis".
           https://mc-stan.org/docs/stan-users-guide/measurement-error.html
    .. [2] Stan Development Team. Prior Choice Recommendations.
           https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations

    .. versionchanged:: 0.0.5

        - The backend moved from PyStan 3 to CmdStanPy. PyStan's sampler argument
          names (``num_samples``, ``num_warmup``, ``num_chains``, ``num_thin``)
          are rejected with a message naming their replacements.
        - ``tau2`` is now the between-group variance rather than its square root,
          and sampling variances are converted to standard deviations before
          being passed to Stan. Both were wrong before, so posterior estimates
          change.
        - ``groups`` accepts scalar labels of any type, not only integers in
          ``1..k``, and :meth:`fit_dataset` now passes ``dataset.g`` rather than
          dropping it.
        - ``ci`` now sets the width of the reported credible interval. It was
          previously accepted and ignored.
    """

    _dataset_attr_map = {"groups": "g"}

    _parameter_constraints = {
        "tau_prior_scale": Interval(0.0, np.inf, closed="neither", allow_none=True),
    }

    def __init__(self, tau_prior_scale=None, **sampling_kwargs):
        renamed = {k: v for k, v in PYSTAN_SAMPLING_KWARGS.items() if k in sampling_kwargs}
        if renamed:
            raise TypeError(
                "These are PyStan argument names, which StanMetaRegression no longer accepts: "
                + ", ".join(
                    f"{old!r} (CmdStanPy calls it {new!r})" for old, new in renamed.items()
                )
                + "."
            )

        self.tau_prior_scale = tau_prior_scale
        self.sampling_kwargs = sampling_kwargs
        self.model = None
        self.result_ = None
        self._validate_params()

    def compile(self, force=False):
        """Compile the Stan model.

        Parameters
        ----------
        force : :obj:`bool`, optional
            Whether to recompile even when an up-to-date executable already
            exists. Default = False.

        Returns
        -------
        :obj:`~pymare.estimators.StanMetaRegression`
            The instance, so that ``compile()`` can be chained.

        Notes
        -----
        Called by :meth:`fit` when needed, so it never has to be called
        directly. Calling it in advance is worthwhile when the same estimator
        will be fitted to several datasets, because the compiled executable does
        not depend on the data.

        The executable is written beside the installed ``.stan`` file, where
        CmdStanPy finds and reuses it on later runs. If that directory is not
        writable -- a read-only ``site-packages``, for instance -- it falls back
        to ``~/.pymare/stan`` and warns once.
        """
        cmdstanpy = _import_cmdstanpy()

        try:
            self.model = cmdstanpy.CmdStanModel(stan_file=STAN_MODEL_PATH, force_compile=force)
            return self
        except Exception as unwritable:
            # Deliberately broad: CmdStanPy reports any failed make invocation
            # as ValueError, so catching OSError would never fire here.
            first_failure = unwritable

        # Compile a copy. exe_file= would not help: it names an executable to
        # reuse rather than a destination to build into, and make writes its
        # intermediates beside the source either way. copy2 preserves the mtime,
        # so the copy never looks newer than its own executable and CmdStanPy's
        # timestamp check keeps the cached build across processes.
        fallback_dir = op.join(op.expanduser("~"), ".pymare", "stan")
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_source = op.join(fallback_dir, op.basename(STAN_MODEL_PATH))
            shutil.copy2(STAN_MODEL_PATH, fallback_source)
            model = cmdstanpy.CmdStanModel(stan_file=fallback_source, force_compile=force)
        except Exception:
            # Somewhere writable failed too, so the first failure was not about
            # writing. Report that one -- it names the real problem.
            raise first_failure

        warn(
            f"Could not compile the Stan model beside {STAN_MODEL_PATH}, most likely because "
            f"that directory is not writable. Compiled into {fallback_dir} instead.",
            stacklevel=2,
        )
        self.model = model
        return self

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
        groups : None or array-like of shape (K,), optional
            One scalar label per observation, identifying the groups of
            observations in the y/v/X inputs. Labels may be strings, integers
            or any other hashable that numpy stores as a single element, and
            need not be consecutive; they are encoded internally in order of
            first occurrence by :func:`~pymare.stats.encode_groups`. Composite
            labels such as tuples are not accepted, because numpy reads a
            sequence of them as a 2-dimensional array. When None (default),
            each observation in the inputs is treated as a separate group.

        Returns
        -------
        :obj:`~pymare.estimators.StanMetaRegression`
            The fitted instance.

        Warns
        -----
        UserWarning
            If the sampler reported divergent transitions. Divergences mean the
            sampler could not explore part of the posterior, so the reported
            means and intervals may be biased; refitting with a larger
            ``adapt_delta`` is the usual remedy.

        Notes
        -----
        This estimator supports (simple) hierarchical models. When multiple
        observations belong to at least one common sampling unit, the `groups`
        argument can specify the nesting structure (i.e., which rows in `y`,
        `v`, and `X` belong to each group).

        The raw CmdStanPy fit is kept on ``self.result_``, so its diagnostics
        remain reachable -- ``est.result_.diagnose()`` reports R-hat, effective
        sample size, E-BFMI and treedepth alongside divergences.

        .. versionchanged:: 0.0.5
            ``groups`` accepts arbitrary hashable labels, and passing a numpy
            array no longer raises.
        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        self.data = _build_stan_data(y, v, X, groups=groups, tau_prior_scale=self.tau_prior_scale)

        if self.model is None:
            self.compile()

        self.result_ = self.model.sample(data=self.data, **self.sampling_kwargs)

        # CmdStanPy logs its own diagnostic warnings. Reraising this one through
        # the warnings module puts it under the caller's warning filters and
        # makes it assertable, which a log record is not.
        divergences = int(np.sum(self.result_.divergences))
        if divergences:
            warn(
                f"The sampler reported {divergences} divergent transition(s). The posterior "
                "summaries may be biased. Refit with a larger adapt_delta (e.g. "
                "StanMetaRegression(adapt_delta=0.99)), and see result_.diagnose() for the "
                "full diagnostic report.",
                stacklevel=2,
            )

        return self

    def summary(self, ci=95):
        """Generate a BayesianMetaRegressionResults object from the fitted estimator.

        Parameters
        ----------
        ci : :obj:`float`, optional
            Desired width of the credible interval, as a percentage.
            Default = 95.0 (95%).

        Returns
        -------
        :obj:`~pymare.results.BayesianMetaRegressionResults`
        """
        if self.result_ is None:
            name = self.__class__.__name__
            raise ValueError(
                "This {} instance hasn't been fitted yet. Please "
                "call fit() before summary().".format(name)
            )
        return BayesianMetaRegressionResults(self.result_, self.dataset_, ci)
