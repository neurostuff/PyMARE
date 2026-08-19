"""Miscellaneous statistical functions.

Vocabulary
----------
A **group** is the unit of dependence: rows sharing a label in ``g`` came from
one sampling unit and are not independent of each other. That is the word used
throughout for the unit itself, and it matches the ``g``/``groups`` arguments.

The word **cluster** is retained only where it is the established term for a
method rather than a description of the data -- "cluster-robust variance
estimation", :func:`cluster_robust_cov`, ``MIN_CLUSTERS_FOR_RVE``, and the
CR0/CR2 family. Those names tie the code to its own citations and are not
renamed. A group and a cluster are the same thing; only the provenance of the
word differs.
"""

import warnings
from typing import NamedTuple

import numpy as np
import scipy.stats as ss
from scipy.optimize import Bounds, minimize

# At or below this many clusters, robust variance estimation is known to be
# anti-conservative; see Hedges, Tipton & Johnson (2010) and Tipton (2015).
MIN_CLUSTERS_FOR_RVE = 10

# Tipton (2015) established by simulation that the Satterthwaite approximation
# holds its nominal level only while the degrees of freedom exceed roughly this
# value. Below it the reference distribution itself is outside its validated
# range, which the cluster count alone will not reveal: a dataset with a
# comfortable number of groups can still produce a dof of 2 for an unbalanced
# group-level predictor.
MIN_DOF_FOR_SATTERTHWAITE = 4.0

# A cluster whose leverage approaches one leaves no residual to learn from, so
# the CR2 adjustment diverges. Floor it rather than emit infinities.
_MIN_LEVERAGE_COMPLEMENT = 1e-10

# Assumed correlation between estimates within a group, used only to collapse
# groups before estimating tau^2. Results are very weakly sensitive to it; 0.8
# is the conventional choice for correlated effects.
DEFAULT_RHO = 0.8

#: How an estimator obtained tau^2. Recorded on the fitted estimator as
#: ``tau2_model_`` so that :mod:`pymare.results` uses the same reduction for the
#: interval and for Q instead of re-deriving it from the weight scheme and
#: drifting out of step with it.
TAU2_INDEPENDENT = "independent"
TAU2_AGGREGATE = "aggregate"
TAU2_CORRELATED = "correlated-effects"


class WeightedInterceptCR2Statistics(NamedTuple):
    """Reusable sufficient statistics for signed intercept-only CR2 tests."""

    weighted_values: np.ndarray
    adjusted_values: np.ndarray
    adjusted_sum_squares: np.ndarray
    adjusted_weight_sum: float
    total_weight: float


def broadcast_columns(arr, n_datasets):
    """Expand a single shared column to one column per parallel dataset.

    Parameters
    ----------
    arr : None or :obj:`numpy.ndarray` of shape (K, 1) or (K, D)
        A per-observation quantity such as ``v``, ``n`` or a weight array.
    n_datasets : :obj:`int`
        The number of parallel datasets the caller is working with.

    Returns
    -------
    None or :obj:`numpy.ndarray` of shape (K, D)
        A read-only view when a shared column was expanded, else the input.

    Notes
    -----
    A single column applies to every parallel dataset, which is what NumPy's own
    broadcasting rules say about a length-one axis. Normalizing once, at the
    boundary, leaves the code downstream with exactly one shape to reason about.
    Re-deriving the convention at each site is what left the looping estimators
    honouring it for ``n`` and not for ``v``.
    """
    if arr is None:
        return None
    arr = ensure_2d(np.asarray(arr, dtype=float))
    if arr.shape[1] == 1 and n_datasets > 1:
        return np.broadcast_to(arr, (arr.shape[0], n_datasets))
    return arr


def encode_groups(groups, n_observations=None):
    """Encode arbitrary group labels in order of first occurrence.

    Parameters
    ----------
    groups : None or array-like of shape (K,) or (K, 1)
        One hashable label per observation. If None, every observation is
        assigned its own group and ``n_observations`` is required.
    n_observations : :obj:`int`, optional
        Expected number of observations.

    Returns
    -------
    codes : :obj:`numpy.ndarray` of shape (K,)
        Consecutive integer codes.
    labels : :obj:`numpy.ndarray` of shape (G,)
        Original labels in order of first occurrence.
    """
    if groups is None:
        if n_observations is None:
            raise ValueError("n_observations is required when groups is None.")
        if not isinstance(n_observations, (int, np.integer)) or n_observations < 0:
            raise ValueError("n_observations must be a non-negative integer.")
        labels = np.arange(n_observations)
        return labels.copy(), labels

    groups = np.asarray(groups, dtype=object)
    if groups.ndim > 2 or (groups.ndim == 2 and 1 not in groups.shape):
        raise ValueError("groups must be one-dimensional.")
    groups = groups.ravel()
    if n_observations is not None and groups.size != n_observations:
        raise ValueError(
            f"groups must contain one label per observation: expected {n_observations}, "
            f"got {groups.size}."
        )

    codes = np.empty(groups.size, dtype=np.intp)
    labels = []
    label_codes = {}
    for observation, label in enumerate(groups):
        try:
            if label not in label_codes:
                label_codes[label] = len(labels)
                labels.append(label)
            codes[observation] = label_codes[label]
        except TypeError as exc:
            raise ValueError("Group labels must be hashable scalars.") from exc
    return codes, np.asarray(labels, dtype=object)


def group_mean(values, groups):
    """Return one arithmetic mean per group in first-occurrence order."""
    values = np.asarray(values)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")

    codes, labels = encode_groups(groups, n_observations=values.shape[0])
    means = np.zeros((labels.size, values.shape[1]), dtype=np.result_type(values, float))
    np.add.at(means, codes, values)
    means /= np.bincount(codes, minlength=labels.size)[:, None]
    return means[:, 0] if one_dimensional else means


def normalize_group_weights(weights, groups):
    """Divide each row's weight by the number of rows in its group."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim not in (1, 2):
        raise ValueError("weights must be one- or two-dimensional.")
    codes, labels = encode_groups(groups, n_observations=weights.shape[0])
    sizes = np.bincount(codes, minlength=labels.size)[codes]
    reshape = (sizes.size,) + (1,) * (weights.ndim - 1)
    return weights / sizes.reshape(reshape)


def one_sample_t_from_sufficient_statistics(sums, sum_squares, n_observations):
    """Calculate one-sample t statistics from sums and sums of squares."""
    sums = np.asarray(sums, dtype=float)
    sum_squares = np.asarray(sum_squares, dtype=float)
    try:
        broadcast_shape = np.broadcast_shapes(sums.shape, sum_squares.shape)
    except ValueError as exc:
        raise ValueError("sum_squares must broadcast to the shape of sums.") from exc
    if broadcast_shape != sums.shape:
        raise ValueError("sum_squares must broadcast to the shape of sums.")
    if not isinstance(n_observations, (int, np.integer)) or n_observations < 2:
        raise ValueError("n_observations must be an integer of at least two.")

    statistics = np.square(sums)
    np.subtract(n_observations * sum_squares, statistics, out=statistics)
    np.maximum(statistics, 0.0, out=statistics)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.sqrt(statistics, out=statistics)
        degenerate = statistics == 0
        np.divide(sums, statistics, out=statistics)
    statistics *= np.sqrt(n_observations - 1)
    statistics[degenerate] = np.nan
    return statistics


def weighted_intercept_cr2_sufficient_statistics(values, weights):
    """Prepare reusable sufficient statistics for intercept-only CR2 WLS."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be two-dimensional.")
    if weights.ndim != 1 or weights.size != values.shape[0]:
        raise ValueError("weights must contain one value per observation.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be finite positive values.")
    if weights.size < 2:
        raise ValueError("At least two observations with positive weight are required.")

    total_weight = float(weights.sum())
    leverage = weights / total_weight
    adjusted_weights = np.square(weights) / (1.0 - leverage)
    return WeightedInterceptCR2Statistics(
        weighted_values=weights[:, None] * values,
        adjusted_values=adjusted_weights[:, None] * values,
        adjusted_sum_squares=(adjusted_weights[:, None] * np.square(values)).sum(axis=0),
        adjusted_weight_sum=float(adjusted_weights.sum()),
        total_weight=total_weight,
    )


def weighted_intercept_cr2(signs, sufficient_statistics):
    r"""Evaluate signed intercept-only CR2 WLS statistics.

    With :math:`W=\sum_g q_g`, :math:`h_g=q_g/W`, and signed observation
    :math:`x_g^*=s_gx_g`, this computes

    .. math::
        \hat\mu = \frac{\sum_g q_gx_g^*}{W},\qquad
        \widehat{V}(\hat\mu) = \frac{1}{W^2}
        \sum_g \frac{q_g^2(x_g^*-\hat\mu)^2}{1-h_g},
        \qquad t=\frac{\hat\mu}{\sqrt{\widehat{V}(\hat\mu)}}.
    """
    signs = np.asarray(signs, dtype=float)
    if signs.ndim == 1:
        signs = signs[None, :]
    if signs.ndim != 2 or signs.shape[1] != sufficient_statistics.weighted_values.shape[0]:
        raise ValueError("signs must contain one value per observation.")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("signs may only contain -1 and 1.")

    weighted_sums = signs @ sufficient_statistics.weighted_values
    adjusted_sums = signs @ sufficient_statistics.adjusted_values
    means = weighted_sums / sufficient_statistics.total_weight
    meat = sufficient_statistics.adjusted_sum_squares[None, :] - 2.0 * means * adjusted_sums
    meat += np.square(means) * sufficient_statistics.adjusted_weight_sum
    np.maximum(meat, 0.0, out=meat)
    degenerate = meat == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = weighted_sums / np.sqrt(meat)
    statistics[degenerate] = np.nan
    return statistics


def correlated_effects_weights(v, groups, tau2=0.0):
    r"""Weight rows so that a group's total does not grow with its row count.

    Parameters
    ----------
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per estimate. Any hashable labels are accepted.
    tau2 : :obj:`float` or :obj:`numpy.ndarray`, optional
        tau^2 estimate to fold into the weights. Default = 0.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (K, D)
        One weight per row, constant within each group.

    See Also
    --------
    correlated_effects_tau2 : The tau^2 estimator derived under these weights.
    collapse_groups : Reduces each group to a single row instead of reweighting
        its rows, which is the stricter alternative when exact invariance to row
        count is required.

    Notes
    -----
    Every row of group :math:`j` receives the same weight, built from the
    group's mean variance :math:`\bar{v}_j`:

    .. math::
        w_{ij} = \frac{1}{n_j(\bar{v}_j + \tau^2)},

    which is equation (7) of :footcite:t:`fisher2015robumeta` and the weighting
    the R package `robumeta <https://cran.r-project.org/package=robumeta>`_
    applies for the correlated effects model of :footcite:t:`hedges2010robust`.
    A group's weights therefore sum to :math:`1/(\bar{v}_j + \tau^2)` however
    many rows it contributed, so replication cannot buy influence.

    The alternative -- keeping each row's own :math:`1/v_i` and dividing by the
    group size -- is more efficient when within-group variance differences are
    genuine, but a group's total is then the *mean of inverse variances*, which
    is unbounded in a single row: one variance reported a thousand times too
    small hands that group almost all of the weight in the analysis. This form
    is bounded by the group's other rows, is the model the tau^2 estimator is
    derived under, and is what the correlated effects model assumes in the first
    place, since dependence arising from shared units implies
    :math:`v_{ij} \approx v_j`.

    Under robust variance estimation the weights affect only efficiency, never
    the validity of the standard errors :footcite:p:`hedges2010robust`.

    References
    ----------
    .. footbibliography::

    """
    v = ensure_2d(np.asarray(v, dtype=float))
    codes, labels = encode_groups(groups, n_observations=v.shape[0])
    sizes = np.bincount(codes, minlength=labels.size)[:, None]
    return (1.0 / (sizes * (group_mean(v, groups) + tau2)))[codes]


def correlated_effects_tau2(y, v, X, groups, rho=DEFAULT_RHO):
    r"""Estimate tau^2 under the correlated-effects working model.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        Estimates.
    v : :obj:`numpy.ndarray` of shape (K, D) or (K, 1)
        Sampling variances. A single column applies to every parallel dataset.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix, at the level the coefficients are fitted.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per estimate.
    rho : :obj:`float`, optional
        Assumed correlation between estimates within a group.
        Default = 0.8.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (D,)
        The tau^2 estimate per parallel dataset, floored at zero.

    Raises
    ------
    ValueError
        If ``rho`` falls outside [0, 1], or there are no more groups than
        predictors.

    See Also
    --------
    collapse_groups : The alternative reduction, which replaces each group with
        one row and therefore requires a design that is constant within a group.
    correlated_effects_weights : The weights this estimator is derived under.

    Notes
    -----
    This is the method-of-moments estimator of :footcite:t:`hedges2010robust`,
    in the form given as equations (8) and (9) of
    :footcite:t:`fisher2015robumeta`. Writing :math:`w_j` for the group weights
    of :func:`correlated_effects_weights` at :math:`\tau^2 = 0`,
    :math:`V = (\sum_j w_j X_j'X_j)^{-1}`, and :math:`J_j` for the
    :math:`n_j \times n_j` matrix of ones,

    .. math::
        Q_E = \sum_j w_j (T_j - X_j b)'(T_j - X_j b),

    .. math::
        \hat{\tau}^2 = \frac{Q_E - m
            + \operatorname{tr}\left(V \sum_j \frac{w_j}{n_j}X_j'X_j\right)
            + \rho \operatorname{tr}\left(V \sum_j \frac{w_j}{n_j}
              \left[X_j'J_jX_j - X_j'X_j\right]\right)}
            {\sum_j n_j w_j
            - \operatorname{tr}\left(V \sum_j w_j^2 X_j'J_jX_j\right)}.

    Every term uses the observation-level design :math:`X_j`, so the estimate
    reflects the model the coefficients were fitted under. This is what
    distinguishes it from :func:`collapse_groups`, which replaces a group's
    predictors with their mean and so cannot see variation within a group --
    the reason :func:`pymare.estimators.estimators._validate_group_design`
    refuses that substitution. The assumed correlation :math:`\rho` enters
    analytically, through the :math:`X_j'J_jX_j` terms, rather than by
    aggregating the data first.

    With one estimate per group the two :math:`\operatorname{tr}` terms
    containing :math:`J_j` coincide with their :math:`X_j'X_j` counterparts, the
    :math:`\rho` term vanishes, and the expression reduces exactly to
    DerSimonian-Laird :footcite:p:`dersimonian1986meta` -- as it must, since
    there is then no dependence to model.

    References
    ----------
    .. footbibliography::

    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1]; got {rho}.")

    y = ensure_2d(np.asarray(y, dtype=float))
    v = ensure_2d(np.asarray(v, dtype=float))
    X = np.asarray(X, dtype=float)
    n_preds = X.shape[1]

    codes, labels = encode_groups(groups, n_observations=y.shape[0])
    n_groups = labels.size
    if n_groups <= n_preds:
        raise ValueError(
            f"The correlated-effects tau^2 needs more groups than predictors: got "
            f"{n_groups} group(s) for {n_preds} predictor(s)."
        )

    sizes = np.bincount(codes, minlength=n_groups)[:, None].astype(float)
    # Equation (7) at tau^2 = 0: the "initial meta-regression" of equation (8).
    group_w = np.broadcast_to(1.0 / (sizes * group_mean(v, groups)), (n_groups, y.shape[1]))
    row_w = group_w[codes]

    # X_j'X_j and X_j'J_jX_j, the latter being the outer product of the group's
    # column sums, since J_j is all ones.
    gram = np.zeros((n_groups, n_preds, n_preds))
    np.add.at(gram, codes, X[:, :, None] * X[:, None, :])
    sums = np.zeros((n_groups, n_preds))
    np.add.at(sums, codes, X)
    outer = sums[:, :, None] * sums[:, None, :]

    def _weighted(scale, moment):
        """Contract a per-group p x p moment with per-group weights."""
        return np.einsum("md,mpq->dpq", scale, moment)

    bread = np.linalg.pinv(_weighted(group_w, gram))
    trace = lambda moment: np.einsum("dpq,dqp->d", bread, moment)  # noqa: E731
    scaled = group_w / sizes
    within = trace(_weighted(scaled, gram))
    across = trace(_weighted(scaled, outer))

    beta = weighted_least_squares(y, v, X, w=row_w)
    q_e = (row_w * np.square(y - X.dot(beta))).sum(0)

    numerator = q_e - n_groups + within + rho * (across - within)
    denominator = (sizes * group_w).sum(0) - trace(_weighted(np.square(group_w), outer))
    return np.maximum(0.0, numerator / denominator)


def collapse_groups(y, v, X, groups, rho=DEFAULT_RHO):
    r"""Aggregate each group to a single effect estimate.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        2d array of estimates (observations x parallel datasets).
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per estimate.
    rho : :obj:`float`, optional
        Assumed correlation between estimates within a group, in [0, 1].
        Default = 0.8, matching the default of the R package `robumeta
        <https://cran.r-project.org/package=robumeta>`_, whose correlated
        effects model this follows.

    Returns
    -------
    :obj:`tuple`
        ``(y, v, X)`` collapsed to one row per group.

    Raises
    ------
    ValueError
        If ``rho`` lies outside [0, 1].

    See Also
    --------
    collapse_groups_by_n : The same reduction for models parameterized by
        sample size rather than by sampling variance.
    correlated_effects_weights : Reweights the rows in place instead of collapsing them,
        which keeps within-group predictor variation available.

    Notes
    -----
    Moment-based estimators of :math:`\tau^2` count every row of ``y`` as an
    independent observation. When a group contributes several rows, that
    pseudo-replication distorts :math:`\tau^2`: the duplicated rows agree with
    each other by construction, so the observed dispersion no longer matches
    what the row count implies. Estimating :math:`\tau^2` from one effect per
    group removes the problem.

    Each group is collapsed to the unweighted mean of its members, whose
    sampling variance is

    .. math::
        \bar{v}_j = \frac{1}{n_j^2}
            \left( \sum_i v_i + \rho \sum_{i \neq k} \sqrt{v_i v_k} \right),

    i.e. the variance of a mean whose terms are correlated at :math:`\rho`.

    The resulting :math:`\tau^2` is only weakly sensitive to ``rho``: sweeping it
    across its whole range typically moves downstream error rates by well under
    a percentage point, because it enters only through the relative weighting of
    already-aggregated groups. Assuming a single value is therefore reasonable
    even when the true within-group correlation varies between groups.

    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1]; got {rho}.")

    groups = np.asarray(groups).ravel()
    # encode_groups, not np.unique: group_mean() below encodes by first
    # occurrence, so using np.unique's sorted-label codes here would pair each
    # collapsed effect with a different group's variance.
    group_codes, group_labels = encode_groups(groups)
    n_groups = group_labels.size

    collapsed_y = group_mean(y, group_codes)
    collapsed_v = np.empty((n_groups, v.shape[1]))
    collapsed_X = group_mean(X, group_codes)

    for group in range(n_groups):
        members = np.flatnonzero(group_codes == group)
        size = members.size
        member_v = v[members]
        if size == 1:
            collapsed_v[group] = member_v[0]
            continue

        sd = np.sqrt(member_v)
        cross = sd.sum(axis=0) ** 2 - member_v.sum(axis=0)
        collapsed_v[group] = (member_v.sum(axis=0) + rho * cross) / size**2

    return collapsed_y, collapsed_v, collapsed_X


def collapse_groups_by_n(y, n, X, groups, rho=DEFAULT_RHO):
    r"""Aggregate each group to one effect with an effective sample size.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        2d array of estimates (observations x parallel datasets).
    n : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sample sizes.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per estimate.
    rho : :obj:`float`, optional
        Assumed correlation between estimates within a group, in [0, 1].
        Default = 0.8.

    Returns
    -------
    :obj:`tuple`
        ``(y, n, X)`` collapsed to one row per group.

    Raises
    ------
    ValueError
        If ``rho`` lies outside [0, 1].

    See Also
    --------
    collapse_groups : The counterpart for models parameterized by sampling
        variance, where :math:`\sigma^2` is not being estimated.

    Notes
    -----
    The counterpart to :func:`~pymare.stats.collapse_groups` for models
    parameterized by sample size, where :math:`v_i = \sigma^2 / n_i` and
    :math:`\sigma^2` is itself being estimated, so :math:`\bar{v}_j` cannot be
    formed yet. Requiring the collapsed effect to satisfy
    :math:`\sigma^2 / n_j^{\text{eff}} = \operatorname{Var}(\bar{y}_j)` gives

    .. math::
        n_j^{\text{eff}} = \frac{m^2}{\sum_i 1/n_i
            + \rho\left[\left(\sum_i 1/\sqrt{n_i}\right)^2
            - \sum_i 1/n_i\right]},

    which is free of :math:`\sigma^2` and so can be formed before the
    likelihood is evaluated.

    For a group of :math:`s` estimates sharing one sample of :math:`n` subjects
    this reduces to :math:`n_j^{\text{eff}} = sn / [1 + \rho(s-1)]`, running from
    :math:`sn` at ``rho=0`` up to :math:`n` at ``rho=1``. The ``rho=1`` endpoint
    is exactly "count the group's sample size once", so a separate function for
    that case is unnecessary -- and choosing it implicitly, by counting ``n``
    once without saying so, assumes perfect within-group correlation and biases
    :math:`\sigma^2` low by :math:`[1 + \rho(s-1)]/s` whenever that is wrong.

    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1]; got {rho}.")

    groups = np.asarray(groups).ravel()
    # encode_groups, not np.unique: group_mean() below encodes by first
    # occurrence, so using np.unique's sorted-label codes here would pair each
    # collapsed effect with a different group's variance.
    group_codes, group_labels = encode_groups(groups)
    n_groups = group_labels.size

    collapsed_y = group_mean(y, group_codes)
    collapsed_n = np.empty((n_groups, n.shape[1]))
    collapsed_X = group_mean(X, group_codes)

    for group in range(n_groups):
        members = np.flatnonzero(group_codes == group)
        size = members.size
        member_n = n[members]
        if size == 1:
            collapsed_n[group] = member_n[0]
            continue

        inverse = (1.0 / member_n).sum(axis=0)
        cross = (1.0 / np.sqrt(member_n)).sum(axis=0) ** 2 - inverse
        collapsed_n[group] = size**2 / (inverse + rho * cross)

    return collapsed_y, collapsed_n, collapsed_X


def estimate_null_correlation(y, groups=None, bias_correct=True):
    r"""Estimate the correlation between estimates under the null.

    Methods that correct for dependence -- Stouffer's with a variance
    inflation term, or Fisher's via Brown's method -- need the correlation the
    inputs would have *under the null*, i.e. the correlation of their noise.
    Correlating the raw rows of ``y`` does not measure that: any effect shared
    across inputs is common signal, and it inflates every pairwise
    correlation, including pairs from unrelated groups that are independent by
    construction.

    Removing the across-dataset mean at each column first strips that shared
    signal, leaving residuals whose correlation estimates the dependence.
    Centering K values induces a spurious :math:`-1/(K-1)` correlation among
    independent rows, which ``bias_correct`` rescales away.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        2d array of estimates (observations x parallel datasets). At least two
        parallel datasets are required to estimate a correlation.
    groups : None or :obj:`numpy.ndarray` of shape (K,), optional
        Group labels, one per estimate. When supplied, the shrinkage that
        centering induces is inverted exactly within each group rather than
        merely rescaled, which matters when K is small.
        Default = None.
    bias_correct : :obj:`bool`, optional
        Whether to undo the spurious :math:`-1/(K-1)` correlation that centering
        K values induces among independent rows.
        Default = True.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (K, K)
        The estimated correlation matrix.

    Notes
    -----
    Centering is a known linear map, :math:`e = Cy` with
    :math:`C = I - J/K`, so the shrinkage it induces can be undone rather than
    merely bounded. When ``groups`` identifies which estimates are exchangeable,
    :func:`~pymare.stats.undo_centering_shrinkage` inverts it in closed form and
    recovers the underlying correlation essentially exactly, even for small K.

    Without ``groups`` the block structure is unknown, so the weaker correction
    below is used: it maps independent estimates to zero but understates genuine
    dependence, increasingly so as K shrinks.

    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must have shape (n_estimates, n_datasets).")
    n_estimates, n_datasets = y.shape
    if n_datasets < 2:
        raise ValueError("At least two parallel datasets are required.")
    if n_estimates < 2:
        raise ValueError("At least two estimates are required.")

    residuals = y - y.mean(axis=0)

    # Rows that are constant after centering carry no information; corrcoef
    # would return NaN for them, so leave them uncorrelated with everything.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(residuals, rowvar=True)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    if bias_correct:
        if groups is None:
            offset = 1.0 / (n_estimates - 1)
            corr = (corr + offset) / (1.0 + offset)
            corr = np.clip(corr, -1.0, 1.0)
        else:
            corr = undo_centering_shrinkage(corr, groups)

    np.fill_diagonal(corr, 1.0)
    return corr


def undo_centering_shrinkage(corr, groups):
    r"""Invert the correlation shrinkage induced by centering.

    Removing the across-dataset mean maps :math:`y` to :math:`e = Cy` with
    :math:`C = I - J/K`, so :math:`\operatorname{Cov}(e) = CRC`. For estimates
    :math:`i \neq j` sharing a group of size :math:`b`, writing
    :math:`S = \sum_h b_h(b_h - 1)\rho_h` for the contribution of every group to
    the grand mean,

    .. math::
        (CRC)_{ij} &= \rho\left(1 - \tfrac{2(b-1)}{K}\right) + c, \\
        (CRC)_{ii} &= \rho\left(-\tfrac{2(b-1)}{K}\right) + 1 + c,
        \qquad c = -\tfrac{2}{K} + \tfrac{K + S}{K^2},

    both affine in :math:`\rho`. Their ratio is the observed residual
    correlation, so :math:`\rho` follows in closed form. :math:`S` couples the
    groups to each other, which a short iteration resolves; each group's own
    share of :math:`S` is kept symbolic rather than folded in, since otherwise
    numerator and denominator vanish together whenever a group holds about half
    the estimates.

    Parameters
    ----------
    corr : :obj:`numpy.ndarray` of shape (K, K)
        Correlation matrix of the centered residuals.
    groups : :obj:`numpy.ndarray` of shape (K,)
        Group labels, one per estimate.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (K, K)
        The de-shrunk correlation matrix.

    Notes
    -----
    Only within-group entries are inverted, since those are what the dependence
    corrections consume. Estimates in different groups are taken to be
    independent, and their entries get the group-agnostic rescaling that maps
    independent estimates to zero.

    """
    corr = np.array(corr, dtype=float, copy=True)
    groups = np.asarray(groups).ravel()
    n_estimates = corr.shape[0]
    if groups.shape[0] != n_estimates:
        raise ValueError(
            f"groups must have one label per estimate: expected {n_estimates}, "
            f"got {groups.shape[0]}."
        )

    # encode_groups, not np.unique: the docstring promises any hashable label,
    # and np.unique additionally requires them to be sortable.
    group_codes, group_labels = encode_groups(groups, n_observations=n_estimates)
    members = [np.flatnonzero(group_codes == g) for g in range(group_labels.size)]
    sizes = np.array([m.size for m in members], dtype=float)

    # Mean observed within-group correlation, used only to drive the shared term.
    observed = np.zeros(len(members))
    for index, member in enumerate(members):
        if member.size < 2:
            continue
        block = corr[np.ix_(member, member)]
        observed[index] = block[~np.eye(member.size, dtype=bool)].mean()

    def solve(r, size, others):
        """Invert the shrinkage for one group, given the other groups' share.

        ``others`` is the grand-mean contribution of every *other* group. The
        group's own contribution stays symbolic, so both sides of the ratio
        remain affine in its rho and the solution is exact. Folding it into
        ``others`` instead would make the numerator and denominator vanish
        together whenever the group holds about half the estimates.
        """
        own = size * (size - 1) / n_estimates**2
        offset = -2.0 / n_estimates + (n_estimates + others) / n_estimates**2
        numerator_slope = 1.0 - 2.0 * (size - 1) / n_estimates + own
        denominator_slope = -2.0 * (size - 1) / n_estimates + own
        denominator = numerator_slope - r * denominator_slope
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = (r * (1.0 + offset) - offset) / denominator
        return np.where(np.abs(denominator) > 1e-12, rho, r)

    # Groups interact only through the grand mean, and each group's own share is
    # handled exactly, so this converges immediately for a single group and in a
    # handful of steps otherwise.
    contributions = np.zeros(len(members))
    for _ in range(50):
        total = contributions.sum()
        updated = np.array(
            [
                (
                    sizes[i]
                    * (sizes[i] - 1)
                    * float(solve(observed[i], sizes[i], total - contributions[i]))
                    if sizes[i] > 1
                    else 0.0
                )
                for i in range(len(members))
            ]
        )
        if np.allclose(updated, contributions, atol=1e-12, rtol=0):
            break
        contributions = updated

    # Between-group pairs keep the group-agnostic rescaling.
    offset = 1.0 / (n_estimates - 1)
    corrected = (corr + offset) / (1.0 + offset)

    # Within-group pairs are inverted exactly, elementwise so that genuine
    # heterogeneity inside a group survives.
    total = contributions.sum()
    for index, member in enumerate(members):
        if member.size < 2:
            continue
        block = corr[np.ix_(member, member)]
        corrected[np.ix_(member, member)] = solve(
            block, sizes[index], total - contributions[index]
        )

    np.fill_diagonal(corrected, 1.0)
    return np.clip(corrected, -1.0, 1.0)


def weighted_least_squares(y, v, X, tau2=0.0, return_cov=False, w=None):
    r"""Perform 2-D weighted least squares.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        2d array of estimates (observations x parallel datasets).
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    tau2 : :obj:`float`, optional
        tau^2 estimate to use for the weights.
        Default = 0.
    return_cov : :obj:`bool`, optional
        Whether to return the covariance matrix of the coefficients alongside
        them. Default = False.
    w : None or :obj:`numpy.ndarray`, optional
        Precomputed weights of the same shape as ``y``, overriding the default
        ``1 / (v + tau2)``. Use
        :func:`~pymare.stats.correlated_effects_weights` to obtain weights that
        do not reward replication within a group.
        Default = None.

    Returns
    -------
    beta : :obj:`numpy.ndarray` of shape (P, D)
        The fixed effect coefficients.
    cov : :obj:`numpy.ndarray` of shape (P, P, D)
        Only when ``return_cov`` is True. Equal to ``(X'WX)^-1``, which is the
        *covariance* of ``beta``.

    See Also
    --------
    cluster_robust_cov : Replaces ``cov`` with a sandwich estimator that does
        not assume the observations are independent.

    Notes
    -----
    All ``D`` parallel datasets are solved in one set of contractions rather
    than in a Python loop, which is what makes the estimators usable when ``D``
    runs to hundreds of thousands. The ``einsum`` subscripts use ``k`` for
    observations, ``p`` and ``q`` for predictors and ``i`` for parallel
    datasets.

    ``(X'WX)^-1`` is the covariance of the coefficients, not an inverse
    covariance: ``X'WX`` is the inverse covariance, so inverting it returns the
    covariance. PyMARE nonetheless stores this quantity under the ``params_``
    key ``"inv_cov"``, a misnomer of long standing that is retained because that
    key is public. Anything taking ``sqrt(diagonal(...))`` of it -- notably
    :attr:`pymare.results.MetaRegressionResults.fe_se` -- is treating it as a
    covariance, which is correct. The function parameters that pass it around
    are named ``model_cov``, which also distinguishes it from the *robust*
    covariance that :func:`cluster_robust_cov` returns.

    """
    w = 1.0 / (v + tau2) if w is None else w

    # Einsum indices: k = observations, p = predictors, i = parallel iterates
    wX = np.einsum("kp,ki->ipk", X, w)
    cov = wX.dot(X)

    # numpy >= 1.8 inverts stacked matrices along the first N - 2 dims, so we
    # can vectorize computation along the second dimension (parallel datasets)
    # (X'WX)^-1, i.e. the covariance of beta. Deliberately not called
    # "precision": in statistics a precision matrix is the *inverse* of a
    # covariance, which is X'WX itself, not this.
    cov_beta = np.linalg.pinv(cov).T

    pwX = np.einsum("ipk,qpi->iqk", wX, cov_beta)
    beta = np.einsum("ipk,ik->ip", pwX, y.T).T

    return (beta, cov_beta) if return_cov else beta


#: Fractions of the search interval evaluated in the coarse scan that precedes
#: the local refinement. The linear part locates the bracket the minimum lives
#: in; the points crowded against each end catch an optimum sitting in the first
#: or last percent of the interval, which is where a variance component pinned
#: near zero ends up.
_SCAN_FRACTIONS = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 1.0, 25),
            [1e-6, 1e-4, 1e-2],
            1.0 - np.array([1e-2, 1e-4, 1e-6]),
        ]
    )
)

#: 1 / golden ratio. Splitting an interval at this fraction from each end makes
#: one of the two interior points of the surviving sub-interval coincide with an
#: interior point of the current one, so each iteration costs a single
#: evaluation instead of two.
_GOLDEN_SECTION = (np.sqrt(5.0) - 1.0) / 2.0


def bounded_scalar_min(f, lower, upper, xtol=1e-10, maxiter=100):
    """Minimize a scalar function over a bounded interval for many datasets at once.

    Parameters
    ----------
    f : :obj:`callable`
        Objective. Takes an :obj:`numpy.ndarray` of shape (D,) holding one
        candidate value per parallel dataset and returns an array of shape (D,)
        holding that dataset's objective at its own candidate. It is called with
        every dataset's current candidate at once, which is what makes this
        cheaper than one minimization per dataset.
    lower, upper : :obj:`numpy.ndarray` of shape (D,)
        Per-dataset search bounds, ``lower <= upper``.
    xtol : :obj:`float`, optional
        Stop once every dataset's bracket is narrower than
        ``xtol * (1 + abs(x))``. Default = 1e-10.
    maxiter : :obj:`int`, optional
        Cap on refinement iterations. Default = 100.

    Returns
    -------
    x : :obj:`numpy.ndarray` of shape (D,)
        The minimizing value per dataset.
    fval : :obj:`numpy.ndarray` of shape (D,)
        The objective there.

    Notes
    -----
    A coarse scan over :data:`_SCAN_FRACTIONS` brackets each dataset's minimum,
    then golden-section search refines every bracket in step. The whole search
    costs ``len(_SCAN_FRACTIONS) + 2 + iterations`` vectorized evaluations of
    ``f`` no matter how many datasets there are, where a per-dataset
    :func:`scipy.optimize.minimize` costs a Python-level optimization each.

    Golden-section search needs the objective to be unimodal on the bracket, not
    globally, which is what the scan buys: a minimum in a narrow dip elsewhere in
    the interval is found by the scan and refined from there. The scan is also
    what makes this less prone to a local minimum than a quasi-Newton run from a
    single starting value.

    The point returned is the better of the refined one and the best scan point,
    so it is never worse than the scan alone.

    ``f`` may return ``nan`` for a degenerate dataset. Those values are treated
    as ``inf`` in the scan so that a ``nan`` never wins the bracket, and the
    comparisons in the refinement are false for them, which leaves that dataset
    parked at its scan bracket rather than propagating into the others.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError(
            "lower and upper must be 1d arrays of the same shape, got "
            f"{lower.shape} and {upper.shape}."
        )

    scan = lower + np.multiply.outer(_SCAN_FRACTIONS, upper - lower)
    scan_vals = np.stack([f(candidate) for candidate in scan])
    best = np.argmin(np.where(np.isnan(scan_vals), np.inf, scan_vals), axis=0)

    # The bracket is the pair of scan points either side of the best one. Both
    # neighbours are clipped into range, so a minimum at an end of the interval
    # brackets as [end, next] rather than reaching outside it.
    left = np.take_along_axis(scan, np.maximum(best - 1, 0)[None], axis=0)[0]
    right = np.take_along_axis(scan, np.minimum(best + 1, scan.shape[0] - 1)[None], axis=0)[0]

    low = left
    high = right
    interior_low = high - _GOLDEN_SECTION * (high - low)
    interior_high = low + _GOLDEN_SECTION * (high - low)
    f_low = f(interior_low)
    f_high = f(interior_high)

    for _ in range(maxiter):
        if np.all(high - low <= xtol * (1.0 + np.abs(low))):
            break

        # Drop the end beyond whichever interior point is worse.
        keep_left = f_low <= f_high
        new_low = np.where(keep_left, low, interior_low)
        new_high = np.where(keep_left, interior_high, high)

        # One interior point of the new interval is an interior point of the old
        # one, so only the other has to be evaluated. Which of the two it is
        # differs per dataset, so both roles are assembled with where() and the
        # fresh points are evaluated in a single call.
        kept = np.where(keep_left, interior_low, interior_high)
        f_kept = np.where(keep_left, f_low, f_high)
        span = _GOLDEN_SECTION * (new_high - new_low)
        fresh = np.where(keep_left, new_high - span, new_low + span)
        f_fresh = f(fresh)

        interior_low = np.where(keep_left, fresh, kept)
        f_low = np.where(keep_left, f_fresh, f_kept)
        interior_high = np.where(keep_left, kept, fresh)
        f_high = np.where(keep_left, f_kept, f_fresh)
        low, high = new_low, new_high

    # Fall back on the best scan point where refinement did not beat it, so the
    # result is never worse than the scan and an optimum sitting exactly on an
    # end of the interval -- tau^2 = 0, say -- is returned exactly rather than
    # approached from inside the bracket.
    take_low = f_low <= f_high
    x = np.where(take_low, interior_low, interior_high)
    fval = np.where(take_low, f_low, f_high)
    scan_x = np.take_along_axis(scan, best[None], axis=0)[0]
    scan_f = np.take_along_axis(scan_vals, best[None], axis=0)[0]
    keep_scan = ~(fval <= scan_f)
    return np.where(keep_scan, scan_x, x), np.where(keep_scan, scan_f, fval)


def _symmetric_sqrt(matrices):
    """Return L with ``L @ L.T`` equal to each symmetric PSD input matrix.

    Parameters
    ----------
    matrices : :obj:`numpy.ndarray` of shape (..., P, P)
        A stack of symmetric positive semi-definite matrices.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (..., P, P)
        A factor of each input, computed from its eigendecomposition.

    Notes
    -----
    Eigenvalues are floored at zero before the square root, so a matrix that is
    only PSD up to rounding does not produce NaNs. Used for two unrelated
    purposes that happen to need the same factor: it factors the quadratic form in
    :func:`satterthwaite_dof` and supplies the ``L`` that
    :func:`_cr2_low_rank_factors` needs, so one call serves both.
    """
    evals, evecs = np.linalg.eigh(matrices)
    return evecs * np.sqrt(np.maximum(evals, 0.0))[..., None, :]


def _cr2_low_rank_factors(group_X, chol):
    r"""Factor the CR2 adjustment using only :math:`p \times p` work.

    Parameters
    ----------
    group_X : :obj:`numpy.ndarray` of shape (..., n, p)
        Whitened design rows for one group.
    chol : :obj:`numpy.ndarray` of shape (..., p, p)
        Any factor with ``chol @ chol.T == M``, e.g. from
        :func:`_symmetric_sqrt`.

    Returns
    -------
    b_factor : :obj:`numpy.ndarray` of shape (..., n, p)
        The ``B`` of the factorization below.
    middle : :obj:`numpy.ndarray` of shape (..., p, p)
        The ``G`` of the factorization below.
    degenerate : :obj:`bool`
        True when a group's leverage reached one, so the adjustment does not
        exist and the eigenvalue floor took over. The caller is responsible for
        warning.

    See Also
    --------
    _cr2_low_rank_apply : Applies the factors without forming the matrix.

    Notes
    -----
    :math:`H_j = \tilde{X}_j M \tilde{X}_j'` is an outer product of an
    :math:`n_j \times p` matrix with itself, so its rank is at most :math:`p`.
    Eigendecomposing the full :math:`n_j \times n_j` block therefore spends
    :math:`O(n_j^3)` to recover at most :math:`p` non-unit eigenvalues plus
    :math:`n_j - p` copies of 1 -- and an eigenvalue of 1 in :math:`I - H_j` is a
    direction the adjustment leaves alone, so that work computes nothing.

    Writing :math:`M = LL'` and :math:`B = \tilde{X}_j L`, the non-zero
    eigenvalues :math:`\mu_i` of :math:`H_j = BB'` are exactly the eigenvalues of
    the :math:`p \times p` matrix :math:`B'B`, whose eigenvectors :math:`q_i` lift
    to :math:`u_i = Bq_i / \sqrt{\mu_i}`. Since the orthogonal complement of the
    :math:`u_i` is an eigenspace of :math:`I - H_j` with eigenvalue 1,

    .. math::
        (I_j - H_j)^{-1/2}
            = I + \sum_i \left[(1 - \mu_i)^{-1/2} - 1\right] u_iu_i'
            = I + BGB', \qquad
        G = Q \operatorname{diag}(c) Q',

    with :math:`c_i = \left[(1 - \mu_i)^{-1/2} - 1\right] / \mu_i`. This turns
    :math:`O(n_j^3)` into :math:`O(n_j p^2)`, which is why the caller only falls
    back to the full block when the group is no larger than the design.

    :math:`c_i` has a removable singularity at :math:`\mu_i = 0`, where the limit
    is :math:`1/2`. Substituting it is cosmetic rather than load-bearing --
    :math:`Bq_i` vanishes with :math:`\mu_i`, since
    :math:`\lVert Bq_i \rVert^2 = \mu_i`, so the whole term is killed either way
    -- but it keeps the intermediate finite.
    """
    b_factor = group_X @ chol
    gram = np.swapaxes(b_factor, -1, -2) @ b_factor
    mu, evecs = np.linalg.eigh(gram)

    complement = 1.0 - mu
    degenerate = bool(np.any(complement < _MIN_LEVERAGE_COMPLEMENT))
    complement = np.maximum(complement, _MIN_LEVERAGE_COMPLEMENT)

    numerator = complement**-0.5 - 1.0
    safe_mu = np.where(np.abs(mu) > _MIN_LEVERAGE_COMPLEMENT, mu, 1.0)
    coefficients = np.where(np.abs(mu) > _MIN_LEVERAGE_COMPLEMENT, numerator / safe_mu, 0.5)

    middle = (evecs * coefficients[..., None, :]) @ np.swapaxes(evecs, -1, -2)
    return b_factor, middle, degenerate


def _cr2_low_rank_apply(b_factor, middle, rhs):
    r"""Apply the factored CR2 adjustment ``I + B G B'`` to a right-hand side.

    Parameters
    ----------
    b_factor : :obj:`numpy.ndarray` of shape (..., n, P)
        The ``B`` returned by :func:`_cr2_low_rank_factors`.
    middle : :obj:`numpy.ndarray` of shape (..., P, P)
        The ``G`` returned by :func:`_cr2_low_rank_factors`.
    rhs : :obj:`numpy.ndarray` of shape (..., n, q)
        What the adjustment is applied to: whitened residuals, or the whitened
        design times the bread.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (..., n, q)
        The adjusted right-hand side.

    See Also
    --------
    _cr2_low_rank_factors : Produces ``B`` and ``G``.

    Notes
    -----
    The adjustment matrix is never formed. Callers only ever need its *action* on
    a specific right-hand side, and the contraction order here keeps every
    intermediate at :math:`(n, P)` rather than materializing an
    :math:`n \times n` matrix per group per parallel dataset.
    """
    return rhs + b_factor @ (middle @ (np.swapaxes(b_factor, -1, -2) @ rhs))


def _cr2_scores(X, w, resid, group_members, bread):
    r"""Compute bias-reduced (CR2) cluster scores.

    Parameters
    ----------
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    w : :obj:`numpy.ndarray` of shape (K, D)
        The weights used to fit the coefficients.
    resid : :obj:`numpy.ndarray` of shape (K, D)
        Residuals ``y - X @ beta``, in the same orientation as ``y``.
    group_members : :obj:`list` of :obj:`numpy.ndarray`
        Row indices belonging to each group, one array per group.
    bread : :obj:`numpy.ndarray` of shape (D, P, P)
        ``(X'WX)^-1`` per parallel dataset.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (D, P, m)
        One adjusted score vector per group and parallel dataset.

    Warns
    -----
    UserWarning
        If any group's leverage reaches one, so that its adjustment does not
        exist and the eigenvalue floor took over. That group then contributes
        nothing to the sandwich, which therefore understates the standard
        errors.

    See Also
    --------
    _cr2_low_rank_factors : Supplies the factored adjustment used for groups
        larger than the design.

    Notes
    -----
    The CR0 score for group :math:`j` is :math:`X_j' W_j e_j`. Because
    :math:`\beta` is fitted using group :math:`j` itself, those residuals are
    shrunk toward zero in proportion to the group's leverage, which biases the
    sandwich downward -- severely when one group carries much of the weight.
    CR2 :footcite:p:`bell2002bias` undoes that shrinkage by inflating the
    residuals with :math:`A_j = (I_j - H_j)^{-1/2}` in the whitened metric,
    giving

    .. math::
        s_j = \tilde{X}_j' (I_j - \tilde{X}_j M \tilde{X}_j')^{-1/2}
              W_j^{1/2} e_j,

    with :math:`\tilde{X}_j = W_j^{1/2} X_j` and :math:`M = (X'WX)^{-1}`.
    Singleton groups reduce to the familiar scalar :math:`1/\sqrt{1 - h_j}`,
    i.e. HC2 :footcite:p:`mackinnon1985some`.

    "Exactly unbiased" is always with respect to a *working model* for the error
    covariance, which the analyst chooses. Bell and McCaffrey pick :math:`A_j`
    to satisfy :math:`A_j (I - H)_j \Phi (I - H)_j' A_j' = \Phi_j`; the form
    above is the solution for :math:`\Phi = I` in the whitened metric, i.e.
    under the assumption that the weights are correct and the observations
    independent -- the same assumption the sandwich exists to avoid relying on.
    That is pragmatic rather than circular: simulation shows the correction
    helps substantially even when the working model is wrong
    :footcite:p:`tipton2015small,imbens2016robust`, and its influence fades as
    the number of groups grows. This form coincides with the correlated-effects
    adjustment :math:`A_j^C` of :footcite:t:`fisher2015robumeta`.

    Because CR2 targets unbiasedness rather than conservatism, it is *not*
    guaranteed to exceed CR0. It does for singleton groups, where the adjustment
    is a scalar :math:`\ge 1` applied to each score; for larger groups the score
    is a projection of the inflated residuals, and in a small fraction of
    designs a given coefficient's CR2 variance comes out below its CR0
    counterpart. :footcite:t:`pustejovsky2018small` place CR2 between CR1, which
    under-corrects, and CR3 :footcite:p:`mancl2001covariance`, which
    over-corrects.

    Implemented from the published formulation rather than ported; the R package
    `clubSandwich <https://cran.r-project.org/package=clubSandwich>`_
    :footcite:p:`pustejovsky2018small` is the reference implementation.

    References
    ----------
    .. footbibliography::

    """
    n_preds = X.shape[1]
    n_iters = w.shape[1]
    n_groups = len(group_members)

    sqrt_w = np.sqrt(w)
    # Whitened design and residuals, oriented (k, i, p) and (k, i).
    whitened_X = X[:, None, :] * sqrt_w[:, :, None]
    whitened_resid = sqrt_w * resid

    # The adjustment (I - H_j) depends on the weights and the design, never on
    # y, so identical weight columns give identical eigendecompositions. Solve
    # the first column only and reuse it: the residuals still vary per dataset,
    # but the expensive part does not. This is the case whenever v arrives as a
    # single shared column, which cluster_robust_cov broadcasts on entry.
    shared_weights = n_iters > 1 and bool(np.all(w == w[:, [0]]))
    if shared_weights:
        design_slice, bread_slice = slice(0, 1), bread[:1]
    else:
        design_slice, bread_slice = slice(None), bread

    # Factor of the bread, reused by every group that takes the low-rank path.
    chol = _symmetric_sqrt(bread_slice)

    degenerate = False
    scores = np.empty((n_iters, n_preds, n_groups))
    for group, members in enumerate(group_members):
        # (i, n_j, p) and (i, n_j)
        group_X = np.transpose(whitened_X[members], (1, 0, 2))
        group_resid = whitened_resid[members].T
        adjust_X = group_X[design_slice]

        if members.size == 1:
            leverage = np.einsum("iap,ipq,iaq->ia", adjust_X, bread_slice, adjust_X)
            complement = 1.0 - leverage
            degenerate = degenerate or bool(np.any(complement < _MIN_LEVERAGE_COMPLEMENT))
            group_resid = group_resid / np.sqrt(np.maximum(complement, _MIN_LEVERAGE_COMPLEMENT))
        elif members.size > n_preds:
            # H_j has rank at most p, so decomposing the p x p Gram matrix
            # recovers the same adjustment for O(n_j p^2) instead of O(n_j^3).
            b_factor, middle, group_degenerate = _cr2_low_rank_factors(adjust_X, chol)
            degenerate = degenerate or group_degenerate
            group_resid = _cr2_low_rank_apply(b_factor, middle, group_resid[..., None])[..., 0]
        else:
            identity = np.eye(members.size)
            adjustment = identity - adjust_X @ bread_slice @ np.transpose(adjust_X, (0, 2, 1))
            # (I - H_j) is symmetric positive semi-definite, so its inverse
            # square root follows from an eigendecomposition. Only reached when
            # the group is no larger than the design, where n_j x n_j is the
            # cheaper of the two decompositions.
            evals, evecs = np.linalg.eigh(adjustment)
            degenerate = degenerate or bool(np.any(evals < _MIN_LEVERAGE_COMPLEMENT))
            evals = np.maximum(evals, _MIN_LEVERAGE_COMPLEMENT)
            inv_sqrt = (evecs * (evals**-0.5)[:, None, :]) @ np.transpose(evecs, (0, 2, 1))
            group_resid = np.einsum("iab,ib->ia", inv_sqrt, group_resid)

        scores[:, :, group] = np.einsum("iap,ia->ip", group_X, group_resid)

    if degenerate:
        warnings.warn(
            "At least one group has full leverage, so its CR2 residual "
            "adjustment does not exist and was floored. That group contributes "
            "nothing to the sandwich, which therefore understates the standard "
            "errors. Treat the affected coefficients as untrustworthy.",
            UserWarning,
            stacklevel=3,
        )

    return scores


def satterthwaite_dof(X, w, groups, model_cov=None):
    r"""Satterthwaite degrees of freedom for CR2 cluster-robust tests.

    Parameters
    ----------
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    w : :obj:`numpy.ndarray` of shape (K,) or (K, D)
        The weights used to fit the coefficients, matching those passed to
        :func:`~pymare.stats.cluster_robust_cov`.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per observation.
    model_cov : None or :obj:`numpy.ndarray` of shape (P, P, D), optional
        The model-based covariance of the coefficients, ``(X'WX)^-1``, reused
        when the caller already has it. Must correspond to the same weights as
        ``w``; pass None to have it rebuilt.
        Default = None.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (P, D)
        Degrees of freedom, one per predictor and parallel dataset, floored at 1.

    Warns
    -----
    UserWarning
        If any group has full leverage on a coefficient, so that the CR2
        adjustment does not exist for it and the degrees of freedom are floored.
    UserWarning
        If any returned value falls below ``MIN_DOF_FOR_SATTERTHWAITE``, outside
        the range in which the approximation is known to hold its nominal level.

    See Also
    --------
    cluster_robust_cov : Produces the variance estimate these degrees of freedom
        describe.
    pymare.results.MetaRegressionResults.fe_dof : Where the result is surfaced
        to users.

    Notes
    -----
    Cluster-robust standard errors are asymptotic in the number of groups, so the
    reference distribution matters when that number is small. The naive
    :math:`m - p` degrees of freedom of :footcite:t:`hedges2010robust` are
    adequate only when weight is spread evenly across groups. When a covariate is
    unbalanced at the group level -- a handful of groups carrying the only
    non-zero values of a predictor, say -- the effective sample size for that
    coefficient is far smaller than :math:`m - p` and the test becomes badly
    anti-conservative.

    The remedy is to match the first two moments of the variance estimate to a
    scaled chi-squared. For a single coefficient :math:`c'\beta`, the CR2
    variance estimate is a quadratic form :math:`u'\left(\sum_j g_jg_j'\right)u`
    in the whitened data :math:`u = W^{1/2}y`, with

    .. math::
        g_j = (I - \tilde{H})b_j, \qquad
        b_j = P_j'A_j\tilde{X}_jMc,

    where :math:`M = (X'WX)^{-1}`, :math:`\tilde{X} = W^{1/2}X`,
    :math:`\tilde{H} = \tilde{X}M\tilde{X}'`, :math:`A_j` is the CR2 adjustment
    of :func:`_cr2_scores`, and :math:`P_j` selects group :math:`j`. Matching
    moments gives

    .. math::
        \nu = \frac{\left(\operatorname{tr}S\right)^2}
                   {\operatorname{tr}\left(S^2\right)},
        \qquad S = G'G, \quad G = [g_1, \ldots, g_m].

    That ratio counts how *evenly* the information is spread rather than how
    much there is: equal eigenvalues return their count exactly, one dominant
    eigenvalue returns nearly 1.

    Credit divides three ways. The moment-matching approximation is
    :footcite:t:`satterthwaite1946approximate` (and, independently,
    :footcite:t:`welch1951comparison`, hence "Welch-Satterthwaite"); applying it
    to cluster-robust variance estimates is :footcite:t:`bell2002bias`; and
    establishing that it works for *meta-regression*, with guidance on when it
    does not, is :footcite:t:`tipton2015small`. The R packages `clubSandwich
    <https://cran.r-project.org/package=clubSandwich>`_
    :footcite:p:`pustejovsky2018small` and `robumeta
    <https://cran.r-project.org/package=robumeta>`_
    :footcite:p:`fisher2015robumeta` use it by default.

    Because the degrees of freedom are covariate-dependent, the number of groups
    alone cannot tell you whether the correction is needed
    :footcite:p:`pustejovsky2018small`, and they should be read before the
    p-values rather than after.

    Forming :math:`G` explicitly would cost a :math:`K \times K` matrix per
    parallel dataset, which is prohibitive when there are many of them. It is
    unnecessary: the :math:`b_j` have disjoint support, so :math:`b_j'b_l`
    vanishes off the diagonal, and :math:`M(\tilde{X}'\tilde{X})M = M` collapses
    the cross terms. Writing :math:`t_j = \tilde{X}_j'b_j`, the whole matrix
    reduces to

    .. math::
        S_{jl} = \delta_{jl}\lVert b_j \rVert^2 - t_j'Mt_l,

    which involves only :math:`p`-vectors. Factoring :math:`M = LL'` and setting
    :math:`R_j = L't_j` gives both traces from :math:`R` alone, so nothing larger
    than :math:`(m, p)` per dataset is ever materialized.

    References
    ----------
    .. footbibliography::

    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    n_observations, n_preds = X.shape
    if w.shape[0] != n_observations:
        raise ValueError("w must have one row per observation.")

    group_codes, group_labels = encode_groups(np.asarray(groups).ravel(), n_observations)
    members = [np.flatnonzero(group_codes == j) for j in range(group_labels.size)]
    n_groups = len(members)
    n_datasets = w.shape[1]

    by_size = {}
    for group, idx in enumerate(members):
        by_size.setdefault(idx.size, []).append((group, idx))

    # The dof depend on the design and the weights, not on y, so identical
    # weight columns give identical dof. Solve once and broadcast: this is the
    # common case when v is supplied as a single column.
    if n_datasets > 1 and np.all(w == w[:, [0]]):
        shared = satterthwaite_dof(
            X, w[:, [0]], groups, None if model_cov is None else model_cov[:, :, [0]]
        )
        return np.repeat(shared, n_datasets, axis=1)

    sqrt_w = np.sqrt(w)
    dof = np.empty((n_preds, n_datasets))
    degenerate = False

    # Chunk over parallel datasets: the working arrays are (chunk, m, p, p).
    chunk = max(1, int(4_000_000 // max(n_groups * n_preds * n_preds, 1)))
    for start in range(0, n_datasets, chunk):
        stop = min(start + chunk, n_datasets)
        sw = sqrt_w[:, start:stop]
        size = stop - start

        if model_cov is None:
            bread = np.linalg.pinv(np.einsum("kp,kd,kq->dpq", X, w[:, start:stop], X))
        else:
            bread = np.moveaxis(np.asarray(model_cov)[:, :, start:stop], -1, 0)

        # M = LL' via its symmetric square root. It factors the quadratic form
        # below (t_j'Mt_l = (L't_j).(L't_l)) and is also the factor the
        # low-rank CR2 path needs, so it is built once for both.
        chol = _symmetric_sqrt(bread)

        # ||b_j||^2 and t_j, both indexed (dataset, group, predictor).
        norms = np.empty((size, n_groups, n_preds))
        scores = np.empty((size, n_groups, n_preds, n_preds))

        # Every group needs an eigendecomposition of its own (I - H_j), but
        # groups of equal size have equal-shaped blocks and can go through
        # numpy in a single batched call. Meta-analytic group sizes repeat
        # heavily, so this usually collapses m calls down to one or two.
        for members_of_size in by_size.values():
            group_index = np.array([group for group, _ in members_of_size])
            rows = np.stack([idx for _, idx in members_of_size])  # (c, s)

            # Whitened design, (d, c, s, p).
            group_X = X[rows][None] * np.transpose(sw[rows], (2, 0, 1))[..., None]
            group_XM = np.einsum("dcsp,dpq->dcsq", group_X, bread)
            group_size = rows.shape[1]
            if group_size == 1:
                # Singleton groups reduce CR2 to HC2: the block is the scalar
                # 1 - h_j, so the matrix inverse square root is just 1/sqrt.
                # Worth special-casing because one group per estimate is a
                # documented usage and eigh on 1x1 blocks is pure overhead.
                complement = 1.0 - np.einsum("dcsq,dcsq->dcs", group_XM, group_X)[..., 0]
                if np.any(complement < _MIN_LEVERAGE_COMPLEMENT):
                    degenerate = True
                scale = np.maximum(complement, _MIN_LEVERAGE_COMPLEMENT) ** -0.5
                b = group_XM * scale[..., None, None]
            elif group_size > n_preds:
                # H_j has rank at most p, so the p x p Gram matrix carries the
                # whole adjustment; see _cr2_low_rank_factors.
                b_factor, middle, group_degenerate = _cr2_low_rank_factors(group_X, chol[:, None])
                degenerate = degenerate or group_degenerate
                b = _cr2_low_rank_apply(b_factor, middle, group_XM)
            else:
                adjustment = np.eye(group_size) - np.einsum("dcsq,dctq->dcst", group_XM, group_X)
                evals, evecs = np.linalg.eigh(adjustment)
                if np.any(evals < _MIN_LEVERAGE_COMPLEMENT):
                    degenerate = True
                evals = np.maximum(evals, _MIN_LEVERAGE_COMPLEMENT)
                # Raise to the power before broadcasting, not after. Written as
                # ``evals[..., None, :] ** -0.5`` the power is evaluated on the
                # broadcast view, so it runs once per matrix *entry* rather than
                # once per eigenvalue -- s times the work for identical bits.
                scale = evals**-0.5
                inv_sqrt = (scale[..., None, :] * evecs) @ np.swapaxes(evecs, -1, -2)
                b = np.einsum("dcst,dctq->dcsq", inv_sqrt, group_XM)
            norms[:, group_index, :] = np.square(b).sum(axis=2)
            scores[:, group_index, :, :] = np.einsum("dcsp,dcsr->dcpr", group_X, b)

        for pred in range(n_preds):
            r = scores[:, :, :, pred] @ chol  # (d, m, p), rows L't_j
            diagonal = np.square(r).sum(axis=2)  # t_j'Mt_j
            nb = norms[:, :, pred]
            gram = np.einsum("dmp,dmq->dpq", r, r)  # R'R, (d, p, p)

            trace = nb.sum(axis=1) - diagonal.sum(axis=1)
            trace_sq = (
                np.square(gram).sum(axis=(1, 2))
                - 2.0 * (nb * diagonal).sum(axis=1)
                + np.square(nb).sum(axis=1)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                dof[pred, start:stop] = np.square(trace) / trace_sq

    if degenerate:
        warnings.warn(
            "At least one group has full leverage on a coefficient, so the CR2 "
            "adjustment does not exist for it. This happens when a single group "
            "supplies all the information about a predictor. The degrees of "
            "freedom are floored at 1 and the corresponding test should not be "
            "trusted.",
            UserWarning,
            stacklevel=2,
        )

    dof = np.where(np.isfinite(dof), dof, 1.0)
    dof = np.maximum(dof, 1.0)

    # A comfortable group count does not imply usable degrees of freedom, so
    # this has to be checked on the dof themselves rather than on m.
    low = dof < MIN_DOF_FOR_SATTERTHWAITE
    if np.any(low):
        predictors = sorted(set(np.flatnonzero(low.any(axis=1)).tolist()))
        warnings.warn(
            f"Satterthwaite degrees of freedom below {MIN_DOF_FOR_SATTERTHWAITE} "
            f"for predictor(s) {predictors} (smallest {dof[low].min():.2f}). The "
            "approximation is only known to control the Type I error rate above "
            "about that value, so the corresponding p-values and intervals are "
            "outside their validated range and should be read as a diagnostic "
            "rather than a result. This usually means a group-level predictor is "
            "carried by very few groups; more groups on the scarce side of the "
            "predictor is the remedy, not a different estimator.",
            UserWarning,
            stacklevel=2,
        )

    return dof


def cluster_robust_cov(
    y,
    v,
    X,
    beta,
    groups,
    tau2=0.0,
    small_sample=True,
    model_cov=None,
    w=None,
    method="CR2",
):
    r"""Compute a cluster-robust ("sandwich") covariance matrix for 2-D WLS.

    Implements robust variance estimation (RVE) for meta-regression with
    dependent effect size estimates :footcite:p:`hedges2010robust`. Estimates
    sharing a group label are treated as statistically dependent, e.g.
    repeated observations contributed by the same sampling unit.

    The estimator is

    .. math::
        V_R = (X'WX)^{-1} \left( \sum_j X_j' W_j e_j e_j' W_j X_j \right)
              (X'WX)^{-1}

    where :math:`j` indexes groups and :math:`e_j = y_j - X_j \beta` are the
    residuals for group :math:`j`. Only the covariance changes: the point
    estimates in ``beta`` and the value of ``tau2`` are taken as given.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        2d array of estimates (observations x parallel datasets).
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    beta : :obj:`numpy.ndarray` of shape (P, D)
        Fixed effect coefficients, as returned by
        :func:`~pymare.stats.weighted_least_squares`.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per observation. Any hashable labels are
        accepted.
    tau2 : :obj:`float` or :obj:`numpy.ndarray`, optional
        tau^2 estimate used for the weights, matching the value used to
        obtain ``beta``.
        Default = 0.
    small_sample : :obj:`bool`, optional
        Whether to apply the ``m / (m - p)`` small-sample scaling, where m is
        the number of groups and p the number of predictors. Only used by
        ``method="CR0"``; CR2 corrects for leverage directly, so the two would
        double-count. Default = True.

        This is the original adjustment of :footcite:t:`hedges2010robust`,
        :math:`A_j = [m / (m - p)]^{1/2} I_j`. Note that it is **not**
        ``clubSandwich``'s ``CR1``, which uses :math:`m / (m - 1)`, nor Stata's
        ``CR1S``, which uses :math:`mK / [(m - 1)(K - p)]`; enabling it will not
        reproduce another package's ``CR1`` column. It is retained for
        reproducing analyses that predate the leverage-based corrections --
        :footcite:t:`fisher2015robumeta` report that simulations find this
        adjustment "inadequate except in very specific cases", which is why
        ``method="CR2"`` is the default.
    model_cov : None or :obj:`numpy.ndarray` of shape (P, P, D), optional
        The model-based covariance of the coefficients, ``(X'WX)^-1``, as
        returned by :func:`~pymare.stats.weighted_least_squares` with
        ``return_cov=True``. Supplying it avoids recomputing a pseudo-inverse
        the caller already has. It must correspond to the same ``tau2`` and the
        same ``w``, since it is used as the bread of the sandwich; a mismatched
        value produces a silently wrong result rather than an error.
        Default = None.
    w : None or :obj:`numpy.ndarray` of shape (K, D), optional
        Precomputed weights overriding the default ``1 / (v + tau2)``. Must be
        the same weights that produced ``beta``, or the residuals will not
        correspond to the fit.
        Default = None.
    method : {"CR2", "CR0"}, optional
        The residual adjustment to apply. The ``CRn`` naming follows the R
        package `clubSandwich <https://cran.r-project.org/package=clubSandwich>`_
        :footcite:p:`pustejovsky2018small`.

        -   ``"CR2"`` (default) inflates each group's residuals by
            :math:`(I_j - H_j)^{-1/2}` to undo the shrinkage caused by fitting
            :math:`\beta` with that group included :footcite:p:`bell2002bias`.
        -   ``"CR0"`` uses the raw residuals with the blunt ``m / (m - p)``
            scaling. This is the historical behaviour.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (P, P, D)
        The robust covariance matrix for the fixed effects, oriented like the
        covariance returned by :func:`~pymare.stats.weighted_least_squares`.

    Raises
    ------
    ValueError
        If ``groups`` does not contain one label per observation.
    ValueError
        If ``method`` is not one of ``"CR2"`` or ``"CR0"``.
    ValueError
        If the number of groups does not exceed the number of predictors, where
        every group has full leverage and the sandwich is undefined.

    Warns
    -----
    UserWarning
        If there are at or below ``MIN_CLUSTERS_FOR_RVE`` groups, where robust
        variance estimation is known to be anti-conservative.

    See Also
    --------
    satterthwaite_dof : The reference distribution these standard errors need.
    correlated_effects_weights : Levels weight across groups, which reduces the bias this
        estimator is subject to more effectively than any choice of ``method``.

    Notes
    -----
    RVE is asymptotic in the number of *groups*, not the number of estimates, and
    is anti-conservative when there are few of them. CR2 substantially reduces
    that bias but does not remove it.

    The dominant driver is not the group count but how unevenly weight is spread
    across groups. When one group carries much of the total weight, the sandwich
    is estimating that group's variance from what is effectively a single
    residual, and no residual adjustment fully rescues it. Weighting the
    estimates with :func:`~pymare.stats.correlated_effects_weights` levels the weight across
    groups and is far more effective than any choice of ``method``.

    For the same reason, a comfortable group count is *not* evidence that the
    small-sample corrections are unnecessary. :footcite:t:`pustejovsky2018small`
    put it directly: because the degrees of freedom are covariate-dependent, it is
    not possible to decide whether a correction is needed from the number of
    groups alone. :footcite:t:`imbens2016robust` find these corrections still
    improve coverage materially at fifty or more groups, and both they and
    :footcite:t:`tipton2015small` recommend using ``method="CR2"`` with the
    Satterthwaite degrees of freedom routinely rather than only when ``m`` looks
    small. Read :func:`~pymare.stats.satterthwaite_dof` before the p-value, not
    the group count.

    Group labels are encoded with :func:`~pymare.stats.encode_groups` rather than
    ``np.unique`` because the docstring promises *any hashable* label, while
    ``np.unique`` additionally requires them to be sortable and so would reject a
    mix of :obj:`str` and :obj:`int`. Every quantity here is a sum over groups, so
    the coding order does not matter.

    References
    ----------
    .. footbibliography::

    """
    groups = np.asarray(groups).ravel()
    if groups.shape[0] != y.shape[0]:
        raise ValueError(
            f"groups must have one label per observation: expected {y.shape[0]} "
            f"labels, got {groups.shape[0]}."
        )

    if method not in ("CR0", "CR2"):
        raise ValueError(f"Invalid method '{method}'; must be one of 'CR2', 'CR0'.")

    # encode_groups, not np.unique: the docstring promises any hashable label,
    # and np.unique additionally requires them to be *sortable*, so a mix of
    # str and int raises. Every quantity below is a sum over groups, so the
    # coding order does not matter.
    group_codes, group_labels = encode_groups(groups, n_observations=y.shape[0])
    n_groups = group_labels.size
    n_preds = X.shape[1]

    if n_groups <= n_preds:
        # Every group's leverage is then exactly one: the design is saturated at
        # the group level, so each group's residuals are fitted away and the
        # meat collapses to zero. CR0 with the m/(m-p) scaling also divides by
        # zero. Left alone this returns a standard error of ~1e-12 and a
        # p-value of ~1e-11 -- maximally significant -- rather than admitting
        # that the sandwich is undefined.
        raise ValueError(
            f"Cluster-robust variance estimation needs more groups than "
            f"predictors, but got {n_groups} group(s) and {n_preds} "
            "predictor(s). The sandwich is undefined here: every group has "
            "full leverage, so there are no residuals left to estimate from."
        )

    if n_groups <= MIN_CLUSTERS_FOR_RVE:
        warnings.warn(
            f"Cluster-robust variance estimation with only {n_groups} groups. "
            "RVE is asymptotic in the number of groups and is known to be "
            f"anti-conservative at or below about {MIN_CLUSTERS_FOR_RVE}; the "
            "residual adjustment only partly compensates, so p-values may "
            "still be too small. If weight is spread unevenly across groups, "
            "consider pymare.stats.correlated_effects_weights.",
            UserWarning,
            stacklevel=2,
        )

    w = 1.0 / (v + tau2) if w is None else w

    # A single column of weights applies to every parallel dataset. The CR0
    # path broadcasts it for free, but CR2 slices per group and needs the
    # dataset axis to be real, so materialize it once here rather than letting
    # the two methods disagree about which inputs they accept. ensure_2d also
    # accepts the 1-D form that satterthwaite_dof documents, so the two
    # functions agree about what weights look like.
    w = ensure_2d(np.asarray(w, dtype=float))
    if w.shape[1] != y.shape[1]:
        w = np.broadcast_to(w, y.shape)

    # Einsum indices: k = observations, p/q = predictors, i = parallel iterates,
    # j = groups.
    wX = np.einsum("kp,ki->ipk", X, w)

    # (X'WX)^-1 is exactly the model-based covariance, so reuse it when the
    # caller already has it rather than repeating the pinv.
    if model_cov is None:
        bread = np.linalg.pinv(wX.dot(X))  # (i, p, p)
    else:
        bread = np.asarray(model_cov).T  # (p, p, i) -> (i, p, p)

    # Residuals, (k, i), matching the orientation of y.
    resid = y - X.dot(beta)

    if method == "CR2":
        group_members = [np.flatnonzero(group_codes == group) for group in range(n_groups)]
        scores = _cr2_scores(X, w, resid, group_members, bread)
    else:
        # Sum the scores within each group to get X_j' W_j e_j, (i, p, m).
        # Sorting is only necessary when a group appears in multiple disjoint
        # blocks. The common contiguous/equal-size layout can instead use a
        # reshape and sum, avoiding both the sort and a full advanced-indexing
        # copy.
        starts = np.r_[0, np.flatnonzero(np.diff(group_codes)) + 1]
        if starts.size == n_groups:
            # The weighted design is no longer needed after the bread is
            # formed, so reuse it for the per-observation scores rather than
            # allocating another array of size (i, p, k).
            wX *= resid.T[:, None, :]
            group_sizes = np.diff(np.r_[starts, group_codes.size])
            if np.all(group_sizes == group_sizes[0]):
                scores = wX.reshape(wX.shape[0], n_preds, n_groups, group_sizes[0]).sum(axis=3)
            else:
                scores = np.add.reduceat(wX, starts, axis=2)
        else:
            # Multiplication into a fresh contiguous array is faster before the
            # advanced-indexing copy required by this uncommon fallback path.
            contrib = wX * resid.T[:, None, :]
            order = np.argsort(group_codes, kind="stable")
            starts = np.searchsorted(group_codes[order], np.arange(n_groups))
            scores = np.add.reduceat(contrib[:, :, order], starts, axis=2)

    # meat[i] = scores[i] @ scores[i].T, as a batched matmul.
    meat = scores @ scores.transpose(0, 2, 1)

    robust_cov = bread @ meat @ bread

    # CR2 removes the leverage-induced downward bias directly, so the blunt
    # m / (m - p) inflation that CR0 needs would double-count here.
    if small_sample and method == "CR0":
        robust_cov = robust_cov * (n_groups / (n_groups - n_preds))

    # Match the (p, p, i) orientation used for the model-based covariance.
    return robust_cov.T


def ensure_2d(arr):
    """Ensure the passed array has 2 dimensions."""
    if arr is None:
        return arr

    try:
        arr = np.array(arr)
    except:
        return arr

    if arr.ndim == 1:
        arr = arr[:, None]

    return arr


def q_profile(y, v, X, alpha=0.05, groups=None):
    """Get the CI for tau^2 via the Q-Profile method.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K,)
        1d array of observation-level estimates
    v : :obj:`numpy.ndarray` of shape (K,)
        1d array of observation-level variances
    X : :obj:`numpy.ndarray` of shape (K[, P])
        1d or 2d array containing observation-level predictors
        (including intercept); has dimensions K x P, where K is the number
        of observations and P is the number of predictor variables.
    alpha : :obj:`float`, optional
        alpha value defining the coverage of the CIs,
        where width(CI) = 1 - alpha. Default = 0.05.
    groups : None or :obj:`numpy.ndarray` of shape (K,), optional
        Group labels marking dependent estimates. When supplied, ``Q`` is the
        correlated-effects statistic of :func:`q_gen` and is referred to the
        number of independent groups rather than the number of rows.
        Default = None.

    Returns
    -------
    :obj:`dict`
        A dictionary with keys 'ci_l' and 'ci_u', corresponding to the lower
        and upper bounds of the tau^2 confidence interval, respectively.

    Notes
    -----
    Following the :footcite:t:`viechtbauer2007confidence` implementation,
    this method returns the interval that gives an equal probability mass at both tails
    (i.e., ``P(tau^2 <= lower_bound)  == P(tau^2 >= upper_bound) == alpha/2``),
    and *not* the smallest possible range of tau^2 values that provides the desired coverage.

    With ``groups``, the same inversion is applied to the correlated-effects
    ``Q``, whose expectation is matched at ``m - p`` by the moment estimator of
    :func:`~pymare.stats.correlated_effects_tau2`. The interval is therefore an
    approximation of the same order as that estimator, and keeps the point
    estimate and its interval describing one quantity.

    References
    ----------
    .. footbibliography::
    """
    k, p = X.shape
    if groups is not None:
        k = encode_groups(groups, n_observations=X.shape[0])[1].size
    df = k - p
    l_crit = ss.chi2.ppf(1 - alpha / 2, df)
    u_crit = ss.chi2.ppf(alpha / 2, df)
    args = (ensure_2d(y), ensure_2d(v), X)
    bds = Bounds([0], [np.inf], keep_feasible=True)

    # Use a point estimate of tau^2 as a starting point; when using a fixed
    # value, minimize() sometimes fails to stay in bounds. It has to be the
    # estimator that matches the Q being inverted, or the search can start on
    # the wrong side of the upper root.
    if groups is None:
        from .estimators import DerSimonianLaird

        ub_start = 2 * DerSimonianLaird().fit(y, v, X).params_["tau2"]
    else:
        ub_start = 2 * correlated_effects_tau2(*args, groups)

    lb = minimize(lambda x: (q_gen(*args, x, groups) - l_crit) ** 2, [0], bounds=bds).x[0]
    ub = minimize(lambda x: (q_gen(*args, x, groups) - u_crit) ** 2, ub_start, bounds=bds).x[0]
    return {"ci_l": lb, "ci_u": ub}


def q_gen(y, v, X, tau2, groups=None):
    """Calculate a generalized form of Cochran's Q-statistic.

    This version of the Q statistic is described in :footcite:t:`veroniki2016methods`.

    Parameters
    ----------
    y : :obj:`numpy.ndarray`
        1d array of observation-level estimates
    v : :obj:`numpy.ndarray`
        1d array of observation-level variances
    X : :obj:`numpy.ndarray`
        1d or 2d array containing observation-level predictors
        (including intercept); has dimensions K x P, where K is the number
        of observations and P is the number of predictor variables.
    tau2 : :obj:`float`
        Between-unit variance. Must be >= 0.
    groups : None or :obj:`numpy.ndarray` of shape (K,), optional
        Group labels marking dependent estimates. When supplied, the
        correlated-effects weights of
        :func:`~pymare.stats.correlated_effects_weights` are used in place of
        ``1 / (v + tau2)``, so that a group's contribution does not grow with
        the number of rows it supplied. This is ``Q_E`` in the notation of
        :footcite:t:`fisher2015robumeta`. Default = None.

    Returns
    -------
    :obj:`float`
        A float giving the value of Cochran's Q-statistic.

    References
    ----------
    .. footbibliography::
    """
    if np.any(tau2 < 0):
        raise ValueError("Value of tau^2 must be >= 0.")

    w = 1.0 / (v + tau2) if groups is None else correlated_effects_weights(v, groups, tau2)
    beta = weighted_least_squares(y, v, X, tau2, w=w)
    return (w * (y - X.dot(beta)) ** 2).sum(0)


def bonferroni(p_values):
    """Perform Bonferroni correction on p values.

    This correction is based on the one described in :footcite:t:`bonferroni1936teoria` and
    :footcite:t:`shaffer1995multiple`.

    .. versionadded:: 0.0.4

    Parameters
    ----------
    p_values : :obj:`numpy.ndarray`
        Uncorrected p values.

    Returns
    -------
    p_corr : :obj:`numpy.ndarray`
        Corrected p values.

    References
    ----------
    .. footbibliography::
    """
    p_corr = p_values * p_values.size
    p_corr[p_corr > 1] = 1
    return p_corr


def fdr(p_values, q=0.05, method="bh"):
    """Perform FDR correction on p values.

    .. versionadded:: 0.0.4

    Parameters
    ----------
    p_values : :obj:`numpy.ndarray`
        Array of p values.
    q : :obj:`float`, optional
        Alpha value. Default is 0.05.
    method : {"bh", "by"}, optional
        Method to use for correction.
        Either "bh" (Benjamini-Hochberg :footcite:p:`benjamini1995controlling`) or
        "by" (Benjamini-Yekutieli :footcite:p:`benjamini2001control`).
        Default is "bh".

    Returns
    -------
    p_adjusted : :obj:`numpy.ndarray`
        Array of adjusted p values.

    Notes
    -----
    This function is adapted from ``statsmodels``, which is licensed under a BSD-3 license.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    statsmodels.stats.multitest.fdrcorrection
    """
    sort_idx = np.argsort(p_values)
    revert_idx = np.argsort(sort_idx)
    p_sorted = p_values[sort_idx]

    n_tests = p_values.size

    # empirical cumulative density function
    ecdf = np.linspace(0, 1, n_tests + 1)[1:]
    if method == "by":
        # NOTE: I don't know what cm stands for
        cm = np.sum(1 / np.arange(1, n_tests + 1))
        ecdffactor = ecdf / cm
    else:
        ecdffactor = ecdf

    p_adjusted = p_sorted / ecdffactor
    p_adjusted = np.minimum.accumulate(p_adjusted[::-1])[::-1]
    # NOTE: Why not this?
    # p_adjusted = np.maximum.accumulate(p_adjusted)

    p_adjusted[p_adjusted > 1] = 1
    p_adjusted = p_adjusted[revert_idx]

    return p_adjusted


def var_to_ci(y, v, n):
    """Convert sampling variance to 95% CI."""
    term = 1.96 * np.sqrt(v) / np.sqrt(n)
    return y - term, y + term
