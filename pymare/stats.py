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


class WeightedInterceptCR2Statistics(NamedTuple):
    """Reusable sufficient statistics for signed intercept-only CR2 tests."""

    weighted_values: np.ndarray
    adjusted_values: np.ndarray
    adjusted_sum_squares: np.ndarray
    adjusted_weight_sum: float
    total_weight: float


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


def group_weights(v, groups, tau2=0.0):
    r"""Rescale inverse-variance weights so replication does not buy influence.

    The ordinary weight :math:`1 / (v_i + \tau^2)` gives a group that
    contributed :math:`n_j` estimates :math:`n_j` times the pull of a group
    that contributed one, because the weights of its members are summed. This
    divides each weight by its group size,

    .. math::
        w_i = \frac{1}{n_j (v_i + \tau^2)},

    so a group's total weight is the *mean* of its members' weights rather
    than their sum. Replication therefore stops buying influence in proportion
    to row count, while genuinely more precise groups still count more.

    This follows the "correlated effects" weighting of
    :footcite:t:`hedges2010robust`, but is not identical to the version in the R
    package `robumeta <https://cran.r-project.org/package=robumeta>`_
    :footcite:p:`fisher2015robumeta`, which assigns every row in a group the
    *same* weight built from the group's mean variance,

    .. math::
        w_{ij}^{\text{robumeta}} = \frac{1}{n_j(\bar{v}_j + \tau^2)},
        \qquad \bar{v}_j = \frac{1}{n_j}\sum_i v_{ij}.

    The two agree exactly for singleton groups and for groups whose ``v`` is
    constant, and differ otherwise: this function's group total is
    :math:`\operatorname{mean}_i(1/v_i)` while robumeta's is
    :math:`1/\operatorname{mean}_i(v_i)`, so by the arithmetic-harmonic mean
    inequality this function never gives a group *less* total weight than
    robumeta does. Keeping the row-specific :math:`v_i` preserves genuine
    precision differences between rows; robumeta equalizes them because its
    working model assumes :math:`v_{ij} \approx v_j` within a group. Expect
    the weights to differ when within-group variances differ.

    Notes
    -----
    Invariance to duplication is exact only when the duplicated row's precision
    equals its group's mean precision -- which is automatic when ``v`` is
    constant within the group. With heterogeneous within-group variances,
    duplicating an above-average-precision row still raises its group's total
    weight somewhat (and duplicating a below-average one lowers it). This is a
    property of correlated-effects weighting rather than of this implementation;
    robumeta behaves the same way. Use ``weight_scheme="collapse"`` if one row per
    group is what the design actually warrants.

    Parameters
    ----------
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group labels, one per estimate. Any hashable labels are accepted.
    tau2 : :obj:`float` or :obj:`numpy.ndarray`, optional
        tau^2 estimate to use for the weights.
        Default = 0.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (K, D)
        The rescaled weights.

    References
    ----------
    .. footbibliography::

    """
    return normalize_group_weights(1.0 / (v + tau2), groups)


def collapse_groups(y, v, X, groups, rho=DEFAULT_RHO):
    r"""Aggregate each group to a single effect estimate.

    Moment-based estimators of :math:`\tau^2` count every row of ``y`` as an
    independent observation. When a group contributes several rows, that
    pseudo-replication biases :math:`\tau^2` downward: the duplicated rows agree
    with each other by construction, so the observed dispersion looks smaller
    than the number of observations would imply. Estimating :math:`\tau^2` from one
    effect per cluster removes the problem.

    Each group is collapsed to the unweighted mean of its members, whose
    sampling variance is

    .. math::
        \bar{v}_j = \frac{1}{n_j^2}
            \left( \sum_i v_i + \rho \sum_{i \neq k} \sqrt{v_i v_k} \right),

    i.e. the variance of a mean whose terms are correlated at :math:`\rho`.

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

    Notes
    -----
    The resulting :math:`\tau^2` is only weakly sensitive to ``rho``: sweeping it
    across its whole range typically moves downstream error rates by well under
    a percentage point, because it enters only through the relative weighting of
    already-aggregated clusters. Assuming a single value is therefore reasonable
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

    The counterpart to :func:`~pymare.stats.collapse_groups` for models
    parameterized by sample size rather than sampling variance, where
    :math:`v_i = \sigma^2 / n_i` and :math:`\sigma^2` is itself being
    estimated. Requiring the collapsed effect to satisfy
    :math:`\sigma^2 / n_j^{\text{eff}} = \operatorname{Var}(\bar{y}_j)` gives

    .. math::
        n_j^{\text{eff}} = \frac{m^2}{\sum_i 1/n_i
            + \rho\left[\left(\sum_i 1/\sqrt{n_i}\right)^2
            - \sum_i 1/n_i\right]},

    which is free of :math:`\sigma^2` and so can be formed before the
    likelihood is evaluated.

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

    _, group_codes = np.unique(groups, return_inverse=True)
    group_codes = np.ravel(group_codes)
    members = [np.flatnonzero(group_codes == g) for g in range(group_codes.max() + 1)]
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
    """Perform 2-D weighted least squares.

    Parameters
    ----------
    y : :obj:`numpy.ndarray`
        2-d array of estimates (observations x parallel datasets)
    v : :obj:`numpy.ndarray`
        2-d array of sampling variances
    X : :obj:`numpy.ndarray`
        Fixed effect design matrix
    tau2 : :obj:`float`, optional
        tau^2 estimate to use for weights.
        Default = 0.
    return_cov : :obj:`bool`, optional
        Whether or not to return the covariance matrix of the coefficients.
        Default = False.
    w : None or :obj:`numpy.ndarray`, optional
        Precomputed weights of the same shape as ``y``, overriding the default
        ``1 / (v + tau2)``. Use :func:`~pymare.stats.group_weights` to obtain
        weights that do not reward replication within a cluster.
        Default = None.

    Returns
    -------
    params[, cov]
        If return_cov is True, returns both the fixed parameter estimates and
        ``(X'WX)^-1``, which is the *covariance* matrix of those estimates; if
        False, only the parameter estimates.

        Note that PyMARE stores this quantity under the key ``"inv_cov"``, which
        is a misnomer of long standing: ``X'WX`` is the inverse covariance, so
        its inverse is the covariance itself. Anything that takes
        ``sqrt(diagonal(...))`` of it -- :attr:`pymare.results.MetaRegressionResults.fe_se`,
        for instance -- is treating it as a covariance, which is correct.
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


def _symmetric_sqrt(matrices):
    """Return L with ``L @ L.T`` equal to each symmetric PSD input matrix."""
    evals, evecs = np.linalg.eigh(matrices)
    return evecs * np.sqrt(np.maximum(evals, 0.0))[..., None, :]


def _cr2_low_rank_factors(group_X, chol):
    r"""Factor the CR2 adjustment using only :math:`p \times p` work.

    :math:`H_j = \tilde{X}_j M \tilde{X}_j'` is an outer product of an
    :math:`n_j \times p` matrix with itself, so its rank is at most :math:`p`.
    Eigendecomposing the full :math:`n_j \times n_j` block therefore spends
    :math:`O(n_j^3)` to recover at most :math:`p` non-unit eigenvalues plus
    :math:`n_j - p` copies of 1. Writing :math:`M = LL'` and
    :math:`B = \tilde{X}_j L`, the non-zero eigenvalues :math:`\mu_i` of
    :math:`H_j = BB'` are exactly the eigenvalues of the :math:`p \times p`
    matrix :math:`B'B`, whose eigenvectors :math:`q_i` lift to
    :math:`u_i = Bq_i / \sqrt{\mu_i}`. Since the orthogonal complement of the
    :math:`u_i` is an eigenspace of :math:`I - H_j` with eigenvalue 1, and
    therefore needs no adjustment at all,

    .. math::
        (I_j - H_j)^{-1/2}
            = I + \sum_i \left[(1 - \mu_i)^{-1/2} - 1\right] u_iu_i'
            = I + BGB', \qquad
        G = Q \operatorname{diag}(c) Q',

    with :math:`c_i = \left[(1 - \mu_i)^{-1/2} - 1\right] / \mu_i`. The
    adjustment is never formed: callers apply ``I + B G B'`` to a right-hand
    side, so nothing larger than :math:`(n_j, p)` is materialized.

    ``c_i`` has a removable singularity at :math:`\mu_i = 0`, where the limit
    is :math:`1/2`. Substituting it is cosmetic rather than load-bearing --
    :math:`Bq_i` vanishes with :math:`\mu_i`, since
    :math:`\lVert Bq_i \rVert^2 = \mu_i`, so the whole term is killed either
    way -- but it keeps the intermediate finite.

    Parameters
    ----------
    group_X : :obj:`numpy.ndarray` of shape (..., n, p)
        Whitened design rows for one group.
    chol : :obj:`numpy.ndarray` of shape (..., p, p)
        Any factor with ``chol @ chol.T == M``, e.g. from
        :func:`_symmetric_sqrt`.

    Returns
    -------
    :obj:`tuple`
        ``(B, G, degenerate)``. ``degenerate`` flags a group whose leverage
        reached one, where the adjustment does not exist and the eigenvalue
        floor took over.
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
    """Apply ``I + B G B'`` to ``rhs`` of shape ``(..., n, q)``."""
    return rhs + b_factor @ (middle @ (np.swapaxes(b_factor, -1, -2) @ rhs))


def _cr2_scores(X, w, resid, group_members, bread):
    r"""Compute bias-reduced (CR2) cluster scores.

    The CR0 score for cluster :math:`j` is :math:`X_j' W_j e_j`. Because
    :math:`\beta` is fitted using cluster :math:`j` itself, those residuals are
    shrunk toward zero in proportion to the cluster's leverage, which biases
    the sandwich downward -- severely when one cluster carries much of the
    weight. CR2 :footcite:p:`bell2002bias` undoes that shrinkage exactly under
    the working model by inflating the residuals with
    :math:`A_j = (I_j - H_j)^{-1/2}` in the whitened metric, giving

    .. math::
        s_j = \tilde{X}_j' (I_j - \tilde{X}_j M \tilde{X}_j')^{-1/2}
              W_j^{1/2} e_j,

    with :math:`\tilde{X}_j = W_j^{1/2} X_j` and :math:`M = (X'WX)^{-1}`.
    Singleton clusters reduce to the familiar scalar :math:`1/\sqrt{1 - h_j}`,
    i.e. HC2 :footcite:p:`mackinnon1985some`.

    "Exactly unbiased" is always with respect to a *working model* for the error
    covariance, which the analyst chooses. Bell and McCaffrey pick :math:`A_j` to
    satisfy :math:`A_j (I - H)_j \Phi (I - H)_j' A_j' = \Phi_j`; the form above is
    the solution for :math:`\Phi = I` in the whitened metric, i.e. under the
    assumption that the weights are correct and the observations independent --
    the same assumption the sandwich exists to avoid relying on. That is not
    circular so much as pragmatic: simulation shows the correction helps
    substantially even when the working model is wrong
    :footcite:p:`tipton2015small,imbens2016robust`, and its influence fades as the
    number of clusters grows. This form coincides with the correlated-effects
    adjustment :math:`A_j^C` of :footcite:t:`fisher2015robumeta`.

    Because CR2 targets unbiasedness rather than conservatism, it is *not*
    guaranteed to exceed CR0. It does for singleton clusters, where the
    adjustment is a scalar :math:`\ge 1` applied to each score; for larger
    clusters the score is a projection of the inflated residuals, and in a small
    fraction of designs a given coefficient's CR2 variance comes out below its
    CR0 counterpart. :footcite:t:`pustejovsky2018small` place CR2 between CR1,
    which under-corrects, and CR3 :footcite:p:`mancl2001covariance`, which
    over-corrects.

    Implemented from the published formulation rather than ported; the R
    package `clubSandwich <https://cran.r-project.org/package=clubSandwich>`_
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

    Cluster-robust standard errors are asymptotic in the number of groups, so
    the reference distribution matters when that number is small. The naive
    :math:`m - p` degrees of freedom of :footcite:t:`hedges2010robust` are
    adequate only when weight is spread evenly across groups. When a covariate
    is unbalanced at the group level -- a handful of groups carrying the only
    non-zero values of a predictor, say -- the effective sample size for that
    coefficient is far smaller than :math:`m - p` and the test becomes badly
    anti-conservative.

    The remedy is to match the first two moments of the variance estimate to a
    scaled chi-squared, which is what this function computes and what the R
    packages `clubSandwich <https://cran.r-project.org/package=clubSandwich>`_
    :footcite:p:`pustejovsky2018small` and `robumeta
    <https://cran.r-project.org/package=robumeta>`_
    :footcite:p:`fisher2015robumeta` use by default. Credit divides three ways:
    the moment-matching approximation is :footcite:t:`satterthwaite1946approximate`
    (and, independently, :footcite:t:`welch1951comparison`, which is why it is
    often called Welch-Satterthwaite); applying it to cluster-robust variance
    estimates is :footcite:t:`bell2002bias`; and establishing that it works for
    *meta-regression*, together with the guidance on when it does not, is
    :footcite:t:`tipton2015small`.

    For a single coefficient :math:`c'\beta`, the CR2 variance estimate is a
    quadratic form :math:`u'\left(\sum_j g_jg_j'\right)u` in the whitened data
    :math:`u = W^{1/2}y`, with

    .. math::
        g_j = (I - \tilde{H})b_j, \qquad
        b_j = P_j'A_j\tilde{X}_jMc,

    where :math:`M = (X'WX)^{-1}`, :math:`\tilde{X} = W^{1/2}X`,
    :math:`\tilde{H} = \tilde{X}M\tilde{X}'`, :math:`A_j` is the CR2
    adjustment of :func:`_cr2_scores`, and :math:`P_j` selects group
    :math:`j`. Matching the first two moments of that form to a scaled
    chi-squared gives

    .. math::
        \nu = \frac{\left(\operatorname{tr}S\right)^2}
                   {\operatorname{tr}\left(S^2\right)},
        \qquad S = G'G, \quad G = [g_1, \ldots, g_m].

    Notes
    -----
    Forming :math:`G` explicitly would cost a :math:`K \times K` matrix per
    parallel dataset, which is prohibitive when there are many of them. It is
    unnecessary: the :math:`b_j` have disjoint support, so :math:`b_j'b_l`
    vanishes off the diagonal, and :math:`M(\tilde{X}'\tilde{X})M = M`
    collapses the cross terms. Writing :math:`t_j = \tilde{X}_j'b_j`, the
    whole matrix reduces to

    .. math::
        S_{jl} = \delta_{jl}\lVert b_j \rVert^2 - t_j'Mt_l,

    which involves only :math:`p`-vectors. Factoring :math:`M = LL'` and
    setting :math:`R_j = L't_j` gives both traces from :math:`R` alone, so
    nothing larger than :math:`(m, p)` per dataset is ever materialized.

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
        Degrees of freedom, one per predictor and parallel dataset.

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
        covariance returned by
        :func:`~pymare.stats.weighted_least_squares`.

    Notes
    -----
    RVE is asymptotic in the number of *groups*, not the number of estimates,
    and is anti-conservative when there are few of them. CR2 substantially
    reduces that bias but does not remove it.

    The dominant driver is not the group count but how unevenly weight is
    spread across groups. When one group carries much of the total weight, the
    sandwich is estimating that group's variance from what is effectively a
    single residual, and no residual adjustment fully rescues it. Weighting the
    estimates with :func:`~pymare.stats.group_weights` levels the weight
    across groups and is far more effective than any choice of ``method``.

    For the same reason, a comfortable group count is *not* evidence that the
    small-sample corrections are unnecessary. :footcite:t:`pustejovsky2018small`
    put it directly: because the degrees of freedom are covariate-dependent, it
    is not possible to decide whether a correction is needed from the number of
    groups alone. :footcite:t:`imbens2016robust` find these corrections still
    improve coverage materially at fifty or more groups, and both they and
    :footcite:t:`tipton2015small` recommend using ``method="CR2"`` with the
    Satterthwaite degrees of freedom routinely rather than only when ``m`` looks
    small. Read :func:`~pymare.stats.satterthwaite_dof` before the p-value, not
    the group count.

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
            "consider pymare.stats.group_weights.",
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


def q_profile(y, v, X, alpha=0.05):
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

    References
    ----------
    .. footbibliography::
    """
    k, p = X.shape
    df = k - p
    l_crit = ss.chi2.ppf(1 - alpha / 2, df)
    u_crit = ss.chi2.ppf(alpha / 2, df)
    args = (ensure_2d(y), ensure_2d(v), X)
    bds = Bounds([0], [np.inf], keep_feasible=True)

    # Use the D-L estimate of tau^2 as a starting point; when using a fixed
    # value, minimize() sometimes fails to stay in bounds.
    from .estimators import DerSimonianLaird

    ub_start = 2 * DerSimonianLaird().fit(y, v, X).params_["tau2"]

    lb = minimize(lambda x: (q_gen(*args, x) - l_crit) ** 2, [0], bounds=bds).x[0]
    ub = minimize(lambda x: (q_gen(*args, x) - u_crit) ** 2, ub_start, bounds=bds).x[0]
    return {"ci_l": lb, "ci_u": ub}


def q_gen(y, v, X, tau2):
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

    beta = weighted_least_squares(y, v, X, tau2)
    w = 1.0 / (v + tau2)
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
