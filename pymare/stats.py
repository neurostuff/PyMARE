"""Miscellaneous statistical functions."""

import warnings

import numpy as np
import scipy.stats as ss
from scipy.optimize import Bounds, minimize

# At or below this many clusters, robust variance estimation is known to be
# anti-conservative; see Hedges, Tipton & Johnson (2010) and Tipton (2015).
MIN_CLUSTERS_FOR_RVE = 10

# A cluster whose leverage approaches one leaves no residual to learn from, so
# the CR2 adjustment diverges. Floor it rather than emit infinities.
_MIN_LEVERAGE_COMPLEMENT = 1e-10

# Assumed correlation between estimates within a cluster, used only to collapse
# clusters before estimating tau^2. Results are very weakly sensitive to it; 0.8
# is the conventional choice for correlated effects.
DEFAULT_CLUSTER_RHO = 0.8


def cluster_weights(v, groups, tau2=0.0):
    r"""Rescale inverse-variance weights so replication does not buy influence.

    The ordinary weight :math:`1 / (v_i + \tau^2)` gives a cluster that
    contributed :math:`n_j` estimates :math:`n_j` times the pull of a cluster
    that contributed one, because the weights of its members are summed. This
    divides each weight by its cluster size,

    .. math::
        w_i = \frac{1}{n_j (v_i + \tau^2)},

    so a cluster's total weight is the *mean* of its members' weights rather
    than their sum. Duplicated estimates therefore no longer buy influence,
    while genuinely more precise clusters still count more. This is the
    "correlated effects" weighting of :footcite:t:`hedges2010robust`, as
    implemented in the R package `robumeta
    <https://cran.r-project.org/package=robumeta>`_.

    Parameters
    ----------
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Cluster labels, one per estimate. Any hashable labels are accepted.
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
    groups = np.asarray(groups).ravel()
    if groups.shape[0] != v.shape[0]:
        raise ValueError(
            f"groups must have one label per estimate: expected {v.shape[0]}, "
            f"got {groups.shape[0]}."
        )

    _, group_codes = np.unique(groups, return_inverse=True)
    sizes = np.bincount(np.ravel(group_codes))[np.ravel(group_codes)]
    return 1.0 / ((v + tau2) * sizes[:, None])


def collapse_clusters(y, v, X, groups, rho=DEFAULT_CLUSTER_RHO):
    r"""Aggregate each cluster to a single effect estimate.

    Moment-based estimators of :math:`\tau^2` count every row of ``y`` as an
    independent study. When a cluster contributes several rows, that
    pseudo-replication biases :math:`\tau^2` downward: the duplicated rows agree
    with each other by construction, so the observed dispersion looks smaller
    than the number of "studies" would imply. Estimating :math:`\tau^2` from one
    effect per cluster removes the problem.

    Each cluster is collapsed to the unweighted mean of its members, whose
    sampling variance is

    .. math::
        \bar{v}_j = \frac{1}{n_j^2}
            \left( \sum_i v_i + \rho \sum_{i \neq k} \sqrt{v_i v_k} \right),

    i.e. the variance of a mean whose terms are correlated at :math:`\rho`.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        2d array of estimates (studies x parallel datasets).
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Cluster labels, one per estimate.
    rho : :obj:`float`, optional
        Assumed correlation between estimates within a cluster, in [0, 1].
        Default = 0.8, matching the default of the R package `robumeta
        <https://cran.r-project.org/package=robumeta>`_, whose correlated
        effects model this follows.

    Returns
    -------
    :obj:`tuple`
        ``(y, v, X)`` collapsed to one row per cluster.

    Notes
    -----
    The resulting :math:`\tau^2` is only weakly sensitive to ``rho``: sweeping it
    across its whole range typically moves downstream error rates by well under
    a percentage point, because it enters only through the relative weighting of
    already-aggregated clusters. Assuming a single value is therefore reasonable
    even when the true within-cluster correlation varies between clusters.

    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1]; got {rho}.")

    groups = np.asarray(groups).ravel()
    _, group_codes = np.unique(groups, return_inverse=True)
    group_codes = np.ravel(group_codes)
    n_groups = int(group_codes.max()) + 1

    collapsed_y = np.empty((n_groups, y.shape[1]))
    collapsed_v = np.empty((n_groups, v.shape[1]))
    collapsed_X = np.empty((n_groups, X.shape[1]))

    for group in range(n_groups):
        members = np.flatnonzero(group_codes == group)
        size = members.size
        collapsed_y[group] = y[members].mean(axis=0)
        collapsed_X[group] = X[members].mean(axis=0)

        member_v = v[members]
        if size == 1:
            collapsed_v[group] = member_v[0]
            continue

        sd = np.sqrt(member_v)
        cross = sd.sum(axis=0) ** 2 - member_v.sum(axis=0)
        collapsed_v[group] = (member_v.sum(axis=0) + rho * cross) / size**2

    return collapsed_y, collapsed_v, collapsed_X


def collapse_clusters_by_n(y, n, X, groups, rho=DEFAULT_CLUSTER_RHO):
    r"""Aggregate each cluster to one effect with an effective sample size.

    The counterpart to :func:`~pymare.stats.collapse_clusters` for models
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
        2d array of estimates (studies x parallel datasets).
    n : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sample sizes.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Cluster labels, one per estimate.
    rho : :obj:`float`, optional
        Assumed correlation between estimates within a cluster, in [0, 1].
        Default = 0.8.

    Returns
    -------
    :obj:`tuple`
        ``(y, n, X)`` collapsed to one row per cluster.

    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1]; got {rho}.")

    groups = np.asarray(groups).ravel()
    _, group_codes = np.unique(groups, return_inverse=True)
    group_codes = np.ravel(group_codes)
    n_groups = int(group_codes.max()) + 1

    collapsed_y = np.empty((n_groups, y.shape[1]))
    collapsed_n = np.empty((n_groups, n.shape[1]))
    collapsed_X = np.empty((n_groups, X.shape[1]))

    for group in range(n_groups):
        members = np.flatnonzero(group_codes == group)
        size = members.size
        collapsed_y[group] = y[members].mean(axis=0)
        collapsed_X[group] = X[members].mean(axis=0)

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
    across studies is common signal, and it inflates every pairwise
    correlation, including pairs from unrelated studies that are independent by
    construction.

    Removing the across-dataset mean at each column first strips that shared
    signal, leaving residuals whose correlation estimates the dependence.
    Centering K values induces a spurious :math:`-1/(K-1)` correlation among
    independent rows, which ``bias_correct`` rescales away.

    Parameters
    ----------
    y : :obj:`numpy.ndarray` of shape (K, D)
        2d array of estimates (studies x parallel datasets). At least two
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
        2-d array of estimates (studies x parallel datasets)
    v : :obj:`numpy.ndarray`
        2-d array of sampling variances
    X : :obj:`numpy.ndarray`
        Fixed effect design matrix
    tau2 : :obj:`float`, optional
        tau^2 estimate to use for weights.
        Default = 0.
    return_cov : :obj:`bool`, optional
        Whether or not to return the inverse cov matrix.
        Default = False.
    w : None or :obj:`numpy.ndarray`, optional
        Precomputed weights of the same shape as ``y``, overriding the default
        ``1 / (v + tau2)``. Use :func:`~pymare.stats.cluster_weights` to obtain
        weights that do not reward replication within a cluster.
        Default = None.

    Returns
    -------
    params[, cov]
        If return_cov is True, returns both fixed parameter estimates and the
        inverse covariance matrix; if False, only the parameter estimates.
    """
    w = 1.0 / (v + tau2) if w is None else w

    # Einsum indices: k = studies, p = predictors, i = parallel iterates
    wX = np.einsum("kp,ki->ipk", X, w)
    cov = wX.dot(X)

    # numpy >= 1.8 inverts stacked matrices along the first N - 2 dims, so we
    # can vectorize computation along the second dimension (parallel datasets)
    precision = np.linalg.pinv(cov).T

    pwX = np.einsum("ipk,qpi->iqk", wX, precision)
    beta = np.einsum("ipk,ik->ip", pwX, y.T).T

    return (beta, precision) if return_cov else beta


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
    i.e. HC2.

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

    scores = np.empty((n_iters, n_preds, n_groups))
    for group, members in enumerate(group_members):
        # (i, n_j, p) and (i, n_j)
        group_X = np.transpose(whitened_X[members], (1, 0, 2))
        group_resid = whitened_resid[members].T

        if members.size == 1:
            leverage = np.einsum("iap,ipq,iaq->ia", group_X, bread, group_X)
            group_resid = group_resid / np.sqrt(
                np.maximum(1.0 - leverage, _MIN_LEVERAGE_COMPLEMENT)
            )
        else:
            identity = np.eye(members.size)
            adjustment = identity - group_X @ bread @ np.transpose(group_X, (0, 2, 1))
            # (I - H_j) is symmetric positive semi-definite, so its inverse
            # square root follows from an eigendecomposition.
            evals, evecs = np.linalg.eigh(adjustment)
            evals = np.maximum(evals, _MIN_LEVERAGE_COMPLEMENT)
            inv_sqrt = (evecs * (evals**-0.5)[:, None, :]) @ np.transpose(evecs, (0, 2, 1))
            group_resid = np.einsum("iab,ib->ia", inv_sqrt, group_resid)

        scores[:, :, group] = np.einsum("iap,ia->ip", group_X, group_resid)

    return scores


def cluster_robust_cov(
    y,
    v,
    X,
    beta,
    groups,
    tau2=0.0,
    small_sample=True,
    inv_cov=None,
    w=None,
    method="CR2",
):
    r"""Compute a cluster-robust ("sandwich") covariance matrix for 2-D WLS.

    Implements robust variance estimation (RVE) for meta-regression with
    dependent effect size estimates :footcite:p:`hedges2010robust`. Estimates
    sharing a group label are treated as statistically dependent, e.g.
    multiple contrasts contributed by the same study.

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
        2d array of estimates (studies x parallel datasets).
    v : :obj:`numpy.ndarray` of shape (K, D)
        2d array of sampling variances.
    X : :obj:`numpy.ndarray` of shape (K, P)
        Fixed effect design matrix.
    beta : :obj:`numpy.ndarray` of shape (P, D)
        Fixed effect coefficients, as returned by
        :func:`~pymare.stats.weighted_least_squares`.
    groups : :obj:`numpy.ndarray` of shape (K,) or (K, 1)
        Group (cluster) labels, one per study. Any hashable labels are
        accepted.
    tau2 : :obj:`float` or :obj:`numpy.ndarray`, optional
        tau^2 estimate used for the weights, matching the value used to
        obtain ``beta``.
        Default = 0.
    small_sample : :obj:`bool`, optional
        Whether to apply the ``m / (m - p)`` small-sample scaling, where m is
        the number of groups and p the number of predictors
        :footcite:p:`tipton2015small`. Only used by ``method="CR0"``; CR2
        corrects for leverage directly, so the two would double-count.
        Default = True.
    inv_cov : None or :obj:`numpy.ndarray` of shape (P, P, D), optional
        The model-based inverse covariance ``(X'WX)^-1``, as returned by
        :func:`~pymare.stats.weighted_least_squares` with ``return_cov=True``.
        Supplying it avoids recomputing a pseudo-inverse that the caller
        already has. It must correspond to the same ``tau2``.
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
        inverse covariance returned by
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
    estimates with :func:`~pymare.stats.cluster_weights` levels the weight
    across groups and is far more effective than any choice of ``method``.

    References
    ----------
    .. footbibliography::

    """
    groups = np.asarray(groups).ravel()
    if groups.shape[0] != y.shape[0]:
        raise ValueError(
            f"groups must have one label per study: expected {y.shape[0]} "
            f"labels, got {groups.shape[0]}."
        )

    if method not in ("CR0", "CR2"):
        raise ValueError(f"Invalid method '{method}'; must be one of 'CR2', 'CR0'.")

    _, group_codes = np.unique(groups, return_inverse=True)
    group_codes = np.ravel(group_codes)
    n_groups = int(group_codes.max()) + 1
    n_preds = X.shape[1]

    if small_sample and method == "CR0" and n_groups <= n_preds:
        raise ValueError(
            f"Cannot apply the small-sample correction with {n_groups} "
            f"group(s) and {n_preds} predictor(s); the number of groups "
            "must exceed the number of predictors. Pass "
            "small_sample=False to skip the correction."
        )

    if n_groups <= MIN_CLUSTERS_FOR_RVE:
        warnings.warn(
            f"Cluster-robust variance estimation with only {n_groups} groups. "
            "RVE is asymptotic in the number of groups and is known to be "
            f"anti-conservative at or below about {MIN_CLUSTERS_FOR_RVE}; the "
            "residual adjustment only partly compensates, so p-values may "
            "still be too small. If weight is spread unevenly across groups, "
            "consider pymare.stats.cluster_weights.",
            UserWarning,
            stacklevel=2,
        )

    w = 1.0 / (v + tau2) if w is None else w

    # Einsum indices: k = studies, p/q = predictors, i = parallel iterates,
    # j = groups.
    wX = np.einsum("kp,ki->ipk", X, w)

    # (X'WX)^-1 is exactly the model-based covariance, so reuse it when the
    # caller already has it rather than repeating the pinv.
    if inv_cov is None:
        bread = np.linalg.pinv(wX.dot(X))  # (i, p, p)
    else:
        bread = np.asarray(inv_cov).T  # (p, p, i) -> (i, p, p)

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
            # formed, so reuse it for the per-study scores rather than
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

    # Match the (p, p, i) orientation used for inv_cov elsewhere.
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
        1d array of study-level estimates
    v : :obj:`numpy.ndarray` of shape (K,)
        1d array of study-level variances
    X : :obj:`numpy.ndarray` of shape (K[, P])
        1d or 2d array containing study-level predictors
        (including intercept); has dimensions K x P, where K is the number
        of studies and P is the number of predictor variables.
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
        1d array of study-level estimates
    v : :obj:`numpy.ndarray`
        1d array of study-level variances
    X : :obj:`numpy.ndarray`
        1d or 2d array containing study-level predictors
        (including intercept); has dimensions K x P, where K is the number
        of studies and P is the number of predictor variables.
    tau2 : :obj:`float`
        Between-study variance. Must be >= 0.

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
