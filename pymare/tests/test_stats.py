"""Tests for pymare.stats."""

import warnings

import numpy as np
import pytest

from pymare import stats
from pymare.estimators import DerSimonianLaird
from pymare.estimators.estimators import _dersimonian_laird_tau2
from pymare.stats import (
    _SCAN_FRACTIONS,
    MIN_DOF_FOR_SATTERTHWAITE,
    _cr2_low_rank_apply,
    _cr2_low_rank_factors,
    _symmetric_sqrt,
    bounded_scalar_min,
    cluster_robust_cov,
    collapse_groups,
    collapse_groups_by_n,
    correlated_effects_tau2,
    correlated_effects_weights,
    encode_groups,
    estimate_null_correlation,
    group_mean,
    normalize_group_weights,
    one_sample_t_from_sufficient_statistics,
    satterthwaite_dof,
    undo_centering_shrinkage,
    weighted_intercept_cr2,
    weighted_intercept_cr2_sufficient_statistics,
    weighted_least_squares,
)


def test_bounded_scalar_min_finds_each_datasets_own_minimum():
    """Every parallel dataset gets its own optimum out of the one shared search."""
    targets = np.array([0.0, 0.25, 0.5, 1.0, 1e-7, 0.999])
    x, fval = bounded_scalar_min(
        lambda x: (x - targets) ** 2, np.zeros(targets.size), np.ones(targets.size)
    )

    assert np.allclose(x, targets, atol=1e-8)
    assert np.allclose(fval, 0.0, atol=1e-14)


def test_bounded_scalar_min_returns_optima_on_the_bounds_exactly():
    """A monotone objective is minimized at an end of the interval, not just near it.

    The refinement narrows a bracket from the inside, so the end itself is only
    ever reached in the limit. It comes back exactly because the coarse scan
    evaluated it and the better of the two is returned.
    """
    x, _ = bounded_scalar_min(lambda x: np.array([x[0], -x[1]]), np.zeros(2), np.array([1.0, 4.0]))

    assert np.array_equal(x, [0.0, 4.0])


def test_bounded_scalar_min_finds_a_minimum_off_the_linear_scan():
    """A dip between two scan points is found by refining the bracket around it."""
    x, _ = bounded_scalar_min(lambda x: np.abs(x - 0.3141592) ** 0.5, np.zeros(1), np.ones(1))

    assert np.allclose(x, 0.3141592, atol=1e-6)


def test_bounded_scalar_min_does_not_refine_a_constant_objective():
    """There is nothing to locate, so the refinement stops before its first step."""
    calls = []

    def objective(x):
        """Score a candidate, and record that it was scored."""
        calls.append(1)
        return np.zeros_like(x)

    x, fval = bounded_scalar_min(objective, np.zeros(1), np.ones(1))

    assert np.allclose(fval, 0.0)
    assert np.isfinite(x).all()
    assert len(calls) == _SCAN_FRACTIONS.size


def test_bounded_scalar_min_converges_on_a_very_flat_minimum():
    """A minimum with almost no curvature must not stall the refinement.

    Parabolic interpolation fits such a minimum badly and can creep towards it by
    ever-smaller steps that never shrink the bracket. The step-size safeguard is
    what bounds the iteration count, so this is asserted under a low ``maxiter``:
    without the safeguard the search is still far away when it runs out.
    """
    x, _ = bounded_scalar_min(lambda x: np.abs(x - 0.4) ** 6, np.zeros(1), np.ones(1), maxiter=25)

    assert np.allclose(x, 0.4, atol=1e-3)


def test_bounded_scalar_min_isolates_a_degenerate_dataset():
    """A dataset whose objective is nan must not disturb the others."""
    targets = np.array([0.25, np.nan, 0.75])
    x, _ = bounded_scalar_min(lambda x: (x - targets) ** 2, np.zeros(3), np.ones(3))

    assert np.allclose(x[[0, 2]], [0.25, 0.75], atol=1e-8)
    assert np.isfinite(x[1])


def test_bounded_scalar_min_refines_a_bracket_with_equal_ends():
    """Equal values at the two ends of the bracket do not make it flat.

    An objective steeper on one side of its minimum than the other can hold both
    ends at the same height while the middle point sits far below them. Reading
    that as flat stops the refinement before its first step and returns the scan
    point, which is a whole scan cell out.
    """
    # Slopes 10 and 1 either side of the minimum, placed so that the two scan
    # points bracketing 8/24 come out at exactly equal height.
    minimum = 79.0 / 264.0

    def asymmetric(x):
        """Score a candidate on a V with a steep left arm and a gentle right one."""
        return np.where(x < minimum, 10.0 * (minimum - x), x - minimum)

    ends = asymmetric(np.array([7.0 / 24.0, 9.0 / 24.0]))
    assert abs(ends[0] - ends[1]) < 1e-15
    assert asymmetric(np.array([8.0 / 24.0]))[0] < ends[0] / 2

    x, _ = bounded_scalar_min(asymmetric, np.zeros(1), np.ones(1))

    assert np.allclose(x, minimum, atol=1e-6)


def test_bounded_scalar_min_rejects_reversed_bounds():
    """A descending interval flips every ordering the refinement relies on.

    It would not announce itself: the tolerance changes sign along with the
    interval, so the first convergence test passes and a scan point comes back as
    though it had been refined.
    """
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        bounded_scalar_min(lambda x: (x - 0.3) ** 2, np.ones(1), np.zeros(1))


def test_bounded_scalar_min_accepts_a_degenerate_interval():
    """A single point is an ordered interval, and the only answer it can give."""
    x, fval = bounded_scalar_min(lambda t: (t - 0.3) ** 2, np.full(1, 0.5), np.full(1, 0.5))

    assert np.allclose(x, 0.5)
    assert np.allclose(fval, 0.04)


def test_bounded_scalar_min_requires_matching_1d_bounds():
    """Bounds carry the dataset count, so their shape is not inferred."""
    with pytest.raises(ValueError, match="1d arrays of the same shape"):
        bounded_scalar_min(lambda x: x, 0.0, 1.0)

    with pytest.raises(ValueError, match="1d arrays of the same shape"):
        bounded_scalar_min(lambda x: x, np.zeros(3), np.ones(2))


def test_weighted_least_squares_handles_a_collinear_design():
    """A rank-deficient design still gets the pseudo-inverse answer.

    The ordinary inverse the fast path uses is undefined there, and the fallback
    is what keeps a duplicated predictor from turning the covariance into
    infinities.
    """
    rng = np.random.RandomState(0)
    y = rng.standard_normal((8, 3))
    v = np.abs(rng.standard_normal((8, 3))) + 0.5
    repeated = rng.standard_normal((8, 1))
    X = np.c_[np.ones(8), repeated, repeated]

    beta, cov = weighted_least_squares(y, v, X, return_cov=True)

    assert np.isfinite(beta).all()
    assert np.isfinite(cov).all()
    # The minimum-norm solution splits a duplicated predictor's coefficient
    # evenly between its two copies, which an ordinary inverse cannot do.
    assert np.allclose(beta[1], beta[2])


def test_q_gen(vars_with_intercept):
    """Test pymare.stats.q_gen."""
    result = stats.q_gen(*vars_with_intercept, 8)
    assert round(result[0], 4) == 8.0161


def test_q_profile(vars_with_intercept):
    """Test pymare.stats.q_profile."""
    bounds = stats.q_profile(*vars_with_intercept, 0.05)
    assert set(bounds.keys()) == {"ci_l", "ci_u"}
    assert round(bounds["ci_l"], 4) == 3.8076
    assert round(bounds["ci_u"], 2) == 59.61


def test_var_to_ci():
    """Test pymare.stats.var_to_ci.

    This is basically a smoke test. We should improve it.
    """
    ci = stats.var_to_ci(0.05, 0.5, n=20)
    assert round(ci[0], 4) == -0.2599
    assert round(ci[1], 4) == 0.3599


# -----------------------------------------------------------------------------
# Generic group primitives
# -----------------------------------------------------------------------------


def test_encode_groups_preserves_first_occurrence_order():
    """Arbitrary labels should map stably without imposing sorted order."""
    codes, labels = encode_groups(["later", "later", "first", "last", "first"])

    assert np.array_equal(codes, [0, 0, 1, 2, 1])
    assert np.array_equal(labels, ["later", "first", "last"])


def test_group_mean_uses_ten_distinct_observations_from_one_group():
    """A large group may contain genuinely different observations, not duplicates."""
    first_group = np.arange(30, dtype=float).reshape(10, 3)
    values = np.vstack([first_group, [[100.0, 200.0, 300.0]]])
    groups = np.array(["many"] * 10 + ["single"])

    means = group_mean(values, groups)

    assert np.allclose(means[0], first_group.mean(axis=0))
    assert np.allclose(means[1], [100.0, 200.0, 300.0])


def test_normalize_group_weights_divides_by_observation_count():
    """Normalization should preserve row-specific weights while removing multiplicity."""
    weights = np.array([2.0, 4.0, 6.0, 10.0])
    groups = np.array([0, 0, 0, 1])

    normalized = normalize_group_weights(weights, groups)

    assert np.allclose(normalized, [2 / 3, 4 / 3, 2, 10])


def test_weighted_intercept_cr2_matches_explicit_signed_formula():
    """The generic vectorized CR2 primitive must recompute residuals after signs."""
    values = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [4.0, -1.0]])
    weights = np.array([20.0, 40.0, 80.0, 25.0])
    signs = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]])
    sufficient_statistics = weighted_intercept_cr2_sufficient_statistics(values, weights)

    observed = weighted_intercept_cr2(signs, sufficient_statistics)

    expected = []
    total_weight = weights.sum()
    leverage = weights / total_weight
    for sign in signs:
        signed = sign[:, None] * values
        mean = (weights[:, None] * signed).sum(axis=0) / total_weight
        residuals = signed - mean
        meat = (
            np.square(weights)[:, None] * np.square(residuals) / (1.0 - leverage)[:, None]
        ).sum(axis=0)
        expected.append(mean / (np.sqrt(meat) / total_weight))

    assert np.allclose(observed, expected)

    # Only one -1/+1 sign per independent input is accepted.
    with pytest.raises(ValueError, match="-1 and 1"):
        weighted_intercept_cr2(np.array([[1.0, 0.0, -1.0, 1.0]]), sufficient_statistics)


def test_one_sample_t_from_sufficient_statistics_matches_direct_formula():
    """The reusable sufficient-statistic form must equal an ordinary one-sample t."""
    values = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [4.0, -1.0]])

    observed = one_sample_t_from_sufficient_statistics(
        values.sum(axis=0),
        np.square(values).sum(axis=0),
        values.shape[0],
    )
    expected = values.mean(axis=0) / (values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]))

    assert np.allclose(observed, expected)
    batched = one_sample_t_from_sufficient_statistics(
        np.vstack([values.sum(axis=0), -values.sum(axis=0)]),
        np.square(values).sum(axis=0),
        values.shape[0],
    )
    assert np.allclose(batched, np.vstack([expected, -expected]))


@pytest.mark.parametrize(
    "collapse, second",
    [
        (lambda y, s, X, g: collapse_groups(y, s, X, g, rho=0.0), np.array([[1.0], [1.0]])),
        (lambda y, s, X, g: collapse_groups_by_n(y, s, X, g, rho=0.0), None),
        (lambda y, s, X, g: collapse_groups_by_n(y, s, X, g), None),
    ],
)
def test_collapse_keeps_effects_paired_with_their_own_group(collapse, second):
    """Effects and their variances must not be ordered by different rules.

    group_mean encodes labels by first occurrence. Indexing the second array by
    np.unique's sorted-label codes instead silently pairs each collapsed effect
    with a different group's variance whenever the labels do not happen to
    appear in sorted order.
    """
    groups = np.array([1, 1, 0, 0])  # first occurrence order != sorted order
    y = np.array([[10.0], [10.0], [100.0], [100.0]])
    X = np.ones((4, 1))
    values = np.array([[1.0], [1.0], [9.0], [9.0]])

    collapsed_y, collapsed_second, _ = collapse(y, values, X, groups)

    # Group 1 (y=10) carries the small value; group 0 (y=100) the large one.
    assert collapsed_y[0, 0] == 10.0
    assert collapsed_y[1, 0] == 100.0
    assert collapsed_second[0, 0] < collapsed_second[1, 0]


def test_collapse_groups_matches_hand_calculation():
    """The collapsed variance is that of a mean of correlated terms."""
    y = np.array([[1.0], [3.0], [10.0]])
    v = np.array([[2.0], [8.0], [5.0]])
    X = np.ones((3, 1))
    groups = np.array([0, 0, 1])
    rho = 0.5

    c_y, c_v, c_X = collapse_groups(y, v, X, groups, rho=rho)

    assert np.allclose(c_y.ravel(), [2.0, 10.0])
    # Var((y1 + y2) / 2) = (v1 + v2 + 2*rho*sqrt(v1*v2)) / 4
    expected = (2.0 + 8.0 + 2 * rho * np.sqrt(2.0 * 8.0)) / 4
    assert np.isclose(c_v.ravel()[0], expected)
    assert np.isclose(c_v.ravel()[1], 5.0)  # singletons pass through
    assert np.allclose(c_X, np.ones((2, 1)))


def test_collapse_groups_by_n_matches_hand_calculation():
    """The effective sample size reproduces the collapsed variance."""
    y = np.array([[1.0], [3.0], [10.0]])
    n = np.array([[20.0], [80.0], [50.0]])
    X = np.ones((3, 1))
    groups = np.array([0, 0, 1])
    rho, sigma2 = 0.5, 4.0

    _, c_n, _ = collapse_groups_by_n(y, n, X, groups, rho=rho)

    # sigma^2 / n_eff must equal Var of the mean of the two members.
    v = sigma2 / n[:2].ravel()
    expected_var = (v.sum() + 2 * rho * np.sqrt(v[0] * v[1])) / 4
    assert np.isclose(sigma2 / c_n.ravel()[0], expected_var)
    assert np.isclose(c_n.ravel()[1], 50.0)  # singletons pass through


def test_collapse_groups_by_n_at_rho_one_replaces_the_deleted_special_case():
    """Counting a group's ``n`` once -- the old separate function -- is just rho=1.

    That function also demanded one ``n`` per group and raised otherwise, while
    the effective-sample-size form collapses unequal within-group ``n`` instead.
    """
    y = np.array([[1.0], [3.0], [10.0]])
    X = np.ones((3, 1))
    groups = np.array([0, 0, 1])

    collapsed_y, collapsed_n, collapsed_X = collapse_groups_by_n(
        y, np.array([[20.0], [20.0], [80.0]]), X, groups, rho=1.0
    )
    assert np.allclose(collapsed_y.ravel(), [2.0, 10.0])
    assert np.allclose(collapsed_n.ravel(), [20.0, 80.0])
    assert np.allclose(collapsed_X, np.ones((2, 1)))

    _, unequal, _ = collapse_groups_by_n(y, np.array([[20.0], [80.0], [40.0]]), X, groups, rho=1.0)
    # n_eff is the harmonic-style combination, between the two inputs.
    assert 20.0 < unequal[0, 0] < 80.0
    assert np.isclose(unequal[1, 0], 40.0)


def test_collapse_groups_rejects_out_of_range_rho():
    """The assumed within-cluster correlation must lie in [0, 1]."""
    y = np.ones((2, 1))
    for collapse, second in ((collapse_groups, y), (collapse_groups_by_n, y * 10)):
        with pytest.raises(ValueError, match="rho must lie"):
            collapse(y, second, np.ones((2, 1)), np.array([0, 1]), rho=1.5)


# -----------------------------------------------------------------------------
# The correlated-effects working model
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n_preds", [1, 2, 3])
def test_correlated_effects_tau2_reduces_to_dersimonian_laird(n_preds):
    """One estimate per group is no dependence at all, so the two must agree.

    Every term of the moment estimator that carries the within-group structure
    is built from ``X_j'J_jX_j``, which coincides with ``X_j'X_j`` when a group
    has one row. The rho term then cancels and the expression collapses to
    DerSimonian-Laird -- the check that the assembled traces are the right ones.
    """
    rng = np.random.default_rng(0)
    n_estimates = 30
    y = rng.standard_normal((n_estimates, 4)) * 2.0
    v = np.abs(rng.standard_normal((n_estimates, 4))) + 0.5
    X = np.c_[np.ones(n_estimates), rng.standard_normal((n_estimates, n_preds - 1))]
    singletons = np.arange(n_estimates)

    assert np.allclose(
        correlated_effects_tau2(y, v, X, singletons, rho=0.8),
        _dersimonian_laird_tau2(y, v, X),
    )
    # With nothing to correlate, the assumed correlation cannot matter either.
    assert np.allclose(
        correlated_effects_tau2(y, v, X, singletons, rho=0.0),
        correlated_effects_tau2(y, v, X, singletons, rho=1.0),
    )


def test_correlated_effects_tau2_uses_the_observation_level_design():
    """tau^2 has to describe the model that was fitted.

    ``collapse_groups`` replaces each group's predictors with their mean, which
    is why "collapse" refuses a design that varies within a group. Estimating
    tau^2 that way under "rescale" -- where every row is kept for the
    coefficients -- returned the same tau^2 for designs whose slopes differ by
    an order of magnitude, because both share the collapsed design.
    """
    rng = np.random.default_rng(2)
    n_groups, size = 12, 2
    groups = np.repeat(np.arange(n_groups), size)
    n_estimates = groups.size
    x_between = np.repeat(rng.standard_normal(n_groups), size)
    y = (2.0 * x_between + rng.standard_normal(n_estimates) * 2.0)[:, None]
    v = np.full((n_estimates, 1), 0.2)
    # Same group means, but a large within-group spread the fit does use.
    x_within = x_between + np.tile([-2.0, 2.0], n_groups)

    fits = [
        DerSimonianLaird(weight_scheme="rescale").fit(
            y=y, v=v, X=np.c_[np.ones(n_estimates), x], g=groups
        )
        for x in (x_between, x_within)
    ]
    slopes = [float(np.ravel(f.params_["fe_params"])[1]) for f in fits]
    tau2s = [float(np.ravel(f.params_["tau2"])[0]) for f in fits]

    assert not np.isclose(slopes[0], slopes[1], rtol=0.5), "designs must differ to be a test"
    assert not np.isclose(tau2s[0], tau2s[1])
    assert all(f.tau2_model_ == "correlated-effects" for f in fits)


def test_correlated_effects_tau2_is_increasing_in_rho():
    """Rho enters analytically, through the term that pairs rows within a group.

    Assuming more within-group correlation means the same spread of estimates
    carries less independent information, so the heterogeneity it implies is
    larger. Aggregating the data first cannot express this; it is why the
    published estimator keeps the observation-level design.
    """
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(15), 3)
    n_estimates = groups.size
    y = np.repeat(rng.standard_normal(15), 3) * 1.5 + rng.standard_normal(n_estimates) * 0.5
    v = np.abs(rng.standard_normal((n_estimates, 1))) + 0.5
    X = np.c_[np.ones(n_estimates), rng.standard_normal(n_estimates)]

    tau2s = [
        float(correlated_effects_tau2(y[:, None], v, X, groups, rho=r)[0])
        for r in (0.0, 0.4, 0.8, 1.0)
    ]

    assert np.all(np.diff(tau2s) > 0)


def test_correlated_effects_weights_give_a_group_one_total():
    """Every row of a group carries the same weight, built from its mean variance.

    This is the weighting the published tau^2 estimator is derived under, so a
    group's weights sum to ``1 / (mean v + tau^2)`` however many rows it
    supplied.
    """
    groups = np.repeat(np.arange(4), 3)
    v = np.abs(np.random.default_rng(0).standard_normal((12, 1))) + 0.5

    w = correlated_effects_weights(v, groups, tau2=0.3)

    for group in range(4):
        members = groups == group
        assert np.allclose(w[members], w[members][0])  # constant within the group
        assert np.isclose(w[members].sum(), 1.0 / (v[members].mean() + 0.3))


def test_correlated_effects_weights_bound_a_group_influence():
    """One mis-reported variance must not hand a group the whole analysis.

    A group's weights sum to ``1 / (mean v + tau^2)``, which its other rows
    bound. Weighting each row by its own ``1 / v_i`` instead makes a group's
    total the mean of inverse variances, which is unbounded in a single row: a
    variance reported a thousand times too small took 79% of the weight in the
    analysis, and 99.7% at ten thousand times, with standard errors that stayed
    perfectly calibrated for the estimator it had silently become.
    """
    groups = np.repeat(np.arange(10), 3)
    v = np.full((30, 1), 0.1)

    shares = []
    for mistake in (0.1, 0.001, 1e-5):
        v[0] = mistake
        w = correlated_effects_weights(v, groups)
        shares.append(w[:3].sum() / w.sum())

    assert np.isclose(shares[0], 0.1)  # ten groups, no mistake, one tenth each
    assert max(shares) < 0.2  # and the mistake cannot run away with it


def test_correlated_effects_tau2_validates_its_inputs():
    """The moment estimator needs more groups than predictors, and rho in [0, 1]."""
    y = np.ones((6, 1))
    v = np.full((6, 1), 0.2)
    X = np.c_[np.ones(6), np.repeat([0.0, 1.0], 3)]

    with pytest.raises(ValueError, match="more groups than predictors"):
        correlated_effects_tau2(y, v, X, np.repeat([0, 1], 3))

    with pytest.raises(ValueError, match="rho must lie"):
        correlated_effects_tau2(y, v, X, np.arange(6), rho=1.5)


# -----------------------------------------------------------------------------
# cluster_robust_cov
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "groups",
    [
        np.repeat(np.arange(4), 3),
        np.repeat([3, 1, 4, 2], [2, 4, 3, 3]),
        np.array([0, 1, 2, 3] * 3),
        np.arange(12),
    ],
    ids=["equal-contiguous", "unequal-contiguous", "noncontiguous", "singleton"],
)
@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_cluster_robust_cov_matches_explicit_reference(groups, explicit_cluster_robust_cov):
    """Every grouping path must reproduce a plain per-dataset sandwich loop.

    With singleton groups that reference is the ordinary HC0 estimator, so the
    same comparison also pins the heteroskedasticity-robust limiting case.
    """
    rng = np.random.RandomState(3)
    n_estimates, n_datasets, n_preds = 12, 5, 3
    y = rng.randn(n_estimates, n_datasets)
    v = np.abs(rng.randn(n_estimates, n_datasets)) + 0.5
    X = np.c_[np.ones(n_estimates), rng.randn(n_estimates, n_preds - 1)]
    beta = weighted_least_squares(y, v, X)

    robust = cluster_robust_cov(y, v, X, beta, groups, small_sample=False, method="CR0")

    assert robust.shape == (n_preds, n_preds, n_datasets)
    assert np.allclose(robust, explicit_cluster_robust_cov(y, v, X, beta, groups))


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("method", ["CR0", "CR2"])
def test_cluster_robust_cov_shared_weights_take_the_fast_path(method):
    """One column of v serves every dataset, and deduplicating it changes nothing.

    (I - H_j) depends on the weights and the design but never on y, so
    identical weight columns give identical eigendecompositions and only the
    first needs solving.

    Checked against a per-column reference rather than bitwise: numpy's own
    reductions are not bit-stable across batch shapes, so a ``(p, p, d)``
    contraction and ``d`` separate ``(p, p, 1)`` ones disagree in the last
    place regardless of this optimization -- CR0, which never touches the
    deduplicated code, disagrees by the same ~1e-17. Bitwise stability of the
    fast path at a *fixed* shape is asserted at the end.
    """
    rng = np.random.RandomState(3)
    n_estimates, n_datasets = 16, 6
    y = rng.randn(n_estimates, n_datasets)
    v = np.abs(rng.randn(n_estimates, 1)) + 0.5
    X = np.c_[np.ones(n_estimates), rng.randn(n_estimates)]
    groups = np.repeat(np.arange(8), 2)

    beta, model_cov = weighted_least_squares(y, v, X, return_cov=True)
    kwargs = dict(model_cov=model_cov, method=method, small_sample=False)
    fast = cluster_robust_cov(y, v, X, beta, groups, **kwargs)
    assert fast.shape == (X.shape[1], X.shape[1], n_datasets)

    # The same weights, repeated so they are no longer detected as identical:
    # the numbers are unchanged but the deduplication no longer applies.
    per_column = np.repeat(v, n_datasets, axis=1)
    assert np.allclose(fast, cluster_robust_cov(y, per_column, X, beta, groups, **kwargs))

    # Solving one dataset at a time leaves the shared path unused entirely.
    slow = np.stack(
        [
            cluster_robust_cov(
                y[:, [i]],
                per_column[:, [i]],
                X,
                beta[:, [i]],
                groups,
                model_cov=model_cov[:, :, [0]],
                method=method,
                small_sample=False,
            )[:, :, 0]
            for i in range(n_datasets)
        ],
        axis=-1,
    )
    assert np.allclose(fast, slow, rtol=0, atol=1e-15)

    # At a fixed shape the fast path is exactly reproducible run to run.
    assert np.array_equal(fast, cluster_robust_cov(y, v, X, beta, groups, **kwargs))


def test_cluster_robust_cov_small_sample_scaling(dependent_data):
    """The small-sample correction should scale the matrix by m / (m - p)."""
    y, v, X, groups = dependent_data(np.random.RandomState(0))
    beta = weighted_least_squares(y, v, X)

    uncorrected = cluster_robust_cov(y, v, X, beta, groups, small_sample=False, method="CR0")
    corrected = cluster_robust_cov(y, v, X, beta, groups, small_sample=True, method="CR0")

    n_groups = np.unique(groups).size
    assert np.allclose(corrected, uncorrected * n_groups / (n_groups - X.shape[1]))


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("method", ["CR0", "CR2"])
@pytest.mark.parametrize("small_sample", [True, False])
@pytest.mark.parametrize("groups", [np.repeat([0, 1], 3), np.zeros(6)], ids=["m==p", "one-group"])
def test_cluster_robust_cov_rejects_saturated_group_designs(method, small_sample, groups):
    """With m <= p the sandwich is undefined and must not return a number.

    Every group then has leverage one, so its residuals are fitted away and
    the meat collapses to ~1e-24. Returning that yields a standard error of
    ~1e-12 and a p-value of ~1e-11 -- maximally significant -- from data that
    carries no information about the coefficients at all.
    """
    rng = np.random.RandomState(0)
    n_estimates = 6
    y = rng.randn(n_estimates, 1)
    v = np.ones((n_estimates, 1))
    X = np.c_[np.ones(n_estimates), rng.randn(n_estimates)]
    beta = weighted_least_squares(y, v, X)

    with pytest.raises(ValueError, match="more groups than predictors"):
        cluster_robust_cov(y, v, X, beta, groups, method=method, small_sample=small_sample)


def test_cluster_robust_cov_rejects_bad_labels_and_methods(dependent_data):
    """Group labels are one per observation, and the adjustment must be known."""
    y, v, X, groups = dependent_data(np.random.RandomState(0))
    beta = weighted_least_squares(y, v, X)

    with pytest.raises(ValueError, match="one label per observation"):
        cluster_robust_cov(y, v, X, beta, np.arange(3))

    with pytest.raises(ValueError, match="Invalid method"):
        cluster_robust_cov(y, v, X, beta, groups, method="CR9")


def test_cluster_robust_cov_accepts_unsortable_labels_and_flat_weights():
    """The docstring promises hashable labels and (K,) weights, not sortable ones."""
    rng = np.random.RandomState(0)
    y = rng.randn(12, 1)
    v = np.abs(rng.randn(12, 1)) + 0.5
    X = np.ones((12, 1))
    beta = weighted_least_squares(y, v, X)
    mixed = np.array([1, 1, "b", "b", 2, 2, "d", "d", 3, 3, "f", "f"], dtype=object)

    # RVE is anti-conservative with few groups, so every call warns about it.
    with pytest.warns(UserWarning, match="anti-conservative"):
        from_mixed = cluster_robust_cov(y, v, X, beta, mixed)
    with pytest.warns(UserWarning, match="Cluster-robust"):
        from_ints = cluster_robust_cov(y, v, X, beta, np.repeat(np.arange(6), 2))
    with pytest.warns(UserWarning, match="Cluster-robust"):
        flat_weights = cluster_robust_cov(
            y, v, X, beta, np.repeat(np.arange(6), 2), w=(1.0 / v).ravel()
        )
    with pytest.warns(UserWarning, match="Cluster-robust"):
        column_weights = cluster_robust_cov(y, v, X, beta, np.repeat(np.arange(6), 2), w=1.0 / v)

    assert np.allclose(from_mixed, from_ints)
    assert np.array_equal(flat_weights, column_weights)


def test_cr2_reduces_to_hc2_with_singleton_groups():
    """One group per estimate makes CR2 the familiar HC2 estimator.

    The expected value is computed here from the HC2 definition. It has also
    been checked against ``statsmodels.regression.linear_model.OLSResults``
    ``cov_HC2``, and the CR0 path against ``cov_type="cluster"`` with both
    corrections disabled, but statsmodels is not a test dependency so the
    reference is reproduced explicitly.
    """
    rng = np.random.default_rng(11)
    n_estimates, n_preds = 30, 3
    X = np.c_[np.ones(n_estimates), rng.standard_normal((n_estimates, n_preds - 1))]
    y = (X @ np.array([1.0, 0.5, -0.3]))[:, None] + rng.standard_normal((n_estimates, 1))
    v = np.ones((n_estimates, 1))

    beta, model_cov = weighted_least_squares(y, v, X, 0.0, return_cov=True)
    robust = cluster_robust_cov(
        y,
        v,
        X,
        beta,
        np.arange(n_estimates),
        model_cov=model_cov,
        method="CR2",
        small_sample=False,
    )

    # HC2: the sandwich with each residual inflated by 1 / sqrt(1 - h_ii).
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    resid = (y - X @ beta).ravel() / np.sqrt(1.0 - np.diag(hat))
    meat = X.T @ np.diag(resid**2) @ X
    expected = np.linalg.inv(X.T @ X) @ meat @ np.linalg.inv(X.T @ X)

    assert np.allclose(robust[:, :, 0], expected)


def test_cr2_is_larger_than_cr0_under_leverage(dependent_data):
    """Undoing residual shrinkage can only widen the sandwich."""
    y, v, X, groups = dependent_data(np.random.default_rng(5), n_groups=6, n_per_group=4)

    beta, model_cov = weighted_least_squares(y, v, X, 0.0, return_cov=True)
    kwargs = dict(model_cov=model_cov, small_sample=False)
    cr0 = cluster_robust_cov(y, v, X, beta, groups, method="CR0", **kwargs)
    cr2 = cluster_robust_cov(y, v, X, beta, groups, method="CR2", **kwargs)

    assert np.all(np.diagonal(cr2, axis1=0, axis2=1) >= np.diagonal(cr0, axis1=0, axis2=1))


@pytest.mark.parametrize("n_rows,n_preds", [(2, 1), (3, 1), (5, 1), (4, 2), (6, 2), (8, 3)])
def test_cr2_low_rank_factorization_matches_the_full_eigendecomposition(n_rows, n_preds):
    """The p x p route must reproduce the n x n adjustment it replaces.

    H_j is an outer product of an (n_j, p) matrix, so it has at most p
    non-unit eigenvalues and the remaining n_j - p directions need no
    adjustment at all. Decomposing the Gram matrix recovers the same operator
    for O(n_j p^2) instead of O(n_j^3).
    """
    rng = np.random.RandomState(n_rows * 10 + n_preds)
    group_X = rng.randn(n_rows, n_preds)
    bread = np.eye(n_preds) / (10.0 * n_rows)  # keeps leverage well below one

    hat = group_X @ bread @ group_X.T
    evals, evecs = np.linalg.eigh(np.eye(n_rows) - hat)
    expected = (evecs * np.maximum(evals, 1e-10) ** -0.5) @ evecs.T

    b_factor, middle, degenerate = _cr2_low_rank_factors(group_X, _symmetric_sqrt(bread))
    observed = _cr2_low_rank_apply(b_factor, middle, np.eye(n_rows))

    assert not degenerate
    assert np.allclose(observed, expected, rtol=0, atol=1e-12)


def test_cr2_low_rank_handles_rank_deficient_and_degenerate_groups():
    """Duplicated rows and full leverage are the two ways the rank drops."""
    # Three identical rows: H_j has rank 1, not 3.
    duplicated = np.ones((3, 1))
    bread = np.array([[0.2]])
    hat = duplicated @ bread @ duplicated.T
    evals, evecs = np.linalg.eigh(np.eye(3) - hat)
    expected = (evecs * np.maximum(evals, 1e-10) ** -0.5) @ evecs.T
    b_factor, middle, degenerate = _cr2_low_rank_factors(duplicated, _symmetric_sqrt(bread))
    assert not degenerate
    assert np.allclose(_cr2_low_rank_apply(b_factor, middle, np.eye(3)), expected, atol=1e-12)

    # Leverage exactly one: the adjustment does not exist and must be flagged.
    _, _, degenerate = _cr2_low_rank_factors(np.ones((2, 1)), _symmetric_sqrt(np.array([[0.5]])))
    assert degenerate

    # A design contributing nothing leaves the identity, with no 0/0.
    b_factor, middle, _ = _cr2_low_rank_factors(np.zeros((4, 2)), _symmetric_sqrt(np.eye(2)))
    identity = _cr2_low_rank_apply(b_factor, middle, np.eye(4))
    assert np.isfinite(identity).all()
    assert np.allclose(identity, np.eye(4))


# -----------------------------------------------------------------------------
# Satterthwaite degrees of freedom
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_satterthwaite_dof_collapses_for_unbalanced_group_level_predictors(group_level_design):
    """The whole point of Satterthwaite dof: m - p is blind to imbalance.

    When only a few groups carry the non-zero values of a predictor, that
    coefficient is informed by far fewer than m groups. m - p does not notice,
    which is what makes the naive reference reject far too often.
    """
    slopes = [satterthwaite_dof(*group_level_design(n))[1, 0] for n in (10, 5, 3, 2)]

    # m - p would report 18 for every one of these designs.
    assert np.isclose(slopes[0], 20 - 2, rtol=0.05)
    assert np.all(np.diff(slopes) < 0)
    assert slopes[-1] < 2.0


@pytest.mark.parametrize(
    "n_nonzero, match, floored",
    [(2, "degrees of freedom below", None), (1, "full leverage", 1.0)],
    ids=["below-validated-range", "single-group"],
)
def test_satterthwaite_dof_warns_when_a_coefficient_is_thinly_supported(
    n_nonzero, match, floored, group_level_design
):
    """A usable group count can still give dof outside the validated range.

    Tipton (2015) established the approximation only holds its level above a dof
    of roughly four. The number of groups does not reveal when that floor is
    crossed -- here 20 groups produce a dof near two for the slope -- so the
    check has to be on the dof themselves. A predictor supported by a single
    group is the extreme case, where no CR2 adjustment exists at all.
    """
    with pytest.warns(UserWarning, match=match):
        dof = satterthwaite_dof(*group_level_design(n_nonzero))

    assert dof[1, 0] < MIN_DOF_FOR_SATTERTHWAITE
    # The intercept is comfortably identified, so the warning names only the slope.
    assert dof[0, 0] > MIN_DOF_FOR_SATTERTHWAITE
    if floored is not None:
        # Floored rather than reported as a comfortable m - p = 18.
        assert dof[1, 0] == floored


def test_satterthwaite_dof_is_quiet_when_the_design_is_balanced(group_level_design):
    """The warning must not fire on a design that is genuinely well identified."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dof = satterthwaite_dof(*group_level_design(10))

    assert np.all(dof > MIN_DOF_FOR_SATTERTHWAITE)


def test_satterthwaite_dof_is_shared_when_weights_do_not_vary():
    """Identical weight columns must give identical dof, via the fast path."""
    m, size = 12, 3
    groups = np.repeat(np.arange(m), size)
    X = np.c_[np.ones(m * size), np.repeat(np.arange(m) % 3, size).astype(float)]
    w = np.ones((m * size, 1))

    one = satterthwaite_dof(X, w, groups)
    many = satterthwaite_dof(X, np.repeat(w, 7, axis=1), groups)

    assert many.shape == (2, 7)
    assert np.allclose(many, one)


# -----------------------------------------------------------------------------
# Estimating the null correlation
# -----------------------------------------------------------------------------


def test_estimate_null_correlation_ignores_shared_signal(correlated_block_data):
    """Independent estimates must not look correlated just because they agree."""
    n_estimates = 10
    y, _ = correlated_block_data(n_estimates, 4000)

    off_diagonal = ~np.eye(n_estimates, dtype=bool)
    naive = np.corrcoef(y)[off_diagonal].mean()
    estimated = estimate_null_correlation(y)[off_diagonal].mean()

    assert naive > 0.5  # the shared signal dominates the raw correlation
    assert abs(estimated) < 0.05  # the truth is zero


def test_estimate_null_correlation_recovers_within_group_dependence(correlated_block_data):
    """Genuine shared noise should still be detected, and only within the block."""
    n_estimates, block_size = 20, 4
    y, _ = correlated_block_data(n_estimates, 4000, block_size=block_size, rho=0.8, seed=1)

    corr = estimate_null_correlation(y)
    within = corr[:block_size, :block_size][~np.eye(block_size, dtype=bool)].mean()
    between = corr[block_size:, block_size:][~np.eye(n_estimates - block_size, dtype=bool)].mean()

    assert 0.6 < within < 0.9
    assert abs(between) < 0.1


def test_estimate_null_correlation_with_groups_beats_the_generic_correction(
    correlated_block_data,
):
    """With few estimates the generic rescaling badly understates dependence."""
    block_size, rho = 4, 0.8
    y, groups = correlated_block_data(8, 20000, block_size=block_size, rho=rho)

    off = ~np.eye(block_size, dtype=bool)
    generic = estimate_null_correlation(y)[:block_size, :block_size][off].mean()
    grouped = estimate_null_correlation(y, groups=groups)[:block_size, :block_size][off].mean()

    assert abs(grouped - rho) < 0.05
    assert generic < rho - 0.15

    # The de-shrinking step iterated with np.unique, so it rejected the mixed
    # but hashable labels the rest of the pipeline accepts.
    labels = np.array(["b" if g == 0 else g for g in groups], dtype=object)
    assert np.allclose(
        estimate_null_correlation(y, groups=labels), estimate_null_correlation(y, groups=groups)
    )


@pytest.mark.parametrize("n_estimates", [6, 8, 20])
@pytest.mark.parametrize("block_size", [2, 4, 5])
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.8, 1.0])
def test_undo_centering_shrinkage_is_exact(
    n_estimates, block_size, rho, block_correlation, centering_shrinkage
):
    """Centering is a known linear map, so it inverts exactly.

    Builds the true correlation matrix, applies the centering map analytically,
    and checks that the inverse recovers the correlation it started from.
    """
    corr, groups = block_correlation(n_estimates, [(block_size, rho)])

    recovered = undo_centering_shrinkage(centering_shrinkage(corr), groups)

    off_diagonal = ~np.eye(block_size, dtype=bool)
    assert np.allclose(recovered[:block_size, :block_size][off_diagonal], rho, atol=1e-8)


def test_undo_centering_shrinkage_handles_several_blocks(block_correlation, centering_shrinkage):
    """The blocks share a grand mean, which the fixed point has to resolve."""
    blocks = [(6, 0.7), (4, 0.2), (3, 0.9)]
    corr, groups = block_correlation(20, blocks)

    recovered = undo_centering_shrinkage(centering_shrinkage(corr), groups)

    start = 0
    for size, rho in blocks:
        block = recovered[start : start + size, start : start + size]
        assert np.allclose(block[~np.eye(size, dtype=bool)], rho, atol=1e-6)
        start += size
