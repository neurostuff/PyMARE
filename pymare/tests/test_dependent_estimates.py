"""Tests for dependent-estimate support (cluster-robust variance, Brown's method)."""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from pymare import Dataset, meta_regression
from pymare.estimators import (
    DerSimonianLaird,
    FisherCombinationTest,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    StoufferCombinationTest,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)
from pymare.estimators.estimators import _collapse_n_inputs, _dersimonian_laird_tau2
from pymare.results import _expand_unit_order
from pymare.stats import (
    DEFAULT_RHO,
    MIN_DOF_FOR_SATTERTHWAITE,
    _cr2_low_rank_apply,
    _cr2_low_rank_factors,
    _symmetric_sqrt,
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

# -----------------------------------------------------------------------------
# Combination tests: group aggregation
# -----------------------------------------------------------------------------


def test_stouffer_group_level_matches_two_stage_hand_calculation():
    """Groups are standardized first and then receive one external weight."""
    z = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]])
    groups = np.array([0, 0, 1])
    weights = np.sqrt(np.array([20.0, 20.0, 80.0]))[:, None]
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = 0.5

    result = StoufferCombinationTest(group_level=True).fit(z, w=weights, g=groups, corr=corr)

    # Var((z_1 + z_2) / 2) = (1 + 1 + 2 rho) / 4 = 3/4.
    group_z = np.vstack([z[:2].mean(axis=0) / np.sqrt(0.75), z[2]])
    group_weights = np.sqrt(np.array([20.0, 80.0]))
    expected_z = (group_z * group_weights[:, None]).sum(axis=0) / np.sqrt(
        np.square(group_weights).sum()
    )

    assert np.allclose(result.params_["z"], expected_z)
    assert np.allclose(result.params_["p"], ss.norm.sf(expected_z))


@pytest.mark.parametrize("mode", ["directed", "concordant"])
def test_group_weights_are_not_multiplied_by_duplicates(group_combination_estimator, mode):
    """Repeating a perfectly correlated observation cannot buy its group influence.

    One external weight per group is allocated across the group's rows, so the
    expanded fit has to reproduce the collapsed one exactly.
    """
    z = np.array([[1.5, -0.5], [0.25, 2.0]])
    w = np.array([[2.0], [4.0]])
    n_duplicates = 4
    expanded_z = np.vstack([np.repeat(z[[0]], n_duplicates, axis=0), z[[1]]])
    expanded_w = np.vstack([np.repeat(w[[0]], n_duplicates, axis=0), w[[1]]])
    corr = np.eye(n_duplicates + 1)
    corr[:n_duplicates, :n_duplicates] = 1.0

    expected = (
        group_combination_estimator(mode=mode).fit(z, w=w, g=np.arange(2), corr=np.eye(2)).params_
    )
    observed = (
        group_combination_estimator(mode=mode)
        .fit(expanded_z, w=expanded_w, g=np.r_[np.zeros(n_duplicates, dtype=int), 1], corr=corr)
        .params_
    )

    assert np.allclose(observed["z"], expected["z"])
    assert np.allclose(observed["p"], expected["p"])


def test_group_combination_permutation_flips_groups_as_units(group_combination_estimator):
    """Exact permutation inference is invariant to duplicated group rows."""
    z = np.array([[1.5], [0.25], [-0.5]])
    w = np.sqrt(np.array([[25.0], [100.0], [50.0]]))
    collapsed = (
        group_combination_estimator()
        .fit_dataset(Dataset(y=z, n=w, g=np.arange(3)), corr=np.eye(3))
        .summary()
    )

    n_duplicates = 4
    corr = np.eye(n_duplicates + 2)
    corr[:n_duplicates, :n_duplicates] = 1.0
    expanded = (
        group_combination_estimator()
        .fit_dataset(
            Dataset(
                y=np.vstack([np.repeat(z[[0]], n_duplicates, axis=0), z[1:]]),
                n=np.vstack([np.repeat(w[[0]], n_duplicates, axis=0), w[1:]]),
                g=np.array([0, 0, 0, 0, 1, 2]),
            ),
            corr=corr,
        )
        .summary()
    )
    assert np.array_equal(expanded.estimator.corr_, corr)

    collapsed_perm = collapsed.permutation_test(n_perm=20)
    expanded_perm = expanded.permutation_test(n_perm=20)
    assert collapsed_perm.exact
    assert expanded_perm.exact
    assert collapsed_perm.n_perm == expanded_perm.n_perm == 8
    assert np.allclose(collapsed_perm.perm_p["fe_p"], expanded_perm.perm_p["fe_p"])


def test_group_combination_rejects_inconsistent_group_weights(group_combination_estimator):
    """A group has one weight, not one independently varying row weight."""
    with pytest.raises(ValueError, match="one weight per group"):
        group_combination_estimator().fit(
            np.ones((3, 2)),
            w=np.array([[2.0], [3.0], [4.0]]),
            g=np.array([0, 0, 1]),
            corr=np.eye(3),
        )


def test_combination_permutation_accepts_weights(combination_estimator):
    """Weights are per observation and must be repeated across permutations.

    They live on the Dataset as ``n`` with one column per parallel dataset,
    while the permuted z array has one column per permutation, so passing the
    stored array straight through only lines up by coincidence. Fisher takes
    one scalar weight per observation, so only Stouffer sees the (K, D) case.
    """
    rng = np.random.RandomState(0)
    z = rng.randn(8, 3)
    n_weight_columns = 3 if combination_estimator is StoufferCombinationTest else 1
    w = np.abs(rng.randn(8, n_weight_columns)) + 1.0

    result = combination_estimator().fit_dataset(Dataset(y=z, n=w)).summary()
    perm = result.permutation_test(n_perm=40)

    assert perm.perm_p["fe_p"].shape == np.ravel(result.p).shape
    assert np.all(perm.perm_p["fe_p"] > 0)
    assert np.all(perm.perm_p["fe_p"] <= 1)


def test_combination_permutation_rejects_undirected_mode(combination_estimator):
    """abs(z) makes the statistic invariant to sign flips, so there is no null."""
    rng = np.random.RandomState(0)
    result = (
        combination_estimator(mode="undirected").fit_dataset(Dataset(y=rng.randn(8, 2))).summary()
    )

    with pytest.raises(ValueError, match="not available for mode='undirected'"):
        result.permutation_test(n_perm=40)


def test_combination_permutation_survives_saturated_p_values():
    """Concordant p caps at 1, so its z is -inf and cannot be compared.

    Ranking on z made every permutation tie at -inf, which read as "more
    extreme than nothing" and returned the smallest achievable p-value.
    """
    # Perfectly balanced z: neither tail wins, so both directed p-values are
    # 0.5 and the doubled minimum is capped at exactly 1.
    z = np.array([0.4, -0.4] * 5)[:, None]
    result = FisherCombinationTest(mode="concordant").fit_dataset(Dataset(y=z)).summary()

    assert np.ravel(result.p)[0] == 1.0
    assert not np.isfinite(result.z).all()  # the condition that used to break it

    # Ranking on z gave 1 / n_perm here. This data is as far from significant
    # as it gets, so the permutation p-value should sit at the other end.
    perm = result.permutation_test(n_perm=200)
    assert perm.perm_p["fe_p"].ravel()[0] == 1.0


def test_combination_permutation_holds_the_correlation_fixed():
    """Re-estimating corr from the permuted array reads the wrong axis.

    y_perm's second axis is permutations, and whole groups share a sign, so
    rows within a group come back near-perfectly correlated along it. That
    inflates the variance correction, and with n_perm < 2 it cannot be
    estimated at all.
    """
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(3), 3)
    z = rng.standard_normal((9, 4))
    results = StoufferCombinationTest().fit_dataset(Dataset(y=z, g=groups)).summary()

    # Used to raise "The number of features must be greater than 1."
    single = results.permutation_test(n_perm=1)
    assert np.all(np.isfinite(single.perm_p["fe_p"]))


def test_stouffer_inflation_uses_each_group_own_weights():
    """The pairwise loop must index the group's weights, not the array's head.

    upper_indices are positions within a group's block, so indexing the full
    weight array with them reuses rows 0..n_j-1 for every group. That both
    understates the variance inflation and makes the answer depend on the
    order the rows happen to arrive in.
    """
    z = np.array([[1.0, 0.5], [2.0, 1.5], [0.5, 2.0], [1.0, 0.0]])
    weights = np.repeat(np.array([[1.0], [1.0], [5.0], [5.0]]), 2, axis=1)
    groups = np.repeat(np.array([[0], [0], [1], [1]]), 2, axis=1)
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.6
    corr[2, 3] = corr[3, 2] = 0.6

    observed = StoufferCombinationTest().fit(z, w=weights, g=groups, corr=corr).params_["p"]

    # Reference: Var(sum w_i z_i) = w' C w.
    flat = weights[:, 0]
    expected = ss.norm.sf((z * flat[:, None]).sum(0) / np.sqrt(flat @ corr @ flat))
    assert np.allclose(observed, expected)

    # Row order must not matter, whether corr is supplied or estimated.
    order = np.array([2, 3, 0, 1])
    reordered = StoufferCombinationTest().fit(
        z[order], w=weights[order], g=groups[order], corr=corr[np.ix_(order, order)]
    )
    assert np.allclose(reordered.params_["p"], expected)

    estimated = StoufferCombinationTest().fit(z, w=weights, g=groups).params_["p"]
    reordered_estimated = StoufferCombinationTest().fit(
        z[order], w=weights[order], g=groups[order]
    )
    assert np.allclose(reordered_estimated.params_["p"], estimated)


def test_stouffer_inflation_is_computed_per_feature():
    """Weights may vary by feature, so one shared inflation term will not do.

    The diagonal term ``(w**2).sum(0)`` is already per feature. Taking the
    off-diagonal term from the first weight column alone left every other
    feature with a variance that did not match its own weights -- here the
    second feature's p-value came out about half what it should be.
    """
    z = np.array([[1.0, 0.5], [2.0, 1.5], [0.5, 2.0], [1.0, 0.0]])
    weights = np.array([[1.0, 9.0], [1.0, 9.0], [5.0, 1.0], [5.0, 1.0]])
    groups = np.repeat(np.array([[0], [0], [1], [1]]), 2, axis=1)
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.6
    corr[2, 3] = corr[3, 2] = 0.6

    observed = StoufferCombinationTest().fit(z, w=weights, g=groups, corr=corr).params_["p"]

    # Reference: Var(sum w_i z_i) = w' C w, one column of w at a time.
    expected = [
        ss.norm.sf((z[:, c] * weights[:, c]).sum() / np.sqrt(weights[:, c] @ corr @ weights[:, c]))
        for c in range(z.shape[1])
    ]
    assert np.allclose(observed, expected)


def test_fisher_does_not_underflow_on_extreme_z():
    """Going via p collapses to exactly 0 once any |z| exceeds about 38."""
    z = np.zeros((100, 2))
    z[0] = 40.0

    fitted = FisherCombinationTest().fit(z).params_

    assert np.all(fitted["p"] > 0)
    assert np.all(np.isfinite(fitted["z"]))
    assert np.allclose(fitted["p"], 1.035e-244, rtol=1e-3)


def test_constant_estimates_cannot_yield_a_correlation(combination_estimator):
    """A row that never varies has no correlation, so do not return NaN."""
    z = np.full((4, 5), 1.3)
    groups = np.array([0, 0, 1, 1])
    if combination_estimator is StoufferCombinationTest:
        groups = np.tile(groups[:, None], (1, 5))

    with pytest.raises(ValueError, match="constant across features"):
        combination_estimator().p_value(z, g=groups)


def test_combination_tests_accept_unsortable_group_labels(combination_estimator):
    """Any hashable label, as encode_groups documents and Dataset accepts.

    These estimators iterated with np.unique, which needs an ordering, so a mix
    of str and int labels raised TypeError from inside fit -- for labels the
    regression estimators handle without complaint.
    """
    z = np.random.RandomState(0).randn(6, 8)
    mixed = np.array([1, 1, "b", "b", 2, 2], dtype=object)

    from_mixed = combination_estimator().fit_dataset(Dataset(y=z, g=mixed)).summary()
    from_ints = (
        combination_estimator().fit_dataset(Dataset(y=z, g=np.repeat(np.arange(3), 2))).summary()
    )

    assert np.allclose(np.ravel(from_mixed.p), np.ravel(from_ints.p))
    assert np.allclose(
        from_mixed.permutation_test(n_perm=8).perm_p["fe_p"],
        from_ints.permutation_test(n_perm=8).perm_p["fe_p"],
    )


def test_stouffer_validates_its_inputs_like_fisher():
    """Silently accepting these produced NaN or a plausible wrong answer."""
    z = np.random.RandomState(0).randn(4, 3)

    with pytest.raises(ValueError, match="same for every feature"):
        StoufferCombinationTest().p_value(
            z, g=np.array([[0, 1, 0], [0, 0, 0], [1, 1, 1], [1, 0, 1]])
        )

    with pytest.raises(ValueError, match="finite positive"):
        StoufferCombinationTest().p_value(z, w=np.full((4, 3), -1.0))

    with pytest.raises(ValueError, match="must have shape"):
        StoufferCombinationTest().p_value(
            z, g=np.tile(np.array([0, 0, 1, 1])[:, None], (1, 3)), corr=np.eye(5)[:4, :5]
        )


# -----------------------------------------------------------------------------
# Brown's method
# -----------------------------------------------------------------------------


def test_kost_polynomial_matches_the_closed_form_covariances():
    """The empirical cubic is pinned wherever cov(-2 ln p_i, -2 ln p_j) is known.

    These three points are the justification for the coefficients, so they are
    asserted rather than left as a claim in the docstring.
    """
    kost = FisherCombinationTest._kost_covariance

    # Independence: no contribution, so Fisher's method is recovered exactly.
    assert kost(0.0) == 0.0

    # Comonotone: identical p-values, so the covariance is Var(chi^2_2) = 4.
    assert np.isclose(kost(1.0), 4.0, atol=1e-12)

    # Countermonotone: the Frechet lower bound for two chi^2_2 variates.
    assert np.isclose(kost(-1.0), 4 * (1 - np.pi**2 / 6), atol=5e-4)

    # Strictly increasing in between, so the variance correction never inverts.
    grid = np.linspace(-1.0, 1.0, 1001)
    assert np.all(np.diff(kost(grid)) > 0)


def test_brown_moments_match_the_chi2_2_definition():
    """Mean 2 and variance 4 per term come from -2 ln p ~ chi^2_2 under the null."""
    estimator = FisherCombinationTest()
    z = np.zeros((5, 3))

    expectation, variance = estimator._brown_moments(z, None)

    assert np.isclose(expectation, 2.0 * 5)
    assert np.isclose(variance, 4.0 * 5)

    # Under independence variance == 2 * expectation, so Brown's scale is 1 and
    # its degrees of freedom collapse to 2k -- exactly Fisher's reference.
    assert np.isclose(variance / (2.0 * expectation), 1.0)
    assert np.isclose(2.0 * expectation**2 / variance, 2 * 5)


@pytest.mark.parametrize(
    "group_size, corr",
    [(1, None), (2, np.eye(10))],
    ids=["singleton-groups", "independent-equal-groups"],
)
def test_brown_reduces_to_fisher_without_dependence(group_size, corr):
    """Independent groups of equal size carry no correction, whatever their size."""
    rng = np.random.RandomState(0)
    n_estimates, n_datasets = 10, 50
    z = rng.randn(n_estimates, n_datasets)
    groups = np.tile(
        np.repeat(np.arange(n_estimates // group_size), group_size)[:, None], (1, n_datasets)
    )

    plain = FisherCombinationTest().fit(z).params_["p"]
    blocked = FisherCombinationTest().fit(z, g=groups, corr=corr).params_["p"]

    assert np.allclose(plain, blocked)


def test_fisher_accepts_one_dimensional_and_positional_group_arguments():
    """Documented one-dimensional labels, and released positional calls, must both work."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 20)
    groups = np.repeat(np.arange(3), 2)

    # Either shape of label, with rho estimated across the features.
    expected = FisherCombinationTest().fit(z, g=groups[:, None]).params_["p"]
    assert np.allclose(FisherCombinationTest().fit(z, g=groups).params_["p"], expected)

    # And the generic weight must not reinterpret released positional calls.
    corr = np.eye(6)
    keyword = FisherCombinationTest().fit(z, g=groups, corr=corr).params_["p"]
    assert np.allclose(FisherCombinationTest().fit(z, groups, corr).params_["p"], keyword)


def test_brown_validates_its_groups_and_correlation():
    """Groups describe rows, and corr has one row and column per estimate."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 10)
    groups = np.repeat(np.arange(3), 2)

    feature_specific = np.tile(groups[:, None], (1, z.shape[1]))
    feature_specific[0, -1] = 99
    with pytest.raises(ValueError, match="same for every feature"):
        FisherCombinationTest().fit(z, g=feature_specific)

    # A mismatched size and a non-square matrix fail the same check.
    for corr, match in ((np.eye(4), "same length"), (np.ones((6, 2)), r"shape \(6, 6\)")):
        with pytest.raises(ValueError, match=match):
            FisherCombinationTest().fit(z, g=groups, corr=corr)

    # Without a correlation matrix, rho is estimated across features.
    with pytest.raises(ValueError, match="number of features"):
        FisherCombinationTest().fit(z[:, :1], g=groups)

    # And a correlation matrix is meaningless without group labels.
    with pytest.warns(UserWarning, match="without groups"):
        FisherCombinationTest().fit(z, corr=np.eye(6))


def test_brown_is_conservative_in_the_upper_tail():
    """Positive dependence must shrink significance where it matters.

    Brown's reference distribution keeps Fisher's mean but has a larger
    variance, so the ordering of p-values only holds in the upper tail -- which
    is the only region used for inference.
    """
    rng = np.random.RandomState(1)
    n_estimates, n_datasets = 10, 2000
    shared = rng.randn(n_estimates // 2, n_datasets)
    z = np.repeat(shared, 2, axis=0) + 0.2 * rng.randn(n_estimates, n_datasets)

    groups = np.tile(np.repeat(np.arange(n_estimates // 2), 2)[:, None], (1, n_datasets))
    corr = np.eye(n_estimates)
    for idx in range(0, n_estimates, 2):
        corr[idx, idx + 1] = corr[idx + 1, idx] = 0.95

    plain = FisherCombinationTest().fit(z).params_["p"]
    blocked = FisherCombinationTest().fit(z, g=groups, corr=corr).params_["p"]

    significant = plain < 0.05
    assert significant.sum() > 50, "test data produced too few positives"
    assert np.all(blocked[significant] > plain[significant])

    # And the false positive rate should drop substantially.
    assert (blocked < 0.05).mean() < (plain < 0.05).mean() / 1.5


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


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "estimator, scheme",
    [
        (DerSimonianLaird, "collapse"),
        (Hedges, "rescale"),
        (Hedges, "collapse"),
        (VarianceBasedLikelihoodEstimator, "rescale"),
        (VarianceBasedLikelihoodEstimator, "collapse"),
    ],
    ids=["DL-collapse", "HE-rescale", "HE-collapse", "ML-rescale", "ML-collapse"],
)
def test_aggregating_estimators_reject_a_within_group_varying_design(estimator, scheme):
    """The guard belongs to the reduction, not to one scheme's name.

    Every estimator except DerSimonian-Laird reaches tau^2 through the same
    one-row-per-group aggregate under both schemes, so the design restriction
    applies to both. It used to be checked only for "collapse", whose error
    message then recommended "rescale" -- routing users into the unguarded path.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 2)
    n_estimates = groups.size
    y = rng.standard_normal((n_estimates, 1))
    v = np.full((n_estimates, 1), 0.2)
    varying = np.c_[np.ones(n_estimates), np.tile([-1.0, 1.0], 6)]

    with pytest.raises(ValueError, match="constant within each group"):
        estimator(weight_scheme=scheme).fit(y=y, v=v, X=varying, g=groups)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_dersimonian_laird_accepts_a_within_group_varying_design_under_rescale():
    """The remedy the other estimators' error message points at has to work."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 2)
    n_estimates = groups.size
    varying = np.c_[np.ones(n_estimates), np.tile([-1.0, 1.0], 6)]

    fitted = DerSimonianLaird(weight_scheme="rescale").fit(
        y=rng.standard_normal((n_estimates, 1)),
        v=np.full((n_estimates, 1), 0.2),
        X=varying,
        g=groups,
    )

    assert fitted.tau2_model_ == "correlated-effects"
    assert np.isfinite(fitted.params_["tau2"]).all()


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "scheme, expected",
    [("individual", "independent"), ("rescale", "correlated-effects"), ("collapse", "aggregate")],
)
def test_results_statistics_follow_the_model_the_estimator_recorded(scheme, expected):
    """The reduction is declared once and read, not re-derived and hoped to match.

    Whichever way the dependence was modelled, Q has to count independent groups
    and the interval has to describe the same quantity as the estimate it
    accompanies.
    """
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(12), 4)
    n_estimates = groups.size
    y = rng.normal(size=n_estimates) + np.repeat(rng.normal(scale=0.3, size=12), 4)
    dataset = Dataset(y=y, v=np.full(n_estimates, 0.2), g=groups)

    results = DerSimonianLaird(weight_scheme=scheme).fit_dataset(dataset).summary()
    assert results.estimator.tau2_model_ == expected
    assert results._tau2_model() == expected

    _, _, _, analysis_groups = results._analysis_arrays()
    assert (analysis_groups is not None) is (expected == "correlated-effects")

    re_stats = results.get_re_stats()
    tau2 = float(np.ravel(re_stats["tau^2"])[0])
    assert float(np.ravel(re_stats["ci_l"])[0]) <= tau2 <= float(np.ravel(re_stats["ci_u"])[0])

    if scheme != "individual":
        # Q counts the 12 groups, not the 48 rows, however it got there.
        raw = DerSimonianLaird().fit_dataset(dataset).summary().get_heterogeneity_stats()
        assert float(np.ravel(results.get_heterogeneity_stats()["Q"])[0]) < float(
            np.ravel(raw["Q"])[0]
        )


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
# Estimators: group labels and weighting schemes
# -----------------------------------------------------------------------------


def test_estimators_groups_change_only_inference(grouped_estimator):
    """Group labels must move the covariance but not the point estimates.

    Omitting them must reproduce the previous behaviour exactly.
    """
    estimator, _, build_inputs = grouped_estimator
    kwargs, groups = build_inputs()

    naive = estimator().fit(**kwargs)
    robust = estimator().fit(**kwargs, g=groups)

    assert naive.n_groups_ is None
    assert "n_groups" not in naive.params_
    assert robust.n_groups_ == np.unique(groups).size
    assert np.allclose(naive.params_["fe_params"], robust.params_["fe_params"])
    assert np.allclose(naive.params_["tau2"], robust.params_["tau2"])
    assert not np.allclose(naive.params_["inv_cov"], robust.params_["inv_cov"])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_permutation_test_handles_parallel_datasets(grouped_estimator):
    """tau^2 is reported as (D,) by some estimators and (1, D) by others.

    permutation_test read ``tau2[i]`` as dataset i's value, which is a whole row
    of a two-dimensional array: with one parallel dataset it broadcast by luck,
    and with more it raised before producing any p-value.
    """
    estimator, second_arg, build_inputs = grouped_estimator
    kwargs, groups = build_inputs(n_groups=6, n_per_group=2, n_datasets=3)
    dataset = Dataset(
        y=kwargs["y"],
        X=kwargs["X"],
        g=groups,
        add_intercept=False,
        **{second_arg: kwargs[second_arg]},
    )

    perm = estimator().fit_dataset(dataset).summary().permutation_test(n_perm=16)

    assert perm.perm_p["fe_p"].shape[-1] == 3
    assert np.all(perm.perm_p["fe_p"] > 0)
    assert perm.perm_p["tau2_p"].shape == (3,)


def test_estimators_groups_widen_standard_errors(grouped_estimator):
    """With dependent estimates, robust SEs should exceed model-based ones."""
    estimator, _, build_inputs = grouped_estimator
    kwargs, groups = build_inputs(n_groups=12, n_per_group=4)

    naive = estimator().fit(**kwargs).summary()
    robust = estimator().fit(**kwargs, g=groups).summary()

    assert np.all(robust.fe_se > naive.fe_se)


def test_estimator_rejects_unknown_weight_scheme(grouped_estimator):
    """An unrecognised weighting scheme should fail at construction."""
    with pytest.raises(ValueError, match="Invalid weight_scheme"):
        grouped_estimator[0](weight_scheme="nonsense")


@pytest.mark.parametrize("rho", [-0.5, 1.5])
def test_rho_is_checked_by_the_same_mechanism_as_weight_scheme(grouped_estimator, rho):
    """Both constructor arguments are declared, so both are checked together.

    weight_scheme was validated at construction and rho was not, so an
    out-of-range correlation survived until some later call happened to read it.
    """
    with pytest.raises(ValueError, match="Invalid rho"):
        grouped_estimator[0](weight_scheme="rescale", rho=rho)


def test_rho_warns_when_the_weight_scheme_cannot_use_it(grouped_estimator):
    """Setting rho under the default scheme did nothing, and said nothing."""
    with pytest.warns(UserWarning, match="weight_scheme='individual'"):
        estimator = grouped_estimator[0](rho=0.5)

    # Still honoured as the value to use if the scheme is changed later.
    assert estimator.rho == 0.5


def test_loopable_estimators_accept_positional_arguments():
    """fit(y, v, X) is the documented call signature and must work."""
    rng = np.random.RandomState(0)
    y = rng.randn(10, 1)
    v = np.abs(rng.randn(10, 1)) + 0.5
    X = np.ones((10, 1))

    positional = VarianceBasedLikelihoodEstimator().fit(y, v, X)
    keyword = VarianceBasedLikelihoodEstimator().fit(y=y, v=v, X=X)

    assert np.allclose(positional.params_["fe_params"], keyword.params_["fe_params"])
    assert np.allclose(positional.params_["tau2"], keyword.params_["tau2"])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_estimators_accept_unsortable_group_labels(weight_scheme):
    """Any hashable label, as encode_groups documents -- not just sortable ones.

    np.unique needs an ordering, so a mix of str and int raised TypeError from
    inside the weight-scheme helpers even though the group primitives handle it.
    """
    rng = np.random.RandomState(0)
    y = rng.randn(12, 1)
    mixed = np.array([1, 1, "b", "b", 2, 2, "d", "d", 3, 3, "f", "f"], dtype=object)

    fitted = DerSimonianLaird(weight_scheme=weight_scheme).fit(
        y=y, v=np.full((12, 1), 0.2), X=np.ones((12, 1)), g=mixed
    )
    assert fitted.n_groups_ == 6

    equivalent = DerSimonianLaird(weight_scheme=weight_scheme).fit(
        y=y, v=np.full((12, 1), 0.2), X=np.ones((12, 1)), g=np.repeat(np.arange(6), 2)
    )
    assert np.allclose(fitted.params_["fe_params"], equivalent.params_["fe_params"])

    # The result statistics have to encode the labels the same way; counting
    # them with np.unique raised TypeError only once the fit was over.
    dataset = Dataset(y=y, v=np.full((12, 1), 0.2), g=mixed)
    results = DerSimonianLaird(weight_scheme=weight_scheme).fit_dataset(dataset).summary()
    assert np.isfinite(np.ravel(results.get_re_stats()["tau^2"])).all()
    assert np.all(results.permutation_test(n_perm=64).perm_p["fe_p"] > 0)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("n_perm", [100000, 2**8], ids=["more-than-exact", "exactly-exact"])
def test_permutation_flips_whole_groups_and_leaves_the_estimator_alone(weight_scheme, n_perm):
    """Dependent rows are exchangeable only as complete groups.

    Flipping them independently builds a null about half as wide as the truth,
    which showed up as ~25% rejection at a nominal 5%. Separately, refitting
    through the live estimator overwrote params_["dof"] with a
    permutation-shaped array, so fe_dof and to_df broke afterwards.

    ``n_perm == 2**m`` is the boundary: the test counts as exact there, but the
    sign patterns were only built on the strict inequality, so asking for
    exactly the exhaustive number raised UnboundLocalError.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 2)
    dataset = Dataset(y=rng.normal(size=16), v=np.full(16, 0.2), g=groups)
    results = DerSimonianLaird(weight_scheme=weight_scheme).fit_dataset(dataset).summary()

    before = np.ravel(results.fe_dof).copy()
    perm = results.permutation_test(n_perm=n_perm)

    # 8 groups, not 16 rows: 2**8 sign patterns exhaust the null.
    assert perm.n_perm == 2**8
    assert perm.exact
    assert np.allclose(np.ravel(results.fe_dof), before)
    perm.to_df()


def test_cluster_weighting_removes_the_replication_advantage():
    """A group contributing many estimates must not outvote a group with one."""
    n_singletons = 8
    groups = np.array([0] * 6 + list(range(1, n_singletons + 1)))
    n_estimates = groups.size
    y = np.zeros((n_estimates, 1))
    y[:6] = 1.0  # the big group is the only one with a non-zero effect
    v = np.ones((n_estimates, 1))
    dataset = Dataset(y=y, v=v, g=groups)

    individual = WeightedLeastSquares().fit_dataset(dataset).summary()
    clustered = WeightedLeastSquares(weight_scheme="rescale").fit_dataset(dataset).summary()

    # Six of fourteen estimates, but only one of nine groups.
    assert np.isclose(individual.get_fe_stats()["est"].ravel()[0], 6 / 14)
    assert np.isclose(clustered.get_fe_stats()["est"].ravel()[0], 1 / 9)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "estimator",
    [WeightedLeastSquares, DerSimonianLaird, Hedges, VarianceBasedLikelihoodEstimator],
    ids=["WLS", "DL", "HE", "ML"],
)
def test_variance_estimators_group_mode_matches_explicit_collapse(estimator):
    """Group mode must fit the algebraically equivalent one-row-per-group model."""
    y = np.array([[1.0], [3.0], [6.0], [10.0], [20.0], [22.0]])
    v = np.array([[1.0], [4.0], [2.0], [8.0], [5.0], [3.0]])
    X = np.ones((6, 1))
    groups = np.array([0, 0, 1, 1, 2, 2])
    rho = 0.4
    collapsed_y, collapsed_v, collapsed_X = collapse_groups(y, v, X, groups, rho=rho)

    expected = estimator().fit(y=collapsed_y, v=collapsed_v, X=collapsed_X, g=np.arange(3))
    observed = estimator(weight_scheme="collapse", rho=rho).fit(y=y, v=v, X=X, g=groups)

    assert observed.n_groups_ == 3
    for key in ("fe_params", "tau2", "inv_cov"):
        assert np.allclose(observed.params_[key], expected.params_[key])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_sample_size_likelihood_group_mode_matches_explicit_collapse():
    """The likelihood receives one effective n per independent group.

    Effective, not raw. Rows in a group share subjects, so n must not be
    counted once per row -- but they are not perfect duplicates either, and
    pinning n at the group's raw value assumes they are. The effective size
    ``s*n / (1 + rho(s-1))`` interpolates between the two.
    """
    y = np.array([[1.0], [3.0], [6.0], [10.0], [20.0], [22.0]])
    n = np.array([[20.0], [20.0], [50.0], [50.0], [100.0], [100.0]])
    X = np.ones((6, 1))
    groups = np.array([0, 0, 1, 1, 2, 2])
    collapsed_y, collapsed_n, collapsed_X = collapse_groups_by_n(y, n, X, groups, rho=DEFAULT_RHO)

    expected = SampleSizeBasedLikelihoodEstimator().fit(
        y=collapsed_y, n=collapsed_n, X=collapsed_X, g=np.arange(3)
    )
    observed = SampleSizeBasedLikelihoodEstimator(weight_scheme="collapse").fit(
        y=y, n=n, X=X, g=groups
    )

    for key in ("fe_params", "sigma2", "tau2", "inv_cov"):
        assert np.allclose(observed.params_[key], expected.params_[key])


def test_collapse_mode_honours_rho_for_sample_sizes():
    """rho=1 is the only value for which a group's raw n is its effective n.

    Rows in a group share subjects, so n must not be counted once per row --
    but they are not perfect duplicates either. For s estimates from the same
    n subjects correlated at rho, Var(ybar) = (sigma^2 / n)(1 + rho(s-1))/s,
    i.e. an effective size s*n / (1 + rho(s-1)) running from n at rho=1 up to
    s*n at rho=0. Pinning it at n biases sigma^2 low by (1 + rho(s-1))/s --
    a factor of five for four uncorrelated estimates per group. Observations
    from one group share subjects but measure different outcomes, so rho is
    well below one and the bias is real.
    """
    rng = np.random.RandomState(0)
    n_groups, group_size = 10, 4
    groups = np.repeat(np.arange(n_groups), group_size)
    n_estimates = groups.size
    n = np.repeat(rng.randint(40, 160, n_groups).astype(float), group_size)[:, None]
    y = rng.randn(n_estimates, 1)
    X = np.ones((n_estimates, 1))

    effective = {}
    for rho in (0.0, 0.5, 1.0):
        _, collapsed_n, _, _ = _collapse_n_inputs(y, n, X, groups, "collapse", rho)
        effective[rho] = collapsed_n

    # Less assumed correlation means more independent information per group.
    assert np.all(effective[0.0] > effective[0.5])
    assert np.all(effective[0.5] > effective[1.0])

    # The endpoints are the two formulas being interpolated between, and the
    # closed form holds in between.
    _, raw_n, _ = collapse_groups_by_n(y, n, X, groups, rho=1.0)
    assert np.allclose(effective[1.0], raw_n)
    assert np.allclose(effective[0.0], group_size * raw_n)
    for rho in (0.0, 0.5, 1.0):
        assert np.allclose(effective[rho], group_size * raw_n / (1.0 + rho * (group_size - 1)))


def test_group_collapse_rejects_a_saturated_collapsed_design(variance_estimator):
    """Collapsing can saturate a design that was identified before it.

    Nine rows and three predictors is unremarkable, but three groups leave
    m == p, where the moment estimators divide by zero and report tau^2 = inf
    with zero standard errors.
    """
    groups = np.repeat([0, 1, 2], 3)
    X = np.c_[np.ones(9), np.repeat([0.0, 1.0, 2.0], 3), np.repeat([0.0, 0.0, 1.0], 3)]
    y = np.random.RandomState(0).randn(9, 1)

    with pytest.raises(ValueError, match="number of groups must exceed"):
        variance_estimator(weight_scheme="collapse").fit(
            y=y, v=np.full((9, 1), 0.1), X=X, g=groups
        )


def test_group_design_check_tolerates_floating_point_noise():
    """A predictor constant in intent may differ in its last bits."""
    groups = np.repeat([0, 1, 2], 3)
    y = np.random.RandomState(0).randn(9, 1)
    v = np.full((9, 1), 0.1)

    X = np.c_[np.ones(9), np.repeat([1.0, 2.0, 3.0], 3)]
    X[1, 1] += 1e-15
    Hedges(weight_scheme="collapse").fit(y=y, v=v, X=X, g=groups)  # must not raise

    # Genuine within-group variation is still rejected.
    X[1, 1] = 99.0
    with pytest.raises(ValueError, match="constant"):
        Hedges(weight_scheme="collapse").fit(y=y, v=v, X=X, g=groups)


def test_variance_components_are_insensitive_to_duplication(variance_estimator):
    """Duplicating a group's estimate must not shrink tau^2.

    Repeated estimates from one group agree with each other by construction. An
    estimator that counts them as independent sees less dispersion than the row
    count implies and shrinks tau^2 toward zero, which sharpens the weights and
    makes downstream inference anti-conservative.
    """
    rng = np.random.default_rng(4)
    n_estimates, n_datasets = 12, 8
    y = rng.standard_normal((n_estimates, n_datasets)) * 2.0
    v = np.full((n_estimates, n_datasets), 0.5)
    groups = np.arange(n_estimates)

    single = variance_estimator(weight_scheme="rescale")
    single.fit_dataset(Dataset(y=y, v=v, g=groups))

    # Group 0 now contributes four identical estimates instead of one.
    dupe_idx = np.r_[np.zeros(4, dtype=int), np.arange(1, n_estimates)]
    duped = variance_estimator(weight_scheme="rescale")
    duped.fit_dataset(Dataset(y=y[dupe_idx], v=v[dupe_idx], g=groups[dupe_idx]))

    assert np.allclose(np.mean(single.summary().tau2), np.mean(duped.summary().tau2), rtol=0.05)


def test_sample_size_identifiability_is_checked_on_the_fitted_values():
    """sigma^2 and tau^2 are identified by the n the likelihood actually sees.

    Under weight_scheme='rescale' those are effective sample sizes, which vary
    with group size even when every raw n is identical.
    """
    sizes = [1] * 6 + [20] * 6
    groups = np.concatenate([[j] * s for j, s in enumerate(sizes)])
    n_estimates = groups.size
    n = np.full((n_estimates, 1), 50.0)  # every raw n identical
    X = np.ones((n_estimates, 1))
    y = 0.5 + np.random.RandomState(2).randn(n_estimates, 1) * 0.4

    fitted = SampleSizeBasedLikelihoodEstimator(weight_scheme="rescale", rho=0.3).fit(
        y=y, n=n, X=X, g=groups
    )
    assert np.isfinite(fitted.params_["sigma2"]).all()

    # Genuinely unidentifiable input is still refused.
    with pytest.raises(ValueError, match="all-equal sample sizes"):
        SampleSizeBasedLikelihoodEstimator().fit(y=y, n=n, X=X)


def test_near_equal_sample_sizes_warn_rather_than_abort():
    """``raise Warning`` aborts the fit; this path should only warn."""
    n = np.full((20, 1), 100.0)
    n[0] = 101.0
    y = np.random.RandomState(0).randn(20, 1)

    with pytest.warns(UserWarning, match="too close"):
        SampleSizeBasedLikelihoodEstimator().fit(y=y, n=n, X=np.ones((20, 1)))


# -----------------------------------------------------------------------------
# Dataset and results integration
# -----------------------------------------------------------------------------


def test_dataset_rejects_invalid_group_labels():
    """One label per observation, and only one dimension of them."""
    with pytest.raises(ValueError, match="one group label per observation"):
        Dataset(y=np.arange(6.0), v=np.ones(6), g=np.tile(np.arange(3), (6, 1))[:, :2])

    with pytest.raises(ValueError, match="same number of rows"):
        Dataset(y=[1.0, 2.0, 3.0], v=[1.0, 1.0, 1.0], g=[0, 1])

    # fit() takes arrays directly, with no Dataset to check them. Validating the
    # count against the labels themselves made the check vacuous, and the fit
    # then failed on a boolean-index IndexError instead.
    with pytest.raises(ValueError, match="one label per observation"):
        Hedges(weight_scheme="collapse").fit(
            y=np.zeros((9, 1)),
            v=np.full((9, 1), 0.1),
            X=np.c_[np.ones(9), np.repeat([1.0, 2.0, 3.0], 3)],
            g=np.repeat([0, 1], 3),
        )


def test_dataset_group_labels_round_trip_through_a_dataframe():
    """Labels survive to_df, repeated once per parallel dataset, and come back."""
    groups = np.array([0, 0, 1, 1])
    parallel = Dataset(y=np.arange(8.0).reshape(4, 2), v=np.ones((4, 2)), g=groups).to_df()
    assert parallel["g"].tolist() == np.tile(groups, 2).tolist()

    frame = Dataset(y=np.arange(4.0), v=np.ones(4), g=groups).to_df()
    assert "g" in frame.columns
    assert Dataset(data=frame).g.ravel().tolist() == groups.tolist()

    # `g or "g"` called bool() on the array and raised.
    without_labels = pd.DataFrame({"y": np.arange(4.0), "v": np.ones(4)})
    assert np.array_equal(np.ravel(Dataset(data=without_labels, g=groups).g), groups)


def test_group_labels_flow_through_dataset_and_meta_regression(dependent_data):
    """Groups must reach the estimator through both the object and functional APIs."""
    y, v, X, groups = dependent_data(np.random.RandomState(0), n_datasets=1)
    n_groups = np.unique(groups).size

    dataset = Dataset(y=y, v=v, X=X, g=groups, add_intercept=False)
    assert dataset.g.shape == (y.shape[0], 1)
    assert WeightedLeastSquares().fit_dataset(dataset).n_groups_ == n_groups

    results = meta_regression(y=y, v=v, X=X, add_intercept=False, method="WLS", g=groups)
    assert results.estimator.n_groups_ == n_groups


def test_results_use_t_reference_with_groups(dependent_data):
    """Robust fits should report a t reference with Satterthwaite dof."""
    y, v, X, groups = dependent_data(np.random.RandomState(0), n_datasets=1)

    dataset = Dataset(y=y, v=v, X=X, g=groups, add_intercept=False)
    robust = WeightedLeastSquares().fit_dataset(dataset).summary()
    naive_dataset = Dataset(y=y, v=v, X=X, add_intercept=False)
    naive = WeightedLeastSquares().fit_dataset(naive_dataset).summary()

    # This design is balanced -- one predictor, equal-sized groups -- which is
    # the regime where m - p is adequate, so the two should nearly agree.
    n_groups = np.unique(groups).size
    assert robust.fe_dof.shape == robust.fe_params.shape
    assert np.all(robust.fe_dof <= n_groups - X.shape[1])
    assert np.allclose(robust.fe_dof, n_groups - X.shape[1], rtol=0.15)
    assert naive.fe_dof is None

    # A t reference with finite df is heavier tailed than a normal, so for the
    # same statistic it yields a larger p-value and a wider interval.
    robust_stats = robust.get_fe_stats()
    naive_stats = naive.get_fe_stats()
    if np.allclose(robust_stats["z"], naive_stats["z"]):  # only if the SEs coincide
        assert np.all(robust_stats["p"] >= naive_stats["p"])
    width_robust = robust_stats["ci_u"] - robust_stats["ci_l"]
    width_naive = naive_stats["ci_u"] - naive_stats["ci_l"]
    assert np.all(width_robust > width_naive)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_one_shared_column_serves_every_parallel_dataset(grouped_estimator):
    """A single column of v or n applies to all of them, and must be read that way.

    Dataset allows it, cluster_robust_cov and satterthwaite_dof document it, and
    the vectorized estimators accepted it -- but the looping ones sliced ``v``
    per dataset regardless, so the same convention held for ``n`` and not for
    ``v``. The degrees of freedom have to come back one per parallel dataset
    either way, or they no longer line up with the coefficients beside them.
    """
    estimator, second_arg, build_inputs = grouped_estimator
    shared, groups = build_inputs(shared_second=True)
    n_datasets = shared["y"].shape[1]
    expanded = dict(shared)
    expanded[second_arg] = np.repeat(shared[second_arg], n_datasets, axis=1)

    from_shared = estimator().fit(**shared, g=groups)
    from_expanded = estimator().fit(**expanded, g=groups)

    assert np.allclose(from_shared.params_["fe_params"], from_expanded.params_["fe_params"])
    assert from_shared.params_["dof"].shape == from_shared.params_["fe_params"].shape


def test_heterogeneity_is_undefined_when_the_design_exhausts_the_df():
    """Q on zero degrees of freedom is rounding noise, not zero heterogeneity.

    Collapsing to one row per group makes df = m - p, which reaches zero far
    more easily than the old K - p. Reported as-is it came out I^2 = 100% and
    H = inf, which reads as total heterogeneity rather than no information.
    """
    rng = np.random.default_rng(0)
    X = np.c_[np.ones(3), rng.normal(size=3), rng.normal(size=3)]  # K == p == 3
    dataset = Dataset(y=rng.normal(size=3), v=np.full(3, 0.2), X=X, add_intercept=False)
    results = WeightedLeastSquares().fit_dataset(dataset).summary()

    stats = results.get_heterogeneity_stats()
    assert all(np.all(np.isnan(stats[key])) for key in ("Q", "p(Q)", "I^2", "H"))


def test_undefined_standard_errors_do_not_read_as_significant():
    """Dividing by a zero standard error yields p = 0, i.e. maximal certainty."""
    rng = np.random.default_rng(0)
    dataset = Dataset(y=rng.normal(size=8), v=np.ones(8))
    results = WeightedLeastSquares().fit_dataset(dataset).summary()
    results.fe_cov = np.zeros_like(results.fe_cov)  # force a degenerate SE
    results.__dict__.pop("fe_se", None)

    stats = results.get_fe_stats()
    assert np.all(np.isnan(stats["p"]))
    assert np.all(np.isnan(stats["ci_l"])) and np.all(np.isnan(stats["ci_u"]))


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("scheme", ["rescale", "collapse"])
def test_tau2_interval_and_heterogeneity_use_the_same_units_as_tau2(scheme):
    """A point estimate must not fall outside its own confidence interval.

    Both "rescale" and "collapse" estimate tau^2 from one aggregate per group.
    Results statistics have to collapse on the same condition, or the interval
    and Q describe a different set of units than the estimate they accompany.
    """
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(12), 4)
    n_estimates = groups.size
    y = rng.normal(size=n_estimates) + np.repeat(rng.normal(scale=0.3, size=12), 4)
    dataset = Dataset(y=y, v=np.full(n_estimates, 0.2), g=groups)

    results = DerSimonianLaird(weight_scheme=scheme).fit_dataset(dataset).summary()
    re_stats = results.get_re_stats()
    tau2 = float(np.ravel(re_stats["tau^2"])[0])
    assert float(np.ravel(re_stats["ci_l"])[0]) <= tau2 <= float(np.ravel(re_stats["ci_u"])[0])

    # Q must be referred to the number of independent groups, not rows.
    heterogeneity = results.get_heterogeneity_stats()
    raw = DerSimonianLaird().fit_dataset(dataset).summary().get_heterogeneity_stats()
    assert float(np.ravel(heterogeneity["Q"])[0]) < float(np.ravel(raw["Q"])[0])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_group_weighted_heterogeneity_matches_collapsed_reference():
    """Q and its degrees of freedom must use independent group aggregates."""
    y = np.array([[1.0], [3.0], [8.0], [10.0], [20.0]])
    v = np.array([[1.0], [4.0], [2.0], [8.0], [5.0]])
    X = np.ones((5, 1))
    groups = np.array([0, 0, 1, 1, 2])
    rho = 0.4

    observed = (
        WeightedLeastSquares(weight_scheme="collapse", rho=rho)
        .fit_dataset(Dataset(y=y, v=v, X=X, g=groups, add_intercept=False))
        .summary()
        .get_heterogeneity_stats()
    )

    collapsed_y, collapsed_v, collapsed_X = collapse_groups(y, v, X, groups, rho=rho)
    expected = (
        WeightedLeastSquares()
        .fit_dataset(Dataset(y=collapsed_y, v=collapsed_v, X=collapsed_X, add_intercept=False))
        .summary()
        .get_heterogeneity_stats()
    )

    for key in ("Q", "p(Q)", "I^2", "H"):
        assert np.allclose(observed[key], expected[key])


def test_heterogeneity_uses_the_aggregation_the_estimator_fitted():
    """Sample-size estimators collapse by n, so results must not use v."""
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(12), 3)
    n = np.repeat(rng.randint(30, 200, 12).astype(float), 3)[:, None]
    y = (rng.randn(12, 1) * 0.5).repeat(3, axis=0) + rng.randn(groups.size, 1) * 0.2

    results = (
        SampleSizeBasedLikelihoodEstimator(weight_scheme="collapse")
        .fit_dataset(Dataset(y=y, n=n, g=groups))
        .summary()
    )
    _, analysis_v, _, analysis_groups = results._analysis_arrays()
    assert analysis_groups is None  # aggregated, not reweighted

    _, collapsed_n, _ = collapse_groups_by_n(y, n, results.dataset.X, groups, rho=DEFAULT_RHO)
    sigma2 = np.asarray(results.estimator.params_["sigma2"], dtype=float)
    assert np.allclose(analysis_v, sigma2 / collapsed_n)


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


# -----------------------------------------------------------------------------
# Permutation refits: same model, same units
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "scheme, keeps_every_row",
    [("individual", True), ("rescale", True), ("collapse", False)],
)
def test_permutation_refits_the_model_that_produced_the_estimate(scheme, keeps_every_row):
    """The null has to be the observed statistic recomputed, not another one.

    Only "collapse" fits one row per group. "rescale" keeps every row and merely
    reweights it, so reducing it to group means built the null from a different
    model: the identity permutation did not even return the observed
    coefficient once predictors or variances varied within a group.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), [1, 5, 2, 4, 3, 3])  # deliberately unequal
    n_estimates = groups.size
    y = rng.normal(size=n_estimates)
    v = np.abs(rng.normal(size=n_estimates)) + 0.3  # varies within a group
    results = (
        DerSimonianLaird(weight_scheme=scheme).fit_dataset(Dataset(y=y, v=v, g=groups)).summary()
    )

    analysis_y, analysis_v, analysis_X, analysis_groups, _ = results._permutation_arrays()
    assert (analysis_y.shape[0] == n_estimates) is keeps_every_row

    # The identity permutation is one of the permutations the null is built
    # from, so it must reproduce the reported coefficient exactly.
    identity = copy.copy(results.estimator).fit(
        y=analysis_y, v=analysis_v, X=analysis_X, g=analysis_groups
    )
    assert np.allclose(identity.params_["fe_params"], results.fe_params)


def test_group_permutation_keeps_each_group_on_its_own_rows():
    """A permuted dataset is refit against the original labels.

    ``_expand_unit_order`` therefore has to write each source group into the
    positions of the group it replaces. Concatenating groups in the permuted
    order instead assumes the labels are contiguous and equally sized: with
    interleaved labels it split a group across two of them even under the
    identity permutation.
    """
    codes = np.array([0, 1, 0, 1])  # interleaved, so order != layout

    identity = _expand_unit_order(np.array([0, 1]), codes, codes.size)
    assert np.array_equal(identity, np.arange(codes.size))

    swapped = _expand_unit_order(np.array([1, 0]), codes, codes.size)
    # Every row labelled 0 now holds an estimate that came from group 1.
    assert np.array_equal(codes[swapped], 1 - codes)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_moderator_permutation_only_swaps_equally_sized_groups():
    """A group of three rows cannot take the place of a group of two.

    The refit is vectorized over permutations and so sees one design and one
    set of labels for every column. Enumerating all m! orders regardless of
    size silently reassigned rows to the wrong group.
    """
    groups = np.repeat(np.arange(4), [2, 2, 3, 3])
    x = np.repeat([0.0, 1.0, 0.0, 1.0], [2, 2, 3, 3])
    rng = np.random.default_rng(0)
    dataset = Dataset(
        y=rng.normal(size=groups.size),
        v=np.full(groups.size, 0.2),
        X=np.c_[np.ones(groups.size), x],
        g=groups,
        add_intercept=False,
    )
    results = WeightedLeastSquares().fit_dataset(dataset).summary()

    perm = results.permutation_test(n_perm=1000)

    # 2! within each size class, not 4!.
    assert perm.exact
    assert perm.n_perm == 4

    # Sizes that are all distinct leave only the identity, which is no test.
    singleton = Dataset(
        y=rng.normal(size=6),
        v=np.full(6, 0.2),
        X=np.c_[np.ones(6), np.repeat([0.0, 1.0, 0.0], [1, 2, 3])],
        g=np.repeat([0, 1, 2], [1, 2, 3]),
        add_intercept=False,
    )
    with pytest.raises(ValueError, match="No non-trivial permutation"):
        WeightedLeastSquares().fit_dataset(singleton).summary().permutation_test(n_perm=100)


def test_combination_permutation_freezes_the_correlation_the_estimator_used():
    """Identical rows are read as perfectly correlated, not as uncorrelated.

    Both estimators skip centering when every row is the same, because centering
    leaves zeros whose correlation is undefined. Centering unconditionally here
    froze an identity matrix instead, so the permutation null was drawn without
    the variance inflation the observed statistic carried.
    """
    z = np.tile(np.random.default_rng(0).normal(size=6), (4, 1))
    groups = np.repeat([0, 1], 2)
    # Identical rows correlate at one, which is what the estimator reads off
    # them and therefore what the permutation refits have to be given.
    corr = np.eye(4)
    corr[:2, :2] = corr[2:, 2:] = 1.0

    estimated = StoufferCombinationTest().fit_dataset(Dataset(y=z, g=groups)).summary()
    supplied = StoufferCombinationTest().fit_dataset(Dataset(y=z, g=groups), corr=corr).summary()
    assert np.allclose(np.ravel(estimated.p), np.ravel(supplied.p))

    # 2**2 sign patterns, so both permutation tests are exact and comparable.
    assert np.allclose(
        estimated.permutation_test(n_perm=4).perm_p["fe_p"],
        supplied.permutation_test(n_perm=4).perm_p["fe_p"],
    )


def test_dataset_exports_a_shared_sample_size_column():
    """One column of n serves every parallel dataset, so to_df must not index it."""
    dataset = Dataset(y=np.arange(8.0).reshape(4, 2), n=np.full((4, 1), 10.0))

    frame = dataset.to_df()

    assert frame["n"].tolist() == [10.0] * 8
    assert frame["set"].tolist() == [0] * 4 + [1] * 4


# -----------------------------------------------------------------------------
# Alignment with robumeta
# -----------------------------------------------------------------------------

#: Reference values from the R package robumeta 2.1, ``robu(..., modelweights =
#: "CORR", small = TRUE)``, on ``pymare/tests/data/robumeta_correlated_effects.csv``.
#: Regenerate with the Docker harness in ``validation/robumeta``; robumeta is not
#: a test dependency, so the numbers are pinned rather than recomputed here.
#: Each row is (model, rho, variance column, tau^2, coefficients, standard
#: errors, Satterthwaite degrees of freedom).
ROBUMETA_REFERENCE = [
    (
        "intercept",
        0,
        "var_constant_within_study",
        1.6028426057839684,
        (0.40471056035924913),
        (0.33099091499096078),
        (8.9756413413106486),
    ),
    (
        "within",
        0,
        "var_constant_within_study",
        1.0958589607913456,
        (0.52706873462747961, 0.65256410459011283),
        (0.30387369766137007, 0.11162812333369045),
        (8.2577982913301433, 3.6375001189847995),
    ),
    (
        "both",
        0,
        "var_constant_within_study",
        0.36200604143599374,
        (-0.011018138717982739, 0.63480056686773689, -0.91134267844738193),
        (0.18567544639284098, 0.15023749633945144, 0.18376565771199688),
        (2.6754498054212048, 3.7006875407608133, 1.7106511808281428),
    ),
    (
        "intercept",
        0.4,
        "var_constant_within_study",
        1.6064045505974576,
        (0.40467282737319688),
        (0.33097861010228691),
        (8.9757345435900078),
    ),
    (
        "within",
        0.4,
        "var_constant_within_study",
        1.1000823546144964,
        (0.52696317263348147, 0.65257321465117712),
        (0.30382938091253558, 0.11162865665086494),
        (8.2579609313791327, 3.637205209769236),
    ),
    (
        "both",
        0.4,
        "var_constant_within_study",
        0.37059879946922641,
        (-0.011418847047311109, 0.63511340620445467, -0.91087832100858623),
        (0.18554675471372578, 0.15017903802755506, 0.18374152318727882),
        (2.674255871990928, 3.6982109910603636, 1.7115225969755608),
    ),
    (
        "intercept",
        0.8,
        "var_constant_within_study",
        1.6099664954109469,
        (0.40463524670403489),
        (0.33096635808306268),
        (8.9758272107119197),
    ),
    (
        "within",
        0.8,
        "var_constant_within_study",
        1.1043057484376473,
        (0.52685832153343115, 0.65258227077691433),
        (0.30378536030489106, 0.11162917895275425),
        (8.2581218098461431, 3.6369122085702452),
    ),
    (
        "both",
        0.8,
        "var_constant_within_study",
        0.37919155750245914,
        (-0.011808637045980008, 0.63541679890689351, -0.91042783478109079),
        (0.18542258314073815, 0.15012210662207862, 0.18371848909744803),
        (2.6730960482294388, 3.6958057543088572, 1.7123678737471151),
    ),
    (
        "intercept",
        1,
        "var_constant_within_study",
        1.6117474678176915,
        (0.40461651319907282),
        (0.33096025179349187),
        (8.9758733448713706),
    ),
    (
        "within",
        1,
        "var_constant_within_study",
        1.1064174453492228,
        (0.52680616031860494, 0.6525867787583457),
        (0.30376346011797389, 0.11162943602684719),
        (8.258201595908762, 3.6367664176856507),
    ),
    (
        "both",
        1,
        "var_constant_within_study",
        0.38348793651907548,
        (-0.011999578720277904, 0.6355650853060808, -0.91020759282478725),
        (0.18536212066885069, 0.15009419419321809, 0.18370736610828231),
        (2.672528452028569, 3.694628921058003, 1.7127810638504646),
    ),
    (
        "intercept",
        0,
        "var_within_study",
        1.5091511990162025,
        (0.41841377265357016),
        (0.33581946793342782),
        (8.9048085664842009),
    ),
    (
        "within",
        0,
        "var_within_study",
        1.0613759144209511,
        (0.54859012187796075, 0.64453537453536569),
        (0.31404547449514841, 0.11307561709829847),
        (8.1859075363796485, 3.7324773492814334),
    ),
    (
        "both",
        0,
        "var_within_study",
        0.23321637308995152,
        (0.0081835962802159212, 0.60128069478156643, -0.94708552700557513),
        (0.17917280179078104, 0.13874486566474509, 0.16671110057652813),
        (2.6813532610297015, 3.9213210954440694, 1.6219220695919043),
    ),
    (
        "intercept",
        0.4,
        "var_within_study",
        1.5129951360130969,
        (0.41834928907453423),
        (0.33579348025581851),
        (8.9051707539191973),
    ),
    (
        "within",
        0.4,
        "var_within_study",
        1.0661700294514225,
        (0.5483958918930808, 0.64457313311463127),
        (0.31396643875506963, 0.11307475976109986),
        (8.1865311957234539, 3.7318151373088844),
    ),
    (
        "both",
        0.4,
        "var_within_study",
        0.24359149939051292,
        (0.0074386891609868888, 0.60217339209870124, -0.94586804049042628),
        (0.17916970202188001, 0.13906404067480177, 0.16700185979133597),
        (2.6799074856248408, 3.9144998938396487, 1.6246328880102403),
    ),
    (
        "intercept",
        0.8,
        "var_within_study",
        1.5168390730099912,
        (0.4182850700023929),
        (0.33576761642262837),
        (8.9055308532481163),
    ),
    (
        "within",
        0.8,
        "var_within_study",
        1.0709641444818936,
        (0.54820310408238437, 0.64461064363983778),
        (0.31388795201504255, 0.11307388027521514),
        (8.18714868174537, 3.7311576161940954),
    ),
    (
        "both",
        0.8,
        "var_within_study",
        0.25396662569107431,
        (0.0067175121550294703, 0.60303848969352369, -0.94469907374351059),
        (0.1791662991434663, 0.13936505111212369, 0.16728171712866918),
        (2.6785181647604874, 3.9078721964279026, 1.6272497948928502),
    ),
    (
        "intercept",
        1,
        "var_within_study",
        1.5187610415084385,
        (0.41825305913214095),
        (0.33575473067628292),
        (8.9057101249480333),
    ),
    (
        "within",
        1,
        "var_within_study",
        1.0733612019971293,
        (0.5481072459186862, 0.64462930661099405),
        (0.31384891267003273, 0.11307343236774971),
        (8.1874551351746874, 3.730830599232112),
    ),
    (
        "both",
        1,
        "var_within_study",
        0.25915418884135499,
        (0.0063654525518125338, 0.60346103883343161, -0.94413179688008408),
        (0.17916449556168593, 0.13950916609190969, 0.16741775234816347),
        (2.6778434546330101, 3.9046295486129234, 1.6285243426993139),
    ),
]


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "model, rho, variances, tau2, beta, se, dof",
    ROBUMETA_REFERENCE,
    ids=[
        f"{row[0]}-rho{row[1]}-{'shared' if row[2].startswith('var_c') else 'varying'}-v"
        for row in ROBUMETA_REFERENCE
    ],
)
def test_matches_robumeta_correlated_effects(
    model, rho, variances, tau2, beta, se, dof, robumeta_dataset
):
    """The correlated-effects model must reproduce the implementation it cites.

    This pins far more than the tau^2 estimator: the coefficients come from the
    correlated-effects weights, the standard errors from the CR2 sandwich and
    the degrees of freedom from the Satterthwaite approximation, so agreement
    here exercises the whole path against
    :footcite:t:`hedges2010robust` as implemented by robumeta.

    Both variance columns are checked: one constant within a study, one varying
    sharply within it. PyMARE weights both the way robumeta does, so the two
    agree throughout rather than only where the weightings happen to coincide.
    """
    frame, designs = robumeta_dataset
    fitted = (
        DerSimonianLaird(weight_scheme="rescale", rho=rho)
        .fit(
            y=frame["effect"].to_numpy()[:, None],
            v=frame[variances].to_numpy()[:, None],
            X=designs[model],
            g=frame["study"].to_numpy(),
        )
        .summary()
    )

    assert np.allclose(np.ravel(fitted.tau2), tau2, rtol=1e-10, atol=0)
    assert np.allclose(np.ravel(fitted.fe_params), beta, rtol=1e-10, atol=0)
    assert np.allclose(np.ravel(fitted.fe_se), se, rtol=1e-10, atol=0)
    assert np.allclose(np.ravel(fitted.fe_dof), dof, rtol=1e-10, atol=0)
