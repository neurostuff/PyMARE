"""Tests for pymare.estimators.combination."""

import numpy as np
import pytest
import scipy.stats as ss

from pymare import Dataset
from pymare.estimators import FisherCombinationTest, StoufferCombinationTest

_z1 = np.array([2.1, 0.7, -0.2, 4.1, 3.8])[:, None]
_z2 = np.c_[_z1, np.array([-0.6, -1.61, -2.3, -0.8, -4.01])[:, None]]

_params = [
    (StoufferCombinationTest, _z1, "directed", [4.69574]),
    (StoufferCombinationTest, _z1, "undirected", [4.87462819]),
    (StoufferCombinationTest, _z1, "concordant", [4.55204117]),
    (StoufferCombinationTest, _z2, "directed", [4.69574275, -4.16803071]),
    (StoufferCombinationTest, _z2, "undirected", [4.87462819, 4.16803071]),
    (StoufferCombinationTest, _z2, "concordant", [4.55204117, -4.00717817]),
    (FisherCombinationTest, _z1, "directed", [5.22413541]),
    (FisherCombinationTest, _z1, "undirected", [5.27449962]),
    (FisherCombinationTest, _z1, "concordant", [5.09434911]),
    (FisherCombinationTest, _z2, "directed", [5.22413541, -3.30626405]),
    (FisherCombinationTest, _z2, "undirected", [5.27449962, 4.27572965]),
    (FisherCombinationTest, _z2, "concordant", [5.09434911, -4.11869468]),
]


@pytest.mark.parametrize("Cls,data,mode,expected", _params)
def test_combination_test(Cls, data, mode, expected):
    """Test CombinationTest Estimators with numpy data."""
    results = Cls(mode).fit(data).params_
    assert np.allclose(results["z"], expected, atol=1e-5)


@pytest.mark.parametrize("Cls,data,mode,expected", _params)
def test_combination_test_from_dataset(Cls, data, mode, expected):
    """Test CombinationTest Estimators with PyMARE Datasets."""
    dset = Dataset(y=data)
    est = Cls(mode).fit_dataset(dset)
    results = est.summary()
    assert np.allclose(results.z, expected, atol=1e-5)


def test_stouffer_adjusted():
    """Test StoufferCombinationTest with weights and groups."""
    # Test with weights and groups
    data = np.array(
        [
            [2.1, 0.7, -0.2, 4.1, 3.8],
            [1.1, 0.2, 0.4, 1.3, 1.5],
            [-0.6, -1.6, -2.3, -0.8, -4.0],
            [2.5, 1.7, 2.1, 2.3, 2.5],
            [3.1, 2.7, 3.1, 3.3, 3.5],
            [3.6, 3.2, 3.6, 3.8, 4.0],
        ]
    )
    weights = np.tile(np.array([4, 3, 4, 10, 15, 10]), (data.shape[1], 1)).T
    groups = np.tile(np.array([0, 0, 1, 2, 2, 2]), (data.shape[1], 1)).T

    results = StoufferCombinationTest("directed").fit(z=data, w=weights, g=groups).params_

    # These values changed when _inflation_term was fixed to index each group's
    # own weights. Group 2 occupies rows 3-5 with weights (10, 15, 10), but the
    # pairwise loop indexed the full weight array with block-local positions and
    # so used rows 0-2's weights (4, 3, 4) instead. That understated the variance
    # inflation, inflating z by ~50%. Verified against Var(sum w_i z_i) = w'Cw.
    z_expected = np.array([3.34419412, 2.47665061, 2.71143136, 3.65341755, 3.47017404])
    assert np.allclose(results["z"], z_expected, atol=1e-5)

    # Test with weights and no groups. Limiting cases.
    # Limiting case 1: all correlations are one.
    n_maps_l1 = 5
    common_sample = np.array([2.1, 0.7, -0.2])
    data_l1 = np.tile(common_sample, (n_maps_l1, 1))
    groups_l1 = np.tile(np.array([0, 0, 0, 0, 0]), (data_l1.shape[1], 1)).T

    results_l1 = StoufferCombinationTest("directed").fit(z=data_l1, g=groups_l1).params_

    sigma_l1 = n_maps_l1 * (n_maps_l1 - 1)  # Expected inflation term
    z_expected_l1 = n_maps_l1 * common_sample / np.sqrt(n_maps_l1 + sigma_l1)
    assert np.allclose(results_l1["z"], z_expected_l1, atol=1e-5)

    # Test with correlation matrix and groups.
    data_corr = data - data.mean(0)
    corr = np.corrcoef(data_corr, rowvar=True)
    results_corr = (
        StoufferCombinationTest("directed").fit(z=data, w=weights, g=groups, corr=corr).params_
    )

    z_corr_expected = np.array([3.34419412, 2.47665061, 2.71143136, 3.65341755, 3.47017404])
    assert np.allclose(results_corr["z"], z_corr_expected, atol=1e-5)

    # Test with no correlation matrix and groups, but only one feature.
    with pytest.raises(ValueError):
        StoufferCombinationTest("directed").fit(z=data[:, :1], w=weights[:, :1], g=groups)

    # Test with correlation matrix and groups of different shapes.
    with pytest.raises(ValueError):
        StoufferCombinationTest("directed").fit(z=data, w=weights, g=groups, corr=corr[:-2, :-2])

    # Test with correlation matrix and no groups.
    results1 = StoufferCombinationTest("directed").fit(z=_z1, corr=corr).params_

    assert np.allclose(results1["z"], [4.69574], atol=1e-5)


# -----------------------------------------------------------------------------
# Dependent estimates: group aggregation
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


def test_fisher_does_not_underflow_on_many_moderate_z():
    """Its own combined tail underflows too, and much sooner than one input can.

    Converting each input in logs, as the test above checks, still leaves
    ``chi2.sf`` -- and ``chi2.logsf``, which logs it afterwards -- at the end of
    the statistic. Two hundred inputs at z = 3 put the combined chi-squared
    where a double-precision tail is exactly zero, which came back as
    ``logp = -inf`` and ``z = inf``: less informative than any single input.
    """
    fitted = FisherCombinationTest().fit(np.full((200, 2), 3.0)).params_

    assert np.all(fitted["p"] == 0.0)  # the representation, not the evidence
    assert np.allclose(fitted["logp"], -749.1910315, rtol=1e-8)
    assert np.allclose(fitted["z"], 38.5906313, rtol=1e-7)


def test_stouffer_does_not_underflow_on_many_moderate_z():
    """Fisher had this guard already; Stouffer reaches the same wall sooner.

    The combined statistic is a weighted *sum* divided by the square root of the
    weight, so it grows like ``sqrt(k)``: four hundred inputs at z = 3 -- not an
    unusual meta-analysis -- combine to z = 60, whose one-tailed p-value is
    about 1e-785. Going via ``norm.sf`` returned exactly 0 there, and the z
    rebuilt from that 0 was ``+inf``, so the answer was less informative than
    any single one of its inputs.
    """
    z = np.full((400, 2), 3.0)

    fitted = StoufferCombinationTest().fit(z).params_

    assert np.all(fitted["p"] == 0.0)  # the representation, not the evidence
    assert np.all(np.isfinite(fitted["logp"]))
    assert np.allclose(fitted["z"], 60.0)
    assert np.allclose(fitted["logp"], ss.norm.logsf(60.0))


def test_public_p_value_survives_the_move_to_log_space():
    """Subclasses now implement log_p_value, but p_value stays part of the API."""
    est = StoufferCombinationTest()

    assert np.allclose(est.p_value(_z2), np.exp(est.log_p_value(_z2)))


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
