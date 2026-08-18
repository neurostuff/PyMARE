"""Tests for dependent-estimate support (cluster-robust variance, Brown's method)."""

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
from pymare.estimators.estimators import _collapse_n_inputs
from pymare.stats import (
    DEFAULT_RHO,
    MIN_DOF_FOR_SATTERTHWAITE,
    _cr2_low_rank_apply,
    _cr2_low_rank_factors,
    _symmetric_sqrt,
    cluster_robust_cov,
    collapse_groups,
    collapse_groups_by_n,
    encode_groups,
    estimate_null_correlation,
    group_mean,
    group_weights,
    normalize_group_weights,
    one_sample_t_from_sufficient_statistics,
    satterthwaite_dof,
    undo_centering_shrinkage,
    weighted_intercept_cr2,
    weighted_intercept_cr2_sufficient_statistics,
    weighted_least_squares,
)


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


def test_stouffer_group_level_is_invariant_to_perfect_duplicates():
    """Duplicating a perfectly correlated observation cannot buy group influence."""
    z = np.array([[1.5, -0.5], [0.25, 2.0]])
    weights = np.sqrt(np.array([25.0, 100.0]))[:, None]
    expected = StoufferCombinationTest(group_level=True).fit(
        z,
        w=weights,
        g=np.array([0, 1]),
        corr=np.eye(2),
    )

    expanded = np.vstack([np.repeat(z[[0]], 4, axis=0), z[[1]]])
    expanded_weights = np.sqrt(np.array([25.0] * 4 + [100.0]))[:, None]
    corr = np.eye(5)
    corr[:4, :4] = 1.0
    observed = StoufferCombinationTest(group_level=True).fit(
        expanded,
        w=expanded_weights,
        g=np.array([0, 0, 0, 0, 1]),
        corr=corr,
    )

    assert np.allclose(observed.params_["z"], expected.params_["z"])
    assert np.allclose(observed.params_["p"], expected.params_["p"])


def test_stouffer_group_level_permutation_flips_groups_as_units():
    """Exact permutation inference is invariant to duplicated group rows."""
    collapsed_z = np.array([[1.5], [0.25], [-0.5]])
    collapsed_weights = np.sqrt(np.array([25.0, 100.0, 50.0]))[:, None]
    collapsed_estimator = StoufferCombinationTest(group_level=True)
    collapsed_result = collapsed_estimator.fit_dataset(
        Dataset(
            y=collapsed_z,
            n=collapsed_weights,
            g=np.arange(3),
        ),
        corr=np.eye(3),
    ).summary()

    expanded_z = np.vstack([np.repeat(collapsed_z[[0]], 4, axis=0), collapsed_z[1:]])
    expanded_weights = np.sqrt(np.array([25.0] * 4 + [100.0, 50.0]))[:, None]
    expanded_groups = np.array([0, 0, 0, 0, 1, 2])
    corr = np.eye(6)
    corr[:4, :4] = 1.0
    expanded_result = (
        StoufferCombinationTest(group_level=True)
        .fit_dataset(
            Dataset(y=expanded_z, n=expanded_weights, g=expanded_groups),
            corr=corr,
        )
        .summary()
    )

    collapsed_perm = collapsed_result.permutation_test(n_perm=20)
    expanded_perm = expanded_result.permutation_test(n_perm=20)
    assert collapsed_perm.exact
    assert expanded_perm.exact
    assert collapsed_perm.n_perm == expanded_perm.n_perm == 8
    assert np.allclose(collapsed_perm.perm_p["fe_p"], expanded_perm.perm_p["fe_p"])


def test_stouffer_group_level_rejects_inconsistent_group_weights():
    """A group has one weight, not one independently varying row weight."""
    with pytest.raises(ValueError, match="one weight per group"):
        StoufferCombinationTest(group_level=True).fit(
            np.ones((3, 2)),
            w=np.array([2.0, 3.0, 4.0])[:, None],
            g=np.array([0, 0, 1]),
            corr=np.eye(3),
        )


def test_combination_permutation_accepts_weights():
    """Weights are per observation and must be repeated across permutations.

    They live on the Dataset as ``n`` with one column per parallel dataset,
    while the permuted z array has one column per permutation, so passing the
    stored array straight through only lines up by coincidence.
    """
    rng = np.random.RandomState(0)
    z = rng.randn(8, 3)
    w = np.abs(rng.randn(8, 3)) + 1.0

    result = StoufferCombinationTest().fit_dataset(Dataset(y=z, n=w)).summary()
    perm = result.permutation_test(n_perm=40)

    assert perm.perm_p["fe_p"].shape == np.ravel(result.p).shape
    assert np.all(perm.perm_p["fe_p"] > 0)
    assert np.all(perm.perm_p["fe_p"] <= 1)


@pytest.mark.parametrize("estimator", [StoufferCombinationTest, FisherCombinationTest])
def test_combination_permutation_rejects_undirected_mode(estimator):
    """abs(z) makes the statistic invariant to sign flips, so there is no null."""
    rng = np.random.RandomState(0)
    result = estimator(mode="undirected").fit_dataset(Dataset(y=rng.randn(8, 2))).summary()

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

    order = np.array([2, 3, 0, 1])
    reordered = StoufferCombinationTest().fit(
        z[order], w=weights[order], g=groups[order], corr=corr[np.ix_(order, order)]
    )
    assert np.allclose(reordered.params_["p"], expected)


def test_fisher_does_not_underflow_on_extreme_z():
    """Going via p collapses to exactly 0 once any |z| exceeds about 38."""
    z = np.zeros((100, 2))
    z[0] = 40.0

    p = FisherCombinationTest().fit(z).params_["p"]

    assert np.all(p > 0)
    assert np.all(np.isfinite(FisherCombinationTest().fit(z).params_["z"]))
    assert np.allclose(p, 1.035e-244, rtol=1e-3)


@pytest.mark.parametrize("estimator", [StoufferCombinationTest, FisherCombinationTest])
def test_constant_estimates_cannot_yield_a_correlation(estimator):
    """A row that never varies has no correlation, so do not return NaN."""
    z = np.full((4, 5), 1.3)
    groups = np.array([0, 0, 1, 1])
    if estimator is StoufferCombinationTest:
        groups = np.tile(groups[:, None], (1, 5))

    with pytest.raises(ValueError, match="constant across features"):
        estimator().p_value(z, g=groups)


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


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("weight_scheme", ["individual", "rescale", "collapse"])
def test_permutation_flips_whole_groups_and_leaves_the_estimator_alone(weight_scheme):
    """Dependent rows are exchangeable only as complete groups.

    Flipping them independently builds a null about half as wide as the truth,
    which showed up as ~25% rejection at a nominal 5%. Separately, refitting
    through the live estimator overwrote params_["dof"] with a
    permutation-shaped array, so fe_dof and to_df broke afterwards.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 2)
    dataset = Dataset(y=rng.normal(size=16), v=np.full(16, 0.2), g=groups)
    results = DerSimonianLaird(weight_scheme=weight_scheme).fit_dataset(dataset).summary()

    before = np.ravel(results.fe_dof).copy()
    perm = results.permutation_test(n_perm=100000)

    # 8 groups, not 16 rows: 2**8 sign patterns exhaust the null.
    assert perm.n_perm == 2**8
    assert perm.exact
    assert np.allclose(np.ravel(results.fe_dof), before)
    perm.to_df()


@pytest.mark.parametrize("estimator", [StoufferCombinationTest, FisherCombinationTest])
def test_combination_permutation_accepts_weights_for_both_estimators(estimator):
    """Fisher takes one scalar weight per observation, not one per permutation."""
    rng = np.random.default_rng(0)
    dataset = Dataset(y=rng.normal(size=(12, 4)), n=np.full((12, 1), 3.0))

    perm = estimator().fit_dataset(dataset).summary().permutation_test(n_perm=100)

    assert np.all(perm.perm_p["fe_p"] > 0)


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


def test_dataset_rejects_group_labels_with_a_second_dimension():
    """One label per observation; a (K, D) array was silently truncated."""
    with pytest.raises(ValueError, match="one group label per observation"):
        Dataset(y=np.arange(6.0), v=np.ones(6), g=np.tile(np.arange(3), (6, 1))[:, :2])


def test_dataset_accepts_group_labels_alongside_a_dataframe():
    """`g or "g"` called bool() on the array and raised."""
    frame = pd.DataFrame({"y": np.arange(6.0), "v": np.ones(6)})
    dataset = Dataset(data=frame, g=np.array([0, 0, 1, 1, 2, 2]))
    assert np.array_equal(np.ravel(dataset.g), [0, 0, 1, 1, 2, 2])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("weight_scheme", ["individual", "rescale", "collapse"])
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

    integers = np.repeat(np.arange(6), 2)
    equivalent = DerSimonianLaird(weight_scheme=weight_scheme).fit(
        y=y, v=np.full((12, 1), 0.2), X=np.ones((12, 1)), g=integers
    )
    assert np.allclose(fitted.params_["fe_params"], equivalent.params_["fe_params"])


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


def _dependent_data(rng, n_groups=10, n_per_group=3, n_datasets=4, rho_sd=1.0):
    """Build data where estimates within a group share a common offset."""
    n_estimates = n_groups * n_per_group
    shared = rng.normal(0, rho_sd, size=(n_groups, n_datasets))
    noise = rng.normal(0, 0.25, size=(n_estimates, n_datasets))
    y = np.repeat(shared, n_per_group, axis=0) + noise
    v = np.abs(rng.normal(0, 0.2, size=(n_estimates, n_datasets))) + 0.5
    X = np.ones((n_estimates, 1))
    groups = np.repeat(np.arange(n_groups), n_per_group)
    return y, v, X, groups


# -----------------------------------------------------------------------------
# Generic group primitives
# -----------------------------------------------------------------------------


def test_encode_groups_preserves_first_occurrence_order():
    """Arbitrary labels should map stably without imposing sorted order."""
    codes, labels = encode_groups(["later", "later", "first", "last", "first"])

    assert np.array_equal(codes, [0, 0, 1, 2, 1])
    assert np.array_equal(labels, ["later", "first", "last"])


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


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("method", ["CR0", "CR2"])
def test_cluster_robust_cov_shared_weights_take_the_fast_path(method):
    """Deduplicating identical weight columns must not change the answer.

    (I - H_j) depends on the weights and the design but never on y, so
    identical weight columns give identical eigendecompositions and only the
    first needs solving.

    Checked against a per-column reference rather than bitwise: numpy's own
    reductions are not bit-stable across batch shapes, so a ``(p, p, d)``
    contraction and ``d`` separate ``(p, p, 1)`` ones disagree in the last
    place regardless of this optimization -- CR0, which never touches the
    deduplicated code, disagrees by the same ~1e-17. Bitwise stability of the
    fast path at a *fixed* shape is covered separately.
    """
    rng = np.random.RandomState(3)
    n_estimates, n_datasets = 16, 6
    y = rng.randn(n_estimates, n_datasets)
    v = np.abs(rng.randn(n_estimates, 1)) + 0.5
    X = np.c_[np.ones(n_estimates), rng.randn(n_estimates)]
    groups = np.repeat(np.arange(8), 2)

    beta, model_cov = weighted_least_squares(y, v, X, return_cov=True)
    fast = cluster_robust_cov(
        y, v, X, beta, groups, model_cov=model_cov, method=method, small_sample=False
    )

    # Same weights, but perturbed so the columns are no longer detected as
    # identical -- forces the per-column path for the same numbers.
    per_column = np.repeat(v, n_datasets, axis=1)
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
    again = cluster_robust_cov(
        y, v, X, beta, groups, model_cov=model_cov, method=method, small_sample=False
    )
    assert np.array_equal(fast, again)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("method", ["CR0", "CR2"])
def test_cluster_robust_cov_accepts_one_shared_variance_column(method):
    """A single column of v applies to every parallel dataset, for both methods."""
    rng = np.random.RandomState(0)
    n_estimates, n_datasets = 12, 5
    y = rng.randn(n_estimates, n_datasets)
    v = np.abs(rng.randn(n_estimates, 1)) + 0.5
    X = np.ones((n_estimates, 1))
    groups = np.repeat(np.arange(6), 2)

    beta, model_cov = weighted_least_squares(y, v, X, return_cov=True)
    shared = cluster_robust_cov(
        y, v, X, beta, groups, model_cov=model_cov, method=method, small_sample=False
    )
    expanded = cluster_robust_cov(
        y,
        np.repeat(v, n_datasets, axis=1),
        X,
        beta,
        groups,
        model_cov=model_cov,
        method=method,
        small_sample=False,
    )

    assert shared.shape == (1, 1, n_datasets)
    assert np.allclose(shared, expanded)


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


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("weight_scheme", ["rescale", "collapse"])
def test_tau2_interval_and_heterogeneity_use_the_same_units_as_tau2(weight_scheme):
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

    results = DerSimonianLaird(weight_scheme=weight_scheme).fit_dataset(dataset).summary()
    re_stats = results.get_re_stats()
    tau2 = float(np.ravel(re_stats["tau^2"])[0])
    lower = float(np.ravel(re_stats["ci_l"])[0])
    upper = float(np.ravel(re_stats["ci_u"])[0])
    assert lower <= tau2 <= upper

    # Q must be referred to the number of independent groups, not rows.
    heterogeneity = results.get_heterogeneity_stats()
    raw = DerSimonianLaird().fit_dataset(dataset).summary().get_heterogeneity_stats()
    assert float(np.ravel(heterogeneity["Q"])[0]) < float(np.ravel(raw["Q"])[0])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("estimator", [WeightedLeastSquares, DerSimonianLaird, Hedges])
def test_dof_matches_fe_params_when_v_is_a_shared_column(estimator):
    """A single column of v still has to yield one dof per parallel dataset."""
    y = np.repeat(np.arange(12.0)[:, None] / 10, 4, axis=1) + np.array([0, 0.1, 0.2, 0.3])
    fitted = estimator().fit(
        y=y,
        X=np.ones((12, 1)),
        v=np.full((12, 1), 0.1),
        g=np.repeat(np.arange(6), 2),
    )
    assert fitted.params_["dof"].shape == fitted.params_["fe_params"].shape


@pytest.mark.parametrize("estimator", [Hedges, DerSimonianLaird, VarianceBasedLikelihoodEstimator])
def test_group_collapse_rejects_a_saturated_collapsed_design(estimator):
    """Collapsing can saturate a design that was identified before it.

    Nine rows and three predictors is unremarkable, but three groups leave
    m == p, where the moment estimators divide by zero and report tau^2 = inf
    with zero standard errors.
    """
    groups = np.repeat([0, 1, 2], 3)
    X = np.c_[np.ones(9), np.repeat([0.0, 1.0, 2.0], 3), np.repeat([0.0, 0.0, 1.0], 3)]
    y = np.random.RandomState(0).randn(9, 1)

    with pytest.raises(ValueError, match="number of groups must exceed"):
        estimator(weight_scheme="collapse").fit(y=y, v=np.full((9, 1), 0.1), X=X, g=groups)


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
    _, analysis_v, _ = results._analysis_arrays()

    _, collapsed_n, _ = collapse_groups_by_n(y, n, results.dataset.X, groups, rho=DEFAULT_RHO)
    sigma2 = np.asarray(results.estimator.params_["sigma2"], dtype=float)
    assert np.allclose(analysis_v, sigma2 / collapsed_n)


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


def test_weighted_intercept_cr2_rejects_invalid_signs():
    """Only one -1/+1 sign per independent input is accepted."""
    statistics = weighted_intercept_cr2_sufficient_statistics(
        np.ones((3, 2)),
        np.ones(3),
    )

    with pytest.raises(ValueError, match="-1 and 1"):
        weighted_intercept_cr2(np.array([[1.0, 0.0, -1.0]]), statistics)


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


# -----------------------------------------------------------------------------
# cluster_robust_cov
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_cluster_robust_cov_matches_explicit_reference():
    """Check the vectorized sandwich against a plain per-dataset loop."""
    rng = np.random.RandomState(0)
    n_estimates, n_datasets, n_preds = 12, 3, 2
    y = rng.randn(n_estimates, n_datasets)
    v = np.abs(rng.randn(n_estimates, n_datasets)) + 0.5
    X = np.c_[np.ones(n_estimates), rng.randn(n_estimates)]
    groups = np.repeat(np.arange(4), 3)

    beta = weighted_least_squares(y, v, X)
    robust = cluster_robust_cov(y, v, X, beta, groups, small_sample=False, method="CR0")

    assert robust.shape == (n_preds, n_preds, n_datasets)

    for i_dataset in range(n_datasets):
        weights = 1.0 / v[:, i_dataset]
        bread = np.linalg.pinv(X.T @ np.diag(weights) @ X)
        resid = y[:, i_dataset] - X @ beta[:, i_dataset]
        meat = np.zeros((n_preds, n_preds))
        for group in np.unique(groups):
            members = groups == group
            score = (X[members].T @ np.diag(weights[members]) @ resid[members]).reshape(-1, 1)
            meat += score @ score.T
        expected = bread @ meat @ bread
        assert np.allclose(robust[:, :, i_dataset], expected)


@pytest.mark.parametrize(
    "groups",
    [
        np.repeat(np.arange(4), 3),
        np.repeat([3, 1, 4, 2], [2, 4, 3, 3]),
        np.array([0, 1, 2, 3] * 3),
    ],
    ids=["equal-contiguous", "unequal-contiguous", "noncontiguous"],
)
@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_cluster_robust_cov_group_layouts_match_explicit_reference(groups):
    """Optimized and fallback grouping paths must produce the same sandwich."""
    rng = np.random.RandomState(3)
    n_estimates, n_datasets, n_preds = 12, 5, 3
    y = rng.randn(n_estimates, n_datasets)
    v = np.abs(rng.randn(n_estimates, n_datasets)) + 0.5
    X = np.c_[np.ones(n_estimates), rng.randn(n_estimates, n_preds - 1)]
    beta = weighted_least_squares(y, v, X)

    robust = cluster_robust_cov(y, v, X, beta, groups, small_sample=False, method="CR0")

    for i_dataset in range(n_datasets):
        weights = 1.0 / v[:, i_dataset]
        bread = np.linalg.pinv(X.T @ np.diag(weights) @ X)
        resid = y[:, i_dataset] - X @ beta[:, i_dataset]
        meat = np.zeros((n_preds, n_preds))
        for group in np.unique(groups):
            members = groups == group
            score = (X[members] * (weights[members] * resid[members])[:, None]).sum(0)
            meat += np.outer(score, score)
        assert np.allclose(robust[:, :, i_dataset], bread @ meat @ bread)


def test_cluster_robust_cov_small_sample_scaling():
    """The small-sample correction should scale the matrix by m / (m - p)."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng)
    beta = weighted_least_squares(y, v, X)

    uncorrected = cluster_robust_cov(y, v, X, beta, groups, small_sample=False, method="CR0")
    corrected = cluster_robust_cov(y, v, X, beta, groups, small_sample=True, method="CR0")

    n_groups = np.unique(groups).size
    n_preds = X.shape[1]
    assert np.allclose(corrected, uncorrected * n_groups / (n_groups - n_preds))


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_cluster_robust_cov_one_group_per_estimate_is_heteroskedasticity_robust():
    """With singleton groups the sandwich reduces to the HC0 estimator."""
    rng = np.random.RandomState(0)
    n_estimates = 8
    y = rng.randn(n_estimates, 1)
    v = np.abs(rng.randn(n_estimates, 1)) + 0.5
    X = np.ones((n_estimates, 1))
    beta = weighted_least_squares(y, v, X)

    robust = cluster_robust_cov(
        y, v, X, beta, np.arange(n_estimates), small_sample=False, method="CR0"
    )

    weights = 1.0 / v[:, 0]
    bread = np.linalg.pinv(X.T @ np.diag(weights) @ X)
    resid = y[:, 0] - X @ beta[:, 0]
    meat = (X * (weights * resid)[:, None]).T @ (X * (weights * resid)[:, None])
    assert np.allclose(robust[:, :, 0], bread @ meat @ bread)


def test_cluster_robust_cov_wrong_group_length():
    """Group labels must be one per observation."""
    rng = np.random.RandomState(0)
    y, v, X, _ = _dependent_data(rng)
    beta = weighted_least_squares(y, v, X)

    with pytest.raises(ValueError, match="one label per observation"):
        cluster_robust_cov(y, v, X, beta, np.arange(3))


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("method", ["CR0", "CR2"])
@pytest.mark.parametrize("small_sample", [True, False])
def test_cluster_robust_cov_rejects_saturated_group_designs(method, small_sample):
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
    groups = np.repeat([0, 1], 3)  # 2 groups, 2 predictors

    with pytest.raises(ValueError, match="more groups than predictors"):
        cluster_robust_cov(y, v, X, beta, groups, method=method, small_sample=small_sample)


def test_cluster_robust_cov_rejects_a_single_group():
    """One group is the extreme case: nothing to compare against."""
    rng = np.random.RandomState(0)
    y = rng.randn(10, 1)
    v = np.ones((10, 1))
    X = np.ones((10, 1))
    beta = weighted_least_squares(y, v, X)

    with pytest.raises(ValueError, match="more groups than predictors"):
        cluster_robust_cov(y, v, X, beta, np.zeros(10))


def test_cluster_robust_cov_accepts_unsortable_labels():
    """The docstring promises hashable labels, not sortable ones."""
    rng = np.random.RandomState(0)
    y = rng.randn(12, 1)
    v = np.ones((12, 1))
    X = np.ones((12, 1))
    beta = weighted_least_squares(y, v, X)
    mixed = np.array([1, 1, "b", "b", 2, 2, "d", "d", 3, 3, "f", "f"], dtype=object)

    with pytest.warns(UserWarning, match="Cluster-robust"):
        from_mixed = cluster_robust_cov(y, v, X, beta, mixed)
    with pytest.warns(UserWarning, match="Cluster-robust"):
        from_ints = cluster_robust_cov(y, v, X, beta, np.repeat(np.arange(6), 2))

    assert np.allclose(from_mixed, from_ints)


def test_cluster_robust_cov_accepts_one_dimensional_weights():
    """satterthwaite_dof documents (K,) weights, so the sandwich must too."""
    rng = np.random.RandomState(0)
    y = rng.randn(12, 1)
    v = np.abs(rng.randn(12, 1)) + 0.5
    X = np.ones((12, 1))
    beta = weighted_least_squares(y, v, X)
    groups = np.repeat(np.arange(6), 2)

    with pytest.warns(UserWarning, match="Cluster-robust"):
        flat = cluster_robust_cov(y, v, X, beta, groups, w=(1.0 / v).ravel())
    with pytest.warns(UserWarning, match="Cluster-robust"):
        column = cluster_robust_cov(y, v, X, beta, groups, w=1.0 / v)

    assert np.array_equal(flat, column)


def test_cluster_robust_cov_warns_with_few_groups():
    """RVE is anti-conservative with few groups, so warn about it."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng, n_groups=4, n_per_group=3)
    beta = weighted_least_squares(y, v, X)

    with pytest.warns(UserWarning, match="anti-conservative"):
        cluster_robust_cov(y, v, X, beta, groups, method="CR0")


# -----------------------------------------------------------------------------
# Estimators
# -----------------------------------------------------------------------------

WLS_ESTIMATORS = [
    (WeightedLeastSquares, "v"),
    (DerSimonianLaird, "v"),
    (Hedges, "v"),
    (VarianceBasedLikelihoodEstimator, "v"),
    (SampleSizeBasedLikelihoodEstimator, "n"),
]


def _fit_kwargs(second_arg, y, v, X, rng):
    """Build fit kwargs, since one estimator takes n instead of v."""
    if second_arg == "v":
        return {"y": y, "v": v, "X": X}
    # Sample sizes must vary for the sample-size-based estimator.
    sample_sizes = rng.randint(20, 200, size=y.shape).astype(float)
    return {"y": y, "n": sample_sizes, "X": X}


@pytest.mark.parametrize(("estimator", "second_arg"), WLS_ESTIMATORS)
def test_estimators_default_is_unchanged(estimator, second_arg):
    """Omitting g must reproduce the previous behaviour exactly."""
    rng = np.random.RandomState(0)
    y, v, X, _ = _dependent_data(rng)
    kwargs = _fit_kwargs(second_arg, y, v, X, np.random.RandomState(1))

    fitted = estimator().fit(**kwargs)

    assert fitted.n_groups_ is None
    assert "n_groups" not in fitted.params_


@pytest.mark.parametrize(("estimator", "second_arg"), WLS_ESTIMATORS)
def test_estimators_groups_change_only_inference(estimator, second_arg):
    """Group labels must move the covariance but not the point estimates."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng)
    kwargs = _fit_kwargs(second_arg, y, v, X, np.random.RandomState(1))

    naive = estimator().fit(**kwargs)
    robust = estimator().fit(**kwargs, g=groups)

    assert np.allclose(naive.params_["fe_params"], robust.params_["fe_params"])
    assert np.allclose(naive.params_["tau2"], robust.params_["tau2"])
    assert not np.allclose(naive.params_["inv_cov"], robust.params_["inv_cov"])
    assert robust.n_groups_ == np.unique(groups).size


@pytest.mark.parametrize(("estimator", "second_arg"), WLS_ESTIMATORS)
def test_estimators_groups_widen_standard_errors(estimator, second_arg):
    """With dependent estimates, robust SEs should exceed model-based ones."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng, n_groups=12, n_per_group=4)
    kwargs = _fit_kwargs(second_arg, y, v, X, np.random.RandomState(1))

    naive = estimator().fit(**kwargs).summary()
    robust = estimator().fit(**kwargs, g=groups).summary()

    assert np.all(robust.fe_se > naive.fe_se)


def test_estimator_groups_via_dataset():
    """Groups should flow through Dataset and fit_dataset."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng, n_datasets=1)

    dataset = Dataset(y=y, v=v, X=X, g=groups, add_intercept=False)
    assert dataset.g.shape == (y.shape[0], 1)

    fitted = WeightedLeastSquares().fit_dataset(dataset)

    assert fitted.n_groups_ == np.unique(groups).size


def test_dataset_groups_from_dataframe():
    """A 'g' column in a DataFrame should be picked up."""
    dataset = Dataset(
        y=[1.0, 2.0, 3.0, 4.0],
        v=[1.0, 1.0, 1.0, 1.0],
        g=[0, 0, 1, 1],
    )
    frame = dataset.to_df()

    assert "g" in frame.columns
    assert Dataset(data=frame).g.ravel().tolist() == [0, 0, 1, 1]


def test_dataset_groups_in_multi_dataset_dataframe():
    """Group labels must be repeated for every set in a multi-dataset export."""
    groups = np.array([0, 0, 1, 1])
    dataset = Dataset(
        y=np.arange(8.0).reshape(4, 2),
        v=np.ones((4, 2)),
        g=groups,
    )

    frame = dataset.to_df()

    assert frame["g"].tolist() == np.tile(groups, 2).tolist()


def test_meta_regression_accepts_groups():
    """The functional API must route group labels through Dataset to the estimator."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng, n_datasets=1)

    results = meta_regression(
        y=y,
        v=v,
        X=X,
        add_intercept=False,
        method="WLS",
        g=groups,
    )

    assert results.estimator.n_groups_ == np.unique(groups).size


def test_dataset_groups_wrong_length():
    """Group labels must have one entry per estimate."""
    with pytest.raises(ValueError, match="same number of rows"):
        Dataset(y=[1.0, 2.0, 3.0], v=[1.0, 1.0, 1.0], g=[0, 1])


# -----------------------------------------------------------------------------
# Results: t reference
# -----------------------------------------------------------------------------


def test_results_use_t_reference_with_groups():
    """Robust fits should report a t reference with Satterthwaite dof."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng, n_datasets=1)

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
    same_z = np.allclose(robust_stats["z"], naive_stats["z"])
    if same_z:  # only true if the SEs happen to coincide
        assert np.all(robust_stats["p"] >= naive_stats["p"])
    width_robust = robust_stats["ci_u"] - robust_stats["ci_l"]
    width_naive = naive_stats["ci_u"] - naive_stats["ci_l"]
    assert np.all(width_robust > width_naive)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_satterthwaite_dof_collapses_for_unbalanced_group_level_predictors():
    """The whole point of Satterthwaite dof: m - p is blind to imbalance.

    When only a few groups carry the non-zero values of a predictor, that
    coefficient is informed by far fewer than m groups. m - p does not notice,
    which is what makes the naive reference reject far too often.
    """
    m, size = 20, 4
    groups = np.repeat(np.arange(m), size)

    slopes = []
    for n_nonzero in (10, 5, 3, 2):
        x = np.repeat((np.arange(m) < n_nonzero).astype(float), size)
        X = np.c_[np.ones(m * size), x]
        slopes.append(satterthwaite_dof(X, np.ones(m * size), groups)[1, 0])

    # m - p would report 18 for every one of these designs.
    assert np.isclose(slopes[0], m - 2, rtol=0.05)
    assert np.all(np.diff(slopes) < 0)
    assert slopes[-1] < 2.0


def test_satterthwaite_dof_warns_below_the_validated_range():
    """A usable group count can still give dof outside the validated range.

    Tipton (2015) established the approximation only holds its level above a dof
    of roughly four. The number of groups does not reveal when that floor is
    crossed -- here 20 groups produce a dof near two for the slope -- so the
    check has to be on the dof themselves.
    """
    m, size = 20, 4
    groups = np.repeat(np.arange(m), size)
    x = np.repeat((np.arange(m) < 2).astype(float), size)
    X = np.c_[np.ones(m * size), x]

    with pytest.warns(UserWarning, match="degrees of freedom below"):
        dof = satterthwaite_dof(X, np.ones(m * size), groups)

    assert dof[1, 0] < MIN_DOF_FOR_SATTERTHWAITE
    # The intercept is comfortably identified, so the warning names only the slope.
    assert dof[0, 0] > MIN_DOF_FOR_SATTERTHWAITE


def test_satterthwaite_dof_is_quiet_when_the_design_is_balanced():
    """The warning must not fire on a design that is genuinely well identified."""
    m, size = 20, 4
    groups = np.repeat(np.arange(m), size)
    x = np.repeat((np.arange(m) < 10).astype(float), size)
    X = np.c_[np.ones(m * size), x]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dof = satterthwaite_dof(X, np.ones(m * size), groups)

    assert np.all(dof > MIN_DOF_FOR_SATTERTHWAITE)


def test_satterthwaite_dof_warns_when_one_group_determines_a_coefficient():
    """A predictor supported by a single group leaves no CR2 adjustment."""
    m, size = 20, 4
    groups = np.repeat(np.arange(m), size)
    x = np.repeat((np.arange(m) < 1).astype(float), size)
    X = np.c_[np.ones(m * size), x]

    with pytest.warns(UserWarning, match="full leverage"):
        dof = satterthwaite_dof(X, np.ones(m * size), groups)

    # Floored rather than reported as a comfortable m - p = 18.
    assert dof[1, 0] == 1.0


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


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_group_weighted_heterogeneity_matches_collapsed_reference():
    """Q and its degrees of freedom must use independent group aggregates."""
    y = np.array([[1.0], [3.0], [8.0], [10.0], [20.0]])
    v = np.array([[1.0], [4.0], [2.0], [8.0], [5.0]])
    X = np.ones((5, 1))
    groups = np.array([0, 0, 1, 1, 2])
    rho = 0.4
    dataset = Dataset(y=y, v=v, X=X, g=groups, add_intercept=False)

    observed = (
        WeightedLeastSquares(weight_scheme="collapse", rho=rho)
        .fit_dataset(dataset)
        .summary()
        .get_heterogeneity_stats()
    )

    collapsed_y, collapsed_v, collapsed_X = collapse_groups(y, v, X, groups, rho=rho)
    expected = (
        WeightedLeastSquares()
        .fit_dataset(
            Dataset(
                y=collapsed_y,
                v=collapsed_v,
                X=collapsed_X,
                add_intercept=False,
            )
        )
        .summary()
        .get_heterogeneity_stats()
    )

    for key in ("Q", "p(Q)", "I^2", "H"):
        assert np.allclose(observed[key], expected[key])


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


def test_brown_reduces_to_fisher_with_singleton_groups():
    """One group per estimate means no dependence, so Fisher's result stands."""
    rng = np.random.RandomState(0)
    n_estimates, n_datasets = 10, 50
    z = rng.randn(n_estimates, n_datasets)
    groups = np.tile(np.arange(n_estimates)[:, None], (1, n_datasets))

    plain = FisherCombinationTest().fit(z).params_["p"]
    blocked = FisherCombinationTest().fit(z, g=groups).params_["p"]

    assert np.allclose(plain, blocked)


def test_brown_equal_sized_groups_reduce_to_fisher_with_zero_correlation():
    """Equal-sized independent groups must recover Fisher's method exactly."""
    rng = np.random.RandomState(0)
    n_estimates, n_datasets = 10, 50
    z = rng.randn(n_estimates, n_datasets)
    groups = np.tile(np.repeat(np.arange(5), 2)[:, None], (1, n_datasets))

    plain = FisherCombinationTest().fit(z).params_["p"]
    blocked = FisherCombinationTest().fit(z, g=groups, corr=np.eye(n_estimates)).params_["p"]

    assert np.allclose(plain, blocked)


def test_brown_accepts_one_dimensional_groups():
    """Direct Fisher calls should accept the documented one-dimensional labels."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 20)
    groups = np.repeat(np.arange(3), 2)

    one_dimensional = FisherCombinationTest().fit(z, g=groups).params_["p"]
    two_dimensional = FisherCombinationTest().fit(z, g=groups[:, None]).params_["p"]

    assert np.allclose(one_dimensional, two_dimensional)


def test_fisher_preserves_group_and_correlation_positional_arguments():
    """The new generic weight must not reinterpret released positional calls."""
    z = np.arange(24, dtype=float).reshape(6, 4) / 10
    groups = np.repeat(np.arange(3), 2)
    corr = np.eye(6)

    expected = FisherCombinationTest().fit(z, g=groups, corr=corr).params_["p"]
    observed = FisherCombinationTest().fit(z, groups, corr).params_["p"]

    assert np.allclose(observed, expected)


@pytest.mark.parametrize("mode", ["directed", "concordant"])
def test_brown_perfect_duplicates_have_unit_group_weight(mode):
    """Repeating a perfectly dependent input must not increase its group's weight."""
    rng = np.random.RandomState(4)
    n_datasets = 100
    collapsed = rng.randn(2, n_datasets)
    n_duplicates = 4
    expanded = np.vstack([collapsed[[0]], np.repeat(collapsed[[1]], n_duplicates, axis=0)])
    groups = np.r_[0, np.repeat(1, n_duplicates)]

    corr = np.eye(expanded.shape[0])
    corr[1:, 1:] = 1.0

    expected = FisherCombinationTest(mode=mode).fit(collapsed).params_["p"]
    actual = FisherCombinationTest(mode=mode).fit(expanded, g=groups, corr=corr).params_["p"]

    assert np.allclose(actual, expected)


def test_brown_permutation_flips_groups_as_units():
    """Exact Brown inference is invariant to repeated rows from one group."""
    collapsed = np.array([[1.5], [0.25], [-0.5]])
    collapsed_result = (
        FisherCombinationTest()
        .fit_dataset(Dataset(y=collapsed, g=np.arange(3)), corr=np.eye(3))
        .summary()
    )

    expanded = np.vstack([np.repeat(collapsed[[0]], 4, axis=0), collapsed[1:]])
    groups = np.array([0, 0, 0, 0, 1, 2])
    corr = np.eye(6)
    corr[:4, :4] = 1.0
    expanded_result = (
        FisherCombinationTest().fit_dataset(Dataset(y=expanded, g=groups), corr=corr).summary()
    )
    assert np.array_equal(expanded_result.estimator.corr_, corr)

    collapsed_perm = collapsed_result.permutation_test(n_perm=20)
    expanded_perm = expanded_result.permutation_test(n_perm=20)
    assert collapsed_perm.exact
    assert expanded_perm.exact
    assert collapsed_perm.n_perm == expanded_perm.n_perm == 8
    assert np.allclose(collapsed_perm.perm_p["fe_p"], expanded_perm.perm_p["fe_p"])


def test_brown_applies_one_external_weight_per_group():
    """Generic group weights are allocated across rows without multiplicity."""
    collapsed = np.array([[1.5, -0.5], [0.25, 2.0]])
    expected = FisherCombinationTest().fit(
        collapsed,
        w=np.array([2.0, 4.0]),
    )

    expanded = np.vstack([np.repeat(collapsed[[0]], 3, axis=0), collapsed[[1]]])
    groups = np.array([0, 0, 0, 1])
    corr = np.eye(4)
    corr[:3, :3] = 1.0
    observed = FisherCombinationTest().fit(
        expanded,
        w=np.array([2.0, 2.0, 2.0, 4.0]),
        g=groups,
        corr=corr,
    )

    assert np.allclose(observed.params_["p"], expected.params_["p"])
    assert np.allclose(observed.params_["z"], expected.params_["z"])


def test_brown_rejects_feature_specific_groups():
    """Groups describe rows and therefore must not vary across features."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 20)
    groups = np.tile(np.repeat(np.arange(3), 2)[:, None], (1, z.shape[1]))
    groups[0, -1] = 99

    with pytest.raises(ValueError, match="same for every feature"):
        FisherCombinationTest().fit(z, g=groups)


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


def test_brown_warns_on_corr_without_groups():
    """A correlation matrix is meaningless without group labels."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 10)

    with pytest.warns(UserWarning, match="without groups"):
        FisherCombinationTest().fit(z, corr=np.eye(6))


def test_brown_rejects_mismatched_corr():
    """The correlation matrix must match the number of estimates."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 10)
    groups = np.tile(np.repeat(np.arange(3), 2)[:, None], (1, 10))

    with pytest.raises(ValueError, match="same length as the correlation matrix"):
        FisherCombinationTest().fit(z, g=groups, corr=np.eye(4))


def test_brown_rejects_nonsquare_corr():
    """Correlation matrices must have one row and column per estimate."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 10)
    groups = np.repeat(np.arange(3), 2)

    with pytest.raises(ValueError, match="shape.*6.*6"):
        FisherCombinationTest().fit(z, g=groups, corr=np.ones((6, 2)))


def test_brown_needs_multiple_features_without_corr():
    """Without a correlation matrix, rho is estimated across features."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 1)
    groups = np.tile(np.repeat(np.arange(3), 2)[:, None], (1, 1))

    with pytest.raises(ValueError, match="number of features"):
        FisherCombinationTest().fit(z, g=groups)


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
    groups = np.arange(n_estimates)

    beta, model_cov = weighted_least_squares(y, v, X, 0.0, return_cov=True)
    robust = cluster_robust_cov(
        y, v, X, beta, groups, model_cov=model_cov, method="CR2", small_sample=False
    )

    # HC2: the sandwich with each residual inflated by 1 / sqrt(1 - h_ii).
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    resid = (y - X @ beta).ravel() / np.sqrt(1.0 - np.diag(hat))
    meat = X.T @ np.diag(resid**2) @ X
    expected = np.linalg.inv(X.T @ X) @ meat @ np.linalg.inv(X.T @ X)

    assert np.allclose(robust[:, :, 0], expected)


def test_cr2_is_larger_than_cr0_under_leverage():
    """Undoing residual shrinkage can only widen the sandwich."""
    rng = np.random.default_rng(5)
    y, v, X, groups = _dependent_data(rng, n_groups=6, n_per_group=4)[:4]

    beta, model_cov = weighted_least_squares(y, v, X, 0.0, return_cov=True)
    kwargs = dict(model_cov=model_cov, small_sample=False)
    cr0 = cluster_robust_cov(y, v, X, beta, groups, method="CR0", **kwargs)
    cr2 = cluster_robust_cov(y, v, X, beta, groups, method="CR2", **kwargs)

    assert np.all(np.diagonal(cr2, axis1=0, axis2=1) >= np.diagonal(cr0, axis1=0, axis2=1))


def test_cluster_robust_cov_rejects_unknown_method():
    """An unrecognised residual adjustment should fail loudly."""
    rng = np.random.default_rng(0)
    y, v, X, groups = _dependent_data(rng)[:4]
    beta = weighted_least_squares(y, v, X, 0.0)

    with pytest.raises(ValueError, match="Invalid method"):
        cluster_robust_cov(y, v, X, beta, groups, method="CR9")


def test_group_weights_equalize_group_totals():
    """Every group should end up with the mean of its members' weights."""
    v = np.array([[1.0], [1.0], [1.0], [2.0]])
    groups = np.array([0, 0, 0, 1])

    w = group_weights(v, groups)

    assert np.isclose(w[:3].sum(), 1.0)  # three estimates of variance 1
    assert np.isclose(w[3].sum(), 0.5)  # one estimate of variance 2


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
def test_group_weighting_matches_explicit_group_collapse():
    """Group mode must fit the algebraically equivalent one-row-per-group model."""
    y = np.array([[1.0], [3.0], [8.0], [10.0], [20.0]])
    v = np.array([[1.0], [4.0], [2.0], [8.0], [5.0]])
    X = np.ones((5, 1))
    groups = np.array([0, 0, 1, 1, 2])
    rho = 0.4

    collapsed_y, collapsed_v, collapsed_X = collapse_groups(y, v, X, groups, rho=rho)
    expected = WeightedLeastSquares().fit(
        collapsed_y,
        collapsed_X,
        v=collapsed_v,
        g=np.arange(3),
    )
    observed = WeightedLeastSquares(
        weight_scheme="collapse",
        rho=rho,
    ).fit(y, X, v=v, g=groups)

    assert np.allclose(observed.params_["fe_params"], expected.params_["fe_params"])
    assert np.allclose(observed.params_["inv_cov"], expected.params_["inv_cov"])
    assert observed.n_groups_ == 3


@pytest.mark.parametrize(
    "estimator",
    [DerSimonianLaird, Hedges, VarianceBasedLikelihoodEstimator],
)
@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_variance_estimators_group_mode_matches_explicit_collapse(estimator):
    """Every variance-based algorithm must use the same grouped inputs."""
    y = np.array([[1.0], [3.0], [6.0], [10.0], [20.0], [22.0]])
    v = np.array([[1.0], [4.0], [2.0], [8.0], [5.0], [3.0]])
    X = np.ones((6, 1))
    groups = np.array([0, 0, 1, 1, 2, 2])
    rho = 0.4
    collapsed_y, collapsed_v, collapsed_X = collapse_groups(y, v, X, groups, rho=rho)

    expected = estimator().fit(
        y=collapsed_y,
        v=collapsed_v,
        X=collapsed_X,
        g=np.arange(3),
    )
    observed = estimator(weight_scheme="collapse", rho=rho).fit(
        y=y,
        v=v,
        X=X,
        g=groups,
    )

    assert np.allclose(observed.params_["fe_params"], expected.params_["fe_params"])
    assert np.allclose(observed.params_["tau2"], expected.params_["tau2"])
    assert np.allclose(observed.params_["inv_cov"], expected.params_["inv_cov"])


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
        y=collapsed_y,
        n=collapsed_n,
        X=collapsed_X,
        g=np.arange(3),
    )
    observed = SampleSizeBasedLikelihoodEstimator(weight_scheme="collapse").fit(
        y=y,
        n=n,
        X=X,
        g=groups,
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
    a factor of five for four uncorrelated estimates per group. Images from
    one study share subjects but measure different contrasts, so rho is well
    below one and the bias is real.
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

    # The endpoints are the two formulas being interpolated between.
    _, raw_n, _ = collapse_groups_by_n(y, n, X, groups, rho=1.0)
    assert np.allclose(effective[1.0], raw_n)
    assert np.allclose(effective[0.0], group_size * raw_n)

    # And the closed form, checked directly.
    for rho in (0.0, 0.5, 1.0):
        expected = group_size * raw_n / (1.0 + rho * (group_size - 1))
        assert np.allclose(effective[rho], expected)


def test_estimator_rejects_unknown_weight_scheme():
    """An unrecognised weighting scheme should fail at construction."""
    for estimator in (
        WeightedLeastSquares,
        DerSimonianLaird,
        Hedges,
        VarianceBasedLikelihoodEstimator,
        SampleSizeBasedLikelihoodEstimator,
    ):
        with pytest.raises(ValueError, match="Invalid weight_scheme"):
            estimator(weight_scheme="nonsense")


def test_estimate_null_correlation_ignores_shared_signal():
    """Independent estimates must not look correlated just because they agree."""
    rng = np.random.default_rng(0)
    n_estimates, n_datasets = 10, 4000
    signal = rng.standard_normal(n_datasets) * 2.0
    y = np.array([signal + rng.standard_normal(n_datasets) for _ in range(n_estimates)])

    off_diagonal = ~np.eye(n_estimates, dtype=bool)
    naive = np.corrcoef(y)[off_diagonal].mean()
    estimated = estimate_null_correlation(y)[off_diagonal].mean()

    assert naive > 0.5  # the shared signal dominates the raw correlation
    assert abs(estimated) < 0.05  # the truth is zero


def test_estimate_null_correlation_recovers_within_group_dependence():
    """Genuine shared noise should still be detected."""
    rng = np.random.default_rng(1)
    n_estimates, n_datasets, rho = 20, 4000, 0.8
    signal = rng.standard_normal(n_datasets) * 2.0
    shared = rng.standard_normal(n_datasets)
    y = np.array([signal + rng.standard_normal(n_datasets) for _ in range(n_estimates)])
    for i in range(4):
        y[i] = signal + np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(n_datasets)

    corr = estimate_null_correlation(y)
    within = corr[:4, :4][~np.eye(4, dtype=bool)].mean()
    between = corr[4:, 4:][~np.eye(n_estimates - 4, dtype=bool)].mean()

    assert 0.6 < within < 0.9
    assert abs(between) < 0.1


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


def test_collapse_groups_by_n_at_rho_one_returns_the_raw_n():
    """The deleted ``collapse_groups_by_n`` special case is just rho=1.

    Counting a group's ``n`` once -- the old separate function -- is what the
    effective sample size reduces to when the rows are assumed perfectly
    correlated, so it needs no code of its own.
    """
    y = np.array([[1.0], [3.0], [10.0]])
    n = np.array([[20.0], [20.0], [80.0]])
    X = np.ones((3, 1))
    groups = np.array([0, 0, 1])

    collapsed_y, collapsed_n, collapsed_X = collapse_groups_by_n(y, n, X, groups, rho=1.0)

    assert np.allclose(collapsed_y.ravel(), [2.0, 10.0])
    assert np.allclose(collapsed_n.ravel(), [20.0, 80.0])
    assert np.allclose(collapsed_X, np.ones((2, 1)))


def test_collapse_groups_by_n_handles_unequal_n_within_a_group():
    """Unequal within-group ``n`` is now well defined rather than rejected.

    The removed function required one ``n`` per group and raised otherwise. The
    effective-sample-size form has no such restriction, so a group whose rows
    report different ``n`` collapses to a sensible value instead of failing.
    """
    y = np.ones((3, 1))
    n = np.array([[20.0], [80.0], [40.0]])
    X = np.ones((3, 1))
    groups = np.array([0, 0, 1])

    _, collapsed_n, _ = collapse_groups_by_n(y, n, X, groups, rho=1.0)

    # rho=1: n_eff is the harmonic-style combination, between the two inputs.
    assert 20.0 < collapsed_n[0, 0] < 80.0
    assert np.isclose(collapsed_n[1, 0], 40.0)


def test_collapse_groups_rejects_out_of_range_rho():
    """The assumed within-cluster correlation must lie in [0, 1]."""
    y = np.ones((2, 1))
    for collapse, second in ((collapse_groups, y), (collapse_groups_by_n, y * 10)):
        with pytest.raises(ValueError, match="rho must lie"):
            collapse(y, second, np.ones((2, 1)), np.array([0, 1]), rho=1.5)


@pytest.mark.parametrize("estimator", [DerSimonianLaird, Hedges, VarianceBasedLikelihoodEstimator])
def test_variance_components_are_insensitive_to_duplication(estimator):
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

    single = estimator(weight_scheme="rescale")
    single.fit_dataset(Dataset(y=y, v=v, g=groups))

    # Group 0 now contributes four identical estimates instead of one.
    dupe_idx = np.r_[np.zeros(4, dtype=int), np.arange(1, n_estimates)]
    duped = estimator(weight_scheme="rescale")
    duped.fit_dataset(Dataset(y=y[dupe_idx], v=v[dupe_idx], g=groups[dupe_idx]))

    assert np.allclose(np.mean(single.summary().tau2), np.mean(duped.summary().tau2), rtol=0.05)


@pytest.mark.parametrize("n_estimates", [6, 8, 20])
@pytest.mark.parametrize("block_size", [2, 4, 5])
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.8, 1.0])
def test_undo_centering_shrinkage_is_exact(n_estimates, block_size, rho):
    """Centering is a known linear map, so it inverts exactly.

    Builds the true correlation matrix, applies the centering map analytically,
    and checks that the inverse recovers the correlation it started from.
    """
    R = np.eye(n_estimates)
    R[:block_size, :block_size] = rho + (1 - rho) * np.eye(block_size)
    C = np.eye(n_estimates) - np.ones((n_estimates, n_estimates)) / n_estimates
    shrunk = C @ R @ C
    # Normalize to a correlation matrix, as np.corrcoef would.
    scale = np.sqrt(np.diag(shrunk))
    shrunk = shrunk / np.outer(scale, scale)

    groups = np.array([0] * block_size + list(range(1, n_estimates - block_size + 1)))
    recovered = undo_centering_shrinkage(shrunk, groups)

    off_diagonal = ~np.eye(block_size, dtype=bool)
    assert np.allclose(recovered[:block_size, :block_size][off_diagonal], rho, atol=1e-8)


def test_undo_centering_shrinkage_handles_several_blocks():
    """The blocks share a grand mean, which the fixed point has to resolve."""
    n_estimates = 20
    blocks = [(range(0, 6), 0.7), (range(6, 10), 0.2), (range(10, 13), 0.9)]
    R = np.eye(n_estimates)
    for members, rho in blocks:
        for i in members:
            for j in members:
                if i != j:
                    R[i, j] = rho

    C = np.eye(n_estimates) - np.ones((n_estimates, n_estimates)) / n_estimates
    shrunk = C @ R @ C
    scale = np.sqrt(np.diag(shrunk))
    shrunk = shrunk / np.outer(scale, scale)

    groups = np.zeros(n_estimates, dtype=int)
    for label, (members, _) in enumerate(blocks):
        groups[list(members)] = label
    groups[13:] = np.arange(len(blocks), len(blocks) + n_estimates - 13)

    recovered = undo_centering_shrinkage(shrunk, groups)
    for members, rho in blocks:
        members = list(members)
        block = recovered[np.ix_(members, members)]
        off = ~np.eye(len(members), dtype=bool)
        assert np.allclose(block[off], rho, atol=1e-6)


def test_estimate_null_correlation_with_groups_beats_the_generic_correction():
    """With few estimates the generic rescaling badly understates dependence."""
    rng = np.random.default_rng(0)
    n_estimates, n_datasets, block_size, rho = 8, 20000, 4, 0.8
    signal = rng.standard_normal(n_datasets) * 2.0
    shared = rng.standard_normal(n_datasets)
    y = np.array([signal + rng.standard_normal(n_datasets) for _ in range(n_estimates)])
    for i in range(block_size):
        y[i] = signal + np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(n_datasets)

    groups = np.array([0] * block_size + list(range(1, n_estimates - block_size + 1)))
    off = ~np.eye(block_size, dtype=bool)
    generic = estimate_null_correlation(y)[:block_size, :block_size][off].mean()
    grouped = estimate_null_correlation(y, groups=groups)[:block_size, :block_size][off].mean()

    assert abs(grouped - rho) < 0.05
    assert generic < rho - 0.15
