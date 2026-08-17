"""Tests for dependent-estimate support (cluster-robust variance, Brown's method)."""

import numpy as np
import pytest

from pymare import Dataset, meta_regression
from pymare.estimators import (
    DerSimonianLaird,
    FisherCombinationTest,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)
from pymare.stats import (
    cluster_robust_cov,
    cluster_weights,
    collapse_clusters,
    collapse_clusters_by_n,
    estimate_null_correlation,
    undo_centering_shrinkage,
    weighted_least_squares,
)


def _dependent_data(rng, n_groups=10, n_per_group=3, n_datasets=4, rho_sd=1.0):
    """Build data where estimates within a group share a common offset."""
    n_studies = n_groups * n_per_group
    shared = rng.normal(0, rho_sd, size=(n_groups, n_datasets))
    noise = rng.normal(0, 0.25, size=(n_studies, n_datasets))
    y = np.repeat(shared, n_per_group, axis=0) + noise
    v = np.abs(rng.normal(0, 0.2, size=(n_studies, n_datasets))) + 0.5
    X = np.ones((n_studies, 1))
    groups = np.repeat(np.arange(n_groups), n_per_group)
    return y, v, X, groups


# -----------------------------------------------------------------------------
# cluster_robust_cov
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_cluster_robust_cov_matches_explicit_reference():
    """Check the vectorized sandwich against a plain per-dataset loop."""
    rng = np.random.RandomState(0)
    n_studies, n_datasets, n_preds = 12, 3, 2
    y = rng.randn(n_studies, n_datasets)
    v = np.abs(rng.randn(n_studies, n_datasets)) + 0.5
    X = np.c_[np.ones(n_studies), rng.randn(n_studies)]
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
    n_studies, n_datasets, n_preds = 12, 5, 3
    y = rng.randn(n_studies, n_datasets)
    v = np.abs(rng.randn(n_studies, n_datasets)) + 0.5
    X = np.c_[np.ones(n_studies), rng.randn(n_studies, n_preds - 1)]
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
def test_cluster_robust_cov_one_group_per_study_is_heteroskedasticity_robust():
    """With singleton groups the sandwich reduces to the HC0 estimator."""
    rng = np.random.RandomState(0)
    n_studies = 8
    y = rng.randn(n_studies, 1)
    v = np.abs(rng.randn(n_studies, 1)) + 0.5
    X = np.ones((n_studies, 1))
    beta = weighted_least_squares(y, v, X)

    robust = cluster_robust_cov(
        y, v, X, beta, np.arange(n_studies), small_sample=False, method="CR0"
    )

    weights = 1.0 / v[:, 0]
    bread = np.linalg.pinv(X.T @ np.diag(weights) @ X)
    resid = y[:, 0] - X @ beta[:, 0]
    meat = (X * (weights * resid)[:, None]).T @ (X * (weights * resid)[:, None])
    assert np.allclose(robust[:, :, 0], bread @ meat @ bread)


def test_cluster_robust_cov_wrong_group_length():
    """Group labels must be one per study."""
    rng = np.random.RandomState(0)
    y, v, X, _ = _dependent_data(rng)
    beta = weighted_least_squares(y, v, X)

    with pytest.raises(ValueError, match="one label per study"):
        cluster_robust_cov(y, v, X, beta, np.arange(3))


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_cluster_robust_cov_too_few_groups_for_correction():
    """The small-sample correction needs more groups than predictors."""
    rng = np.random.RandomState(0)
    n_studies = 6
    y = rng.randn(n_studies, 1)
    v = np.ones((n_studies, 1))
    X = np.c_[np.ones(n_studies), rng.randn(n_studies)]
    beta = weighted_least_squares(y, v, X)
    groups = np.repeat([0, 1], 3)  # 2 groups, 2 predictors

    with pytest.raises(ValueError, match="must exceed the number of predictors"):
        cluster_robust_cov(y, v, X, beta, groups, method="CR0")


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

    assert fitted.n_clusters_ is None
    assert "n_clusters" not in fitted.params_


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
    assert robust.n_clusters_ == np.unique(groups).size


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

    assert fitted.n_clusters_ == np.unique(groups).size


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

    assert results.estimator.n_clusters_ == np.unique(groups).size


def test_dataset_groups_wrong_length():
    """Group labels must have one entry per estimate."""
    with pytest.raises(ValueError, match="same number of rows"):
        Dataset(y=[1.0, 2.0, 3.0], v=[1.0, 1.0, 1.0], g=[0, 1])


# -----------------------------------------------------------------------------
# Results: t reference
# -----------------------------------------------------------------------------


def test_results_use_t_reference_with_groups():
    """Robust fits should report a t reference with m - p degrees of freedom."""
    rng = np.random.RandomState(0)
    y, v, X, groups = _dependent_data(rng, n_datasets=1)

    dataset = Dataset(y=y, v=v, X=X, g=groups, add_intercept=False)
    robust = WeightedLeastSquares().fit_dataset(dataset).summary()
    naive_dataset = Dataset(y=y, v=v, X=X, add_intercept=False)
    naive = WeightedLeastSquares().fit_dataset(naive_dataset).summary()

    n_groups = np.unique(groups).size
    assert robust.fe_dof == n_groups - X.shape[1]
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


# -----------------------------------------------------------------------------
# Brown's method
# -----------------------------------------------------------------------------


def test_brown_reduces_to_fisher_with_singleton_groups():
    """One group per study means no dependence, so Fisher's result stands."""
    rng = np.random.RandomState(0)
    n_studies, n_datasets = 10, 50
    z = rng.randn(n_studies, n_datasets)
    groups = np.tile(np.arange(n_studies)[:, None], (1, n_datasets))

    plain = FisherCombinationTest().fit(z).params_["p"]
    blocked = FisherCombinationTest().fit(z, g=groups).params_["p"]

    assert np.allclose(plain, blocked)


def test_brown_equal_sized_groups_reduce_to_fisher_with_zero_correlation():
    """Equal-sized independent groups must recover Fisher's method exactly."""
    rng = np.random.RandomState(0)
    n_studies, n_datasets = 10, 50
    z = rng.randn(n_studies, n_datasets)
    groups = np.tile(np.repeat(np.arange(5), 2)[:, None], (1, n_datasets))

    plain = FisherCombinationTest().fit(z).params_["p"]
    blocked = FisherCombinationTest().fit(z, g=groups, corr=np.eye(n_studies)).params_["p"]

    assert np.allclose(plain, blocked)


def test_brown_accepts_one_dimensional_groups():
    """Direct Fisher calls should accept the documented one-dimensional labels."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 20)
    groups = np.repeat(np.arange(3), 2)

    one_dimensional = FisherCombinationTest().fit(z, g=groups).params_["p"]
    two_dimensional = FisherCombinationTest().fit(z, g=groups[:, None]).params_["p"]

    assert np.allclose(one_dimensional, two_dimensional)


@pytest.mark.parametrize("mode", ["directed", "concordant"])
def test_brown_perfect_duplicates_have_unit_group_weight(mode):
    """Repeating a perfectly dependent input must not increase its study's weight."""
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


def test_brown_rejects_feature_specific_groups():
    """Groups describe studies and therefore must not vary across features."""
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
    n_studies, n_datasets = 10, 2000
    shared = rng.randn(n_studies // 2, n_datasets)
    z = np.repeat(shared, 2, axis=0) + 0.2 * rng.randn(n_studies, n_datasets)

    groups = np.tile(np.repeat(np.arange(n_studies // 2), 2)[:, None], (1, n_datasets))
    corr = np.eye(n_studies)
    for idx in range(0, n_studies, 2):
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
    """The correlation matrix must match the number of studies."""
    rng = np.random.RandomState(0)
    z = rng.randn(6, 10)
    groups = np.tile(np.repeat(np.arange(3), 2)[:, None], (1, 10))

    with pytest.raises(ValueError, match="same length as the correlation matrix"):
        FisherCombinationTest().fit(z, g=groups, corr=np.eye(4))


def test_brown_rejects_nonsquare_corr():
    """Correlation matrices must have one row and column per study."""
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
    n_studies, n_preds = 30, 3
    X = np.c_[np.ones(n_studies), rng.standard_normal((n_studies, n_preds - 1))]
    y = (X @ np.array([1.0, 0.5, -0.3]))[:, None] + rng.standard_normal((n_studies, 1))
    v = np.ones((n_studies, 1))
    groups = np.arange(n_studies)

    beta, inv_cov = weighted_least_squares(y, v, X, 0.0, return_cov=True)
    robust = cluster_robust_cov(
        y, v, X, beta, groups, inv_cov=inv_cov, method="CR2", small_sample=False
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

    beta, inv_cov = weighted_least_squares(y, v, X, 0.0, return_cov=True)
    kwargs = dict(inv_cov=inv_cov, small_sample=False)
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


def test_cluster_weights_equalize_group_totals():
    """Every group should end up with the mean of its members' weights."""
    v = np.array([[1.0], [1.0], [1.0], [2.0]])
    groups = np.array([0, 0, 0, 1])

    w = cluster_weights(v, groups)

    assert np.isclose(w[:3].sum(), 1.0)  # three estimates of variance 1
    assert np.isclose(w[3].sum(), 0.5)  # one estimate of variance 2


def test_cluster_weighting_removes_the_replication_advantage():
    """A study contributing many estimates must not outvote a study with one."""
    n_singletons = 8
    groups = np.array([0] * 6 + list(range(1, n_singletons + 1)))
    n_studies = groups.size
    y = np.zeros((n_studies, 1))
    y[:6] = 1.0  # the big study is the only one with a non-zero effect
    v = np.ones((n_studies, 1))
    dataset = Dataset(y=y, v=v, g=groups)

    individual = WeightedLeastSquares().fit_dataset(dataset).summary()
    clustered = WeightedLeastSquares(weight_scheme="cluster").fit_dataset(dataset).summary()

    # Six of fourteen estimates, but only one of nine studies.
    assert np.isclose(individual.get_fe_stats()["est"].ravel()[0], 6 / 14)
    assert np.isclose(clustered.get_fe_stats()["est"].ravel()[0], 1 / 9)


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


def test_collapse_clusters_matches_hand_calculation():
    """The collapsed variance is that of a mean of correlated terms."""
    y = np.array([[1.0], [3.0], [10.0]])
    v = np.array([[2.0], [8.0], [5.0]])
    X = np.ones((3, 1))
    groups = np.array([0, 0, 1])
    rho = 0.5

    c_y, c_v, c_X = collapse_clusters(y, v, X, groups, rho=rho)

    assert np.allclose(c_y.ravel(), [2.0, 10.0])
    # Var((y1 + y2) / 2) = (v1 + v2 + 2*rho*sqrt(v1*v2)) / 4
    expected = (2.0 + 8.0 + 2 * rho * np.sqrt(2.0 * 8.0)) / 4
    assert np.isclose(c_v.ravel()[0], expected)
    assert np.isclose(c_v.ravel()[1], 5.0)  # singletons pass through
    assert np.allclose(c_X, np.ones((2, 1)))


def test_collapse_clusters_by_n_matches_hand_calculation():
    """The effective sample size reproduces the collapsed variance."""
    y = np.array([[1.0], [3.0], [10.0]])
    n = np.array([[20.0], [80.0], [50.0]])
    X = np.ones((3, 1))
    groups = np.array([0, 0, 1])
    rho, sigma2 = 0.5, 4.0

    _, c_n, _ = collapse_clusters_by_n(y, n, X, groups, rho=rho)

    # sigma^2 / n_eff must equal Var of the mean of the two members.
    v = sigma2 / n[:2].ravel()
    expected_var = (v.sum() + 2 * rho * np.sqrt(v[0] * v[1])) / 4
    assert np.isclose(sigma2 / c_n.ravel()[0], expected_var)
    assert np.isclose(c_n.ravel()[1], 50.0)  # singletons pass through


def test_collapse_clusters_rejects_out_of_range_rho():
    """The assumed within-cluster correlation must lie in [0, 1]."""
    y = np.ones((2, 1))
    for collapse, second in ((collapse_clusters, y), (collapse_clusters_by_n, y * 10)):
        with pytest.raises(ValueError, match="rho must lie"):
            collapse(y, second, np.ones((2, 1)), np.array([0, 1]), rho=1.5)


@pytest.mark.parametrize("estimator", [DerSimonianLaird, Hedges, VarianceBasedLikelihoodEstimator])
def test_variance_components_are_insensitive_to_duplication(estimator):
    """Duplicating a study's estimate must not shrink tau^2.

    Repeated estimates from one study agree with each other by construction. An
    estimator that counts them as independent sees less dispersion than the row
    count implies and shrinks tau^2 toward zero, which sharpens the weights and
    makes downstream inference anti-conservative.
    """
    rng = np.random.default_rng(4)
    n_studies, n_datasets = 12, 8
    y = rng.standard_normal((n_studies, n_datasets)) * 2.0
    v = np.full((n_studies, n_datasets), 0.5)
    groups = np.arange(n_studies)

    single = estimator(weight_scheme="cluster")
    single.fit_dataset(Dataset(y=y, v=v, g=groups))

    # Study 0 now contributes four identical estimates instead of one.
    dupe_idx = np.r_[np.zeros(4, dtype=int), np.arange(1, n_studies)]
    duped = estimator(weight_scheme="cluster")
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
