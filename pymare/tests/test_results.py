"""Tests for pymare.results."""

import copy

import numpy as np
import pytest

from pymare import Dataset
from pymare.estimators import (
    DerSimonianLaird,
    SampleSizeBasedLikelihoodEstimator,
    StoufferCombinationTest,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)
from pymare.results import (
    CombinationTestResults,
    MetaRegressionResults,
    _expand_unit_order,
)
from pymare.stats import DEFAULT_RHO, collapse_groups, collapse_groups_by_n


def test_meta_regression_results_from_arrays(dataset):
    """Ensure that a MetaRegressionResults can be created from arrays.

    This is a regression test for a bug that caused the MetaRegressionResults
    to fail when Estimators were fitted to arrays instead of Datasets.
    See https://github.com/neurostuff/PyMARE/issues/52 for more info.
    """
    est = DerSimonianLaird()
    fitted_estimator = est.fit(y=dataset.y, X=dataset.X, v=dataset.v)
    results = fitted_estimator.summary()
    assert isinstance(results, MetaRegressionResults)
    assert results.fe_params.shape == (2, 1)
    assert results.fe_cov.shape == (2, 2, 1)
    assert results.tau2.shape == (1,)

    # fit overwrites dataset_ attribute with None
    assert fitted_estimator.dataset_ is None
    # fit_dataset overwrites it with the Dataset
    fitted_estimator.fit_dataset(dataset)
    assert isinstance(fitted_estimator.dataset_, Dataset)
    # fit sets it back to None
    fitted_estimator.fit(y=dataset.y, X=dataset.X, v=dataset.v)
    assert fitted_estimator.dataset_ is None

    # Some methods are not available if fit was used
    results = fitted_estimator.summary()
    with pytest.raises(ValueError):
        results.get_re_stats()

    with pytest.raises(ValueError):
        results.get_heterogeneity_stats()

    with pytest.raises(ValueError):
        results.to_df()

    with pytest.raises(ValueError):
        results.permutation_test(1000)


def test_combination_test_results_from_arrays(dataset):
    """Ensure that a CombinationTestResults can be created from arrays.

    This is a regression test for a bug that caused the MetaRegressionResults
    to fail when Estimators were fitted to arrays instead of Datasets.
    See https://github.com/neurostuff/PyMARE/issues/52 for more info.
    """
    fitted_estimator = StoufferCombinationTest().fit(z=dataset.y)
    results = fitted_estimator.summary()
    assert isinstance(results, CombinationTestResults)
    assert results.p.shape == (1,)

    # fit overwrites dataset_ attribute with None
    assert fitted_estimator.dataset_ is None

    # fit_dataset overwrites it with the Dataset
    fitted_estimator.fit_dataset(Dataset(dataset.y))
    assert isinstance(fitted_estimator.dataset_, Dataset)
    # fit sets it back to None
    fitted_estimator.fit(z=dataset.y)
    assert fitted_estimator.dataset_ is None

    # Some methods are not available if fit was used
    with pytest.raises(ValueError):
        fitted_estimator.summary().permutation_test(1000)


def test_meta_regression_results_init_1d(fitted_estimator):
    """Test MetaRegressionResults from 1D data."""
    est = fitted_estimator
    results = MetaRegressionResults(
        est, est.dataset_, est.params_["fe_params"], est.params_["inv_cov"], est.params_["tau2"]
    )
    assert isinstance(est.summary(), MetaRegressionResults)
    assert results.fe_params.shape == (2, 1)
    assert results.fe_cov.shape == (2, 2, 1)
    assert results.tau2.shape == (1,)


def test_meta_regression_results_init_2d(results_2d):
    """Test MetaRegressionResults from 2D data."""
    assert isinstance(results_2d, MetaRegressionResults)
    assert results_2d.fe_params.shape == (2, 3)
    assert results_2d.fe_cov.shape == (2, 2, 3)
    assert results_2d.tau2.shape == (1, 3)


def test_mrr_fe_se(results, results_2d, dataset, dataset_2d):
    """Test MetaRegressionResults fixed-effect standard error estimates."""
    se_1d, se_2d = results.fe_se, results_2d.fe_se
    assert se_1d.shape == (2, 1)
    assert se_2d.shape == (2, 3)
    assert np.allclose(se_1d.T, [3.0071, 1.1180], atol=1e-4)
    assert np.allclose(se_2d[:, 0].T, [3.0031, 1.1165], atol=1e-4)

    # "wald" returns the unadjusted (X'WX)^-1 standard errors, which are what
    # every PyMARE release before the Knapp-Hartung adjustment reported. Pinned
    # here so the escape hatch is checked where a reader looking for the previous
    # numbers will find it.
    unadjusted = DerSimonianLaird(small_sample_correction="wald").fit_dataset(dataset).summary()
    unadjusted_2d = (
        VarianceBasedLikelihoodEstimator(small_sample_correction="wald")
        .fit_dataset(dataset_2d)
        .summary()
    )
    assert np.allclose(unadjusted.fe_se.T, [2.6512, 0.9857], atol=1e-4)
    assert np.allclose(unadjusted_2d.fe_se[:, 0].T, [2.5656, 0.9538], atol=1e-4)


def test_mrr_get_fe_stats(results):
    """Test MetaRegressionResults.get_fe_stats."""
    stats = results.get_fe_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"est", "se", "ci_l", "ci_u", "z", "p"}
    assert np.allclose(stats["ci_l"].T, [-7.4651, -1.9693], atol=1e-4)
    assert np.allclose(stats["p"].T, [0.9728, 0.5186], atol=1e-4)
    # A t reference with K - P = 6 degrees of freedom, not a normal one.
    assert np.all(results.fe_dof == 6)


def test_mrr_get_re_stats(results_2d):
    """Test MetaRegressionResults.get_re_stats."""
    stats = results_2d.get_re_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"tau^2", "ci_l", "ci_u"}
    assert stats["tau^2"].shape == (1, 3)
    assert stats["ci_u"].shape == (3,)
    assert round(stats["tau^2"][0, 2], 4) == 7.7649
    assert round(stats["ci_l"][2], 4) == 3.8076
    assert round(stats["ci_u"][2], 2) == 59.61


def test_mrr_get_heterogeneity_stats(results_2d):
    """Test MetaRegressionResults.get_heterogeneity_stats."""
    stats = results_2d.get_heterogeneity_stats()
    assert len(stats["Q"] == 3)
    assert round(stats["Q"][2], 4) == 53.8052
    assert round(stats["I^2"][0], 4) == 88.8487
    assert round(stats["H"][0], 4) == 2.9946
    assert stats["p(Q)"][0] < 1e-5


def test_mrr_to_df(results):
    """Test conversion of MetaRegressionResults to DataFrame."""
    df = results.to_df()
    assert df.shape == (2, 7)
    col_names = {"estimate", "p-value", "z-score", "ci_0.025", "ci_0.975", "se", "name"}
    assert set(df.columns) == col_names
    assert np.allclose(df["p-value"].values, [0.9728, 0.5186], atol=1e-4)


def test_small_variance_mrr_to_df(small_variance_results, small_variance_dataset):
    """Test conversion of MetaRegressionResults to DataFrame.

    This fixture sets ``y`` equal to a column of the design, so the weighted
    residuals are exactly zero and the Knapp-Hartung scale factor is exactly
    zero with them. The standard errors are then zero, which ``get_fe_stats``
    reports as an undefined p-value rather than as a maximally significant one --
    which is the honest answer, since a dataset with no residual variation at all
    carries no information about how uncertain the coefficients are.

    Under ``"wald"`` the same fixture still reports the tiny standard errors
    the model-based covariance gives, and hence the near-zero p-value that every
    release before the adjustment reported.
    """
    df = small_variance_results.to_df()
    assert df.shape == (2, 7)
    col_names = {"estimate", "p-value", "z-score", "ci_0.025", "ci_0.975", "se", "name"}
    assert set(df.columns) == col_names
    assert np.all(np.isnan(df["p-value"].values))
    assert np.all(small_variance_results.fe_se == 0.0)

    unadjusted = (
        DerSimonianLaird(small_sample_correction="wald")
        .fit_dataset(small_variance_dataset)
        .summary()
    )
    assert np.allclose(
        unadjusted.to_df()["p-value"].values, [1, np.finfo(np.float64).eps], atol=1e-4
    )


def test_estimator_summary(dataset):
    """Test Estimator's summary method."""
    est = WeightedLeastSquares()
    # Fails if we haven't fitted yet
    with pytest.raises(ValueError):
        est.summary()

    est.fit_dataset(dataset)
    summary = est.summary()
    assert isinstance(summary, MetaRegressionResults)


def test_exact_perm_test_2d_no_mods(small_dataset_2d):
    """Test the exact permutation test on 2D data."""
    results = DerSimonianLaird().fit_dataset(small_dataset_2d).summary()
    pmr = results.permutation_test(1000)
    assert pmr.n_perm == 8
    assert pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (1, 2)
    assert pmr.perm_p["tau2_p"].shape == (2,)


def test_approx_perm_test_1d_with_mods(results):
    """Test the approximate permutation test on 2D data."""
    pmr = results.permutation_test(1000)
    assert pmr.n_perm == 1000
    assert not pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (2, 1)
    assert pmr.perm_p["tau2_p"].shape == (1,)


def test_exact_perm_test_1d_no_mods():
    """Test the exact permutation test on 1D data."""
    dataset = Dataset([1, 1, 2, 1.3], [1.5, 1, 2, 4])
    results = DerSimonianLaird().fit_dataset(dataset).summary()
    pmr = results.permutation_test(867)
    assert pmr.n_perm == 16
    assert pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (1, 1)
    assert pmr.perm_p["tau2_p"].shape == (1,)


def test_approx_perm_test_with_n_based_estimator(dataset_n):
    """Test the approximate permutation test on an sample size-based Estimator."""
    results = SampleSizeBasedLikelihoodEstimator().fit_dataset(dataset_n).summary()
    pmr = results.permutation_test(100)
    assert pmr.n_perm == 100
    assert not pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (1, 1)
    assert pmr.perm_p["tau2_p"].shape == (1,)


def test_stouffers_perm_test_exact():
    """Test the exact permutation test on Stouffers Estimator."""
    dataset = Dataset([1, 1, 2, 1.3], [1.5, 1, 2, 4])
    results = StoufferCombinationTest().fit_dataset(dataset).summary()
    pmr = results.permutation_test(2000)
    assert pmr.n_perm == 16
    assert pmr.exact
    assert isinstance(pmr.results, CombinationTestResults)
    assert pmr.perm_p["fe_p"].shape == (1,)
    assert "tau2_p" not in pmr.perm_p


def test_stouffers_perm_test_approx():
    """Test the approximate permutation test on Stouffers Estimator."""
    y = [2.8, -0.2, -1, 4.5, 1.9, 2.38, 0.6, 1.88, -0.4, 1.5, 3.163, 0.7]
    dataset = Dataset(y)
    results = StoufferCombinationTest().fit_dataset(dataset).summary()
    pmr = results.permutation_test(2000)
    assert not pmr.exact
    assert pmr.n_perm == 2000
    assert isinstance(pmr.results, CombinationTestResults)
    assert pmr.perm_p["fe_p"].shape == (1,)
    assert "tau2_p" not in pmr.perm_p


# -----------------------------------------------------------------------------
# Dependent estimates: robust inference
# -----------------------------------------------------------------------------


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
