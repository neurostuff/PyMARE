"""Tests for pymare.results."""
import numpy as np
import pytest

from pymare import Dataset
from pymare.estimators import (
    DerSimonianLaird,
    FisherCombinationTest,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    StanMetaRegression,
    StoufferCombinationTest,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)
from pymare.results import (
    BayesianMetaRegressionResults,
    CombinationTestResults,
    MetaRegressionResults,
)


@pytest.fixture
def fitted_estimator(dataset):
    est = DerSimonianLaird()
    return est.fit_dataset(dataset)


@pytest.fixture
def results(fitted_estimator):
    return fitted_estimator.summary()


@pytest.fixture
def results_2d(fitted_estimator, dataset_2d):
    est = VarianceBasedLikelihoodEstimator()
    return est.fit_dataset(dataset_2d).summary()


def test_meta_regression_results_init_1d(fitted_estimator):
    est = fitted_estimator
    results = MetaRegressionResults(
        est, est.dataset_, est.params_["fe_params"], est.params_["inv_cov"], est.params_["tau2"]
    )
    assert isinstance(est.summary(), MetaRegressionResults)
    assert results.fe_params.shape == (2, 1)
    assert results.fe_cov.shape == (2, 2, 1)
    assert results.tau2.shape == (1,)


def test_meta_regression_results_init_2d(results_2d):
    assert isinstance(results_2d, MetaRegressionResults)
    assert results_2d.fe_params.shape == (2, 3)
    assert results_2d.fe_cov.shape == (2, 2, 3)
    assert results_2d.tau2.shape == (1, 3)


def test_mrr_fe_se(results, results_2d):
    se_1d, se_2d = results.fe_se, results_2d.fe_se
    assert se_1d.shape == (2, 1)
    assert se_2d.shape == (2, 3)
    assert np.allclose(se_1d.T, [2.6512, 0.9857], atol=1e-4)
    assert np.allclose(se_2d[:, 0].T, [2.5656, 0.9538], atol=1e-4)


def test_mrr_get_fe_stats(results):
    stats = results.get_fe_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"est", "se", "ci_l", "ci_u", "z", "p"}
    assert np.allclose(stats["ci_l"].T, [-5.3033, -1.1655], atol=1e-4)
    assert np.allclose(stats["p"].T, [0.9678, 0.4369], atol=1e-4)


def test_mrr_get_re_stats(results_2d):
    stats = results_2d.get_re_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"tau^2", "ci_l", "ci_u"}
    assert stats["tau^2"].shape == (1, 3)
    assert stats["ci_u"].shape == (3,)
    assert round(stats["tau^2"][0, 2], 4) == 7.7649
    assert round(stats["ci_l"][2], 4) == 3.8076
    assert round(stats["ci_u"][2], 2) == 59.61


def test_mrr_get_heterogeneity_stats(results_2d):
    stats = results_2d.get_heterogeneity_stats()
    assert len(stats["Q"] == 3)
    assert round(stats["Q"][2], 4) == 53.8052
    assert round(stats["I^2"][0], 4) == 88.8487
    assert round(stats["H"][0], 4) == 2.9946
    assert stats["p(Q)"][0] < 1e-5


def test_mrr_to_df(results):
    df = results.to_df()
    assert df.shape == (2, 7)
    col_names = {"estimate", "p-value", "z-score", "ci_0.025", "ci_0.975", "se", "name"}
    assert set(df.columns) == col_names
    assert np.allclose(df["p-value"].values, [0.9678, 0.4369], atol=1e-4)


def test_estimator_summary(dataset):
    est = WeightedLeastSquares()
    # Fails if we haven't fitted yet
    with pytest.raises(ValueError):
        results = est.summary()

    est.fit_dataset(dataset)
    summary = est.summary()
    assert isinstance(summary, MetaRegressionResults)


def test_exact_perm_test_2d_no_mods(small_dataset_2d):
    results = DerSimonianLaird().fit_dataset(small_dataset_2d).summary()
    pmr = results.permutation_test(1000)
    assert pmr.n_perm == 8
    assert pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (1, 2)
    assert pmr.perm_p["tau2_p"].shape == (2,)


def test_approx_perm_test_1d_with_mods(results):
    pmr = results.permutation_test(1000)
    assert pmr.n_perm == 1000
    assert not pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (2, 1)
    assert pmr.perm_p["tau2_p"].shape == (1,)


def test_exact_perm_test_1d_no_mods():
    dataset = Dataset([1, 1, 2, 1.3], [1.5, 1, 2, 4])
    results = DerSimonianLaird().fit_dataset(dataset).summary()
    pmr = results.permutation_test(867)
    assert pmr.n_perm == 16
    assert pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (1, 1)
    assert pmr.perm_p["tau2_p"].shape == (1,)


def test_approx_perm_test_with_n_based_estimator(dataset_n):
    results = SampleSizeBasedLikelihoodEstimator().fit_dataset(dataset_n).summary()
    pmr = results.permutation_test(100)
    assert pmr.n_perm == 100
    assert not pmr.exact
    assert isinstance(pmr.results, MetaRegressionResults)
    assert pmr.perm_p["fe_p"].shape == (1, 1)
    assert pmr.perm_p["tau2_p"].shape == (1,)


def test_stouffers_perm_test_exact():
    dataset = Dataset([1, 1, 2, 1.3], [1.5, 1, 2, 4])
    results = StoufferCombinationTest().fit_dataset(dataset).summary()
    pmr = results.permutation_test(2000)
    assert pmr.n_perm == 16
    assert pmr.exact
    assert isinstance(pmr.results, CombinationTestResults)
    assert pmr.perm_p["fe_p"].shape == (1,)
    assert "tau2_p" not in pmr.perm_p


def test_stouffers_perm_test_approx():
    y = [2.8, -0.2, -1, 4.5, 1.9, 2.38, 0.6, 1.88, -0.4, 1.5, 3.163, 0.7]
    dataset = Dataset(y)
    results = StoufferCombinationTest().fit_dataset(dataset).summary()
    pmr = results.permutation_test(2000)
    assert not pmr.exact
    assert pmr.n_perm == 2000
    assert isinstance(pmr.results, CombinationTestResults)
    assert pmr.perm_p["fe_p"].shape == (1,)
    assert "tau2_p" not in pmr.perm_p
