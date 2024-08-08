"""Tests for pymare.results."""

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
from pymare.results import CombinationTestResults, MetaRegressionResults


@pytest.fixture
def fitted_estimator(dataset):
    """Create a fitted Estimator as a fixture."""
    est = DerSimonianLaird()
    return est.fit_dataset(dataset)


@pytest.fixture
def small_variance_estimator(small_variance_dataset):
    """Create a fitted Estimator with small variances as a fixture."""
    est = DerSimonianLaird()
    return est.fit_dataset(small_variance_dataset)


@pytest.fixture
def results(fitted_estimator):
    """Create a results object as a fixture."""
    return fitted_estimator.summary()


@pytest.fixture
def small_variance_results(small_variance_estimator):
    """Create a results object with small variances as a fixture."""
    return small_variance_estimator.summary()


@pytest.fixture
def results_2d(fitted_estimator, dataset_2d):
    """Create a 2D results object as a fixture."""
    est = VarianceBasedLikelihoodEstimator()
    return est.fit_dataset(dataset_2d).summary()


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


def test_mrr_fe_se(results, results_2d):
    """Test MetaRegressionResults fixed-effect standard error estimates."""
    se_1d, se_2d = results.fe_se, results_2d.fe_se
    assert se_1d.shape == (2, 1)
    assert se_2d.shape == (2, 3)
    assert np.allclose(se_1d.T, [2.6512, 0.9857], atol=1e-4)
    assert np.allclose(se_2d[:, 0].T, [2.5656, 0.9538], atol=1e-4)


def test_mrr_get_fe_stats(results):
    """Test MetaRegressionResults.get_fe_stats."""
    stats = results.get_fe_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"est", "se", "ci_l", "ci_u", "z", "p"}
    assert np.allclose(stats["ci_l"].T, [-5.3033, -1.1655], atol=1e-4)
    assert np.allclose(stats["p"].T, [0.9678, 0.4369], atol=1e-4)


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
    assert np.allclose(df["p-value"].values, [0.9678, 0.4369], atol=1e-4)


def test_small_variance_mrr_to_df(small_variance_results):
    """Test conversion of MetaRegressionResults to DataFrame."""
    df = small_variance_results.to_df()
    assert df.shape == (2, 7)
    col_names = {"estimate", "p-value", "z-score", "ci_0.025", "ci_0.975", "se", "name"}
    assert set(df.columns) == col_names
    assert np.allclose(df["p-value"].values, [1, np.finfo(np.float64).eps], atol=1e-4)


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
