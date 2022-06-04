"""Tests for pymare.estimators.estimators."""
import numpy as np
import pytest

from pymare import Dataset
from pymare.estimators import (
    DerSimonianLaird,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)


def test_weighted_least_squares_estimator(dataset):
    """Test WeightedLeastSquares estimator."""
    # ground truth values are from metafor package in R
    est = WeightedLeastSquares().fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert isinstance(tau2, float)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.2725, 0.6935], atol=1e-4)
    assert tau2 == 0.0

    # With non-zero tau^2
    est = WeightedLeastSquares(8.0).fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1071, 0.7657], atol=1e-4)
    assert tau2 == 8.0


def test_dersimonian_laird_estimator(dataset):
    """Test DerSimonianLaird estimator."""
    # ground truth values are from metafor package in R
    est = DerSimonianLaird().fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1,)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2, 8.3627, atol=1e-4)


def test_2d_DL_estimator(dataset_2d):
    """Test DerSimonianLaird estimator on 2D Dataset."""
    results = DerSimonianLaird().fit_dataset(dataset_2d).summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 3)
    assert tau2.shape == (3,)
    assert fe_stats["est"].shape == (2, 3)
    assert fe_stats["se"].shape == (2, 3)
    assert fe_stats["ci_l"].shape == (2, 3)
    assert fe_stats["ci_u"].shape == (2, 3)
    assert fe_stats["z"].shape == (2, 3)
    assert fe_stats["p"].shape == (2, 3)

    # Check output values
    # First and third sets are identical to previous DL test; second set is
    # randomly different.
    assert np.allclose(beta[:, 0], [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2[0], 8.3627, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1070, 0.7664], atol=1e-4)
    assert not np.allclose(tau2[1], 8.3627, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2[2], 8.3627, atol=1e-4)


def test_hedges_estimator(dataset):
    """Test Hedges estimator."""
    # ground truth values are from metafor package in R, except that metafor
    # always gives negligibly different values for tau2, likely due to
    # algorithmic differences in the computation.
    est = Hedges().fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1,)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2, 11.3881, atol=1e-4)


def test_2d_hedges(dataset_2d):
    """Test Hedges estimator on 2D Dataset."""
    results = Hedges().fit_dataset(dataset_2d).summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 3)
    assert tau2.shape == (3,)
    assert fe_stats["est"].shape == (2, 3)
    assert fe_stats["se"].shape == (2, 3)
    assert fe_stats["ci_l"].shape == (2, 3)
    assert fe_stats["ci_u"].shape == (2, 3)
    assert fe_stats["z"].shape == (2, 3)
    assert fe_stats["p"].shape == (2, 3)

    # First and third sets are identical to single dim test; second set is
    # randomly different.
    assert np.allclose(beta[:, 0], [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2[0], 11.3881, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1070, 0.7664], atol=1e-4)
    assert not np.allclose(tau2[1], 11.3881, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2[2], 11.3881, atol=1e-4)


def test_variance_based_maximum_likelihood_estimator(dataset):
    """Test VarianceBasedLikelihoodEstimator estimator."""
    # ground truth values are from metafor package in R
    est = VarianceBasedLikelihoodEstimator(method="ML").fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2, 7.7649, atol=1e-4)


def test_variance_based_restricted_maximum_likelihood_estimator(dataset):
    """Test VarianceBasedLikelihoodEstimator estimator with REML."""
    # ground truth values are from metafor package in R
    est = VarianceBasedLikelihoodEstimator(method="REML").fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)


def test_sample_size_based_maximum_likelihood_estimator(dataset_n):
    """Test SampleSizeBasedLikelihoodEstimator estimator."""
    # test values have not been verified for convergence with other packages
    est = SampleSizeBasedLikelihoodEstimator(method="ML").fit_dataset(dataset_n)
    results = est.summary()
    beta = results.fe_params
    sigma2 = results.estimator.params_["sigma2"]
    tau2 = results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (1, 1)
    assert sigma2.shape == (1, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (1, 1)
    assert fe_stats["se"].shape == (1, 1)
    assert fe_stats["ci_l"].shape == (1, 1)
    assert fe_stats["ci_u"].shape == (1, 1)
    assert fe_stats["z"].shape == (1, 1)
    assert fe_stats["p"].shape == (1, 1)

    # Check output values
    assert np.allclose(beta, [-2.0951], atol=1e-4)
    assert np.allclose(sigma2, 12.777, atol=1e-3)
    assert np.allclose(tau2, 2.8268, atol=1e-4)


def test_sample_size_based_restricted_maximum_likelihood_estimator(dataset_n):
    """Test SampleSizeBasedLikelihoodEstimator REML estimator."""
    # test values have not been verified for convergence with other packages
    est = SampleSizeBasedLikelihoodEstimator(method="REML").fit_dataset(dataset_n)
    results = est.summary()
    beta = results.fe_params
    sigma2 = results.estimator.params_["sigma2"]
    tau2 = results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (1, 1)
    assert sigma2.shape == (1, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (1, 1)
    assert fe_stats["se"].shape == (1, 1)
    assert fe_stats["ci_l"].shape == (1, 1)
    assert fe_stats["ci_u"].shape == (1, 1)
    assert fe_stats["z"].shape == (1, 1)
    assert fe_stats["p"].shape == (1, 1)

    # Check output values
    assert np.allclose(beta, [-2.1071], atol=1e-4)
    assert np.allclose(sigma2, 13.048, atol=1e-3)
    assert np.allclose(tau2, 3.2177, atol=1e-4)


def test_2d_looping(dataset_2d):
    """Test 2D looping in estimators."""
    est = VarianceBasedLikelihoodEstimator().fit_dataset(dataset_2d)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 3)
    assert tau2.shape == (1, 3)
    assert fe_stats["est"].shape == (2, 3)
    assert fe_stats["se"].shape == (2, 3)
    assert fe_stats["ci_l"].shape == (2, 3)
    assert fe_stats["ci_u"].shape == (2, 3)
    assert fe_stats["z"].shape == (2, 3)
    assert fe_stats["p"].shape == (2, 3)

    # Check output values
    # First and third sets are identical to single dim test; 2nd is different
    assert np.allclose(beta[:, 0], [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2[0, 0], 7.7649, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1072, 0.7653], atol=1e-4)
    assert not np.allclose(tau2[0, 1], 7.7649, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2[0, 2], 7.7649, atol=1e-4)


def test_2d_loop_warning(dataset_2d):
    """Test 2D looping warning on certain estimators."""
    est = VarianceBasedLikelihoodEstimator()
    y = np.random.normal(size=(10, 100))
    v = np.random.randint(1, 50, size=(10, 100))
    dataset = Dataset(y, v)
    # Warning is raised when 2nd dim is > 10
    with pytest.warns(UserWarning, match="Input contains"):
        est.fit_dataset(dataset)
    # But not when it's smaller
    est.fit_dataset(dataset_2d)
