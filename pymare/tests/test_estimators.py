import numpy as np
import pytest
from pymare.estimators import (WeightedLeastSquares, DerSimonianLaird,
                               VarianceBasedLikelihoodEstimator,
                               SampleSizeBasedLikelihoodEstimator,
                               StanMetaRegression, Hedges, Stouffers)
from pymare import Dataset


def test_weighted_least_squares_estimator(dataset):
    # ground truth values are from metafor package in R
    est = WeightedLeastSquares().fit(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.2725, 0.6935], atol=1e-4)
    assert tau2 == 0.

    # With non-zero tau^2
    est = WeightedLeastSquares(8.).fit(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1071, 0.7657], atol=1e-4)
    assert tau2 == 8.


def test_dersimonian_laird_estimator(dataset):
    # ground truth values are from metafor package in R
    est = DerSimonianLaird().fit(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2, 8.3627, atol=1e-4)


def test_2d_DL_estimator(dataset_2d):
    results = DerSimonianLaird().fit(dataset_2d).summary()
    beta, tau2 = results.fe_params, results.tau2
    assert beta.shape == (2, 3)
    assert tau2.shape == (3,)

    # First and third sets are identical to previous DL test; second set is
    # randomly different.
    assert np.allclose(beta[:, 0], [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2[0], 8.3627, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1070, 0.7664], atol=1e-4)
    assert not np.allclose(tau2[1], 8.3627, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2[2], 8.3627, atol=1e-4)


def test_hedges_estimator(dataset):
    # ground truth values are from metafor package in R, except that metafor
    # always gives negligibly different values for tau2, likely due to
    # algorithmic differences in the computation.
    est = Hedges().fit(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2, 11.3881, atol=1e-4)


def test_2d_hedges(dataset_2d):
    results = Hedges().fit(dataset_2d).summary()
    beta, tau2 = results.fe_params, results.tau2
    assert beta.shape == (2, 3)
    assert tau2.shape == (3,)

    # First and third sets are identical to single dim test; second set is
    # randomly different.
    assert np.allclose(beta[:, 0], [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2[0], 11.3881, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1070, 0.7664], atol=1e-4)
    assert not np.allclose(tau2[1], 11.3881, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2[2], 11.3881, atol=1e-4)


def test_variance_based_maximum_likelihood_estimator(dataset):
    # ground truth values are from metafor package in R
    est = VarianceBasedLikelihoodEstimator(method='ML').fit(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2, 7.7649, atol=1e-4)


def test_variance_based_restricted_maximum_likelihood_estimator(dataset):
    # ground truth values are from metafor package in R
    est = VarianceBasedLikelihoodEstimator(method='REML').fit(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)


def test_sample_size_based_maximum_likelihood_estimator(dataset_n):
    # test values have not been verified for convergence with other packages
    est = SampleSizeBasedLikelihoodEstimator(method='ML').fit(dataset_n)
    results = est.summary()
    beta = results.fe_params
    sigma2 = results.estimator.params_['sigma2']
    tau2 = results.tau2
    assert np.allclose(beta, [-2.0951], atol=1e-4)
    assert np.allclose(sigma2, 12.777, atol=1e-3)
    assert np.allclose(tau2, 2.8268, atol=1e-4)


def test_sample_size_based_restricted_maximum_likelihood_estimator(dataset_n):
    # test values have not been verified for convergence with other packages
    est = SampleSizeBasedLikelihoodEstimator(method='REML').fit(dataset_n)
    results = est.summary()
    beta = results.fe_params
    sigma2 = results.estimator.params_['sigma2']
    tau2 = results.tau2
    assert np.allclose(beta, [-2.1071], atol=1e-4)
    assert np.allclose(sigma2, 13.048, atol=1e-3)
    assert np.allclose(tau2, 3.2177, atol=1e-4)


def test_stouffers():
    # 1-d
    z = np.array([2.1, 0.7, -0.2, 4.1, 3.8])[:, None]
    results = Stouffers()._fit(z)
    assert np.allclose(results['z'], [4.69574], atol=1e-5)

    # 2-d
    z = np.c_[z, np.array([-0.6, -1.61, -2.3, -0.8, -4.01])[:, None]]
    w = np.array([2, 1, 1, 2, 2])[:, None]
    results = Stouffers()._fit(z, w)['z'].ravel()
    assert np.allclose(results, [5.47885, -3.93675], atol=1e-5)


def test_2d_looping(dataset_2d):
    est = VarianceBasedLikelihoodEstimator().fit(dataset_2d)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert beta.shape == (2, 3)
    assert tau2.shape == (1, 3)

    # First and third sets are identical to single dim test; 2nd is different
    assert np.allclose(beta[:, 0], [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2[0, 0], 7.7649, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1072, 0.7653], atol=1e-4)
    assert not np.allclose(tau2[0, 1], 7.7649, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2[0, 2], 7.7649, atol=1e-4)


def test_2d_loop_warning(dataset_2d):
    est = VarianceBasedLikelihoodEstimator()
    y = np.random.normal(size=(10, 100))
    v = np.random.randint(1, 50, size=(10, 100))
    dataset = Dataset(y, v)
    # Warning is raised when 2nd dim is > 10
    with pytest.warns(UserWarning, match='Input contains'):
        est.fit(dataset)
    # But not when it's smaller
    est.fit(dataset_2d)
