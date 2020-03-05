import numpy as np

from pymare.estimators import (WeightedLeastSquares, DerSimonianLaird,
                               VarianceBasedLikelihoodEstimator,
                               SampleSizeBasedLikelihoodEstimator,
                               StanMetaRegression, Hedges)


def test_weighted_least_squares_estimator(dataset):
    # ground truth values are from metafor package in R
    results = WeightedLeastSquares().fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.2725, 0.6935], atol=1e-4)
    assert tau2 == 0.

    # With non-zero tau^2
    results = WeightedLeastSquares(8.).fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1071, 0.7657], atol=1e-4)
    assert tau2 == 8.


def test_dersimonian_laird_estimator(dataset):
    # ground truth values are from metafor package in R
    results = DerSimonianLaird().fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2, 8.3627, atol=1e-4)


def test_hedges_estimator(dataset):
    # ground truth values are from metafor package in R, except that metafor
    # always gives negligibly different values for tau2, likely due to
    # algorithmic differences in the computation.
    results = Hedges().fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2, 11.3881, atol=1e-4)


def test_variance_based_maximum_likelihood_estimator(dataset):
    # ground truth values are from metafor package in R
    results = VarianceBasedLikelihoodEstimator(method='ML').fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2, 7.7649, atol=1e-4)


def test_variance_based_restricted_maximum_likelihood_estimator(dataset):
    # ground truth values are from metafor package in R
    results = VarianceBasedLikelihoodEstimator(method='REML').fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)


def test_sample_size_based_maximum_likelihood_estimator(dataset_n):
    # test values have not been verified for convergence with other packages
    results = SampleSizeBasedLikelihoodEstimator(method='ML').fit(dataset_n)
    beta = results['beta']['est']
    sigma2 = results['sigma2']['est']
    tau2 = results['tau2']['est']
    assert np.allclose(beta, [-2.0951], atol=1e-4)
    assert np.allclose(sigma2, 12.777, atol=1e-4)
    assert np.allclose(tau2, 2.8268, atol=1e-4)


def test_sample_size_based_restricted_maximum_likelihood_estimator(dataset_n):
    # test values have not been verified for convergence with other packages
    results = SampleSizeBasedLikelihoodEstimator(method='REML').fit(dataset_n)
    beta = results['beta']['est']
    sigma2 = results['sigma2']['est']
    tau2 = results['tau2']['est']
    assert np.allclose(beta, [-2.1071], atol=1e-4)
    assert np.allclose(sigma2, 13.048, atol=1e-4)
    assert np.allclose(tau2, 3.2177, atol=1e-4)
