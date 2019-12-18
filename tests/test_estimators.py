import numpy as np

from pymare.estimators import (WeightedLeastSquares, DerSimonianLaird,
                         LikelihoodBased, StanMetaRegression)


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


def test_maximum_likelihood_estimator(dataset):
    # ground truth values are from metafor package in R
    results = LikelihoodBased(method='ML').fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2, 7.7649, atol=1e-4)


def test_restricted_maximum_likelihood_estimator(dataset):
    # ground truth values are from metafor package in R
    results = LikelihoodBased(method='REML').fit(dataset)
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)


def test_stan_estimator(dataset):
    # no ground truth here, so we use sanity checks and rough bounnds
    results = StanMetaRegression(iter=2500).fit(dataset)
    assert 'BayesianMetaRegressionResults' == results.__class__.__name__
    summary = results.summary(['beta', 'tau2'])
    beta1, beta2, tau2 = summary['mean'].values[:3]
    assert -0.5 < beta1 < 0.1
    assert 0.6 < beta2 < 0.9
    assert 2 < tau2 < 6
