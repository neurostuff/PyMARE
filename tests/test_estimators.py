import numpy as np

from pymare.estimators import (weighted_least_squares, dersimonian_laird,
                               likelihood_based, stan, StanMetaRegression)


def test_weighted_least_squares_estimator(vars_with_intercept):
    # ground truth values are from metafor package in R
    results = weighted_least_squares(*vars_with_intercept)
    beta, tau2 = results['beta'], results['tau2']
    assert np.allclose(beta, [-0.2725, 0.6935], atol=1e-4)
    assert tau2 == 0.

    # With non-zero tau^2
    results = weighted_least_squares(*vars_with_intercept, tau2=8.)
    beta, tau2 = results['beta'], results['tau2']
    assert np.allclose(beta, [-0.1071, 0.7657], atol=1e-4)
    assert tau2 == 8.


def test_dersimonian_laird_estimator(vars_with_intercept):
    # ground truth values are from metafor package in R
    results = dersimonian_laird(*vars_with_intercept)
    beta, tau2 = results['beta'], results['tau2']
    assert np.allclose(beta, [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2, 8.3627, atol=1e-4)


def test_maximum_likelihood_estimator(vars_with_intercept):
    # ground truth values are from metafor package in R
    results = likelihood_based(*vars_with_intercept, method='ML')
    beta, tau2 = results['beta'], results['tau2']
    assert np.allclose(beta, [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2, 7.7649, atol=1e-4)


def test_restricted_maximum_likelihood_estimator(vars_with_intercept):
    # ground truth values are from metafor package in R
    results = likelihood_based(*vars_with_intercept, method='REML')
    beta, tau2 = results['beta'], results['tau2']
    assert np.allclose(beta, [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)


def test_stan_estimator(vars_with_intercept):
    # no ground truth here, so we use sanity checks and rough bounnds
    results = stan(*vars_with_intercept, iter=1200)
    assert 'StanFit4Model' == results.__class__.__name__
    summary = results.summary(['beta', 'tau2'])
    beta = summary['summary'][:2, 0]
    tau2 = summary['summary'][2, 0]
    assert -0.4 < beta[0] < 0.1
    assert 0.6 < beta[1] < 0.9
    assert 2 < tau2 < 6
