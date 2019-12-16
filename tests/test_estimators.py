import numpy as np
import pytest

from pymare.estimators import (weighted_least_squares, dersimonian_laird,
                               likelihood_based)


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
