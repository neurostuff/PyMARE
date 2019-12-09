import numpy as np

from pymeta import Dataset, meta_regression


def result_matches_target(result, target):
    beta_r, tau_r = result
    beta_t, tau_t = target
    print(beta_r, tau_r, beta_t, tau_t)
    assert np.allclose(beta_r, beta_t, 4)
    assert np.allclose(tau_r, tau_t, 4)


def test_meta_regression_smoke_test():
    """ Smoketest for implemented estimators called via meta_regression(). """
    # ground truth values are from metafor package in R
    y = np.array([-1, 0.5, 0.5, 0.5, 1, 1, 2, 10])
    v = np.array([1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5])
    X = np.array([1, 1, 2, 2, 4, 4, 2.8, 2.8])

    # ML
    results = meta_regression(y, v, X, 'ML')
    beta, tau2 = results.beta, results.tau2
    assert np.allclose(beta, [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2, 7.7649, atol=1e-4)

    # REML
    results = meta_regression(y, v, X, 'REML')
    beta, tau2 = results.beta, results.tau2
    assert np.allclose(beta, [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)

    # DerSimonian-Laird
    results = meta_regression(y, v, X, 'DL')
    beta, tau2 = results.beta, results.tau2
    assert np.allclose(beta, [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2, 8.3627, atol=1e-4)
