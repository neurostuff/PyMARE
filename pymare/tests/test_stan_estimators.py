import numpy as np
import pytest

from pymare.estimators import StanMetaRegression


def test_stan_estimator(dataset):
    # no ground truth here, so we use sanity checks and rough bounds
    est = StanMetaRegression(iter=500).fit(dataset)
    results = est.summary()
    assert 'BayesianMetaRegressionResults' == results.__class__.__name__
    summary = results.summary(['beta', 'tau2'])
    beta1, beta2, tau2 = summary['mean'].values[:3]
    assert -0.5 < beta1 < 0.1
    assert 0.6 < beta2 < 0.9
    assert 2 < tau2 < 6


def test_stan_2d_input_failure(dataset_2d):
    with pytest.raises(ValueError) as exc:
        est = StanMetaRegression(iter=500).fit(dataset_2d)
    assert str(exc.value).startswith('The StanMetaRegression')
