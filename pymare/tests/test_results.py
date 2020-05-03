import pytest
import numpy as np

from pymare.results import MetaRegressionResults, BayesianMetaRegressionResults
from pymare.estimators import (WeightedLeastSquares, DerSimonianLaird,
                               VarianceBasedLikelihoodEstimator,
                               SampleSizeBasedLikelihoodEstimator,
                               StanMetaRegression, Hedges)



@pytest.fixture
def fitted_estimator(dataset):
    est = DerSimonianLaird()
    return est.fit(dataset)


@pytest.fixture
def results(fitted_estimator):
    return fitted_estimator.summary()


@pytest.fixture
def results_2d(fitted_estimator, dataset_2d):
    est = VarianceBasedLikelihoodEstimator()
    return est.fit(dataset_2d).summary()


def test_meta_regression_results_init_1d(fitted_estimator):
    est = fitted_estimator
    results = MetaRegressionResults(est, est.dataset_, est.params_['beta'],
                                    est.params_['inv_cov'], est.params_['tau2'])
    assert isinstance(est.summary(), MetaRegressionResults)
    assert results.fe_params.shape == (2, 1)
    assert results.fe_cov.shape == (2, 2, 1)
    assert results.tau2.shape == (1,)


def test_meta_regression_results_init_2d(results_2d):
    assert isinstance(results_2d, MetaRegressionResults)
    assert results_2d.fe_params.shape == (2, 3)
    assert results_2d.fe_cov.shape == (2, 2, 3)
    assert results_2d.tau2.shape == (3,)


def test_mrr_fe_se(results, results_2d):
    se_1d, se_2d = results.fe_se, results_2d.fe_se
    assert se_1d.shape == (2, 1)
    assert se_2d.shape == (2, 3)
    assert np.allclose(se_1d.T, [2.6512, 0.9857], atol=1e-4)
    assert np.allclose(se_2d[:, 0].T, [2.5656, 0.9538], atol=1e-4)


def test_mrr_get_fe_stats(results):
    stats = results.get_fe_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {'est', 'se', 'ci_l', 'ci_u', 'z', 'p'}
    assert np.allclose(stats['ci_l'].T, [-5.3033, -1.1655], atol=1e-4)
    assert np.allclose(stats['p'].T, [0.9678, 0.4369], atol=1e-4)


def test_mrr_get_re_stats(results_2d):
    stats = results_2d.get_re_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {'tau^2', 'ci_l', 'ci_u'}
    assert stats['tau^2'].shape == stats['ci_u'].shape == (3,)
    assert round(stats['tau^2'][2], 4) == 7.7649
    assert round(stats['ci_l'][2], 4) == 3.8076
    assert round(stats['ci_u'][2], 2) == 59.61


def test_mrr_to_df(results):
    df = results.to_df()
    assert df.shape == (3, 7)
    assert np.isnan(df.iloc[:, 1:].values.ravel()).sum() == 3


def test_estimator_summary(dataset):
    est = WeightedLeastSquares()
    # Fails if we haven't fitted yet
    with pytest.raises(ValueError):
        results = est.summary()
    
    est.fit(dataset)
    summary = est.summary()
    assert isinstance(summary, MetaRegressionResults)
