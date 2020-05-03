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


def test_meta_regression_results_init_1d(fitted_estimator):
    est = fitted_estimator
    results = MetaRegressionResults(est, est.dataset_, est.params_['beta'],
                                    est.params_['inv_cov'], est.params_['tau2'])
    assert isinstance(est.summary(), MetaRegressionResults)
    assert results.fe_params.shape == (2, 1)
    assert results.fe_cov.shape == (2, 2, 1)
    assert results.tau2.shape == (1,)


def test_mrr_compute_stats(results):
    results.compute_stats()
    assert set(results['beta'].keys()) == {'z', 'p', 'se', 'ci_l', 'ci_u', 'est'}
    assert np.all(results['beta']['ci_u'] > results['beta']['ci_l'])
    for val in results['beta'].values():
        assert val.shape == (2, 1)


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
