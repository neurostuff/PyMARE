import numpy as np
import pytest

from pymare import Dataset, meta_regression


def test_dataset_init(variables):
    dataset = Dataset(*variables, names=['bork'])

    # Convenience accessors
    assert np.array_equal(dataset.X, dataset.predictors)
    assert np.array_equal(dataset.y, dataset.estimates)
    assert np.array_equal(dataset.v, dataset.variances)

    n = len(variables[0])
    assert dataset.X.shape == (n, 2)
    assert dataset.names == ['intercept', 'bork']

    dataset = Dataset(*variables, names=['bork'], add_intercept=False,
                      extra_arg=200)
    assert dataset.X.shape == (n, 1)
    assert dataset.names == ['bork']
    assert 'extra_arg' in dataset.kwargs
    assert dataset.extra_arg == 200


def test_meta_regression_1(variables):
    results = meta_regression(*variables, names=['my_cov'], method='REML')
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)
    df = results.to_df()
    assert set(df['name']) == {'my_cov', 'intercept', 'tau^2'}

def test_meta_regression_2(dataset_n):
    y, n = dataset_n.y, dataset_n.n
    df = meta_regression(estimates=y, sample_sizes=n).to_df()
    assert df.shape == (2, 7)
    assert np.isnan(df.iloc[1]['z-score'])
    assert df['ci_0.025'][1] == 0