import numpy as np
import pytest

from pymare import Dataset, meta_regression


def test_dataset_init(variables):
    dataset = Dataset(*variables, X_names=['bork'])

    n = len(variables[0])
    assert dataset.X.shape == (n, 2)
    assert dataset.X_names == ['intercept', 'bork']

    dataset = Dataset(*variables, X_names=['bork'], add_intercept=False,
                      extra_arg=200)
    assert dataset.X.shape == (n, 1)
    assert dataset.X_names == ['bork']
    assert 'extra_arg' in dataset.kwargs
    assert dataset.extra_arg == 200


def test_meta_regression_wrapper(variables):
    results = meta_regression(*variables, X_names=['my_cov'], method='REML')
    beta, tau2 = results['beta']['est'], results['tau2']['est']
    assert np.allclose(beta, [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)
    df = results.to_df()
    assert set(df['name']) == {'my_cov', 'intercept', 'tau^2'}
