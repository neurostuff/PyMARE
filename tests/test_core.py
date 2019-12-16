import numpy as np
import pytest

from pymare import Dataset


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
