import pytest
import numpy as np

from pymare import Dataset


@pytest.fixture(scope='package')
def variables():
    y = np.array([[-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]]).T
    v = np.array([[1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]]).T
    X = np.array([1, 1, 2, 2, 4, 4, 2.8, 2.8])
    return (y, v, X)


@pytest.fixture(scope='package')
def dataset(variables):
    return Dataset(*variables, names=['my_covariate'])


@pytest.fixture(scope='package')
def vars_with_intercept():
    y = np.array([[-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]]).T
    v = np.array([[1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]]).T
    X = np.array([np.ones(8), [1, 1, 2, 2, 4, 4, 2.8, 2.8]]).T
    return (y, v, X)
