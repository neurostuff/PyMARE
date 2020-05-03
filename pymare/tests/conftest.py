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
    return Dataset(*variables, X_names=['my_covariate'])


@pytest.fixture(scope='package')
def dataset_2d(variables):
    y, v, X = variables
    y = np.repeat(y, 3, axis=1)
    y[:, 1] = np.random.randint(-10, 10, size=len(y))
    v = np.repeat(v, 3, axis=1)
    v[:, 1] = np.random.randint(2, 10, size=len(v))
    return Dataset(y, v, X)


@pytest.fixture(scope='package')
def dataset_n():
    y = np.array([[-3., -0.5, 0., -5.01, 0.35, -2., -6., -4., -4.3, -0.1, -1.]]).T
    n = np.array([[16, 16, 20.548, 32.597, 14., 11.118, 4.444, 12.414, 26.963,
                   130.556, 126.76]]).T / 2
    return Dataset(y, n=n)


@pytest.fixture(scope='package')
def vars_with_intercept():
    y = np.array([[-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]]).T
    v = np.array([[1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]]).T
    X = np.array([np.ones(8), [1, 1, 2, 2, 4, 4, 2.8, 2.8]]).T
    return (y, v, X)
