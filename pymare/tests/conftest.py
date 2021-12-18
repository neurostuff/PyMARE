"""Data for tests."""
import numpy as np
import pytest

from pymare import Dataset


@pytest.fixture(scope="package")
def variables():
    """Build basic numpy variables."""
    y = np.array([[-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]]).T
    v = np.array([[1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]]).T
    X = np.array([1, 1, 2, 2, 4, 4, 2.8, 2.8])
    return (y, v, X)


@pytest.fixture(scope="package")
def dataset(variables):
    """Build a Dataset compiled from the variables fixture."""
    return Dataset(*variables, X_names=["my_covariate"])


@pytest.fixture(scope="package")
def small_dataset_2d(variables):
    """Build a small Dataset with 2D data."""
    y = np.array([[1.5, 1.9, 2.2], [4, 2, 1]]).T
    v = np.array([[1, 0.8, 3], [1, 1.5, 1]]).T
    return Dataset(y, v)


@pytest.fixture(scope="package")
def dataset_2d(variables):
    """Build a larger Dataset with 2D data."""
    y, v, X = variables
    y = np.repeat(y, 3, axis=1)
    y[:, 1] = np.random.randint(-10, 10, size=len(y))
    v = np.repeat(v, 3, axis=1)
    v[:, 1] = np.random.randint(2, 10, size=len(v))
    return Dataset(y, v, X)


@pytest.fixture(scope="package")
def dataset_n():
    """Build a Dataset with sample sizes, but no variances."""
    y = np.array([[-3.0, -0.5, 0.0, -5.01, 0.35, -2.0, -6.0, -4.0, -4.3, -0.1, -1.0]]).T
    n = (
        np.array(
            [[16, 16, 20.548, 32.597, 14.0, 11.118, 4.444, 12.414, 26.963, 130.556, 126.76]]
        ).T
        / 2
    )
    return Dataset(y, n=n)


@pytest.fixture(scope="package")
def vars_with_intercept():
    """Build basic numpy variables with intercepts included in the design matrix."""
    y = np.array([[-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]]).T
    v = np.array([[1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]]).T
    X = np.array([np.ones(8), [1, 1, 2, 2, 4, 4, 2.8, 2.8]]).T
    return (y, v, X)
