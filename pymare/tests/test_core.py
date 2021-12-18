"""Tests for pymare.core."""
import numpy as np
import pandas as pd

from pymare import Dataset, meta_regression


def test_dataset_init(variables):
    """Test Dataset creation from numpy arrays."""
    dataset = Dataset(*variables, X_names=["bork"])

    n = len(variables[0])
    assert dataset.X.shape == (n, 2)
    assert dataset.X_names == ["intercept", "bork"]

    dataset = Dataset(*variables, X_names=["bork"], add_intercept=False)
    assert dataset.X.shape == (n, 1)
    assert dataset.X_names == ["bork"]


def test_dataset_init_from_df(variables):
    """Test Dataset creation from a DataFrame."""
    df = pd.DataFrame({"y": [2, 4, 6], "v_alt": [100, 100, 100], "X1": [5, 2, 1], "X7": [9, 8, 7]})
    dataset = Dataset(v="v_alt", X=["X1", "X7"], data=df)
    assert dataset.X.shape == (3, 3)
    assert dataset.X_names == ["intercept", "X1", "X7"]
    assert np.array_equal(dataset.y, np.array([[2, 4, 6]]).T)
    assert np.array_equal(dataset.v, np.array([[100, 100, 100]]).T)


def test_meta_regression_1(variables):
    """Test meta_regression function."""
    results = meta_regression(*variables, X_names=["my_cov"], method="REML")
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)
    df = results.to_df()
    assert set(df["name"]) == {"my_cov", "intercept"}


def test_meta_regression_2(dataset_n):
    """Test meta_regression function."""
    y, n = dataset_n.y, dataset_n.n
    df = meta_regression(y=y, n=n).to_df()
    assert df.shape == (1, 7)
