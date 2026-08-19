"""Tests for pymare.core."""

import numpy as np
import pandas as pd
import pytest

from pymare import Dataset, meta_regression
from pymare.estimators import Hedges, WeightedLeastSquares


def test_dataset_init(variables):
    """Test Dataset creation from numpy arrays."""
    dataset = Dataset(*variables, X_names=["bork"])

    n = len(variables[0])
    assert dataset.X.shape == (n, 2)
    assert dataset.X_names == ["intercept", "bork"]

    dataset = Dataset(*variables, X_names=["bork"], add_intercept=False)
    assert dataset.X.shape == (n, 1)
    assert dataset.X_names == ["bork"]

    df = dataset.to_df()
    assert isinstance(df, pd.DataFrame)


def test_dataset_init_2D():
    """Test Dataset creation from 2D numpy arrays."""
    n_studies, n_tests = 100, 10
    y = np.random.random((n_studies, n_tests))
    v = np.random.random((n_studies, n_tests))
    n = np.random.random((n_studies, n_tests))
    X = np.random.random((n_studies, 2))
    X_names = ["X1", "X2"]
    dataset = Dataset(y=y, v=v, n=n, X=X, X_names=X_names)

    assert dataset.y.shape == (n_studies, n_tests)
    assert dataset.X.shape == (n_studies, 3)
    assert dataset.X_names == ["intercept", "X1", "X2"]

    df = dataset.to_df()
    assert isinstance(df, pd.DataFrame)


def test_dataset_init_from_df(variables):
    """Test Dataset creation from a DataFrame."""
    df = pd.DataFrame(
        {
            "y": [2, 4, 6],
            "v_alt": [100, 100, 100],
            "sample_size": [10, 20, 30],
            "X1": [5, 2, 1],
            "X7": [9, 8, 7],
        }
    )
    dataset = Dataset(v="v_alt", X=["X1", "X7"], n="sample_size", data=df)
    assert dataset.X.shape == (3, 3)
    assert dataset.X_names == ["intercept", "X1", "X7"]
    assert np.array_equal(dataset.y, np.array([[2, 4, 6]]).T)
    assert np.array_equal(dataset.v, np.array([[100, 100, 100]]).T)
    assert np.array_equal(dataset.n, np.array([[10, 20, 30]]).T)

    df2 = dataset.to_df()
    assert isinstance(df2, pd.DataFrame)

    # y is undefined
    df = pd.DataFrame({"v": [100, 100, 100], "X": [5, 2, 1], "n": [10, 20, 30]})
    with pytest.raises(KeyError):
        dataset = Dataset(data=df)

    # X is undefined
    df = pd.DataFrame({"y": [2, 4, 6], "v_alt": [100, 100, 100], "n": [10, 20, 30]})
    dataset = Dataset(v="v_alt", data=df)
    assert dataset.X.shape == (3, 1)
    assert dataset.X_names == ["intercept"]
    assert np.array_equal(dataset.y, np.array([[2, 4, 6]]).T)
    assert np.array_equal(dataset.v, np.array([[100, 100, 100]]).T)

    # X is undefined, but add_intercept is False
    df = pd.DataFrame({"y": [2, 4, 6], "v_alt": [100, 100, 100], "n": [10, 20, 30]})
    with pytest.raises(ValueError):
        dataset = Dataset(v="v_alt", data=df, add_intercept=False)

    # v is undefined
    df = pd.DataFrame({"y": [2, 4, 6], "X": [5, 2, 1], "n": [10, 20, 30]})
    dataset = Dataset(data=df)
    assert dataset.X.shape == (3, 2)
    assert dataset.X_names == ["intercept", "X"]
    assert dataset.v is None
    assert np.array_equal(dataset.y, np.array([[2, 4, 6]]).T)

    # v is undefined
    df = pd.DataFrame({"y": [2, 4, 6], "X": [5, 2, 1], "v": [10, 20, 30]})
    dataset = Dataset(data=df)
    assert dataset.X.shape == (3, 2)
    assert dataset.X_names == ["intercept", "X"]
    assert dataset.n is None
    assert np.array_equal(dataset.y, np.array([[2, 4, 6]]).T)


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


# -----------------------------------------------------------------------------
# Dependent estimates: group labels
# -----------------------------------------------------------------------------


def test_dataset_rejects_invalid_group_labels():
    """One label per observation, and only one dimension of them."""
    with pytest.raises(ValueError, match="one group label per observation"):
        Dataset(y=np.arange(6.0), v=np.ones(6), g=np.tile(np.arange(3), (6, 1))[:, :2])

    with pytest.raises(ValueError, match="same number of rows"):
        Dataset(y=[1.0, 2.0, 3.0], v=[1.0, 1.0, 1.0], g=[0, 1])

    # fit() takes arrays directly, with no Dataset to check them. Validating the
    # count against the labels themselves made the check vacuous, and the fit
    # then failed on a boolean-index IndexError instead.
    with pytest.raises(ValueError, match="one label per observation"):
        Hedges(weight_scheme="collapse").fit(
            y=np.zeros((9, 1)),
            v=np.full((9, 1), 0.1),
            X=np.c_[np.ones(9), np.repeat([1.0, 2.0, 3.0], 3)],
            g=np.repeat([0, 1], 3),
        )


def test_dataset_group_labels_round_trip_through_a_dataframe():
    """Labels survive to_df, repeated once per parallel dataset, and come back."""
    groups = np.array([0, 0, 1, 1])
    parallel = Dataset(y=np.arange(8.0).reshape(4, 2), v=np.ones((4, 2)), g=groups).to_df()
    assert parallel["g"].tolist() == np.tile(groups, 2).tolist()

    frame = Dataset(y=np.arange(4.0), v=np.ones(4), g=groups).to_df()
    assert "g" in frame.columns
    assert Dataset(data=frame).g.ravel().tolist() == groups.tolist()

    # `g or "g"` called bool() on the array and raised.
    without_labels = pd.DataFrame({"y": np.arange(4.0), "v": np.ones(4)})
    assert np.array_equal(np.ravel(Dataset(data=without_labels, g=groups).g), groups)


def test_group_labels_flow_through_dataset_and_meta_regression(dependent_data):
    """Groups must reach the estimator through both the object and functional APIs."""
    y, v, X, groups = dependent_data(np.random.RandomState(0), n_datasets=1)
    n_groups = np.unique(groups).size

    dataset = Dataset(y=y, v=v, X=X, g=groups, add_intercept=False)
    assert dataset.g.shape == (y.shape[0], 1)
    assert WeightedLeastSquares().fit_dataset(dataset).n_groups_ == n_groups

    results = meta_regression(y=y, v=v, X=X, add_intercept=False, method="WLS", g=groups)
    assert results.estimator.n_groups_ == n_groups


def test_dataset_exports_a_shared_sample_size_column():
    """One column of n serves every parallel dataset, so to_df must not index it."""
    dataset = Dataset(y=np.arange(8.0).reshape(4, 2), n=np.full((4, 1), 10.0))

    frame = dataset.to_df()

    assert frame["n"].tolist() == [10.0] * 8
    assert frame["set"].tolist() == [0] * 4 + [1] * 4
