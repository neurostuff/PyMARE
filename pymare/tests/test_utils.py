"""Tests for pymare.utils."""
import os.path as op

import numpy as np
import pytest

from pymare import utils


def test_get_resource_path():
    """Test nimare.utils.get_resource_path."""
    print(utils.get_resource_path())
    assert op.isdir(utils.get_resource_path())


def test_check_inputs_shape():
    """Test nimare.utils._check_inputs_shape."""
    n_rows = 5
    n_columns = 4
    n_pred = 3
    y = np.random.randint(1, 100, size=(n_rows, n_columns))
    v = np.random.randint(1, 100, size=(n_rows + 1, n_columns))
    n = np.random.randint(1, 100, size=(n_rows, n_columns))
    X = np.random.randint(1, 100, size=(n_rows, n_pred))
    X_names = [f"X{x}" for x in range(n_pred)]

    utils._check_inputs_shape(y, X, "y", "X", row=True)
    utils._check_inputs_shape(y, n, "y", "n", row=True, column=True)
    utils._check_inputs_shape(X, np.array(X_names)[None, :], "X", "X_names", column=True)

    # Raise error if the number of rows and columns of v don't match y
    with pytest.raises(ValueError):
        utils._check_inputs_shape(y, v, "y", "v", row=True, column=True)

    # Raise error if neither row or column is True
    with pytest.raises(ValueError):
        utils._check_inputs_shape(y, n, "y", "n")

    # Dataset may be initialized with n or v as None
    utils._check_inputs_shape(y, None, "y", "n", row=True, column=True)
