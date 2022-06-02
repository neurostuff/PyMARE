"""Tests for the pymare.datasets module."""
import pandas as pd

from pymare import datasets


def test_michael2013():
    """Ensure that the Michael 2013 dataset is loadable."""
    data, meta = datasets.michael2013()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (12, 13)
    assert isinstance(meta, dict)
