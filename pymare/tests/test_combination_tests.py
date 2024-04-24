"""Tests for pymare.estimators.combination."""

import numpy as np
import pytest
import scipy.stats as ss

from pymare import Dataset
from pymare.estimators import FisherCombinationTest, StoufferCombinationTest

_z1 = np.array([2.1, 0.7, -0.2, 4.1, 3.8])[:, None]
_z2 = np.c_[_z1, np.array([-0.6, -1.61, -2.3, -0.8, -4.01])[:, None]]

_params = [
    (StoufferCombinationTest, _z1, "directed", [4.69574]),
    (StoufferCombinationTest, _z1, "undirected", [4.87462819]),
    (StoufferCombinationTest, _z1, "concordant", [4.55204117]),
    (StoufferCombinationTest, _z2, "directed", [4.69574275, -4.16803071]),
    (StoufferCombinationTest, _z2, "undirected", [4.87462819, 4.16803071]),
    (StoufferCombinationTest, _z2, "concordant", [4.55204117, 4.00717817]),
    (FisherCombinationTest, _z1, "directed", [5.22413541]),
    (FisherCombinationTest, _z1, "undirected", [5.27449962]),
    (FisherCombinationTest, _z1, "concordant", [5.09434911]),
    (FisherCombinationTest, _z2, "directed", [5.22413541, -3.30626405]),
    (FisherCombinationTest, _z2, "undirected", [5.27449962, 4.27572965]),
    (FisherCombinationTest, _z2, "concordant", [5.09434911, 4.11869468]),
]


@pytest.mark.parametrize("Cls,data,mode,expected", _params)
def test_combination_test(Cls, data, mode, expected):
    """Test CombinationTest Estimators with numpy data."""
    results = Cls(mode).fit(data).params_
    z = ss.norm.isf(results["p"])
    assert np.allclose(z, expected, atol=1e-5)


@pytest.mark.parametrize("Cls,data,mode,expected", _params)
def test_combination_test_from_dataset(Cls, data, mode, expected):
    """Test CombinationTest Estimators with PyMARE Datasets."""
    dset = Dataset(y=data)
    est = Cls(mode).fit_dataset(dset)
    results = est.summary()
    z = ss.norm.isf(results.p)
    assert np.allclose(z, expected, atol=1e-5)


def test_stouffer_adjusted():
    """Test StoufferCombinationTest with weights and groups."""
    # Test with weights and groups
    data = np.array(
        [
            [2.1, 0.7, -0.2, 4.1, 3.8],
            [1.1, 0.2, 0.4, 1.3, 1.5],
            [-0.6, -1.6, -2.3, -0.8, -4.0],
            [2.5, 1.7, 2.1, 2.3, 2.5],
            [3.1, 2.7, 3.1, 3.3, 3.5],
            [3.6, 3.2, 3.6, 3.8, 4.0],
        ]
    )
    weights = np.tile(np.array([4, 3, 4, 10, 15, 10]), (data.shape[1], 1)).T
    groups = np.tile(np.array([0, 0, 1, 2, 2, 2]), (data.shape[1], 1)).T

    results = StoufferCombinationTest("directed").fit(z=data, w=weights, g=groups).params_
    z = ss.norm.isf(results["p"])

    z_expected = np.array([5.00088912, 3.70356943, 4.05465924, 5.4633001, 5.18927878])
    assert np.allclose(z, z_expected, atol=1e-5)

    # Test with weights and no groups. Limiting cases.
    # Limiting case 1: all correlations are one.
    n_maps_l1 = 5
    common_sample = np.array([2.1, 0.7, -0.2])
    data_l1 = np.tile(common_sample, (n_maps_l1, 1))
    groups_l1 = np.tile(np.array([0, 0, 0, 0, 0]), (data_l1.shape[1], 1)).T

    results_l1 = StoufferCombinationTest("directed").fit(z=data_l1, g=groups_l1).params_
    z_l1 = ss.norm.isf(results_l1["p"])

    sigma_l1 = n_maps_l1 * (n_maps_l1 - 1)  # Expected inflation term
    z_expected_l1 = n_maps_l1 * common_sample / np.sqrt(n_maps_l1 + sigma_l1)
    assert np.allclose(z_l1, z_expected_l1, atol=1e-5)

    # Test with correlation matrix and groups.
    data_corr = data - data.mean(0)
    corr = np.corrcoef(data_corr, rowvar=True)
    results_corr = (
        StoufferCombinationTest("directed").fit(z=data, w=weights, g=groups, corr=corr).params_
    )
    z_corr = ss.norm.isf(results_corr["p"])

    z_corr_expected = np.array([5.00088912, 3.70356943, 4.05465924, 5.4633001, 5.18927878])
    assert np.allclose(z_corr, z_corr_expected, atol=1e-5)

    # Test with no correlation matrix and groups, but only one feature.
    with pytest.raises(ValueError):
        StoufferCombinationTest("directed").fit(z=data[:, :1], w=weights[:, :1], g=groups)

    # Test with correlation matrix and groups of different shapes.
    with pytest.raises(ValueError):
        StoufferCombinationTest("directed").fit(z=data, w=weights, g=groups, corr=corr[:-2, :-2])

    # Test with correlation matrix and no groups.
    results1 = StoufferCombinationTest("directed").fit(z=_z1, corr=corr).params_
    z1 = ss.norm.isf(results1["p"])

    assert np.allclose(z1, [4.69574], atol=1e-5)
