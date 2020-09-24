import numpy as np
import scipy.stats as ss
import pytest

from pymare.estimators import StoufferCombinationTest, FisherCombinationTest
from pymare import Dataset


_z1 = np.array([2.1, 0.7, -0.2, 4.1, 3.8])[:, None]
_z2 = np.c_[_z1, np.array([-0.6, -1.61, -2.3, -0.8, -4.01])[:, None]]

_params = [
    (StoufferCombinationTest, _z1, 'directed', [4.69574]),
    (StoufferCombinationTest, _z1, 'undirected', [4.87462819]),
    (StoufferCombinationTest, _z1, 'concordant', [4.55204117]),
    (StoufferCombinationTest, _z2, 'directed', [4.69574275, -4.16803071]),
    (StoufferCombinationTest, _z2, 'undirected', [4.87462819, 4.16803071]),
    (StoufferCombinationTest, _z2, 'concordant', [4.55204117, 4.00717817]),
    (FisherCombinationTest, _z1, 'directed', [5.22413541]),
    (FisherCombinationTest, _z1, 'undirected', [5.27449962]),
    (FisherCombinationTest, _z1, 'concordant', [5.09434911]),
    (FisherCombinationTest, _z2, 'directed', [5.22413541, -3.30626405]),
    (FisherCombinationTest, _z2, 'undirected', [5.27449962, 4.27572965]),
    (FisherCombinationTest, _z2, 'concordant', [5.09434911, 4.11869468]),
]


@pytest.mark.parametrize("Cls,data,mode,expected", _params)
def test_combination_test(Cls, data, mode, expected):
    results = Cls(mode)._fit(data)
    z = ss.norm.isf(results['p'])
    assert np.allclose(z, expected, atol=1e-5)


@pytest.mark.parametrize("Cls,data,mode,expected", _params)
def test_combination_test_from_dataset(Cls, data, mode, expected):
    dset = Dataset(y=data)
    est = Cls(mode).fit(dset)
    results = est.summary()
    z = ss.norm.isf(results.p)
    assert np.allclose(z, expected, atol=1e-5)