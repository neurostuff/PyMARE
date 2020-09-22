import numpy as np
import scipy.stats as ss

from pymare.estimators import Stouffers, Fishers


def test_stouffers():
    # One-tailed, 1-d
    z = np.array([2.1, 0.7, -0.2, 4.1, 3.8])[:, None]
    results = Stouffers()._fit(z)
    res_z = ss.norm.isf(results['p'])
    assert np.allclose(res_z, [4.69574], atol=1e-5)

    # 2-d with weights
    z = np.c_[z, np.array([-0.6, -1.61, -2.3, -0.8, -4.01])[:, None]]
    w = np.array([2, 1, 1, 2, 2])[:, None]
    results = Stouffers()._fit(z, w)
    res_z = ss.norm.isf(results['p'])
    assert np.allclose(res_z, [5.47885, -3.93675], atol=1e-5)


def test_fishers():
    # 1-d
    z = np.array([2.1, 0.7, -0.2, 4.1, 3.8])[:, None]
    results = Fishers()._fit(z)
    res_z = ss.norm.isf(results['p'])
    assert np.allclose(res_z, [5.22413541], atol=1e-5)

    # 2-d
    z = np.c_[z, np.array([-0.6, -1.61, -2.3, -0.8, -4.01])[:, None]]
    results = Fishers()._fit(z)
    res_z = ss.norm.isf(results['p'])
    assert np.allclose(res_z, [5.22413541, -3.30626405], atol=1e-5)
