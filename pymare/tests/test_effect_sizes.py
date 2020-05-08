import pytest
import numpy as np
import pandas as pd

from pymare.effectsize import (OneSampleEffectSizeConverter, solve_system,
                               select_expressions, compute_measure,
                               TwoSampleEffectSizeConverter)
from pymare import Dataset


@pytest.fixture(scope='module')
def one_samp_data():
    return {
        'm': np.array([7, 5, 4]),
        'sd': np.sqrt(np.array([4.2, 1.2, 1.9])),
        'n': np.array([24, 31, 40]),
        'r': np.array([0.2, 0.18, 0.3])
    }


@pytest.fixture(scope='module')
def two_samp_data():
    return {
        'm1': np.array([4, 2]),
        'sd1': np.sqrt(np.array([1, 9])),
        'n1': np.array([12, 15]),
        'm2': np.array([5, 2.5]),
        'sd2': np.sqrt(np.array([4, 16])),
        'n2': np.array([12, 16]),
    }


def test_EffectSizeConverter_smoke_test(two_samp_data):
    data = two_samp_data
    esc = OneSampleEffectSizeConverter(m=data['m1'], sd=data['sd1'], n=data['n1'])
    assert set(esc.known_vars.keys()) == {'m', 'sd', 'n'}
    assert esc.get_sm().shape == data['m1'].shape
    assert not {'d', 'sd'} - set(esc.known_vars.keys())

    esc = TwoSampleEffectSizeConverter(**data)
    assert set(esc.known_vars.keys()) == set(data.keys())
    assert np.allclose(esc.get_d(), np.array([-0.63246, -0.140744]), atol=1e-5)
    assert np.allclose(esc.get_smd(), np.array([-0.61065, -0.13707]), atol=1e-5)


def test_esc_implicit_dtype_conversion():
    esc = OneSampleEffectSizeConverter(m=[10, 12, 18])
    assert 'm' in esc.known_vars
    assert isinstance(esc.known_vars['m'], np.ndarray)
    assert esc.known_vars['m'][1] == 12


def test_EffectSizeConverter_from_df(two_samp_data):
    df = pd.DataFrame(two_samp_data)
    esc = TwoSampleEffectSizeConverter(df)
    assert np.allclose(esc.get_smd(), np.array([-0.61065, -0.13707]), atol=1e-5)


def test_EffectSizeConverter_to_dataset(two_samp_data):
    esc = TwoSampleEffectSizeConverter(**two_samp_data)
    X = np.array([1, 2])
    dataset = esc.to_dataset(X=X, X_names=['dummy'])
    assert dataset.__class__.__name__ == 'Dataset'
    assert dataset.X_names == ['intercept', 'dummy']


def test_2d_array_conversion():
    shape = (10, 2)
    data = {
        'm': np.random.randint(10, size=shape),
        'sd': np.random.randint(1, 10, size=shape),
        'n': np.ones(shape) * 40
    }
    esc = OneSampleEffectSizeConverter(**data)

    sd = esc.get_d()
    assert np.array_equal(sd, data['m'] / data['sd'])

    # smoke test other parameters to make sure all generated numpy funcs can
    # handle 2d inputs.
    for stat in ['sm']:
        result = esc.get(stat)
        assert result.shape == shape


def test_convert_r_and_n_to_rz():
    r=[0.2, 0.16, 0.6]
    n = (68, 165, 17)
    esc = OneSampleEffectSizeConverter(r=r)
    zr = esc.get_zr()
    assert np.allclose(zr, np.arctanh(r))
    # Needs n
    with pytest.raises(ValueError, match="Unable to solve"):
        esc.get_v_zr()
    esc = OneSampleEffectSizeConverter(r=r, n=n)
    v_zr = esc.get('V_ZR')
    assert np.allclose(v_zr, 1 / (np.array(n) - 3))
    ds = esc.to_dataset(measure="ZR")
    assert np.allclose(ds.y.ravel(), zr)
    assert np.allclose(ds.v.ravel(), v_zr)
    assert ds.n is not None


def test_convert_r_to_itself():
    r=np.array([0.2, 0.16, 0.6])
    n = np.array((68, 165, 17))
    esc = OneSampleEffectSizeConverter(r=r)
    also_r = esc.get_r()
    assert np.array_equal(r, also_r)
    # Needs n
    with pytest.raises(ValueError, match="Unable to solve"):
        esc.get_v_r()
    esc = OneSampleEffectSizeConverter(r=r, n=n)
    v_r = esc.get('V_R')
    assert np.allclose(v_r, (1 - r**2) / (n - 2))
    ds = esc.to_dataset(measure="R")
    assert np.allclose(ds.y.ravel(), r)
    assert np.allclose(ds.v.ravel(), v_r)
    assert ds.n is not None


def test_compute_measure(one_samp_data, two_samp_data):
    # Default args
    base_result = compute_measure('SM',**one_samp_data)
    assert isinstance(base_result, tuple)
    assert len(base_result) == 2
    assert base_result[0].shape == one_samp_data['m'].shape

    # Explicit and correct comparison type
    result2 = compute_measure('SM', comparison=1, **one_samp_data)
    assert np.array_equal(np.array(base_result), np.array(result2))

    # Incorrect comparison type fails downstream
    with pytest.raises(ValueError):
        compute_measure('SM', comparison=2, **one_samp_data)

    # Ambiguous comparison type
    with pytest.raises(ValueError, match="Requested measure \(D\)"):
        compute_measure('D', **one_samp_data, **two_samp_data)

    # Works with explicit comparison type: check for both comparison types
    result = compute_measure('D', comparison=1, **one_samp_data, **two_samp_data)
    conv = compute_measure('D', **one_samp_data, return_type='converter')
    assert isinstance(conv, OneSampleEffectSizeConverter)
    assert np.array_equal(result[1], conv.get_v_d())

    result = compute_measure('D', comparison=2, **one_samp_data, **two_samp_data)
    conv = compute_measure('D', **two_samp_data, return_type='converter')
    assert isinstance(conv, TwoSampleEffectSizeConverter)
    assert np.array_equal(result[1], conv.get_v_d())

    # Test other return types
    result = compute_measure('SM', return_type='dict', **one_samp_data)
    assert np.array_equal(base_result[1], result['v'])

    dset = compute_measure('SM', return_type='dataset', **one_samp_data,
                           X=[4, 3, 2], X_names=['my_covar'])
    assert isinstance(dset, Dataset)
    assert np.array_equal(base_result[1], dset.v.ravel())
    assert dset.X.shape == (3, 2)
    assert dset.X_names == ['intercept', 'my_covar']

    # Test with input DataFrame
    df = pd.DataFrame(two_samp_data)
    result = compute_measure('RMD', df)
    assert np.array_equal(result[0], df['m1'].values - df['m2'].values)

    # Override one of the DF columns
    result = compute_measure('RMD', df, m1=[3, 3])
    assert not np.array_equal(result[0], df['m1'].values - df['m2'].values)
