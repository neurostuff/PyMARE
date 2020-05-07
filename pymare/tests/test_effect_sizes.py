import pytest
import numpy as np
import pandas as pd

from pymare.effectsize import (OneSampleEffectSizeConverter, solve_system,
                                select_expressions, TwoSampleEffectSizeConverter)


@pytest.fixture(scope='module')
def data():
    return {
        'm1': np.array([4, 2]),
        'sd1': np.sqrt(np.array([1, 9])),
        'n1': np.array([12, 15]),
        'm2': np.array([5, 2.5]),
        'sd2': np.sqrt(np.array([4, 16])),
        'n2': np.array([12, 16]),
    }


def test_EffectSizeConverter_smoke_test(data):
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


def test_EffectSizeConverter_from_df(data):
    df = pd.DataFrame(data)
    esc = TwoSampleEffectSizeConverter(df)
    assert np.allclose(esc.get_smd(), np.array([-0.61065, -0.13707]), atol=1e-5)


def test_EffectSizeConverter_to_dataset(data):
    esc = TwoSampleEffectSizeConverter(**data)
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
