import pytest
import numpy as np
import pandas as pd

from pymare.effectsizes import (OneSampleEffectSizeConverter, solve_system,
                                select_expressions, TwoSampleEffectSizeConverter)


@pytest.fixture(scope='module')
def data():
    return {
        'y1': np.array([4, 2]),
        'v1': np.array([1, 9]),
        'n1': np.array([12, 15]),
        'y2': np.array([5, 2.5]),
        'v2': np.array([4, 16]),
        'n2': np.array([12, 16]),
        'z': np.array([1.96, -2.58]),
        'p': np.array([0.05, 0.99])
    }


def test_EffectSizeConverter_smoke_test(data):
    esc = OneSampleEffectSizeConverter(y=data['y1'], v=data['v1'], n=data['n1'])
    assert set(esc.known_vars.keys()) == {'y', 'v', 'n'}
    assert esc.get_d().shape == data['y1'].shape
    assert not {'d', 'sd'} - set(esc.known_vars.keys())

    esc = TwoSampleEffectSizeConverter(**data)
    assert set(esc.known_vars.keys()) == set(data.keys())
    assert np.allclose(esc.get_d(), np.array([-0.63246, -0.140744]), atol=1e-5)
    assert np.allclose(esc.get_g(), np.array([-0.61065, -0.13707]), atol=1e-5)

    esc = OneSampleEffectSizeConverter(z=data['z'])
    assert np.allclose(esc.get_p(), np.array([0.975, 0.005]), atol=1e-3)

    esc = OneSampleEffectSizeConverter(p=data['p'])
    assert np.allclose(esc.get_z(), np.array([-1.645, 2.326]), atol=1e-3)


def test_EffectSizeConverter_from_df(data):
    df = pd.DataFrame(data)
    esc = TwoSampleEffectSizeConverter(df)
    assert np.allclose(esc.get_g(), np.array([-0.61065, -0.13707]), atol=1e-5)


def test_EffectSizeConverter_to_dataset(data):
    esc = TwoSampleEffectSizeConverter(**data)
    X = np.array([1, 2])
    dataset = esc.to_dataset(X=X, X_names=['dummy'])
    assert dataset.__class__.__name__ == 'Dataset'
    assert dataset.X_names == ['intercept', 'dummy']
