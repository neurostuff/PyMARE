import pytest
import numpy as np
import pandas as pd

from pymare.effectsizes import (EffectSizeConverter, solve_system,
                                select_expressions)


@pytest.fixture(scope='module')
def data():
    return {
        'y': np.array([4, 2]),
        'v': np.array([1, 9]),
        'n': np.array([12, 15]),
        'y2': np.array([5, 2.5]),
        'v2': np.array([4, 16]),
        'n2': np.array([12, 16]),
        'z': np.array([1.96, -2.58]),
        'p': np.array([0.05, 0.99])
    }


def test_EffectSizeConverter_smoke_test(data):
    esc = EffectSizeConverter(y=data['y'], v=data['v'], n=data['n'])
    assert set(esc.known_vars.keys()) == {'y', 'v', 'n'}
    assert esc.to_d().shape == data['y'].shape
    assert not {'d', 'sd'} - set(esc.known_vars.keys())

    esc = EffectSizeConverter(**data)
    assert set(esc.known_vars.keys()) == set(data.keys())
    assert np.allclose(esc.to_d(), np.array([-0.63246, -0.140744]), atol=1e-5)
    assert np.allclose(esc.to_g(), np.array([-0.61065, -0.13707]), atol=1e-5)

    esc = EffectSizeConverter(z=data['z'])
    assert np.allclose(esc.to_p(), np.array([0.975, 0.005]), atol=1e-3)

    esc = EffectSizeConverter(p=data['p'])
    assert np.allclose(esc.to_z(), np.array([-1.645, 2.326]), atol=1e-3)


def test_EffectSizeConverter_from_df(data):
    df = pd.DataFrame(data)
    esc = EffectSizeConverter(df)
    assert np.allclose(esc.to_g(), np.array([-0.61065, -0.13707]), atol=1e-5)