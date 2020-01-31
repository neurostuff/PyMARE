import pytest
import numpy as np

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
        'n2': np.array([12, 16])
    }


def test_EffectSizeConverter_smoke_test(data):
    esc = EffectSizeConverter(y=data['y'], v=data['v'], n=data['n'])
    assert set(esc.known_vars.keys()) == {'y', 'v', 'n'}
    assert esc.to_d().shape == data['y'].shape
    assert not {'d', 'sd'} - set(esc.known_vars.keys())

    esc = EffectSizeConverter(**data)
    assert set(esc.known_vars.keys()) == set(data.keys())
    print(esc.to_g())
    assert np.allclose(esc.to_d(), np.array([-0.63246, -0.140744]), atol=1e-5)
    assert np.allclose(esc.to_g(), np.array([-0.61065, -0.13707]), atol=1e-5)
