import pytest
from sympy.core.sympify import SympifyError
from sympy import Symbol

from pymare.effectsizes.expressions import Expression, select_expressions


def _symbol_set(*args):
    return set([Symbol(a) for a in args])


def test_Expression_init():
    # Fails because SymPy can't parse expression
    with pytest.raises(SympifyError):
        Expression('This isn"t 29 a valid + expre55!on!')

    exp = Expression("x / 4 * y")
    assert exp.symbols == _symbol_set('x', 'y')
    assert exp.description is None
    assert exp.inputs is None

    exp = Expression("x + y - cos(z)", "Test expression", 1)
    assert exp.symbols == _symbol_set('x', 'y', 'z')
    assert exp.description == "Test expression"
    assert exp.inputs == 1


def test_select_expressions():
    exps = select_expressions('t', {'d', 'n'})
    assert len(exps) == 1
    assert exps[0].symbols == _symbol_set('t', 'd', 'n')

    assert select_expressions('t', {'d'}) is None

    exps = select_expressions('g', known_vars={'y', 'n', 'v'})
    assert len(exps) == 4
    targets = {'j - 1 + 3/(4*n - 5)', 'sd - sqrt(v)', '-d*j + g', 'd - y/sd'}
    assert set([str(e.sympy) for e in exps]) == targets

    # For 2-sample test, need n2 as well
    assert select_expressions('t', {'d', 'n'}, 2) is None

    exps = select_expressions('t', {'d', 'n1', 'n2'}, 2)
    assert len(exps) == 1
    assert exps[0].symbols == _symbol_set('t', 'd', 'n1', 'n2')

    exps = select_expressions('g', {'y1', 'y2', 'n1', 'n2', 'v1', 'v2'}, inputs=2)
    assert len(exps) == 4
    targets = ['sd - sqrt((v1*(n1 - 1) + v2*(n2 - 1))/(n1 + n2 - 2))',
               'd - (y1 - y2)/sd', '-d*j + g',
               'j - 1 + 3/(4*n1 + 4*n2 - 9)']
    assert set([str(e.sympy) for e in exps]) == set(targets)

    exps = select_expressions('p', {'z'})
    assert len(exps) == 1
