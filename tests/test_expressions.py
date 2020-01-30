import pytest
from sympy.core.sympify import SympifyError
from sympy import Symbol

from pymare.effectsizes.expressions import Expression, select_expressions


def test_Expression_init():
    # Fails because SymPy can't parse expression
    with pytest.raises(SympifyError):
        Expression('This isn"t 29 a valid + expre55!on!')

    exp = Expression("x / 4 * y")
    assert exp.symbols == {Symbol('x'), Symbol('y')}
    assert exp.description is None
    assert exp.inputs is None
    assert exp.metric is None

    exp = Expression("x + y - cos(z)", "Test expression", 1, "OR")
    assert exp.symbols == {Symbol('x'), Symbol('y'), Symbol('z')}
    assert exp.description == "Test expression"
    assert exp.inputs == 1
    assert exp.metric == "OR"
