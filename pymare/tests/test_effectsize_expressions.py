"""Tests for pymare.effectsize.expressions."""
import pytest
from sympy import Symbol
from sympy.core.sympify import SympifyError

from pymare.effectsize.expressions import Expression, select_expressions


def _symbol_set(*args):
    return set([Symbol(a) for a in args])


def test_Expression_init():
    # Fails because SymPy can't parse expression
    with pytest.raises(SympifyError):
        Expression('This isn"t 29 a valid + expre55!on!')

    exp = Expression("x / 4 * y")
    assert exp.symbols == _symbol_set("x", "y")
    assert exp.description is None
    assert exp.type == 0

    exp = Expression("x + y - cos(z)", "Test expression", 1)
    assert exp.symbols == _symbol_set("x", "y", "z")
    assert exp.description == "Test expression"
    assert exp.type == 1


def test_select_expressions():
    exps = select_expressions("sd", {"d", "m"})
    assert len(exps) == 1
    assert exps[0].symbols == _symbol_set("sd", "d", "m")

    assert select_expressions("v_d", {"d"}) is None

    exps = select_expressions("sm", known_vars={"m", "n", "sd"})
    assert len(exps) == 3
    targets = {"j - 1 + 3/(4*n - 5)", "-d*j + sm", "d - m/sd"}
    assert set([str(e.sympy) for e in exps]) == targets

    assert select_expressions("v_d", {"d", "n"}, 2) is None

    exps = select_expressions("d", {"m1", "m2", "sd1", "sd2", "n1", "n2"}, 2)
    assert len(exps) == 2
    target = _symbol_set("d", "m1", "m2", "sdp")
    assert exps[0].symbols == target or exps[1].symbols == target
