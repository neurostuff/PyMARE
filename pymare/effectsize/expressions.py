"""Statistical expressions."""
from collections import defaultdict
from itertools import chain
import re
import json
from pathlib import Path

from sympy import sympify, Symbol
from sympy.core.compatibility import exec_


_locals = {}
exec_('from sympy.stats import *', _locals)


class Expression:
    """Represents a single statistical expression.

    Args:
        expr (str): String representation of the mathematical expression.
        description (str, optional): Optional text description of expression.
        type (int, optional): Indicates whether the expression applies
            in the one-sample case (1), two-sample case (2), or both (0).
    """
    def __init__(self, expression, description=None, type=0):
        self.expr = expression
        self.description = description
        self.sympy = sympify(expression, locals=_locals)
        self.type = type
        self.symbols = self.sympy.free_symbols


def _load_expressions():
    expressions = []
    path = Path(__file__).parent / 'expressions.json'
    expr_list = json.load(open(path, 'r'))
    for expr in expr_list:
        expr = Expression(**expr)
        expressions.append(expr)

    one_samp = [e for e in expressions if e.type == 1]
    two_samp = [e for e in expressions if e.type == 2]

    return one_samp, two_samp


# Construct the 1-sample and 2-sample expression sets at import time
one_sample_expressions, two_sample_expressions = _load_expressions()


def select_expressions(target, known_vars, type=1):
    """Select a ~minimal system of expressions needed to solve for the target.

    Args:
        target (str): The named statistic to solve for ('t', 'd', 'g', etc.).
        known_vars (set): A set of strings giving the names of the known
            variables.
        type (int): Restricts the system to expressions that apply in
            the one-sample case (1), two-sample case (2), or both (None).

    Returns:
        A list of Expression instances, or None if there is no solution.
    """

    exp_dict = defaultdict(list)

    exprs = one_sample_expressions if type == 1 else two_sample_expressions

    # make sure target exists before going any further
    all_symbols = set().union(*[e.symbols for e in exprs])
    if Symbol(target) not in all_symbols:
        raise ValueError("Target symbol '{}' cannot be found in any of the "
                         "known expressions).".format(target))

    for exp in exprs:
        for sym in exp.symbols:
            if sym not in known_vars:
                exp_dict[sym.name].append(exp)

    def df_search(sym, exprs, known, visited):
        """Recursively select expressions needed to solve for sym."""

        results = []

        for exp in exp_dict[sym]:

            candidates = []

            sym_names = set(s.name for s in exp.symbols)

            # Abort if we're cycling
            if visited & sym_names:
                continue

            new_exprs = list(exprs) + [exp]
            free_symbols = sym_names - known.union({sym})
            _visited = set(visited) | {sym}

            # If there are no more free symbols, we're done
            if not free_symbols:
                results.append((new_exprs, _visited))
                continue

            # Loop over remaining free symbols and recurse
            candidates = [df_search(child, new_exprs, known, _visited)
                          for child in free_symbols]
            candidates = [c for c in candidates if c is not None]

            # Make sure we've covered all free symbols in the expression
            symbols_found = set().union(*[c[1] for c in candidates])
            if free_symbols - symbols_found:
                continue

            # TODO: compact the resulting set, as it could currently include
            # redundant equations.
            merged = list(set().union(*chain([c[0] for c in candidates])))
            results.append((merged, symbols_found))

        if not results:
            return None

        # Order solutions by number of expressions
        results.sort(key=lambda x: len(x[0]))
        return results[0]

    # base case
    candidates = df_search(target, [], known_vars, set())
    return None if not candidates else candidates[0]
