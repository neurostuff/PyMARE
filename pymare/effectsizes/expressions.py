"""Statistical expressions."""
from collections import defaultdict
from itertools import chain
import re

from sympy import sympify
from sympy.core.compatibility import exec_


_locals = {}
exec_('from sympy.stats import *', _locals)


# Common to one-sample and two-sample procedures
_base_expressions = [
    ('p - cdf(Normal("normal", 0, 1))(z)',),
    ('sd - sqrt(v)',),
    ('sem - sd / sqrt(n)',),
    ('t - y / sem', "One-sample t-test"),
    ('d - y / sd', "Cohen's d (one sample)"),
    ('d - t / sqrt(n)', "Cohen's d (from t)"),
    ('g - d * j', "Hedges' g"),
    # TODO: we currently use Hedges' approximation instead of original J
    # function because the gamma function slows solving down considerably and
    # breaks numpy during lambdification. Need to fix/improve this.
    ('j - (1 - (3 / (4 * (n - 1) - 1)))',
     "Approximate correction factor for Hedges' g"),
]


_two_sample_expressions = [
    ('sd1 - sqrt(v1)',),
    ('sd2 - sqrt(v2)',),
    ('t - (y1 - y2) / sqrt(v1 / n1 + v2 / n2)',
     "Two-sample t-test (unequal variances)"),
    ('sd - sqrt((v1 * (n1 - 1) + v2 * (n2 - 1)) / (n1 + n2 - 2))',
     "Pooled standard deviation (Cohen version)"),
    ('d - (y1 - y2) / sd', "Cohen's d (two-sample)"),
    ('d - t * sqrt(1 / n1 + 1 / n2)', "Cohen's d (two-sample from t)"),
    ('g - d * j', "Hedges' g"),
    ('j - (1 - (3 / (4 * (n1 + n2) - 9)))',
     "Approximate correction factor for Hedges' g")
]


class Expression:
    """Represents a single statistical expression.

    Args:
        expr (str): String representation of the mathematical expression.
        description (str, optional): Optional text description of expression.
        inputs (bool, optional): Indicates whether the expression applies
            in the one-sample case (1), two-sample case (2), or both
            (None).
    """
    def __init__(self, expr, description=None, inputs=None):
        self.expr = expr
        self.description = description
        self.inputs = inputs
        self.sympy = sympify(expr, locals=_locals)
        self.symbols = self.sympy.free_symbols


def _construct_sets():
    one_samp = [Expression(*exp, inputs=1) for exp in _base_expressions]
    two_samp = [Expression(*exp, inputs=2) for exp in _two_sample_expressions]
    for exp in _base_expressions:
        for n in ['1', '2']:
            eq = re.sub(r"(\b(p|d|t|y|n|sd|sem|g|j|v)\b)",
                        r"\g<1>{}".format(n),
                        exp[0])
            two_samp.append(Expression(eq, *exp[1:], inputs=2))
    return one_samp, two_samp


# Construct the 1-sample and 2-sample expression sets at import time
one_sample_expressions, two_sample_expressions = _construct_sets()


def select_expressions(target, known_vars, inputs=1):
    """Select a ~minimal system of expressions needed to solve for the target.

    Args:
        target (str): The named statistic to solve for ('t', 'd', 'g', etc.).
        known_vars (set): A set of strings giving the names of the known
            variables.
        inputs (None, int): Restricts the system to expressions that apply in
            the one-sample case (1), two-sample case (2), or both (None).

    Returns:
        A list of Expression instances, or None if there is no solution.
    """

    exp_dict = defaultdict(list)

    exprs = one_sample_expressions if inputs == 1 else two_sample_expressions
    for exp in exprs:
        if exp.inputs is not None and exp.inputs != inputs:
            continue
        for sym in exp.symbols:
            if sym not in known_vars:
                exp_dict[sym.name].append(exp)

    def df_search(sym, exprs, known, visited):
        """Recursively select expressions needed to solve for sym."""

        if sym not in exp_dict:
            return None

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
