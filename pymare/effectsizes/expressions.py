"""Statistical expressions."""
from collections import defaultdict
from itertools import chain

from sympy import sympify
from sympy.core.compatibility import exec_


_locals = {}
exec_('from sympy.stats import *', _locals)


class Expression:
    """Represents a single statistical expression.
    
    Args:
        expr (str): String representation of the mathematical expression.
        description (str, optional): Optional text description of expression.
        inputs (bool, optional): Indicates whether the expression applies
            in the one-sample case (1), two-sample case (2), or both
            (None).
        metric (str, optional): Name of metric for which this expression
            applies. Defaults to None (all inputs with unlabeled metricss).
    """
    def __init__(self, expr, description=None, inputs=None, metric=None):
        self.expr = expr
        self.description = description
        self.inputs = inputs
        self.metric = metric

        self.sympy = sympify(expr, locals=_locals)
        self.symbols = self.sympy.free_symbols


EXPRESSIONS = [

    # Common to one-sample and two-sample procedures
    Expression('sd - sqrt(v)'),
    Expression('sem - sd / sqrt(n)'),
    Expression('p - cdf(Normal("normal", 0, 1))(z)'),

    # One-sample procedures
    Expression('t - y / sem', "One-sample t-test", inputs=1),
    Expression('d - y / sd', "Cohen's d (one sample)", inputs=1),
    Expression('d - t / sqrt(n)', "Cohen's d (from t)", inputs=1),
    Expression('g - d * j', "Hedges' g"),
    # TODO: we currently use Hedges' approximation instead of original J
    # function because the gamma function slows solving down considerably and
    # breaks numpy during lambdification. Need to fix/improve this.
    Expression('j - (1 - (3 / (4 * (n - 1) - 1)))',
               "Approximate correction factor for Hedges' g", inputs=1),

    # Two-sample procedures
    Expression('sd2 - sqrt(v2)', inputs=2),
    Expression('t - (y - y2) / sqrt(v / n + v2 / n2)',
               "Two-sample t-test (unequal variances)", inputs=2),
    Expression('sd_pooled - sqrt((v * (n - 1) + v2 * (n2 - 1)) / (n + n2 - 2))',
               "Pooled standard deviation (Cohen version)", inputs=2),
    Expression('d - (y - y2) / sd_pooled', "Cohen's d (two-sample)",
               inputs=2),
    Expression('d - t * sqrt(1 / n + 1 / n2)', "Cohen's d (two-sample from t)",
               inputs=2),
    Expression('j - (1 - (3 / (4 * (n + n2) - 9)))',
               "Approximate correction factor for Hedges' g", inputs=2)
]


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

    for exp in EXPRESSIONS:
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
