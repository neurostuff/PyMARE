"""Tools for effect size computation/conversion."""

import warnings
import inspect
from functools import partial

from sympy import sympify, lambdify, nonlinsolve, Symbol

from .expressions import select_expressions


def solve_system(system, known_vars=None):
    """Solve and evaluate a system of SymPy equations given known inputs.
    
    Args:
        system ([sympy.core.expr.Expr]): A list of SymPy expressions defining
            the system to solve.
        known_vars (dict, optional): A dictionary of known variables to use
            when evaluating the solution. Keys are the names of parameters
            (e.g., 'sem', 't'), values are numerical data types (including
            numpy arrays).
    
    Returns:
        A dictionary of newly computed values, where the keys are parameter
        names and the values are numerical data types.

    Notes:
        The returned dictionary contains only keys that were not passed in as
        input (i.e., already known variables will be ignored).
    """
    system = system.copy()

    known_vars = known_vars or {}

    # Get base system of equations and construct symbol dict
    symbols = set().union(*[eq.free_symbols for eq in system])
    symbols = {s.name: s for s in list(symbols)}

    # Add a dummy equation for each known variable
    dummies = set()
    for name in known_vars.keys():
        if name not in symbols:
            continue
        dummy = Symbol('_%s' % name)
        dummies.add(dummy)
        system.append(symbols[name] - dummy)

    # Solve the system for all existing symbols
    symbols = list(symbols.values())
    solutions = nonlinsolve(system, symbols)

    if not len(solutions.args):
        return {}

    # Prepare the dummy list and data args in a fixed order
    dummy_list = list(dummies)
    data_args = [known_vars[var.name.strip('_')] for var in dummy_list]

    # Compute any solved vars via numpy and store in new dict
    results = {}
    for i, sol in enumerate(solutions.args[0]):
        name = symbols[i].name
        free = sol.free_symbols
        if (not (free - dummies) and not
            (len(free) == 1 and list(free)[0].name.strip('_') == name)):
            func = lambdify(dummy_list, sol, 'numpy')
            results[name] = func(*data_args)

    return results


class EffectSizeConverter:
    """Converts between effect size metrics and dependent quantities.

    Args:
        dataset (Dataset, optional): Optional PyMARE dataset to extract input
            quantities from. Other arguments will take precedent over values
            found in the Dataset. Defaults to None.
        y ((float, iterable), optional): Point estimate(s) (e.g., means).
            Defaults to None.
        v ((float, iterable), optional): Variance(s). Defaults to None.
        n ((float, iterable), optional): Sample size(s). Defaults to None.
        t ((float, iterable), optional): t-statistic(s). Defaults to None.
        y2 ((float, iterable), optional): Second set of of point estimates, in
            the two-sample case. Defaults to None.
        v2 ((float, iterable), optional): Second set of variances, in the
            two-sample case. Defaults to None.
        n2 ((float, iterable), optional): [description]. Second set of sample
            sizes, in the two-sample case. Defaults to None.

    Notes:
        All input variables are assumed to reflect study- or analysis-level
        summaries, and are _not_ individual data points. E.g., do not pass in
        a vector of point estimates as `y` and a scalar for the variances `v`.
        The lengths of all inputs must match. This is true even if two sets of
        values are provided (i.e., if y2, v2, or n2 are passed). In this case,
        the pairs reflect study-level summaries for the two groups, and are not
        the raw scores for (potentially different-sized) groups.

        It also follows from this assumption that if two sets of values are
        provided, the data are assumed to come from an independent-samples
        comparison. Paired-sampled comparisons are not handled, and must be
        converted to one-sample summaries prior to initialization.

    """
    def __init__(self, dataset=None, y=None, v=None, n=None, t=None, d=None,
                 y2=None, v2=None, n2=None):
        # Assume equal variances if there are two estimates but only one variance
        if (y is not None and y2 is not None and v is not None
            and v2 is None):
            v2 = v
            warnings.warn("Two sets of estimates were provided, but only one "
                          "variance. Assuming equal variances.")

        # Validate presence of quantities that need to be passed in pairs
        if y2 is not None or v2 is not None or n2 is not None:
            self.inputs = 2
            scope_vars = locals()
            for q1 in ['y', 'v', 'n']:
                q2 = '%s2' % q1
                if ((scope_vars[q1] is not None and scope_vars[q2] is None) or
                    (scope_vars[q2] is None and scope_vars[q1] is not None)):
                    raise ValueError(
                        "There appear to be 2 conditions or groups. Please "
                        "provide either both %s and %s or neither." % (q1, q2))
        else:
            self.inputs = 1

        # Consolidate available variables
        local_vars = locals()
        if dataset is not None:
            dataset_vars = self._extract_from_dataset(dataset)
            local_vars.update(dataset_vars)

        # Set any known variables
        self.known_vars = {}
        args = inspect.getfullargspec(self.__init__).args
        args = list(set(args) - {'self', 'dataset'})
        for var in args:
            if local_vars[var] is not None:
                self.known_vars[var] = local_vars[var]

    def _extract_from_dataset(self, dataset):
        pass

    def __getattr__(self, key):
        if key.startswith('to_'):
            stat = key.replace('to_', '')
            return partial(self.to, stat=stat)

    def to(self, stat):
        """Compute and return values for the specified statistic, if possible.

        Args:
            stat (str): The name of the statistic to compute (e.g., 'd', 'g').
        
        Returns:
            A float or ndarray containing the requested parameter values, if
            successfully computed.

        Notes:
            All values computed via to() are internally cached. Do not try to
            update the instance's known values directly and then recompute
            quantities; any change to input data require initialization of a
            new instance.
        """
        if stat in self.known_vars:
            return self.known_vars[stat]

        known = set(self.known_vars.keys())
        system = select_expressions(target=stat, known_vars=known,
                                    inputs=self.inputs)
        system = [exp.sympy for exp in system]
        result = solve_system(system, self.known_vars)

        if result is None:
            raise ValueError("Unable to solve for statistic '{}' given the "
                             "known quantities ({}).".format(stat, known))

        self.known_vars.update(result)
        return result[stat]
