"""Tools for effect size computation/conversion."""

import warnings
import inspect
from functools import partial

from sympy import sympify, lambdify, Symbol, solve

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

    # Solve the system for all existing symbols.
    # NOTE: previously we used the nonlinsolve() solver instead of solve().
    # for inscrutable reasons, nonlinsolve behaves unpredictably, and sometimes
    # fails to produce solutions even for repeated runs of the exact same
    # inputs. Conclusion: do not use nonlinsolve.
    symbols = list(symbols.values())
    solutions = solve(system, symbols)

    if not len(solutions):
        return {}

    # Prepare the dummy list and data args in a fixed order
    dummy_list = list(dummies)
    data_args = [known_vars[var.name.strip('_')] for var in dummy_list]

    # Compute any solved vars via numpy and store in new dict
    results = {}
    for i, sol in enumerate(solutions[0]):
        name = symbols[i].name
        free = sol.free_symbols
        if (not (free - dummies) and not
            (len(free) == 1 and list(free)[0].name.strip('_') == name)):
            func = lambdify(dummy_list, sol, ['numpy', 'scipy'])
            results[name] = func(*data_args)

    return results


class EffectSizeConverter:
    """Converts between effect size metrics and dependent quantities.

    Args:
        dataset (Dataset, optional): Optional PyMARE dataset to extract input
            quantities from. Other arguments will take precedent over values
            found in the Dataset. Defaults to None.
        **kwargs: Optional keyword arguments providing additional inputs. All
            values must be floats, 1d ndarrays, or any iterable that can be
            converted to an ndarray. All variables must have the same length.
            Allowable variable currently include:
            * y: Point estimate with unbounded distributions--most commonly,
                study- or experiment-level estimates of means.
            * v: Sampling variance of the mean.
            * sd: Sample standard deviation.
            * n: Sample size.
            * sem: Standard error of the mean.
            * d: Cohen's d.
            * g: Hedges' g.
            * t: t-statistic.
            * z: z-score.
            In addition, for most of the above (all but 't', 'd', 'g', 'z'),
            one can pass in a second set of values, representing a second group
            of estimates, by appending any name with '2'--e.g., y2, v2, sd2,
            n2, etc. Note that if any such variable is passed, the
            corresponding estimate for the first group must also be passed--
            e.g., if `v2` is set, `v` must also be provided.

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
    def __init__(self, dataset=None, **kwargs):
        # Assume equal variances if there are two estimates but only one variance
        if (kwargs.get('y') is not None and kwargs.get('y2') is not None
            and kwargs.get('v') is not None and kwargs.get('v2') is None):
            kwargs['v2'] = kwargs['v']
            warnings.warn("Two sets of estimates were provided, but only one "
                          "variance. Assuming equal variances.")

        # Validate presence of quantities that need to be passed in pairs
        var_names = list(kwargs.keys())
        if any([name.endswith('2') for name in var_names]):
            self.inputs = 2
            all_vars = set([v.strip('2') for v in var_names])
            pair_vars = all_vars - {'t', 'z', 'd', 'p'}
            for q1 in pair_vars:
                q2 = '%s2' % q1
                if ((kwargs.get(q1) is not None and kwargs.get(q2) is None) or
                    (kwargs.get(q2) is None and kwargs.get(q1) is not None)):
                    raise ValueError(
                        "There appear to be 2 groups of estimates. Please "
                        "provide both of %s and %s or neither." % (q1, q2))
        else:
            self.inputs = 1

        # Extract variables from dataset if passed
        self.known_vars = {} if dataset is None else self._from_dataset(dataset)

        self.known_vars.update(kwargs)

    def _from_dataset(self, dataset):
        return {}

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
