"""Tools for effect size computation/conversion."""

import warnings
from functools import partial
from abc import ABCMeta

from sympy import sympify, lambdify, Symbol, solve

from .expressions import select_expressions
from pymare import Dataset


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

    if not solutions:
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


class EffectSizeConverter(metaclass=ABCMeta):
    """Base class for effect size converters."""
    def __init__(self, **kwargs):
        self.known_vars = kwargs

    def __getattr__(self, key):
        if key.startswith('to_'):
            stat = key.replace('to_', '')
            return partial(self.to, stat=stat)

    def to_dataset(self, estimate='g', **kwargs):
        y = self.to(estimate)
        v = self.known_vars.get('v_{}'.format(estimate))
        n = self.known_vars.get('n')
        return Dataset(y=y, v=v, n=n, **kwargs)

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
                                    inputs=self._inputs)
        system = [exp.sympy for exp in system]
        result = solve_system(system, self.known_vars)

        if result is None:
            raise ValueError("Unable to solve for statistic '{}' given the "
                             "known quantities ({}).".format(stat, known))

        self.known_vars.update(result)
        return result[stat]


class OneSampleEffectSizeConverter(EffectSizeConverter):
    """Effect size converter for one-sample or paired comparisons.

    Args:
        data (DataFrame): Optional pandas DataFrame to extract variables from.
            Column names must match the controlled names listed below for
            kwargs. If additional kwargs are provided, they will take
            precedence over the values in the data frame.
        **kwargs: Optional keyword arguments providing additional inputs. All
            values must be floats, 1d ndarrays, or any iterable that can be
            converted to an ndarray. All variables must have the same length.
            Allowable variables currently include:
            * y: Point estimate with unbounded distributions--most commonly,
                study- or experiment-level estimates of means.
            * v: Sampling variance of the mean.
            * sd: Sample standard deviation.
            * n: Sample size.
            * se: Standard error of the mean.
            * d: Cohen's d.
            * g: Hedges' g.
            * t: t-statistic.
            * z: z-score.
            * p: p-value.

    Notes:
        All input variables are assumed to reflect study- or analysis-level
        summaries, and are _not_ individual data points. E.g., do not pass in
        a vector of point estimates as `y` and a scalar for the variances `v`.
        The lengths of all inputs must match.
    """
    _inputs = 1

    def __init__(self, data=None, **kwargs):
        
        if data is not None:
            df_cols = {col: data.loc[:, col].values for col in data.columns}
            kwargs = dict(**df_cols, **kwargs)

        super().__init__(**kwargs)


class TwoSampleEffectSizeConverter(EffectSizeConverter):
    """Effect size converter for two-sample comparisons.

    Args:
        data (DataFrame): Optional pandas DataFrame to extract variables from.
            Column names must match the controlled names listed below for
            kwargs. If additional kwargs are provided, they will take
            precedence over the values in the data frame.
        **kwargs: Optional keyword arguments providing additional inputs. All
            values must be floats, 1d ndarrays, or any iterable that can be
            converted to an ndarray. All variables must have the same length.
            All variables must be passed in pairs. Allowable variables
            currently include:
            * y1, y2: Point estimates with unbounded distributions--most
                commonly, study- or experiment-level estimates of means.
            * v1, v2: Sampling variances of the means.
            * sd1, sd2: Sample standard deviations.
            * n1, n2: Sample sizes.
            * sem1, sem2: Standard errors of the means.
            In addition, for most of the above (all but 't', 'd', 'g', 'p', 'z'),
            one can pass in a second set of values, representing a second group
            of estimates, by appending any name with '2'--e.g., y2, v2, sd2,
            n2, etc. Note that if any such variable is passed, the
            corresponding estimate for the first group must also be passed--
            e.g., if `v2` is set, `v` must also be provided.

    Notes:
        All input variables are assumed to reflect study- or analysis-level
        summaries, and are _not_ individual data points. E.g., do not pass in
        a vector of point estimates as `y` and a scalar for the variances `v`.
        The lengths of all inputs must match. Note that the variable pairs
        (e.g., y1 and y2) reflect study-level summaries for the two groups, and
        are not the raw scores for (potentially different-sized) groups.

        When using the TwoSampleEffectSizeConverter, it is assumed that the
        paired inputs are from independent samples. Paired-sampled comparisons
        are not supported (use the OneSampleEffectSizeConverter instead).
    """
    _inputs = 2

    def __init__(self, data=None, **kwargs):

        if data is not None:
            df_cols = {col: data.loc[:, col].values for col in data.columns}
            kwargs = dict(**df_cols, **kwargs)

        # Validate that all inputs were passed in pairs
        var_names = set([v.strip('[12]') for v in kwargs.keys()])
        pair_vars = var_names - {'t', 'z', 'd', 'p'}
        for var in pair_vars:
            name1, name2 = '%s1' % var, '%s2' % var
            var1, var2 = kwargs.get(name1), kwargs.get(name2)
            if (var1 is None) != (var2 is None):
                raise ValueError(
                    "Input variable '{}' must be provided in pairs; please "
                    "provide both {} and {} (or neither)." % (var, q1, q2))

        super().__init__(**kwargs)
