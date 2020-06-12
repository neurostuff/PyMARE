"""Tools for effect size computation/conversion."""

import warnings
from functools import partial
from abc import ABCMeta
from collections import defaultdict

import numpy as np
from sympy import sympify, lambdify, Symbol, solve

from .expressions import select_expressions
from pymare import Dataset


SYMPY_MODULES = ['numpy', 'scipy']


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

    # solver will return a dict if there's only one non-dummy expression
    if isinstance(solutions, dict):
        solutions = [list(solutions.values())]

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
            func = lambdify(dummy_list, sol, modules=SYMPY_MODULES)
            results[name] = func(*data_args)

    return results


class EffectSizeConverter(metaclass=ABCMeta):
    """Base class for effect size converters."""
    def __init__(self, data=None, **kwargs):

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if data is not None:
            kwargs = self._collect_variables(data, kwargs)

        # Do any subclass-specific validation
        kwargs = self._validate(kwargs)

        # Scalars are fine, but lists and tuples break lambdified expressions
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                kwargs[k] = np.array(v)

        self.known_vars = {}
        self._system_cache = defaultdict(dict)
        self.update_data(**kwargs)

    @staticmethod
    def _collect_variables(data, kwargs):
        # consolidate variables from pandas DF and keyword arguments, giving
        # precedence to the latter.
        kwargs = kwargs.copy()
        df_cols = {col: data.loc[:, col].values for col in data.columns}
        df_cols.update(kwargs)
        return df_cols

    def _validate(self, kwargs):
        return kwargs

    def __getattr__(self, key):
        if key.startswith('get_'):
            stat = key.replace('get_', '')
            return partial(self.get, stat=stat)

    def update_data(self, incremental=False, **kwargs):
        """Update instance data.

        Args:
            incremental (bool): If True, updates data incrementally (i.e.,
                existing data will be preserved unless they're overwritten by
                incoming keys). If False, all existing data is dropped first.
            kwargs: Data values or arrays; keys are the names of the
                quantities. All inputs to __init__ are valid.
        """
        if not incremental:
            self.known_vars = {}
        self.known_vars.update(kwargs)

    def _get_system(self, stat):
        # Retrieve a system of equations capable of solving for desired stat.
        known = set([k for k, v in self.known_vars.items() if v is not None])

        # get system from cache if available
        cached = self._system_cache.get(stat, {})
        for k, system in cached.items():
            if known.issuperset(k):
                return system

        # otherwise try to get a sufficient system
        exprs = select_expressions(target=stat, known_vars=known,
                                   type=self._type)
        if exprs is None:
            return None
        system = [exp.sympy for exp in exprs]

        # update the cache
        if system:
            free_syms = set().union(*[exp.symbols for exp in exprs])
            set_key = frozenset([s.name for s in free_syms])
            self._system_cache[stat][set_key] = system

        return system

    def to_dataset(self, measure, **kwargs):
        measure = measure.lower()
        y = self.get(measure)
        v = self.get('v_{}'.format(measure), error=False)
        try:
            n = self.get('n')
        except:
            n = None
        return Dataset(y=y, v=v, n=n, **kwargs)

    def get(self, stat, error=True):
        """Compute and return values for the specified statistic, if possible.

        Args:
            stat (str): The name of the quantity to retrieve.
            error (bool): Specifies behavior in the event that the requested
                quantity cannot be computed. If True (default), raises an
                exception. If False, returns None.

        Returns:
            A float or ndarray containing the requested parameter values, if
            successfully computed.

        Notes:
            All values computed via get() are internally cached. Do not try to
            update the instance's known values directly; any change to input
            data require either initialization of a new instance, or a call to
            update_data().
        """
        stat = stat.lower()

        if stat in self.known_vars:
            return self.known_vars[stat]

        system = self._get_system(stat)
        if system is not None:
            result = solve_system(system, self.known_vars)

        if error and (system is None or result is None):
            known = list(self.known_vars.keys())
            raise ValueError("Unable to solve for statistic '{}' given the "
                             "known quantities ({}).".format(stat, known))

        self.known_vars.update(result)
        return result[stat]


class OneSampleEffectSizeConverter(EffectSizeConverter):
    """Effect size converter for metric involving a single group/set of scores.

    Args:
        data (DataFrame): Optional pandas DataFrame to extract variables from.
            Column names must match the controlled names listed below for
            kwargs. If additional kwargs are provided, they will take
            precedence over the values in the data frame.
        m (array-like): Means or other continuous estimates
        sd (array-like): Standard deviations
        n (array-like): Sample sizes
        r (array-like): Correlation coefficients
        **kwargs: Optional keyword arguments providing additional inputs. All
            values must be floats, 1d ndarrays, or any iterable that can be
            converted to an ndarray. All variables must have the same length.

    Notes:
        All input variables are assumed to reflect study- or analysis-level
        summaries, and are _not_ individual data points. E.g., do not pass in
        a vector of point estimates as `m` and a scalar for the SDs `sd`.
        The lengths of all inputs must match.
    """
    _type = 1

    def __init__(self, data=None, m=None, sd=None, n=None, r=None, **kwargs):
        super().__init__(data, m=m, sd=sd, n=n, r=r, **kwargs)

    def to_dataset(self, measure='RM', **kwargs):
        """Get a Pymare Dataset with y and v mapped to the specified measure.

        Args:
            measure (str): The measure to map to the Dataset's y and v
                attributes (where y is the desired measure, and v is its
                variance). Valid values include:
                    * 'RM': Raw mean of the group.
                    * 'SM': Standardized mean. This is often called Hedges g.
                      (one-sample), or equivalently, Cohen's one-sample d with
                      a bias correction applied.
                    * 'D': Cohen's d. Note that no bias correction is applied
                      (use 'SM' instead).
                    * 'R': Raw correlation coefficient.
                    * 'ZR': Fisher z-transformed correlation coefficient.
            kwargs: Optional keyword arguments to pass onto the Dataset
                initializer. Provides a way of supplementing the generated y
                and v arrays with additional arguments (e.g., X, X_names, n).
                See pymare.Dataset docs for details.

        Returns:
            A pymare.Dataset instance.

        Notes:
            Measures 'RM', 'SM', and 'D' require m, sd, and n as inputs.
            Measures 'R' and 'ZR' require r and n as inputs.
        """
        return super().to_dataset(measure, **kwargs)


class TwoSampleEffectSizeConverter(EffectSizeConverter):
    """Effect size converter for two-sample comparisons.

    Args:
        data (DataFrame): Optional pandas DataFrame to extract variables from.
            Column names must match the controlled names listed below for
            kwargs. If additional kwargs are provided, they will take
            precedence over the values in the data frame.
        m1 (array-like): Means for group 1
        m2 (array-like): Means for group 2
        sd1 (array-like): Standard deviations for group 1
        sd2 (array-like): Standard deviations for group 2
        n1 (array-like): Sample sizes for group 1
        n2 (array-like): Sample sizes for group 2
        **kwargs: Optional keyword arguments providing additional inputs. All
            values must be floats, 1d ndarrays, or any iterable that can be
            converted to an ndarray.

    Notes:
        All input variables are assumed to reflect study- or analysis-level
        summaries, and are _not_ individual data points. E.g., do not pass in
        a vector of point estimates as `m1` and a scalar for the SDs `sd1`.
        The lengths of all inputs must match. All variables must be passed in
        as pairs (e.g., if m1 is provided, m2 must also be provided).

        When using the TwoSampleEffectSizeConverter, it is assumed that the
        variable pairs are from independent samples. Paired-sampled comparisons
        are not currently supported.
    """
    _type = 2

    def __init__(self, data=None, m1=None, m2=None, sd1=None, sd2=None,
                 n1=None, n2=None, **kwargs):
        super().__init__(data, m1=m1, m2=m2, sd1=sd1, sd2=sd2, n1=n1, n2=n2,
                         **kwargs)

    def _validate(self, kwargs):
        # Validate that all inputs were passed in pairs
        var_names = set([v.strip('[12]') for v in kwargs.keys()])
        pair_vars = var_names - {'d'}
        for var in pair_vars:
            name1, name2 = '%s1' % var, '%s2' % var
            var1, var2 = kwargs.get(name1), kwargs.get(name2)
            if (var1 is None) != (var2 is None):
                raise ValueError(
                    "Input variable '{}' must be provided in pairs; please "
                    "provide both {} and {} (or neither)."
                    .format(var, name1, name2))
        return kwargs

    def to_dataset(self, measure='SMD', **kwargs):
        """Get a Pymare Dataset with y and v mapped to the specified measure.

        Args:
            measure (str): The measure to map to the Dataset's y and v
                attributes (where y is the desired measure, and v is its
                variance). Valid values include:
                    * 'RMD': Raw mean difference between groups.
                    * 'SMD': Standardized mean difference between groups. This
                      is often called Hedges g, or equivalently, Cohen's d with
                      a bias correction applied.
                    * 'D': Cohen's d. Note that no bias correction is applied
                      (use 'SMD' instead).
            kwargs: Optional keyword arguments to pass onto the Dataset
                initializer. Provides a way of supplementing the generated y
                and v arrays with additional arguments (e.g., X, X_names, n).
                See pymare.Dataset docs for details.

        Returns:
            A pymare.Dataset instance.

        Notes:
            All measures require that m1, m2, sd1, sd2, n1, and n2 be passed in
            as inputs (or be solvable from the passed inputs).
        """
        return super().to_dataset(measure, **kwargs)


def compute_measure(measure, data=None, comparison='infer', return_type='tuple',
                    m=None, sd=None, n=None, r=None, m1=None, m2=None,
                    sd1=None, sd2=None, n1=None, n2=None, **dataset_kwargs):
    """Wrapper that auto-detects and applies the right converter class.

    Args:
        measure (str): The desired output effect size measure. Valid values are
            listed below, with the required named inputs in parentheses:
            * 'RM' (m, sd, n): Raw mean of the group.
            * 'SM' (m, sd, n): Standardized mean. This is often called Hedges
                g. (one-sample), or equivalently, Cohen's one-sample d with a
                bias correction applied.
            * 'R' (r, n): Raw correlation coefficient.
            * 'ZR' (r, n): Fisher z-transformed correlation coefficient.
            * 'RMD' (m1, m2, sd1, sd2, n1, n2): Raw mean difference between
                groups.
            * 'SMD' (m1, m2, sd1, sd2, n1, n2): Standardized mean difference
                between groups. This is often called Hedges g, or equivalently,
                Cohen's d with a bias correction applied.
            * 'D' (m, sd, n, or m1, m2, sd1, sd2, n1, n2): Cohen's d. No bias
                correction is applied (for that, use 'SM' or 'SMD' instead).
                Note that 'D' can be either one-sample or two-sample. This is
                specified in type, or (if type=='infer'), inferred from the
                passed arguments.
        data (DataFrame, optional): A pandas DataFrame to extract variables
            from. Column names must match the names of other args ('m', 'sd',
            'n2', etc.). If both a DataFrame and keyword arguments are
            provided, the two will be merged, with variables passed as separate
            arguments taking precedence over DataFrame columns in the event of
            a clash.
        comparison (str, int, optional): The type of originating comparison.
            This is currently unnecessary, as the type can be deterministically
            inferred from the input arguments and measure, but may become
            necessary in future, and provides a way of imposing constraints on
            code embedded in larger pipelines. Valid values:
                'infer' (*default): Infer the type of comparison from the input
                arguments. Currently
            1: One-group comparison. Must be accompanied by some/all of the
                following named variables: m, sd, n, r.
            2: Two-group comparison. Independent samples are assumed. Must be
                accompanied by some/all of the following named variables: m1,
                m2, sd1, sd2, n1, n2.
        return_type (str, optional): Controls what gets returned. Valid values:
            'tuple': A 2-tuple, where the first element is a 1-d array
                containing the computed estimates (i.e., y), and the second
                element is a 1-d array containing the associated sampling
                variances.
            'dict': A dictionary with keys 'y' and 'v' that map to the arrays
                described for 'tuple'.
            'dataset': A pymare Dataset instance, with y and v attributes set
                to the corresponding arrays. Note that additional keyword
                arguments can be passed onto the Dataset init via kwargs.
            'converter': The EffectSizeConverter class internally initialized
                to handle the desired computation. The target measures will
                have already been computed (and hence, cached), and can be
                retrieved via get_('{measure}') and get_('v_{measure}')
        m (array-like): Means or other estimates in single-group case
        sd (array-like): Standard deviations in single-group case
        n (array-like): Sample sizes in single-group case
        r (array-like): Correlation coefficients
        m1 (array-like): Means for group 1
        m2 (array-like): Means for group 2
        sd1 (array-like): Standard deviations for group 1
        sd2 (array-like): Standard deviations for group 2
        n1 (array-like): Sample sizes for group 1
        n2 (array-like): Sample sizes for group 2
        dataset_kwargs (optional): Optional keyword arguments passed on to the
            Dataset initializer. Ignored unless return_type == 'dataset'.

        Returns:
            A tuple, dict, or pymare.Dataset, depending on `return_type`.
        """

    var_args = dict(m=m, sd=sd, n=n, r=r, m1=m1, m2=m2, sd1=sd1, sd2=sd2,
                    n1=n1, n2=n2)
    var_args = {k: v for k, v in var_args.items() if v is not None}

    if data is not None:
        var_args = EffectSizeConverter._collect_variables(data, var_args)

    valid_measures = {'RM', 'SM', 'R', 'ZR', 'RMD', 'SMD', 'D'}
    if measure not in valid_measures:
        raise ValueError("Invalid measures '{}'; must be one of {}."
                         .format(measure, valid_measures))

    # Select or infer converter class
    if comparison == 'infer':

        one_samp_inputs = {'m', 'sd', 'n', 'r'}
        two_samp_inputs = {'m1', 'm2', 'sd1', 'sd2', 'n1', 'n2'}

        if measure in {'RM', 'SM', 'R', 'ZR'}:
            comparison = 1
        elif measure in {'RMD', 'SMD'}:
            comparison = 2
        elif measure in {'D'}:
            arg_set = set(var_args.keys())
            if (arg_set & one_samp_inputs) and not (arg_set & two_samp_inputs):
                comparison = 1
            elif (arg_set & two_samp_inputs) and not (arg_set & one_samp_inputs):
                comparison = 2
            else:
                raise ValueError(
                    "Requested measure (D) and provided data arguments ({}) "
                    "are insufficient to determine comparison; either provide"
                    " data consistent with only one-group or two-group "
                    "measures, or explicitly set comparison to 1 or 2."
                    .format(arg_set))

    if comparison == 1:
        conv_cls = OneSampleEffectSizeConverter
    elif comparison == 2:
        conv_cls = TwoSampleEffectSizeConverter
    else:
        raise ValueError("Invalid comparison type '{}'! Valid values are "
                         "'infer', 1, and 2.".format(comparison))

    conv = conv_cls(**var_args)
    y = conv.get(measure)
    v = conv.get('v_{}'.format(measure))

    return_type = return_type.lower()
    if return_type == 'tuple':
        return (y, v)
    elif return_type == 'dict':
        return {'y': y, 'v': v}
    elif return_type == 'dataset':
        return conv.to_dataset(measure, **dataset_kwargs)
    elif return_type == 'converter':
        return conv
    else:
        raise ValueError("Invalid return_type value '{}'. Must be one of "
                         "'tuple', 'dict', 'dataset', or 'converter'."
                         .format(return_type))
