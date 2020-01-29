from typing import Iterable
import warnings
import inspect
from functools import partial

from sympy import sympify, lambdify, nonlinsolve, Symbol

from pymare import Dataset


BASE_EQUATIONS = [
    'sd - sqrt(v)',
    'sem - sd / sqrt(n)',
    't - y / sem'
]

EFFECT_SIZE_EQUATIONS = [
    'd - y / sd',
    'g - y * j / sd',
    'j - 1 - (3 / (4 * (n - 1) -1))'
]


def get_system(mode='base', target=None, known_vars=None):
    # Build list of candidate expressions based on the mode
    if mode == 'base':
        exprs = BASE_EQUATIONS
    else:
        exprs = EFFECT_SIZE_EQUATIONS

    exprs = [sympify(expr) for expr in exprs]

    # If a target is passed, check if any single equation is sufficient
    if target is not None:
        known_vars = known_vars or {}
        known_set = set(known_vars.keys()) | set(target)
        for exp in exprs:
            free = set(s.name for s in exp.free_symbols)
            if target not in free:
                continue
            if not free - known_set:
                return [exp]
    
    return exprs


def solve_system(system, known_vars):

    system = system.copy()

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

    def __init__(
            self,
            dataset: Dataset=None,
            y: Iterable[float]=None,
            v: Iterable[float]=None,
            n: Iterable[float]=None,
            t: Iterable[float]=None,
            d: Iterable[float]=None,
            y2: Iterable[float]=None,
            v2: Iterable[float]=None,
            n2: Iterable[float]=None, 
            paired: bool=True):
        """[summary]
        
        Args:
            dataset (Dataset, optional): [description]. Defaults to None.
            y (Iterable[float], optional): [description]. Defaults to None.
            v (Iterable[float], optional): [description]. Defaults to None.
            n (Iterable[float], optional): [description]. Defaults to None.
            t (Iterable[float], optional): [description]. Defaults to None.
            y2 (Iterable[float], optional): [description]. Defaults to None.
            v2 (Iterable[float], optional): [description]. Defaults to None.
            n2 (Iterable[float], optional): [description]. Defaults to None.
            paired (bool, optional): [description]. Defaults to True.
        """
        # If two ys but only one sample size are provided and the test is
        # paired, assume the same sample size in both conditions
        if (y is not None and y2 is not None
            and n is not None and n2 is None and paired):
            n2 = n

        # Assume equal variances if there are two estimates but only one variance
        if (y is not None and y2 is not None and v is not None
            and v2 is None):
            v2 = v
            warnings.warn("Two sets of estimates were provided, but only one "
                          "variance. Assuming equal variances.")

        # Validate presence of quantities that need to be passed in pairs
        if y2 is not None or v2 is not None or n2 is not None:
            self.two_sample = True
            scope_vars = locals()
            for q1 in ['y', 'v', 'n']:
                q2 = '%s2' % q1
                if ((scope_vars[q1] is not None and scope_vars[q2] is None) or
                    (scope_vars[q2] is None and scope_vars[q1] is not None)):
                    raise ValueError(
                        "There appear to be 2 conditions or groups. Please "
                        "provide either both %s and %s or neither." % (q1, q2))
        else:
            self.two_sample = False

        # Consolidate available variables
        local_vars = locals()
        if dataset is not None:
            dataset_vars = self._extract_from_dataset(dataset)
            local_vars.update(dataset_vars)

        # Set any known variables
        self.known_vars = {}
        args = inspect.getfullargspec(self.__init__).args
        args = list(set(args) - {'self', 'dataset', 'paired'})
        for var in args:
            if local_vars[var] is not None:
                self.known_vars[var] = local_vars[var]

        system = get_system(mode='base')
        new_vars = solve_system(system, self.known_vars)
        self.known_vars.update(new_vars)

    def _extract_from_dataset(self, dataset):
        pass

    def __getattr__(self, key):
        if key.startswith('to_'):
            stat = key.replace('to_', '')
            return partial(self.to, stat=stat)

    def to(self, stat):

        if stat in self.known_vars:
            return self.known_vars[stat]

        if self.two_sample:
            mode = 'twosample'
        else:
            mode = 'paired' if self.paired else 'onesample'

        system = get_system(mode=mode, target=stat, known_vars=self.known_vars)
        result = solve_system(system, self.known_vars)
        self.known_vars.update(result)
        return result[stat]
