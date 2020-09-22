from abc import abstractmethod

import numpy as np
import scipy.stats as ss

from .estimators import BaseEstimator
from ..results import CombinationTestResults


class CombinationTest(BaseEstimator):
    """Base class for methods based on combining p/z values."""
    def __init__(self, mode='one-sided'):
        mode = mode.lower()
        if not (mode.startswith('one') or mode.startswith('two') or
                mode.startswith('conc')):
            raise ValueError("Invalid mode; must be one of 'one-sided', "
                             "'two-sided', or 'concordant' (or 'one', 'two',"
                             "or 'conc' for short).")
        self.mode = mode

    @abstractmethod
    def p_value(self, z, *args, **kwargs):
        pass

    def _fit(self, y, *args, **kwargs):
        if self.mode.startswith('conc'):
            ose = self.__class__(mode='one')
            p1 = ose.p_value(y, *args, **kwargs)
            p2 = ose.p_value(-y, *args, **kwargs)
            p = np.maximum(1, 2 * np.minimum(p1, p2))
        else:
            p = self.p_value(y, *args, **kwargs)
        return {'p': p}

    def summary(self):
        if not hasattr(self, 'params_'):
            name = self.__class__.__name__
            raise ValueError("This {} instance hasn't been fitted yet. Please "
                             "call fit() before summary().".format(name))
        return CombinationTestResults(self, self.dataset_, self.params_['z'])


class Stouffers(CombinationTest):
    """Stouffer's Z-score meta-analysis method.

    Takes study-level z-scores and combines them via Stouffer's method to
    produce a fixed-effect estimate of the combined effect.

    Args:
        input (str): The type of measure passed as the `y` input to fit().
            Must be one of 'p' (p-values) or 'z' (z-scores).
        p_type (str) If input == 'p', p_type indicates the type of passed
            p-values. Valid values:
                * 'right' (default): one-sided, right-tailed p-values
                * 'left': one-sided, left-tailed p-values
                * 'two': two-sided p-values

    Notes:
        * When passing in two-sided p-values as input, note that sign
        information is unavailable, and the null being tested is that at least
        one study deviates from 0 in *either* direction. If one-sided p-value
        can be computed, users are strongly recommended to pass those instead.
        (The same caveat applies to 'z' inputs if originally computed from
        two-sided p-values.)
        * This estimator does not support meta-regression; any moderators
        passed in as the X array will be ignored.
        * The fit() method takes z-scores of p-values as the 'y' input, and
        (optionally) weights as the 'v' input. If no weights are passed, unit
        weights are used.
    """
    def p_value(self, z, w=None):
        if self.mode.startswith('two'):
            z = ss.norm.isf(2 * ss.norm.sf(np.abs(z)))
        if w is None:
            w = np.ones_like(z)
        cz = (z * w).sum(0) / np.sqrt((w**2).sum(0))
        return ss.norm.sf(cz)


class Fishers(CombinationTest):
    """Fisher's method for combining p-values.

    Takes study-level p-values or z-scores and combines them via Fisher's
    method to produce a fixed-effect estimate of the combined effect.

    Args:
        input (str): The type of measure passed as the `y` input to fit().
            Must be one of 'p' (p-values) or 'z' (z-scores).
        p_type (str) If input == 'p', p_type indicates the type of passed
            p-values. Valid values:
                * 'right' (default): one-sided, right-tailed p-values
                * 'left': one-sided, left-tailed p-values
                * 'two': two-sided p-values

    Notes:
        * When passing in two-sided p-values as input, note that sign
        information is unavailable, and the null being tested is that at least
        one study deviates from 0 in *either* direction. If one-sided p-value
        can be computed, users are strongly recommended to pass those instead.
        (The same caveat applies to 'z' inputs if originally computed from
        two-sided p-values.)
        * This estimator does not support meta-regression; any moderators
        passed in as the X array will be ignored.
        * The fit() method takes z-scores or p-values as the `y` input. Studies
        are weighted equally; the `v` argument will be ignored if passed.
    """
    def _z_to_p(self, z):
        # Transforms the z inputs to p values based on mode of test
        p = ss.norm.sf(z)
        if self.mode.startswith('two'):
            p = 2 * np.minimum(p, 1 - p)
        return p

    def p_value(self, z):
        p = self._z_to_p(z)
        chi2 = -2 * np.log(p).sum(0)
        return ss.chi2.sf(chi2, 2 * z.shape[0])
