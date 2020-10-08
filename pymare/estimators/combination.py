from abc import abstractmethod
import warnings

import numpy as np
import scipy.stats as ss

from .estimators import BaseEstimator
from ..results import CombinationTestResults


class CombinationTest(BaseEstimator):
    """Base class for methods based on combining p/z values."""
    def __init__(self, mode='directed'):
        mode = mode.lower()
        if mode not in {'directed', 'undirected', 'concordant'}:
            raise ValueError("Invalid mode; must be one of 'directed', "
                             "'undirected', or 'concordant'.")
        if mode == 'undirected':
            warnings.warn(
                "You have opted to conduct an 'undirected' test. Are you sure "
                "this is what you want? If you're looking for the analog of a "
                "conventional two-tailed test, use 'concordant'.")
        self.mode = mode

    @abstractmethod
    def p_value(self, z, *args, **kwargs):
        pass

    def _z_to_p(self, z):
        return ss.norm.sf(z)

    def fit(self, z, *args, **kwargs):
        if self.mode == 'concordant':
            ose = self.__class__(mode='directed')
            p1 = ose.p_value(z, *args, **kwargs)
            p2 = ose.p_value(-z, *args, **kwargs)
            p = np.minimum(1, 2 * np.minimum(p1, p2))
        else:
            if self.mode == 'undirected':
                z = np.abs(z)
            p = self.p_value(z, *args, **kwargs)
        self.params_ = {'p': p}
        return self

    def summary(self):
        if not hasattr(self, 'params_'):
            name = self.__class__.__name__
            raise ValueError("This {} instance hasn't been fitted yet. Please "
                             "call fit() before summary().".format(name))
        return CombinationTestResults(self, self.dataset_, p=self.params_['p'])


class StoufferCombinationTest(CombinationTest):
    """Stouffer's Z-score meta-analysis method.

    Takes a set of independent z-scores and combines them via Stouffer's method
    to produce a fixed-effect estimate of the combined effect.

    Args:
        mode (str): The type of test to perform-- i.e., what null hypothesis to
            reject. See Winkler et al. (2016) for details. Valid options are:
                * 'directed': tests a directional hypothesis--i.e., that the
                    observed value is consistently greater than 0 in the input
                    studies.
                * 'undirected': tests an undirected hypothesis--i.e., that the
                    observed value differs from 0 in the input studies, but
                    allowing the direction of the deviation to vary by study.
                * 'concordant': equivalent to two directed tests, one for each
                    sign, with correction for 2 tests.

    Notes:
        (1) All input z-scores are assumed to correspond to one-sided p-values.
            Do NOT pass in z-scores that have been directly converted from
            two-tailed p-values, as these do not preserve directional
            information.
        (2) The 'directed' and 'undirected' modes are NOT the same as
            one-tailed and two-tailed tests. In general, users who want to test
            directed hypotheses should use the 'directed' mode, and users who
            want to test for consistent effects in either the positive or
            negative direction should use the 'concordant' mode. The
            'undirected' mode tests a fairly uncommon null that doesn't
            constrain the sign of effects to be consistent across studies
            (one can think of it as a test of extremity). In the vast majority
            of meta-analysis applications, this mode is not appropriate, and
            users should instead opt for 'directed' or 'concordant'.
        (3) This estimator does not support meta-regression; any moderators
            passed in to fit() as the X array will be ignored.
    """

    # Maps Dataset attributes onto fit() args; see BaseEstimator for details.
    _dataset_attr_map = {'z': 'y', 'w': 'v'}

    def fit(self, z, w=None):
        return super().fit(z, w=w)

    def p_value(self, z, w=None):
        if w is None:
            w = np.ones_like(z)
        cz = (z * w).sum(0) / np.sqrt((w**2).sum(0))
        return ss.norm.sf(cz)


class FisherCombinationTest(CombinationTest):
    """Fisher's method for combining p-values.

    Takes a set of independent z-scores and combines them via Fisher's method
    to produce a fixed-effect estimate of the combined effect.

    Args:
        mode (str): The type of test to perform-- i.e., what null hypothesis to
            reject. See Winkler et al. (2016) for details. Valid options are:
                * 'directed': tests a directional hypothesis--i.e., that the
                    observed value is consistently greater than 0 in the input
                    studies.
                * 'undirected': tests an undirected hypothesis--i.e., that the
                    observed value differs from 0 in the input studies, but
                    allowing the direction of the deviation to vary by study.
                * 'concordant': equivalent to two directed tests, one for each
                    sign, with correction for 2 tests.

    Notes:
        (1) All input z-scores are assumed to correspond to one-sided p-values.
            Do NOT pass in z-scores that have been directly converted from
            two-tailed p-values, as these do not preserve directional
            information.
        (2) The 'directed' and 'undirected' modes are NOT the same as
            one-tailed and two-tailed tests. In general, users who want to test
            directed hypotheses should use the 'directed' mode, and users who
            want to test for consistent effects in either the positive or
            negative direction should use the 'concordant' mode. The
            'undirected' mode tests a fairly uncommon null that doesn't
            constrain the sign of effects to be consistent across studies
            (one can think of it as a test of extremity). In the vast majority
            of meta-analysis applications, this mode is not appropriate, and
            users should instead opt for 'directed' or 'concordant'.
        (3) This estimator does not support meta-regression; any moderators
            passed in to fit() as the X array will be ignored.
    """

    # Maps Dataset attributes onto fit() args; see BaseEstimator for details.
    _dataset_attr_map = {'z': 'y'}

    def p_value(self, z):
        p = self._z_to_p(z)
        chi2 = -2 * np.log(p).sum(0)
        return ss.chi2.sf(chi2, 2 * z.shape[0])
