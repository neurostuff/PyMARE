"""Estimators for combination (p/z) tests."""

import warnings
from abc import abstractmethod

import numpy as np
import scipy.stats as ss

from ..results import CombinationTestResults
from .estimators import BaseEstimator


class CombinationTest(BaseEstimator):
    """Base class for methods based on combining p/z values."""

    def __init__(self, mode="directed"):
        mode = mode.lower()
        if mode not in {"directed", "undirected", "concordant"}:
            raise ValueError(
                "Invalid mode; must be one of 'directed', 'undirected', or 'concordant'."
            )
        if mode == "undirected":
            warnings.warn(
                "You have opted to conduct an 'undirected' test. Are you sure "
                "this is what you want? If you're looking for the analog of a "
                "conventional two-tailed test, use 'concordant'."
            )
        self.mode = mode

    @abstractmethod
    def p_value(self, z, *args, **kwargs):
        """Calculate p-values."""
        pass

    def _z_to_p(self, z):
        return ss.norm.sf(z)

    def fit(self, z, *args, **kwargs):
        """Fit the estimator to z-values."""
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        if self.mode == "concordant":
            ose = self.__class__(mode="directed")
            p1 = ose.p_value(z, *args, **kwargs)
            p2 = ose.p_value(-z, *args, **kwargs)
            p = np.minimum(1, 2 * np.minimum(p1, p2))
        else:
            if self.mode == "undirected":
                z = np.abs(z)
            p = self.p_value(z, *args, **kwargs)
        self.params_ = {"p": p}
        return self

    def summary(self):
        """Generate a summary of the estimator results."""
        if not hasattr(self, "params_"):
            name = self.__class__.__name__
            raise ValueError(
                "This {} instance hasn't been fitted yet. Please "
                "call fit() before summary().".format(name)
            )
        return CombinationTestResults(self, self.dataset_, p=self.params_["p"])


class StoufferCombinationTest(CombinationTest):
    """Stouffer's Z-score meta-analysis method.

    Takes a set of independent z-scores and combines them via Stouffer's
    :footcite:p:`stouffer1949american` method to produce a fixed-effect estimate of the combined
    effect.

    Parameters
    ----------
    mode : {"directed", "undirected", "concordant"}, optional
        The type of test to perform-- i.e., what null hypothesis to reject.
        See :footcite:t:`winkler2016non` for details.
        Valid options are:

        -   'directed': tests a directional hypothesis--i.e., that the
            observed value is consistently greater than 0 in the input
            studies. This is the default.
        -   'undirected': tests an undirected hypothesis--i.e., that the
            observed value differs from 0 in the input studies, but
            allowing the direction of the deviation to vary by study.
        -   'concordant': equivalent to two directed tests, one for each
            sign, with correction for 2 tests.

    Notes
    -----
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

    References
    ----------
    .. footbibliography::
    """

    # Maps Dataset attributes onto fit() args; see BaseEstimator for details.
    _dataset_attr_map = {"z": "y", "w": "n", "g": "v"}

    def _inflation_term(self, z, w, g):
        """Calculate the variance inflation term for each group.

        This term is used to adjust the variance of the combined z-score when
        multiple sample come from the same study.

        Parameters
        ----------
        z : :obj:`numpy.ndarray` of shape (n, d)
            Array of z-values.
        w : :obj:`numpy.ndarray` of shape (n, d)
            Array of weights.
        g : :obj:`numpy.ndarray` of shape (n, d)
            Array of group labels.

        Returns
        -------
        sigma : float
            The variance inflation term.
        """
        # Only center if the samples are not all the same, to prevent division by zero
        # when calculating the correlation matrix.
        # This centering is problematic for N=2
        all_samples_same = np.all(np.equal(z, z[0]), axis=0).all()
        z = z if all_samples_same else z - z.mean(0)

        # Use the value from one feature, as all features have the same groups and weights
        groups = g[:, 0]
        weights = w[:, 0]

        # Loop over groups
        unique_groups = np.unique(groups)

        sigma = 0
        for group in unique_groups:
            group_indices = np.where(groups == group)[0]
            group_z = z[group_indices]

            # For groups with only one sample the contribution to the summand is 0
            n_samples = len(group_indices)
            if n_samples < 2:
                continue

            # Calculate the within group correlation matrix and sum the non-diagonal elements
            corr = np.corrcoef(group_z, rowvar=True)
            upper_indices = np.triu_indices(n_samples, k=1)
            non_diag_corr = corr[upper_indices]
            w_i, w_j = weights[upper_indices[0]], weights[upper_indices[1]]

            sigma += (2 * w_i * w_j * non_diag_corr).sum()

        return sigma

    def fit(self, z, w=None, g=None):
        """Fit the estimator to z-values, optionally with weights and groups."""
        return super().fit(z, w=w, g=g)

    def p_value(self, z, w=None, g=None):
        """Calculate p-values."""
        if w is None:
            w = np.ones_like(z)

        # Calculate the variance inflation term, sum of non-diagonal elements of sigma.
        sigma = self._inflation_term(z, w, g) if g is not None else 0

        # The sum of diagonal elements of sigma is given by (w**2).sum(0).
        variance = (w**2).sum(0) + sigma

        cz = (z * w).sum(0) / np.sqrt(variance)
        return ss.norm.sf(cz)


class FisherCombinationTest(CombinationTest):
    """Fisher's method for combining p-values.

    Takes a set of independent z-scores and combines them via Fisher's
    :footcite:p:`fisher1946statistical` method to produce a fixed-effect estimate of the combined
    effect.

    Parameters
    ----------
    mode : {"directed", "undirected", "concordant"}, optional
        The type of test to perform-- i.e., what null hypothesis to reject.
        See :footcite:t:`winkler2016non` for details.
        Valid options are:

            -   'directed': tests a directional hypothesis--i.e., that the
                observed value is consistently greater than 0 in the input
                studies. This is the default.
            -   'undirected': tests an undirected hypothesis--i.e., that the
                observed value differs from 0 in the input studies, but
                allowing the direction of the deviation to vary by study.
            -   'concordant': equivalent to two directed tests, one for each
                sign, with correction for 2 tests.

    Notes
    -----
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

    References
    ----------
    .. footbibliography::
    """

    # Maps Dataset attributes onto fit() args; see BaseEstimator for details.
    _dataset_attr_map = {"z": "y"}

    def p_value(self, z):
        """Calculate p-values."""
        p = self._z_to_p(z)
        chi2 = -2 * np.log(p).sum(0)
        return ss.chi2.sf(chi2, 2 * z.shape[0])
