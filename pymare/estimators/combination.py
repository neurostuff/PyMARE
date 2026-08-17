"""Estimators for combination (p/z) tests."""

import warnings
from abc import abstractmethod

import numpy as np
import scipy.stats as ss
from scipy.special import ndtr

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
        return ndtr(-z)

    def fit(self, z, *args, **kwargs):
        """Fit the estimator to z-values."""
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        if self.mode == "concordant":
            ose = self.__class__(mode="directed")
            p1 = ose.p_value(z, *args, **kwargs)
            p2 = ose.p_value(-z, *args, **kwargs)
            p = np.minimum(1, 2 * np.minimum(p1, p2))
            z_calc = ss.norm.isf(p)
            z_calc[p2 < p1] *= -1
        else:
            if self.mode == "undirected":
                z = np.abs(z)
            p = self.p_value(z, *args, **kwargs)
            z_calc = ss.norm.isf(p)

        self.params_ = {"p": p, "z": z_calc}
        return self

    def summary(self):
        """Generate a summary of the estimator results."""
        if not hasattr(self, "params_"):
            name = self.__class__.__name__
            raise ValueError(
                "This {} instance hasn't been fitted yet. Please "
                "call fit() before summary().".format(name)
            )
        return CombinationTestResults(
            self, self.dataset_, z=self.params_["z"], p=self.params_["p"]
        )


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
    _dataset_attr_map = {"z": "y", "w": "n", "g": "g"}

    def _inflation_term(self, z, w, g, corr=None):
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
        corr : :obj:`numpy.ndarray` of shape (n, n), optional
            The correlation matrix of the z-values. If None, it will be calculated.

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
        groups = np.asarray(g).reshape(g.shape[0], -1)[:, 0]
        weights = np.asarray(w).reshape(w.shape[0], -1)[:, 0]

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
            if corr is None:
                if z.shape[1] < 2:
                    raise ValueError("The number of features must be greater than 1.")
                group_corr = np.corrcoef(group_z, rowvar=True)
            else:
                group_corr = corr[group_indices][:, group_indices]

            upper_indices = np.triu_indices(n_samples, k=1)
            non_diag_corr = group_corr[upper_indices]
            w_i, w_j = weights[upper_indices[0]], weights[upper_indices[1]]

            sigma += (2 * w_i * w_j * non_diag_corr).sum()

        return sigma

    def fit(self, z, w=None, g=None, corr=None):
        """Fit the estimator to z-values, optionally with weights and groups."""
        return super().fit(z, w=w, g=g, corr=corr)

    def p_value(self, z, w=None, g=None, corr=None):
        """Calculate p-values."""
        if w is None:
            w = np.ones_like(z)

        if g is None and corr is not None:
            warnings.warn("Correlation matrix provided without groups. Ignoring.")

        if g is not None and corr is not None and g.shape[0] != corr.shape[0]:
            raise ValueError("Group labels must have the same length as the correlation matrix.")

        # Calculate the variance inflation term, sum of non-diagonal elements of sigma.
        sigma = self._inflation_term(z, w, g, corr=corr) if g is not None else 0

        # The sum of diagonal elements of sigma is given by (w**2).sum(0).
        variance = (w**2).sum(0) + sigma

        cz = (z * w).sum(0) / np.sqrt(variance)
        return ss.norm.sf(cz)


class FisherCombinationTest(CombinationTest):
    """Fisher's method for combining p-values.

    Takes a set of z-scores and combines them via Fisher's
    :footcite:p:`fisher1946statistical` method to produce a fixed-effect estimate of the combined
    effect.

    When group labels are supplied to :meth:`fit`, the statistic is instead referred to the
    scaled chi-squared distribution of :footcite:t:`brown1975method`, which accounts for
    dependence among z-scores within a group. Inputs are weighted by the inverse of their group
    size, so every group has total weight one regardless of how many estimates it contributes.
    Ordinary Fisher inference is recovered when group weights are all one, as when no groups are
    supplied or every input belongs to a different group.

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
    (4) The covariance between the ``-2 ln p`` terms is approximated from the
        correlation of the z-scores using the polynomial of
        :footcite:t:`kost2002combining`.

    References
    ----------
    .. footbibliography::
    """

    # Maps Dataset attributes onto fit() args; see BaseEstimator for details.
    _dataset_attr_map = {"z": "y", "g": "g"}

    @staticmethod
    def _kost_covariance(corr):
        """Approximate cov(-2 ln p_i, -2 ln p_j) from the correlation of z_i, z_j.

        Uses the polynomial fit of :footcite:t:`kost2002combining`, which is
        the standard companion to Brown's method.

        References
        ----------
        .. footbibliography::

        """
        return corr * (3.263 + corr * (0.710 + corr * 0.027))

    @staticmethod
    def _group_weights(g, n_studies):
        """Give every group total weight one, or every input weight one without groups."""
        if g is None:
            return np.ones(n_studies)

        _, group_inverse = np.unique(g, return_inverse=True)
        group_sizes = np.bincount(group_inverse)
        return 1.0 / group_sizes[group_inverse]

    def _brown_moments(self, z, g, corr=None, weights=None):
        """Return the mean and variance of Brown's chi-squared statistic.

        For weights ``w``, the statistic has mean ``2 * sum(w)`` and an
        independent variance contribution of ``4 * sum(w**2)``. Dependence
        within a group adds ``2 * sum_{i<j} w_i * w_j * cov_ij`` to the
        variance, leaving the mean unchanged. Grouped inputs receive inverse
        group-size weights, so each group contributes two to the expectation
        :footcite:p:`brown1975method`.

        References
        ----------
        .. footbibliography::

        """
        n_studies = z.shape[0]
        if weights is None:
            weights = self._group_weights(g, n_studies)

        expectation = 2.0 * weights.sum()
        variance = 4.0 * np.square(weights).sum()

        if g is None:
            return expectation, variance

        groups = g

        # Only center if the samples are not all the same, to prevent division
        # by zero when calculating the correlation matrix.
        all_samples_same = np.all(np.equal(z, z[0]), axis=0).all()
        z_centered = z if all_samples_same else z - z.mean(0)

        for group in np.unique(groups):
            group_indices = np.where(groups == group)[0]

            # Groups with a single sample contribute nothing.
            n_samples = len(group_indices)
            if n_samples < 2:
                continue

            if corr is None:
                if z.shape[1] < 2:
                    raise ValueError("The number of features must be greater than 1.")
                group_corr = np.corrcoef(z_centered[group_indices], rowvar=True)
            else:
                group_corr = corr[group_indices][:, group_indices]

            upper_indices = np.triu_indices(n_samples, k=1)
            non_diag_corr = group_corr[upper_indices]
            group_weights = weights[group_indices]
            pair_weights = group_weights[upper_indices[0]] * group_weights[upper_indices[1]]
            variance += 2.0 * (pair_weights * self._kost_covariance(non_diag_corr)).sum()

        return expectation, variance

    @staticmethod
    def _validate_dependence_inputs(z, g, corr):
        """Normalize group labels and validate an optional correlation matrix."""
        if g is None:
            if corr is not None:
                warnings.warn("Correlation matrix provided without groups. Ignoring.")
            return None, None

        groups = np.asarray(g)
        if groups.ndim == 1:
            pass
        elif groups.ndim == 2 and groups.shape[1] > 0:
            if groups.shape[1] > 1 and not np.all(groups == groups[:, [0]]):
                raise ValueError("Group labels must be the same for every feature.")
            groups = groups[:, 0]
        else:
            raise ValueError("Group labels must be a one- or two-dimensional array.")

        n_studies = z.shape[0]
        if groups.shape[0] != n_studies:
            raise ValueError(
                f"Group labels must contain one label per study: expected {n_studies}, "
                f"got {groups.shape[0]}."
            )

        if corr is not None:
            corr = np.asarray(corr)
            expected_shape = (n_studies, n_studies)
            if corr.shape != expected_shape:
                raise ValueError(
                    "Group labels must have the same length as the correlation matrix; "
                    f"expected shape {expected_shape}, got {corr.shape}."
                )

        return groups, corr

    def fit(self, z, g=None, corr=None):
        """Fit the estimator to z-values, optionally with groups."""
        return super().fit(z, g=g, corr=corr)

    def p_value(self, z, g=None, corr=None):
        """Calculate p-values."""
        g, corr = self._validate_dependence_inputs(z, g, corr)

        p = self._z_to_p(z)
        weights = self._group_weights(g, z.shape[0])
        if g is None:
            chi2 = -2 * np.log(p).sum(0)
        else:
            # p is not needed below, so turn it into the weighted log-p terms
            # in place rather than allocating two additional map-sized arrays.
            np.log(p, out=p)
            p *= weights[:, None]
            chi2 = -2 * p.sum(0)

        expectation, variance = self._brown_moments(z, g, corr=corr, weights=weights)

        # Brown's scaled chi-squared: divide the statistic by c and refer it to
        # f degrees of freedom. With unit weights and independent inputs,
        # variance == 2 * expectation, so c == 1 and f == 2k, recovering
        # Fisher's method exactly.
        scale = variance / (2.0 * expectation)
        dof = 2.0 * expectation**2 / variance

        return ss.chi2.sf(chi2 / scale, dof)
