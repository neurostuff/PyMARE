"""Estimators for combination (p/z) tests."""

import copy
import warnings
from abc import abstractmethod

import numpy as np
from scipy.special import log_ndtr, ndtri_exp

from ..results import CombinationTestResults
from ..stats import encode_groups, log_chi2_sf, normalize_group_weights
from .estimators import BaseEstimator


def _check_estimable_correlation(z, group_indices):
    """Reject a group whose centered rows carry no variance to correlate.

    Parameters
    ----------
    z : :obj:`numpy.ndarray` of shape (K, D)
        The **centered** estimates, as both callers pass them: the per-feature
        mean over rows has already been removed. Passing the raw array instead
        tests a different and largely harmless condition; see Notes.
    group_indices : :obj:`numpy.ndarray`
        Row indices of the one group about to be correlated.

    Raises
    ------
    ValueError
        If any selected row is constant across features, which leaves its
        correlation with the others undefined.

    See Also
    --------
    pymare.stats.estimate_null_correlation : Estimates the same quantity for
        the caller, and applies the corresponding bias correction.

    Notes
    -----
    ``np.corrcoef`` divides each row by its own standard deviation, so a row
    with no variance across features produces ``NaN`` -- with only a
    ``RuntimeWarning`` -- and that ``NaN`` then travels through the variance
    inflation term into the reported p-values without raising.

    The condition is on the *centered* rows, which is not the same as a row
    that is constant in the raw data. Centering subtracts each feature's mean
    over rows, so a raw row of ``[1, 1, 1, 1]`` generally becomes non-constant
    and needs no guard at all. What does trigger it is a row whose centered
    values are constant, i.e. one differing from the per-feature means by a
    fixed offset. With two rows in a group that happens exactly when the rows
    differ by a constant; two rows differing by a varying amount centre to
    mirror images and correlate at exactly ``-1``, which is degenerate but not
    undefined.

    This is a different failure from the "all samples identical" check in the
    callers, which skips centering when *every* row is the same. Those rows
    still vary across features and legitimately correlate at 1, so no ``NaN``
    arises and no guard is needed. The two checks are therefore not redundant,
    and collapsing them would reintroduce one of the two failures.
    """
    if np.ptp(z[group_indices], axis=1).min() == 0:
        raise ValueError(
            "Cannot estimate a within-group correlation: at least one estimate "
            "is constant across features, so its correlation with the others is "
            "undefined. Supply `corr` explicitly if the dependence is known."
        )


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
    def log_p_value(self, z, *args, **kwargs):
        """Calculate natural logarithms of the p-values."""
        pass

    def p_value(self, z, *args, **kwargs):
        """Calculate p-values.

        Underflows to zero where the combined evidence exceeds what a double can
        represent; :meth:`log_p_value` is the same quantity without that limit.
        """
        return np.exp(self.log_p_value(z, *args, **kwargs))

    def fit(self, z, *args, **kwargs):
        """Fit the estimator to z-values.

        .. versionchanged:: 0.0.11
            The ``"concordant"`` statistic was ``norm.isf(p)`` applied to that
            already-doubled p-value, which is a tail mismatch rather than a
            different convention: it shrank the statistic to absorb the
            multiplicity penalty that :footcite:t:`winkler2016non` place on the
            p-value alone. Their concordant statistic is
            ``T = max(-2 sum ln p_k, -2 sum ln (1 - p_k))`` -- the better of the
            two directed combinations, unshrunk -- which is what
            ``norm.isf(p / 2)`` returns. The old form also made the reported
            statistic non-monotone in the evidence: it was negative wherever
            ``p > 0.5``, carried the opposite sign to the effect there, and was
            ``-inf`` wherever the cap put ``p`` at exactly 1, which gave the
            least significant results the largest magnitudes. A single input now
            combines to its own z, as it should.

        References
        ----------
        .. footbibliography::

        """
        # This resets the Estimator's dataset_ attribute. fit_dataset will overwrite if called.
        self.dataset_ = None

        if self.mode == "concordant":
            # Preserve subclass configuration (for example group-level
            # aggregation) while evaluating the two directed tails.
            ose = copy.copy(self)
            ose.mode = "directed"
            log_p1 = ose.log_p_value(z, *args, **kwargs)
            log_p2 = ose.log_p_value(-z, *args, **kwargs)
            # Doubling the smaller tail and capping at 1, in logs: the
            # correction for two tests is an added log(2), the cap a minimum
            # against log(1).
            log_p = np.minimum(0.0, np.log(2.0) + np.minimum(log_p1, log_p2))
            z_calc = -ndtri_exp(log_p - np.log(2.0))
            z_calc = np.where(log_p2 < log_p1, -z_calc, z_calc)
        else:
            if self.mode == "undirected":
                z = np.abs(z)
            log_p = self.log_p_value(z, *args, **kwargs)
            # ``norm.isf(p)`` instead would saturate to +/-inf the moment p
            # underflowed or hit 1; the inverse of the log CDF does not.
            z_calc = -ndtri_exp(log_p)

        self.params_ = {"p": np.exp(log_p), "logp": log_p, "z": z_calc}
        return self

    def summary(self):
        """Generate a summary of the estimator results."""
        if not hasattr(self, "params_"):
            name = self.__class__.__name__
            raise ValueError(
                "This {} instance hasn't been fitted yet. Please "
                "call fit() before summary().".format(name)
            )
        # p is exactly ``exp(logp)``, which the container derives itself, so
        # passing it would store a second copy of one quantity. z is not
        # derivable here: in concordant mode it carries the sign of whichever
        # tail won, which the log p-value alone does not record.
        return CombinationTestResults(
            self, self.dataset_, z=self.params_["z"], logp=self.params_["logp"]
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
            observed value is consistently greater than 0 across the inputs.
            This is the default.
        -   'undirected': tests an undirected hypothesis--i.e., that the
            observed value differs from 0 across the inputs, but
            allowing the direction of the deviation to vary by input.
        -   'concordant': equivalent to two directed tests, one for each
            sign, with correction for 2 tests.
    group_level : :obj:`bool`, optional
        If True and group labels are supplied, first convert every group to
        one variance-standardized equal-weight mean, then apply one externally
        supplied weight per group. Repeated rows in a group must carry the same
        weight. Default is False, which preserves row-level Stouffer behavior.

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
        constrain the sign of effects to be consistent across inputs
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

    def __init__(self, mode="directed", group_level=False):
        super().__init__(mode=mode)
        self.group_level = group_level

    @staticmethod
    def _validate_groups(g, n_rows):
        """Return one group label per row."""
        groups = np.asarray(g)
        if groups.ndim == 2:
            if groups.shape[1] == 0:
                raise ValueError("Group labels cannot have zero columns.")
            if groups.shape[1] > 1 and not np.all(groups == groups[:, [0]]):
                raise ValueError("Group labels must be the same for every feature.")
            groups = groups[:, 0]
        elif groups.ndim != 1:
            raise ValueError("Group labels must be one- or two-dimensional.")
        if groups.shape[0] != n_rows:
            raise ValueError(
                f"Group labels must contain one label per estimate: expected {n_rows}, "
                f"got {groups.shape[0]}."
            )
        return groups

    def _group_statistics(self, z, w, g, corr=None):
        r"""Standardize one equal-weight mean per group.

        For :math:`a_g = 1/k_g`, the group statistic is
        :math:`a_g'z_g / \sqrt{a_g'R_ga_g}`. The supplied row weights
        represent one group-level weight and must therefore be constant within
        a group; this prevents row multiplicity from changing total weight.
        """
        groups = self._validate_groups(g, z.shape[0])
        w = np.asarray(w, dtype=float)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != z.shape[0] or w.shape[1] not in (1, z.shape[1]):
            raise ValueError("Weights must have one row per estimate and one or D columns.")
        if w.shape[1] == 1 and z.shape[1] > 1:
            w = np.broadcast_to(w, z.shape)

        if corr is not None:
            corr = np.asarray(corr, dtype=float)
            if corr.shape != (z.shape[0], z.shape[0]):
                raise ValueError(f"Correlation matrix must have shape {(z.shape[0], z.shape[0])}.")

        # encode_groups, not np.unique: labels only have to be hashable, and
        # np.unique needs an ordering, so a mix of str and int labels -- which
        # Dataset and the regression estimators accept -- raised TypeError here.
        group_codes, group_labels = encode_groups(groups, n_observations=z.shape[0])
        group_z = np.empty((group_labels.size, z.shape[1]), dtype=float)
        group_w = np.empty_like(group_z)
        centered = z if np.all(z == z[0]) else z - z.mean(axis=0)

        for group_idx in range(group_labels.size):
            members = np.flatnonzero(group_codes == group_idx)
            member_w = w[members]
            if not np.allclose(member_w, member_w[[0]], rtol=1e-12, atol=1e-15):
                raise ValueError(
                    "Group-level Stouffer requires one weight per group; repeated "
                    "rows in a group must have equal weights."
                )
            group_w[group_idx] = member_w[0]

            size = members.size
            if size == 1:
                variance = 1.0
            elif corr is not None:
                block_corr = corr[np.ix_(members, members)]
                variance = block_corr.sum() / size**2
            else:
                if z.shape[1] < 2:
                    raise ValueError("The number of features must be greater than 1.")
                block_corr = np.corrcoef(centered[members], rowvar=True)
                variance = block_corr.sum() / size**2

            if not np.isfinite(variance) or variance <= 0:
                raise ValueError(
                    "Each group's aggregated z statistic must have positive variance."
                )
            group_z[group_idx] = z[members].mean(axis=0) / np.sqrt(variance)

        return group_z, group_w

    def _inflation_term(self, z, w, g, corr=None):
        """Calculate the variance inflation term for each group.

        This term is used to adjust the variance of the combined z-score when
        multiple observations come from the same group.

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
        sigma : :obj:`numpy.ndarray` of shape (d,) or (1,)
            The variance inflation term, one per feature. Weights may differ by
            feature, and the diagonal term ``(w**2).sum(0)`` this is added to is
            already per feature, so collapsing it to a scalar taken from the
            first column understated or overstated every other column.
        """
        # Only center if the samples are not all the same, to prevent division by zero
        # when calculating the correlation matrix.
        # This centering is problematic for N=2
        all_samples_same = np.all(np.equal(z, z[0]), axis=0).all()
        z = z if all_samples_same else z - z.mean(0)

        # Groups are the same for every feature; weights are not, so keep their
        # column axis and return one inflation term per feature.
        groups = np.asarray(g).reshape(g.shape[0], -1)[:, 0]
        weights = np.asarray(w, dtype=float).reshape(w.shape[0], -1)

        # Loop over groups, encoded rather than sorted so that any hashable
        # label works, as encode_groups documents and Dataset allows.
        group_codes, group_labels = encode_groups(groups, n_observations=z.shape[0])

        sigma = np.zeros(weights.shape[1])
        for group in range(group_labels.size):
            group_indices = np.flatnonzero(group_codes == group)
            group_z = z[group_indices]

            # For groups with only one sample the contribution to the summand is 0
            n_samples = len(group_indices)
            if n_samples < 2:
                continue

            # Calculate the within group correlation matrix and sum the non-diagonal elements
            if corr is None:
                if z.shape[1] < 2:
                    raise ValueError("The number of features must be greater than 1.")
                _check_estimable_correlation(z, group_indices)
                group_corr = np.corrcoef(group_z, rowvar=True)
            else:
                group_corr = corr[group_indices][:, group_indices]

            upper_indices = np.triu_indices(n_samples, k=1)
            non_diag_corr = group_corr[upper_indices]
            # upper_indices are positions *within* this group's block, so they
            # have to index the group's own weights. Indexing the full weight
            # array with them silently reuses rows 0..n_j-1 of the dataset for
            # every group, which both understates the inflation and makes the
            # result depend on row order. _brown_moments does this correctly.
            group_weights = weights[group_indices]
            w_i, w_j = group_weights[upper_indices[0]], group_weights[upper_indices[1]]

            sigma += (2 * w_i * w_j * non_diag_corr[:, None]).sum(axis=0)

        return sigma

    def fit(self, z, w=None, g=None, corr=None):
        """Fit the estimator to z-values, optionally with weights and groups."""
        self.corr_ = corr
        return super().fit(z, w=w, g=g, corr=corr)

    def log_p_value(self, z, w=None, g=None, corr=None):
        """Calculate natural logarithms of the p-values."""
        if w is None:
            w = np.ones_like(z)
        else:
            # Match FisherCombinationTest, which already rejects these. Silently
            # accepting a negative or non-finite weight yields a NaN p-value or,
            # worse, a plausible-looking one with the sign of the effect flipped.
            w = np.asarray(w, dtype=float)
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("Weights must be finite positive values.")

        if g is not None:
            # Reduces (K, D) labels to (K,), raising if they differ by feature.
            g = self._validate_groups(g, z.shape[0])[:, None]
            if corr is not None:
                corr = np.asarray(corr, dtype=float)
                expected = (z.shape[0], z.shape[0])
                if corr.shape != expected:
                    raise ValueError(f"Correlation matrix must have shape {expected}.")

        if self.group_level and g is not None:
            group_z, group_w = self._group_statistics(z, w, g, corr=corr)
            variance = np.square(group_w).sum(axis=0)
            cz = (group_z * group_w).sum(axis=0) / np.sqrt(variance)
            return log_ndtr(-cz)

        if g is None and corr is not None:
            warnings.warn("Correlation matrix provided without groups. Ignoring.")

        if g is not None and corr is not None and g.shape[0] != corr.shape[0]:
            raise ValueError("Group labels must have the same length as the correlation matrix.")

        # Calculate the variance inflation term, sum of non-diagonal elements of sigma.
        sigma = self._inflation_term(z, w, g, corr=corr) if g is not None else 0

        # The sum of diagonal elements of sigma is given by (w**2).sum(0).
        variance = (w**2).sum(0) + sigma

        cz = (z * w).sum(0) / np.sqrt(variance)
        # log_ndtr, not norm.sf: the combined z is a weighted *sum*, so it grows
        # with the number of observations and passes 38 -- where a double-
        # precision p-value is exactly zero -- on datasets of very ordinary size.
        return log_ndtr(-cz)


class FisherCombinationTest(CombinationTest):
    """Fisher's method for combining p-values.

    Takes a set of z-scores and combines them via Fisher's
    :footcite:p:`fisher1946statistical` method to produce a fixed-effect estimate of the combined
    effect.

    When group labels are supplied to :meth:`fit`, the statistic is instead referred to the
    scaled chi-squared distribution of :footcite:t:`brown1975method`, which accounts for
    dependence among z-scores within a group. By default, inputs are weighted by the inverse of
    their group size, so every group has total weight one regardless of how many rows it
    contributes. Optional positive external weights may assign different total weights to groups;
    repeated rows in a group must carry the same external weight. Ordinary Fisher inference is
    recovered when group weights are all one, as when no groups are supplied or every input
    belongs to a different group.

    Parameters
    ----------
    mode : {"directed", "undirected", "concordant"}, optional
        The type of test to perform-- i.e., what null hypothesis to reject.
        See :footcite:t:`winkler2016non` for details.
        Valid options are:

            -   'directed': tests a directional hypothesis--i.e., that the
                observed value is consistently greater than 0 across the inputs.
                This is the default.
            -   'undirected': tests an undirected hypothesis--i.e., that the
                observed value differs from 0 across the inputs, but
                allowing the direction of the deviation to vary by input.
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
        constrain the sign of effects to be consistent across inputs
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
    _dataset_attr_map = {"z": "y", "w": "n", "g": "g"}

    @staticmethod
    def _kost_covariance(corr):
        r"""Approximate cov(-2 ln p_i, -2 ln p_j) from the correlation of z_i, z_j.

        This covariance has no closed form, so :footcite:t:`brown1975method`
        tabulated it by numerical integration and :footcite:t:`kost2002combining`
        fitted the cubic used here to that table:

        .. math::

            \operatorname{Cov}(-2 \ln p_i, -2 \ln p_j) \approx
                3.263 \rho + 0.710 \rho^2 + 0.027 \rho^3.

        The three coefficients are empirical, but they are pinned at all three
        points where the covariance is known in closed form, which is why the
        approximation can be trusted across the whole range
        :math:`\rho \in [-1, 1]` that a correlation matrix can supply:

        * at :math:`\rho = 0` the polynomial is 0. Independent inputs add
          nothing to the variance, so Fisher's method is recovered exactly.
        * at :math:`\rho = 1` the coefficients sum to exactly 4.000, which is
          :math:`\operatorname{Var}(\chi^2_2)`. Perfectly correlated z-scores
          give identical p-values, so the covariance must equal the variance.
        * at :math:`\rho = -1` the polynomial gives -2.580, against an exact
          countermonotone value of :math:`4(1 - \pi^2/6) = -2.5797`. This is
          the Frechet lower bound for two :math:`\chi^2_2` variates, not
          :math:`-4`; the covariance is asymmetric in :math:`\rho` because
          :math:`-2 \ln p` is skewed, which is why the fit needs the even
          :math:`\rho^2` term.

        In between, the polynomial is strictly increasing (its derivative
        bottoms out at 1.92 on :math:`[-1, 1]`) and agrees with simulated
        Gaussian-copula covariances to within about 0.005 in absolute terms.
        Written in Horner form for numerical stability.

        Note that :math:`\rho` is the correlation of the *z-scores*, not of the
        ``-2 ln p`` terms; converting between the two is exactly what this
        approximation exists to do.

        References
        ----------
        .. footbibliography::

        """
        return corr * (3.263 + corr * (0.710 + corr * 0.027))

    @staticmethod
    def _group_weights(g, n_observations, w=None):
        """Allocate one optional external weight across each group's rows."""
        if w is None:
            external = np.ones(n_observations)
        else:
            external = np.asarray(w, dtype=float).squeeze()
            if external.ndim != 1 or external.size != n_observations:
                raise ValueError("Weights must contain one scalar per observation.")
            if np.any(~np.isfinite(external)) or np.any(external <= 0):
                raise ValueError("Weights must be finite positive values.")

        if g is None:
            return external

        group_codes, labels = encode_groups(g, n_observations=n_observations)
        for group in range(labels.size):
            member_weights = external[group_codes == group]
            if not np.allclose(member_weights, member_weights[0], rtol=1e-12, atol=1e-15):
                raise ValueError(
                    "Grouped Fisher combination requires one weight per group; "
                    "repeated rows in a group must have equal weights."
                )
        return normalize_group_weights(external, group_codes)

    def _brown_moments(self, z, g, corr=None, weights=None):
        r"""Return the mean and variance of Brown's chi-squared statistic.

        The 2 and the 4 below are the mean and variance of a chi-squared
        variate on two degrees of freedom, which is where every term in
        Fisher's method starts. Under the null, each p-value is uniform on
        (0, 1), so :math:`-2 \ln p_i \sim \chi^2_2` and

        .. math::

            E[-2 \ln p_i] = 2, \qquad \operatorname{Var}(-2 \ln p_i) = 4,

        using :math:`E[\chi^2_\nu] = \nu` and
        :math:`\operatorname{Var}(\chi^2_\nu) = 2\nu` at :math:`\nu = 2`.

        The statistic is the weighted sum
        :math:`X = \sum_i w_i (-2 \ln p_i)`, so linearity of expectation and
        the bilinearity of covariance give

        .. math::

            E[X] &= 2 \sum_i w_i, \\
            \operatorname{Var}(X) &= 4 \sum_i w_i^2
                + 2 \sum_{i<j} w_i w_j \operatorname{Cov}(-2 \ln p_i,
                  -2 \ln p_j).

        The first variance term is what independent inputs alone contribute;
        the second is the dependence correction, evaluated per group via
        :meth:`_kost_covariance` and added to ``variance`` in the loop below.
        Dependence shifts no mass, so it leaves ``expectation`` untouched --
        this is precisely why Brown's method rescales rather than recenters.

        Without external weights, grouped inputs receive inverse group-size
        weights, so each group's ``w_i`` sum to one and each group therefore
        contributes exactly 2 to the expectation, matching a single
        independent p-value :footcite:p:`brown1975method`.

        References
        ----------
        .. footbibliography::

        """
        n_observations = z.shape[0]
        if weights is None:
            weights = self._group_weights(g, n_observations)

        # 2 and 4 are the mean and variance of chi^2_2; see the docstring.
        expectation = 2.0 * weights.sum()
        variance = 4.0 * np.square(weights).sum()

        if g is None:
            return expectation, variance

        groups = g

        # Only center if the samples are not all the same, to prevent division
        # by zero when calculating the correlation matrix.
        all_samples_same = np.all(np.equal(z, z[0]), axis=0).all()
        z_centered = z if all_samples_same else z - z.mean(0)

        group_codes, group_labels = encode_groups(groups, n_observations=n_observations)
        for group in range(group_labels.size):
            group_indices = np.flatnonzero(group_codes == group)

            # Groups with a single sample contribute nothing.
            n_samples = len(group_indices)
            if n_samples < 2:
                continue

            if corr is None:
                if z.shape[1] < 2:
                    raise ValueError("The number of features must be greater than 1.")
                _check_estimable_correlation(z_centered, group_indices)
                group_corr = np.corrcoef(z_centered[group_indices], rowvar=True)
            else:
                group_corr = corr[group_indices][:, group_indices]

            upper_indices = np.triu_indices(n_samples, k=1)
            non_diag_corr = group_corr[upper_indices]
            group_weights = weights[group_indices]
            pair_weights = group_weights[upper_indices[0]] * group_weights[upper_indices[1]]
            # The 2 counts each unordered pair twice, since only the upper
            # triangle is enumerated but Var(sum) sums over i != j.
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

        n_observations = z.shape[0]
        if groups.shape[0] != n_observations:
            raise ValueError(
                f"Group labels must contain one label per observation: expected {n_observations}, "
                f"got {groups.shape[0]}."
            )

        if corr is not None:
            corr = np.asarray(corr)
            expected_shape = (n_observations, n_observations)
            if corr.shape != expected_shape:
                raise ValueError(
                    "Group labels must have the same length as the correlation matrix; "
                    f"expected shape {expected_shape}, got {corr.shape}."
                )

        return groups, corr

    def fit(self, z, g=None, corr=None, w=None):
        """Fit the estimator with optional external weights and groups."""
        self.corr_ = corr
        return super().fit(z, g=g, corr=corr, w=w)

    def log_p_value(self, z, g=None, corr=None, w=None):
        """Calculate natural logarithms of the p-values."""
        g, corr = self._validate_dependence_inputs(z, g, corr)

        # Work in log space throughout. Going
        # via p underflows to exactly 0 around z = 38, after which log(p) is
        # -inf and the combined result collapses to p = 0 with z = inf.
        log_p = log_ndtr(-z)
        weights = self._group_weights(g, z.shape[0], w=w)
        if g is None and w is None:
            chi2 = -2 * log_p.sum(0)
        else:
            log_p *= weights[:, None]
            chi2 = -2 * log_p.sum(0)

        expectation, variance = self._brown_moments(z, g, corr=corr, weights=weights)

        # Brown's scaled chi-squared: divide the statistic by c and refer it to
        # f degrees of freedom. c and f come from matching the first two
        # moments of the statistic X to those of c * chi^2_f, using
        # E[c * chi^2_f] = c * f and Var(c * chi^2_f) = 2 * c**2 * f:
        #
        #     Var(X) / E[X] = 2c            ->  c = Var(X) / (2 E[X])
        #     f = E[X] / c                  ->  f = 2 E[X]**2 / Var(X)
        #
        # so both 2s below are the 2 in Var(chi^2_f) = 2f, not free parameters.
        # With unit weights and independent inputs, variance == 2 * expectation,
        # so c == 1 and f == 2k, recovering Fisher's method exactly.
        scale = variance / (2.0 * expectation)
        dof = 2.0 * expectation**2 / variance

        return log_chi2_sf(chi2 / scale, dof)
