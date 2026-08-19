"""Benchmark the functions in :mod:`pymare.stats`.

These are the innermost pieces of every fit, so a slowdown here is a slowdown
everywhere. The cluster-robust entries carry the most risk of regressing,
because their cost is per-group work that has to stay vectorized across
datasets -- ``bench_cluster_robust.py`` explains which mistakes make it grow.
"""

from pymare.stats import (
    cluster_robust_cov,
    collapse_groups,
    correlated_effects_tau2,
    correlated_effects_weights,
    estimate_null_correlation,
    q_gen,
    q_profile,
    satterthwaite_dof,
    weighted_least_squares,
)

from .common import make_data

# asv benchmark attributes, applied to every benchmark in this module. The
# ceilings matter because the Benchmark workflow times the whole suite twice on
# every pull request: bounding the samples keeps that under a couple of minutes,
# and the floor of two samples keeps a single noisy reading from being the whole
# measurement.
repeat = (2, 5, 10.0)
warmup_time = 0.1
timeout = 180


class TimeWeightedLeastSquares:
    """Time the weighted least squares solve every estimator runs."""

    def setup(self):
        """Build the design and the parallel datasets."""
        self.y, self.v, self.X, _ = make_data(n_predictors=3)

    def time_solve(self):
        """Solve for the coefficients alone."""
        weighted_least_squares(self.y, self.v, self.X)

    def time_solve_with_cov(self):
        """Solve and return the model-based covariance the sandwich needs."""
        weighted_least_squares(self.y, self.v, self.X, return_cov=True)


class TimeClusterRobustCov:
    """Time the cluster-robust sandwich.

    Both variance shapes are timed. One ``v`` column shared by every dataset
    takes a fast path, because the weights and the design determine the CR2
    adjustment and ``y`` never enters it; a ``v`` per dataset cannot.
    """

    def setup(self):
        """Fit once, so only the sandwich is timed."""
        self.y, self.v, self.X, self.groups = make_data(n_predictors=3)
        self.beta, self.model_cov = weighted_least_squares(self.y, self.v, self.X, return_cov=True)
        self.shared_v = self.v[:, :1]
        self.shared_beta, self.shared_model_cov = weighted_least_squares(
            self.y, self.shared_v, self.X, return_cov=True
        )

    def time_cr0(self):
        """Time the plain sandwich, with no small-sample adjustment."""
        cluster_robust_cov(
            self.y,
            self.v,
            self.X,
            self.beta,
            self.groups,
            model_cov=self.model_cov,
            method="CR0",
            small_sample=False,
        )

    def time_cr2(self):
        """Time the default: CR2 residuals plus the small-sample scaling."""
        cluster_robust_cov(
            self.y, self.v, self.X, self.beta, self.groups, model_cov=self.model_cov
        )

    def time_cr2_shared_variances(self):
        """One ``v`` column for every dataset, which takes the fast path."""
        cluster_robust_cov(
            self.y,
            self.shared_v,
            self.X,
            self.shared_beta,
            self.groups,
            model_cov=self.shared_model_cov,
        )


class TimeSatterthwaiteDof:
    """Time the Satterthwaite degrees of freedom."""

    def setup(self):
        """Fit once, so only the degrees of freedom are timed."""
        self.y, self.v, self.X, self.groups = make_data(n_predictors=3)
        _, self.model_cov = weighted_least_squares(self.y, self.v, self.X, return_cov=True)
        self.weights = 1.0 / self.v
        self.shared_weights = self.weights[:, :1]
        _, self.shared_model_cov = weighted_least_squares(
            self.y, self.v[:, :1], self.X, return_cov=True
        )

    def time_dof(self):
        """Time a weight column per dataset."""
        satterthwaite_dof(self.X, self.weights, self.groups, model_cov=self.model_cov)

    def time_dof_shared_weights(self):
        """One weight column, so the adjustment is solved once and broadcast."""
        satterthwaite_dof(
            self.X, self.shared_weights, self.groups, model_cov=self.shared_model_cov
        )


class TimeCorrelatedEffects:
    """Time the correlated-effects working model."""

    def setup(self):
        """Build a problem with several observations per group."""
        self.y, self.v, self.X, self.groups = make_data(n_predictors=3)

    def time_tau2(self):
        """Time the method-of-moments tau^2, which reads the observation-level design."""
        correlated_effects_tau2(self.y, self.v, self.X, self.groups, rho=0.8)

    def time_weights(self):
        """Time the weights that give each group one total."""
        correlated_effects_weights(self.v, self.groups, tau2=0.5)

    def time_collapse(self):
        """Reducing every group to a single row."""
        collapse_groups(self.y, self.v, self.X, self.groups, rho=0.8)


class TimeHeterogeneity:
    """Time the heterogeneity statistics and the profile-likelihood interval."""

    def setup(self):
        """Use a single dataset: q_profile solves for a root and does not vectorize."""
        self.y, self.v, self.X, _ = make_data(n_datasets=1, n_predictors=3)

    def time_q_gen(self):
        """Time the generalized Q statistic."""
        q_gen(self.y, self.v, self.X, tau2=0.5)

    def time_q_profile(self):
        """Time the profile-likelihood interval on tau^2."""
        q_profile(self.y, self.v, self.X, alpha=0.05)


class TimeNullCorrelation:
    """Time the null-correlation estimator."""

    def setup(self):
        """Enough datasets that the correlation across them is estimable."""
        self.y, _, _, self.groups = make_data(n_observations=40, n_datasets=200)

    def time_estimate_null_correlation(self):
        """Time the pooled within-group correlation."""
        estimate_null_correlation(self.y, self.groups)

    def time_estimate_null_correlation_ungrouped(self):
        """Time the same estimate with no group structure to exploit."""
        estimate_null_correlation(self.y)


class PeakMemClusterRobustCov:
    """Guard the memory the sandwich needs.

    Nothing of size ``(n_datasets, n_observations, n_observations)`` may be
    materialized, and a peak that jumps says something now is.
    """

    def setup(self):
        """Fit once, so only the sandwich allocates."""
        self.y, self.v, self.X, self.groups = make_data()
        self.beta, self.model_cov = weighted_least_squares(self.y, self.v, self.X, return_cov=True)

    def peakmem_cr2(self):
        """Peak memory of the default sandwich."""
        cluster_robust_cov(
            self.y, self.v, self.X, self.beta, self.groups, model_cov=self.model_cov
        )
