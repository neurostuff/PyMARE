"""Benchmark the estimators end to end.

The pairs with and without ``g`` are the point of this module: passing group
labels swaps the model-based covariance for the cluster-robust one, which is the
most expensive part of a robust fit, so the two timings together say what
dependence costs rather than only what a fit costs.
"""

from pymare import Dataset
from pymare.estimators import (
    DerSimonianLaird,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)

from .common import (
    N_DATASETS_LOOPED,
    make_data,
    make_group_level_design,
    make_sample_sizes,
)

# asv benchmark attributes, applied to every benchmark in this module. The
# ceilings matter because the Benchmark workflow times the whole suite twice on
# every pull request: bounding the samples keeps that under a couple of minutes,
# and the floor of two samples keeps a single noisy reading from being the whole
# measurement.
repeat = (3, 8, 15.0)
warmup_time = 0.1
timeout = 180


class TimeVectorizedEstimators:
    """Time the estimators that vectorize across datasets.

    Parameters
    ----------
    estimator : {"WLS", "DL", "Hedges"}
        The estimator to time, named as asv will label it.
    """

    params = ["WLS", "DL", "Hedges"]
    param_names = ["estimator"]

    ESTIMATORS = {
        "WLS": WeightedLeastSquares,
        "DL": DerSimonianLaird,
        "Hedges": Hedges,
    }

    def setup(self, estimator):
        """Build one problem, reused by every parameter.

        Two designs, because the aggregating schemes only accept one of them:
        ``X`` varies within a group, ``group_X`` does not.
        """
        self.y, self.v, self.X, self.groups = make_data(n_predictors=3)
        self.group_X = make_group_level_design(self.groups)
        self.estimator = self.ESTIMATORS[estimator]

    def time_fit(self, estimator):
        """Fit with no group structure: the model-based covariance."""
        self.estimator().fit(y=self.y, v=self.v, X=self.X)

    def time_fit_with_groups(self, estimator):
        """Fit with group labels: the cluster-robust covariance and Satterthwaite dof."""
        self.estimator().fit(y=self.y, v=self.v, X=self.X, g=self.groups)

    def time_fit_rescale(self, estimator):
        """Fit under the correlated-effects working model."""
        self.estimator(weight_scheme="rescale", rho=0.8).fit(
            y=self.y, v=self.v, X=self.group_X, g=self.groups
        )

    def time_fit_collapse(self, estimator):
        """Fit with every group reduced to one row."""
        self.estimator(weight_scheme="collapse", rho=0.8).fit(
            y=self.y, v=self.v, X=self.group_X, g=self.groups
        )


class TimeCorrelatedEffectsFit:
    """Time the one fit that reads the observation-level design.

    DerSimonian-Laird under ``"rescale"`` is the only estimator that accepts a
    design varying inside a group, and it is the path robumeta's correlated
    effects model corresponds to, so it gets its own entry.
    """

    def setup(self):
        """Build a design that varies within groups."""
        self.y, self.v, self.X, self.groups = make_data(n_predictors=3)

    def time_fit(self):
        """Fit the correlated-effects model."""
        DerSimonianLaird(weight_scheme="rescale", rho=0.8).fit(
            y=self.y, v=self.v, X=self.X, g=self.groups
        )


class TimeLikelihoodEstimators:
    """Time the likelihood estimators, which loop over datasets.

    They optimize per dataset, so they get the smaller second dimension. That
    also makes them the benchmark that would catch a loop appearing in a path
    that used to vectorize.
    """

    params = ["ML", "REML"]
    param_names = ["method"]

    def setup(self, method):
        """Build a problem small enough for a per-dataset optimizer."""
        self.y, self.v, self.X, self.groups = make_data(
            n_observations=40, n_datasets=N_DATASETS_LOOPED, n_predictors=2
        )
        self.n = make_sample_sizes(n_observations=40, n_datasets=N_DATASETS_LOOPED)

    def time_variance_based_fit(self, method):
        """Fit from sampling variances."""
        VarianceBasedLikelihoodEstimator(method=method).fit(y=self.y, v=self.v, X=self.X)

    def time_variance_based_fit_with_groups(self, method):
        """Fit from sampling variances, with cluster-robust inference."""
        VarianceBasedLikelihoodEstimator(method=method).fit(
            y=self.y, v=self.v, X=self.X, g=self.groups
        )

    def time_sample_size_based_fit(self, method):
        """Fit from sample sizes instead of variances."""
        SampleSizeBasedLikelihoodEstimator(method=method).fit(y=self.y, n=self.n, X=self.X)


class TimeSummary:
    """Time what happens after the fit.

    ``summary()`` is where the standard errors, p-values and intervals are
    formed, so it is the other half of the user-visible cost of a fit.
    """

    def setup(self):
        """Fit once, robustly, so the summary has dof to expand.

        Fitted through a Dataset because the heterogeneity statistics need one.
        """
        y, v, X, groups = make_data(n_predictors=3)
        dataset = Dataset(y=y, v=v, X=X, g=groups, add_intercept=False)
        self.fitted = DerSimonianLaird().fit_dataset(dataset)

    def time_summary(self):
        """Build the results object."""
        self.fitted.summary()

    def time_get_fe_stats(self):
        """Form the fixed-effect statistics from it."""
        self.fitted.summary().get_fe_stats()

    def time_get_heterogeneity_stats(self):
        """Form the heterogeneity statistics from it."""
        self.fitted.summary().get_heterogeneity_stats()
