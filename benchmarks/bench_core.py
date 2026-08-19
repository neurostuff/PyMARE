"""Benchmark the Dataset container and the meta_regression entry point.

The container is on the path of every user-facing call, and ``to_df`` is the
export that has to expand one shared column across every parallel dataset.
"""

from pymare import Dataset, meta_regression

from .common import make_data

# asv benchmark attributes, applied to every benchmark in this module. The
# ceilings matter because the Benchmark workflow times the whole suite twice on
# every pull request: bounding the samples keeps that under a couple of minutes,
# and the floor of two samples keeps a single noisy reading from being the whole
# measurement.
repeat = (2, 5, 10.0)
warmup_time = 0.1
timeout = 180


class TimeDataset:
    """Time building a Dataset and exporting it."""

    def setup(self):
        """Build the arrays a Dataset is assembled from."""
        self.y, self.v, self.X, self.groups = make_data(n_datasets=50, n_predictors=3)
        self.dataset = Dataset(y=self.y, v=self.v, X=self.X, g=self.groups)

    def time_init(self):
        """Construct the container, which validates and shapes its inputs."""
        Dataset(y=self.y, v=self.v, X=self.X, g=self.groups)

    def time_to_df(self):
        """Export to a long data frame, one row per observation per dataset."""
        self.dataset.to_df()


class TimeMetaRegression:
    """Time the one-call interface most users start from."""

    def setup(self):
        """Build a problem with dependent observations."""
        self.y, self.v, self.X, self.groups = make_data(n_datasets=50, n_predictors=3)

    def time_meta_regression(self):
        """Fit through the entry point, with no group structure."""
        meta_regression(y=self.y, v=self.v, X=self.X, method="DL")

    def time_meta_regression_with_groups(self):
        """Fit through the entry point, with cluster-robust inference."""
        meta_regression(y=self.y, v=self.v, X=self.X, g=self.groups, method="DL")
