"""Benchmark the combination tests.

The grouped calls are the ones worth watching: accounting for dependence means
inflating each group's variance by its own weights, which is per-group work that
has to stay vectorized across features.
"""

import numpy as np

from pymare.estimators import FisherCombinationTest, StoufferCombinationTest

from .common import make_data

# asv benchmark attributes, applied to every benchmark in this module. The
# ceilings matter because the Benchmark workflow times the whole suite twice on
# every pull request: bounding the samples keeps that under a couple of minutes,
# and the floor of two samples keeps a single noisy reading from being the whole
# measurement.
repeat = (2, 5, 10.0)
warmup_time = 0.1
timeout = 180


class TimeCombinationTests:
    """Time Stouffer's and Fisher's methods, with and without dependence.

    Parameters
    ----------
    mode : {"directed", "undirected", "concordant"}
        The tail handling, which changes how much work the test does.
    """

    params = ["directed", "undirected", "concordant"]
    param_names = ["mode"]

    def setup(self, mode):
        """Build z-scores and group labels."""
        z, _, _, self.groups = make_data(n_predictors=1)
        self.z = z
        self.weights = np.abs(z[:, :1]) + 0.5

    def time_stouffer(self, mode):
        """Stouffer's method over independent estimates."""
        StoufferCombinationTest(mode).fit(z=self.z)

    def time_stouffer_with_groups(self, mode):
        """Stouffer's method with the within-group correlation estimated."""
        StoufferCombinationTest(mode).fit(z=self.z, g=self.groups)

    def time_stouffer_with_groups_and_weights(self, mode):
        """Stouffer's method with groups and per-observation weights."""
        StoufferCombinationTest(mode).fit(z=self.z, w=self.weights, g=self.groups)

    def time_fisher(self, mode):
        """Fisher's method over independent estimates."""
        FisherCombinationTest(mode).fit(z=self.z)

    def time_fisher_with_groups(self, mode):
        """Fisher's method with Brown's correction for dependence."""
        FisherCombinationTest(mode).fit(z=self.z, g=self.groups)
