"""Tests for estimators that use stan.

pystan is an optional dependency, so these tests skip rather than fail when it
is missing. Marked ``stan`` so CI can run them in a job that installs it.
"""

from importlib.util import find_spec

import pytest

from pymare.estimators import StanMetaRegression

pytestmark = pytest.mark.stan

requires_pystan = pytest.mark.skipif(
    find_spec("pystan") is None, reason="requires the optional pystan dependency"
)


@requires_pystan
def test_stan_estimator(dataset):
    """Run smoke test for StanMetaRegression."""
    # no ground truth here, so we use sanity checks and rough bounds
    est = StanMetaRegression(num_samples=3000).fit_dataset(dataset)
    results = est.summary()
    assert "BayesianMetaRegressionResults" == results.__class__.__name__
    summary = results.summary(["beta", "tau2"])
    beta1, beta2, tau2 = summary["mean"].values[:3]
    assert -0.5 < beta1 < 0.1
    assert 0.6 < beta2 < 0.9
    assert 3 < tau2 < 5


def test_stan_2d_input_failure(dataset_2d):
    """Run smoke test for StanMetaRegression on 2D data.

    No pystan needed: the shape is rejected before the model is compiled.
    """
    with pytest.raises(ValueError) as exc:
        StanMetaRegression(num_samples=500).fit_dataset(dataset_2d)
    assert str(exc.value).startswith("The StanMetaRegression")
