"""Fixtures for the PyMARE test suite."""

import os
import os.path as op

import numpy as np
import pandas as pd
import pytest

from pymare import Dataset
from pymare.estimators import (
    DerSimonianLaird,
    FisherCombinationTest,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    StoufferCombinationTest,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)
from pymare.tests.utils import cmdstan_is_available, get_test_data_path


def pytest_collection_modifyitems(config, items):
    """Fail the run where CmdStan is declared present but is not.

    The Stan tests skip when CmdStan is missing, so a contributor without it does
    not see red. In CI that leniency is wrong: a skip is indistinguishable from a
    pass in a job log, which is how this job once reported success while running
    none of the tests it existed to run.

    ``PYMARE_REQUIRE_CMDSTAN=1``, which the Stan CI job sets, asserts that the
    environment should be able to run them. Failing at collection reports that
    once rather than as a quietly shorter run.
    """
    if os.environ.get("PYMARE_REQUIRE_CMDSTAN") != "1":
        return
    if cmdstan_is_available():
        return

    raise pytest.UsageError(
        "PYMARE_REQUIRE_CMDSTAN=1 says this environment should be able to run the "
        "Stan tests, but cmdstanpy or its CmdStan installation is missing, so they "
        "would all skip. Install them with `pip install -e .[stan]` followed by "
        "`python -m cmdstanpy.install_cmdstan`, or unset PYMARE_REQUIRE_CMDSTAN to "
        "allow the skip."
    )


# -----------------------------------------------------------------------------
# Basic data
# -----------------------------------------------------------------------------


@pytest.fixture(scope="package")
def variables():
    """Build basic numpy variables."""
    y = np.array([[-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]]).T
    v = np.array([[1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]]).T
    X = np.array([1, 1, 2, 2, 4, 4, 2.8, 2.8])
    return (y, v, X)


@pytest.fixture(scope="package")
def extreme_effect_results():
    """Return the results of a fit at a chosen effect size.

    The sampling variances are small and the residuals around the effect are
    tiny, so the statistic grows in proportion to the effect and can be driven
    as far into the tail as a caller needs -- including past the point where a
    double-precision p-value is exactly zero.
    """
    v = np.array([[0.02, 0.03, 0.025, 0.04, 0.02]]).T
    residual = np.array([[0.01, -0.02, 0.015, -0.01, 0.005]]).T

    def fit(effect, correction="knapp-hartung"):
        dataset = Dataset(y=effect + residual, v=v)
        estimator = WeightedLeastSquares(tau2=0.0, small_sample_correction=correction)
        return estimator.fit_dataset(dataset).summary()

    return fit


@pytest.fixture(scope="package")
def small_variance_variables(variables):
    """Make highly correlated variables."""
    y, v, X = variables
    y = X.copy()
    v /= 10
    return (y, v, X)


@pytest.fixture(scope="package")
def dataset(variables):
    """Build a Dataset compiled from the variables fixture."""
    return Dataset(*variables, X_names=["my_covariate"])


@pytest.fixture(scope="package")
def small_variance_dataset(small_variance_variables):
    """Build a Dataset compiled from the small variance variables fixture."""
    return Dataset(*small_variance_variables, X_names=["my_covariate"])


@pytest.fixture(scope="package")
def small_dataset_2d(variables):
    """Build a small Dataset with 2D data."""
    y = np.array([[1.5, 1.9, 2.2], [4, 2, 1]]).T
    v = np.array([[1, 0.8, 3], [1, 1.5, 1]]).T
    return Dataset(y, v)


@pytest.fixture(scope="package")
def dataset_2d(variables):
    """Build a larger Dataset with 2D data."""
    y, v, X = variables
    y = np.repeat(y, 3, axis=1)
    y[:, 1] = np.random.randint(-10, 10, size=len(y))
    v = np.repeat(v, 3, axis=1)
    v[:, 1] = np.random.randint(2, 10, size=len(v))
    return Dataset(y, v, X)


@pytest.fixture(scope="package")
def dataset_n():
    """Build a Dataset with sample sizes, but no variances."""
    y = np.array([[-3.0, -0.5, 0.0, -5.01, 0.35, -2.0, -6.0, -4.0, -4.3, -0.1, -1.0]]).T
    n = (
        np.array(
            [[16, 16, 20.548, 32.597, 14.0, 11.118, 4.444, 12.414, 26.963, 130.556, 126.76]]
        ).T
        / 2
    )
    return Dataset(y, n=n)


@pytest.fixture(scope="package")
def vars_with_intercept():
    """Build basic numpy variables with intercepts included in the design matrix."""
    y = np.array([[-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]]).T
    v = np.array([[1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]]).T
    X = np.array([np.ones(8), [1, 1, 2, 2, 4, 4, 2.8, 2.8]]).T
    return (y, v, X)


# -----------------------------------------------------------------------------
# Dependent estimates: estimator sets
# -----------------------------------------------------------------------------


@pytest.fixture(
    params=[StoufferCombinationTest, FisherCombinationTest], ids=["stouffer", "fisher"]
)
def combination_estimator(request):
    """Both combination tests, which share validation and permutation behavior."""
    return request.param


@pytest.fixture(
    params=[(StoufferCombinationTest, {"group_level": True}), (FisherCombinationTest, {})],
    ids=["stouffer-group-level", "brown"],
)
def group_combination_estimator(request):
    """Build a combination test that aggregates whole groups rather than rows."""
    estimator, defaults = request.param
    return lambda **kwargs: estimator(**{**defaults, **kwargs})


@pytest.fixture(
    params=[DerSimonianLaird, Hedges, VarianceBasedLikelihoodEstimator],
    ids=["DL", "HE", "ML"],
)
def variance_estimator(request):
    """Return each estimator that obtains tau^2 from the supplied variances."""
    return request.param


@pytest.fixture(params=["individual", "rescale", "collapse"])
def weight_scheme(request):
    """Every weighting scheme the grouped estimators accept."""
    return request.param


# -----------------------------------------------------------------------------
# Dependent estimates: data builders and references
# -----------------------------------------------------------------------------


@pytest.fixture(scope="package")
def dependent_data():
    """Build data where estimates within a group share a common offset."""

    def _build(rng, n_groups=10, n_per_group=3, n_datasets=4, rho_sd=1.0):
        n_estimates = n_groups * n_per_group
        shared = rng.normal(0, rho_sd, size=(n_groups, n_datasets))
        noise = rng.normal(0, 0.25, size=(n_estimates, n_datasets))
        y = np.repeat(shared, n_per_group, axis=0) + noise
        v = np.abs(rng.normal(0, 0.2, size=(n_estimates, n_datasets))) + 0.5
        X = np.ones((n_estimates, 1))
        groups = np.repeat(np.arange(n_groups), n_per_group)
        return y, v, X, groups

    return _build


@pytest.fixture(
    params=[
        (WeightedLeastSquares, "v"),
        (DerSimonianLaird, "v"),
        (Hedges, "v"),
        (VarianceBasedLikelihoodEstimator, "v"),
        (SampleSizeBasedLikelihoodEstimator, "n"),
    ],
    ids=["WLS", "DL", "HE", "ML", "SSML"],
)
def grouped_estimator(request, dependent_data):
    """Pair an estimator with the argument it takes and a builder for its kwargs.

    ``_inputs(shared_second=True)`` supplies that argument as a single column,
    which every estimator has to read as applying to all parallel datasets.
    """
    estimator, second_arg = request.param

    def _inputs(shared_second=False, **kwargs):
        y, v, X, groups = dependent_data(np.random.RandomState(0), **kwargs)
        if second_arg == "v":
            second = v
        else:
            # Sample sizes must vary for the sample-size-based estimator.
            second = np.random.RandomState(1).randint(20, 200, size=y.shape).astype(float)
        if shared_second:
            second = second[:, :1]
        return {"y": y, second_arg: second, "X": X}, groups

    return estimator, second_arg, _inputs


@pytest.fixture(scope="package")
def group_level_design():
    """Build a design whose slope is carried by only the first ``n_nonzero`` groups."""

    def _build(n_nonzero, n_groups=20, group_size=4):
        groups = np.repeat(np.arange(n_groups), group_size)
        x = np.repeat((np.arange(n_groups) < n_nonzero).astype(float), group_size)
        n_estimates = n_groups * group_size
        return np.c_[np.ones(n_estimates), x], np.ones(n_estimates), groups

    return _build


@pytest.fixture(scope="package")
def explicit_cluster_robust_cov():
    """Compute the CR0 sandwich as a plain per-dataset, per-group loop."""

    def _reference(y, v, X, beta, groups):
        n_preds = X.shape[1]
        robust = np.empty((n_preds, n_preds, y.shape[1]))
        for i_dataset in range(y.shape[1]):
            weights = 1.0 / v[:, i_dataset]
            bread = np.linalg.pinv(X.T @ np.diag(weights) @ X)
            resid = y[:, i_dataset] - X @ beta[:, i_dataset]
            meat = np.zeros((n_preds, n_preds))
            for group in np.unique(groups):
                members = groups == group
                score = (X[members] * (weights[members] * resid[members])[:, None]).sum(0)
                meat += np.outer(score, score)
            robust[:, :, i_dataset] = bread @ meat @ bread
        return robust

    return _reference


@pytest.fixture(scope="package")
def explicit_knapp_hartung():
    """Compute the Knapp-Hartung scale factor as a plain per-dataset, per-row loop.

    Written from the published formula rather than from
    :func:`~pymare.stats.knapp_hartung_cov_and_dof`, so that a test comparing the
    two is a check on the implementation rather than a restatement of it.
    """

    def _reference(y, v, X, beta, tau2=0.0, conservative=False):
        n_obs, n_preds = X.shape
        scale = np.empty(y.shape[1])
        for i_dataset in range(y.shape[1]):
            column = min(i_dataset, v.shape[1] - 1)
            tau = tau2[i_dataset] if np.ndim(tau2) else tau2
            total = 0.0
            for i_obs in range(n_obs):
                weight = 1.0 / (v[i_obs, column] + tau)
                resid = y[i_obs, i_dataset] - X[i_obs] @ beta[:, i_dataset]
                total += weight * resid**2
            scale[i_dataset] = total / (n_obs - n_preds)
            if conservative:
                scale[i_dataset] = max(scale[i_dataset], 1.0)
        return scale

    return _reference


@pytest.fixture(scope="package")
def correlated_block_data():
    """Build estimates that share a signal, the first ``block_size`` of which co-vary."""

    def _build(n_estimates, n_datasets, block_size=0, rho=0.8, seed=0):
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(n_datasets) * 2.0
        shared = rng.standard_normal(n_datasets)
        y = np.array([signal + rng.standard_normal(n_datasets) for _ in range(n_estimates)])
        for i in range(block_size):
            y[i] = (
                signal + np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(n_datasets)
            )
        groups = np.array([0] * block_size + list(range(1, n_estimates - block_size + 1)))
        return y, groups

    return _build


@pytest.fixture(scope="package")
def centering_shrinkage():
    """Apply the centering map that ``np.corrcoef`` implies to a true correlation matrix."""

    def _shrink(corr):
        n_estimates = corr.shape[0]
        centering = np.eye(n_estimates) - np.ones((n_estimates, n_estimates)) / n_estimates
        shrunk = centering @ corr @ centering
        scale = np.sqrt(np.diag(shrunk))
        return shrunk / np.outer(scale, scale)

    return _shrink


@pytest.fixture(scope="package")
def sampled_centering_shrinkage():
    """Estimate the centered correlation from data, the way a caller really would.

    The ``centering_shrinkage`` fixture applies the centering map exactly, so
    every off-diagonal entry of a block comes back equal and a block-mean
    inversion cannot be told apart from an entrywise one. Drawing ``n_datasets``
    samples instead gives the block the genuine spread that separates them.
    """

    def _estimate(corr, n_datasets, seed=0):
        n_estimates = corr.shape[0]
        factor = np.linalg.cholesky(corr + 1e-10 * np.eye(n_estimates))
        y = factor @ np.random.default_rng(seed).standard_normal((n_estimates, n_datasets))
        return np.corrcoef(y - y.mean(axis=0), rowvar=True)

    return _estimate


@pytest.fixture(scope="package")
def block_correlation():
    """Build an equicorrelated-block correlation matrix and its group labels."""

    def _build(n_estimates, blocks):
        corr = np.eye(n_estimates)
        groups = np.empty(n_estimates, dtype=int)
        start = 0
        for label, (size, rho) in enumerate(blocks):
            stop = start + size
            corr[start:stop, start:stop] = rho + (1 - rho) * np.eye(size)
            groups[start:stop] = label
            start = stop
        groups[start:] = np.arange(len(blocks), len(blocks) + n_estimates - start)
        return corr, groups

    return _build


@pytest.fixture(scope="package")
def robumeta_dataset():
    """Load the dataset the robumeta reference values were computed on."""
    frame = pd.read_csv(op.join(get_test_data_path(), "robumeta_correlated_effects.csv"))
    n_estimates = len(frame)
    designs = {
        "intercept": np.ones((n_estimates, 1)),
        "within": np.c_[np.ones(n_estimates), frame["within"].to_numpy()],
        "both": np.c_[
            np.ones(n_estimates), frame["within"].to_numpy(), frame["between"].to_numpy()
        ],
    }
    return frame, designs


@pytest.fixture(scope="package")
def metafor_dataset():
    """Load the designs the metafor reference values were computed on."""
    return pd.read_csv(op.join(get_test_data_path(), "metafor_small_sample.csv"))


# -----------------------------------------------------------------------------
# Fitted estimators and their results
# -----------------------------------------------------------------------------


@pytest.fixture
def fitted_estimator(dataset):
    """Create a fitted Estimator as a fixture."""
    est = DerSimonianLaird()
    return est.fit_dataset(dataset)


@pytest.fixture
def small_variance_estimator(small_variance_dataset):
    """Create a fitted Estimator with small variances as a fixture."""
    est = DerSimonianLaird()
    return est.fit_dataset(small_variance_dataset)


@pytest.fixture
def results(fitted_estimator):
    """Create a results object as a fixture."""
    return fitted_estimator.summary()


@pytest.fixture
def small_variance_results(small_variance_estimator):
    """Create a results object with small variances as a fixture."""
    return small_variance_estimator.summary()


@pytest.fixture
def results_2d(fitted_estimator, dataset_2d):
    """Create a 2D results object as a fixture."""
    est = VarianceBasedLikelihoodEstimator()
    return est.fit_dataset(dataset_2d).summary()


# -----------------------------------------------------------------------------
# Effect size conversion
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_samp_data():
    """Create one-sample data for tests."""
    return {
        "m": np.array([7, 5, 4]),
        "sd": np.sqrt(np.array([4.2, 1.2, 1.9])),
        "n": np.array([24, 31, 40]),
        "r": np.array([0.2, 0.18, 0.3]),
    }


@pytest.fixture(scope="module")
def two_samp_data():
    """Create two-sample data for tests."""
    return {
        "m1": np.array([4, 2]),
        "sd1": np.sqrt(np.array([1, 9])),
        "n1": np.array([12, 15]),
        "m2": np.array([5, 2.5]),
        "sd2": np.sqrt(np.array([4, 16])),
        "n2": np.array([12, 16]),
    }


# -----------------------------------------------------------------------------
# Stan estimator
# -----------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point ``os.path.expanduser("~")`` at a temporary directory, on any platform.

    Returns
    -------
    :obj:`pathlib.Path`
        The directory ``~`` now expands to.

    Notes
    -----
    Both variables are needed. POSIX ``expanduser`` reads ``HOME``; Windows reads
    ``USERPROFILE`` and ignores ``HOME`` entirely. Setting only ``HOME``
    therefore looks portable while silently leaving the real profile in place on
    Windows, so a test can pass on two platforms and write into the runner's
    actual home directory on the third.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="package")
def planted_hierarchical_dataset():
    """Simulate a Dataset from the model ``meta_regression.stan`` encodes.

    Returns
    -------
    :obj:`tuple` of (:obj:`~pymare.core.Dataset`, :obj:`dict`)
        The simulated Dataset, with one group label per observation in ``g``,
        and the parameter values it was generated from.

    Notes
    -----
    The sampling standard deviations are drawn from ``uniform(0.1, 0.4)``, well
    away from 1, and that is load-bearing. The ``variables`` fixture has ``v``
    near 1, where ``v`` and ``sqrt(v)`` differ by a few percent, so a model that
    confuses them fits it about as well as the correct one. Here ``v`` spans 0.01
    to 0.16 against ``sqrt(v)`` from 0.1 to 0.4 -- a factor of 2.5 to 10 in one
    direction -- so the mistake shows up as an inflated tau2. Do not substitute
    ``variables`` here.

    ``tau2`` and ``tau`` are likewise kept apart (0.25 against 0.5) so reporting
    one under the other's name fails a tight interval.
    """
    rng = np.random.default_rng(20250818)

    n_groups, per_group = 30, 3
    beta = np.array([0.5, -0.8])
    tau = 0.5

    groups = np.repeat(np.arange(n_groups), per_group)
    n_observations = groups.size

    moderator = rng.normal(size=n_observations)
    # Dataset prepends the intercept itself, so beta[0] is the intercept and
    # beta[1] the moderator slope in the X the estimator will actually see.
    X = np.column_stack([np.ones(n_observations), moderator])
    theta = rng.normal(0, tau, size=n_groups)
    sigma = rng.uniform(0.1, 0.4, size=n_observations)
    y = X @ beta + theta[groups] + rng.normal(0, sigma)

    dataset = Dataset(y=y, v=sigma**2, X=moderator, X_names=["moderator"], g=groups)
    return dataset, {"beta": beta, "tau": tau, "tau2": tau**2}
