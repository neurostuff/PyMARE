"""Tests for pymare.estimators.estimators."""

import warnings

import numpy as np
import pytest

from pymare import Dataset
from pymare.estimators import (
    DerSimonianLaird,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)
from pymare.estimators.estimators import _collapse_n_inputs
from pymare.stats import (
    DEFAULT_RHO,
    collapse_groups,
    collapse_groups_by_n,
    weighted_least_squares,
)


def test_weighted_least_squares_estimator(dataset):
    """Test WeightedLeastSquares estimator."""
    # ground truth values are from metafor package in R
    est = WeightedLeastSquares().fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert isinstance(tau2, float)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.2725, 0.6935], atol=1e-4)
    assert tau2 == 0.0

    # With non-zero tau^2
    est = WeightedLeastSquares(8.0).fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    assert np.allclose(beta.ravel(), [-0.1071, 0.7657], atol=1e-4)
    assert tau2 == 8.0


def test_dersimonian_laird_estimator(dataset):
    """Test DerSimonianLaird estimator."""
    # ground truth values are from metafor package in R
    est = DerSimonianLaird().fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1,)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2, 8.3627, atol=1e-4)


def test_2d_DL_estimator(dataset_2d):
    """Test DerSimonianLaird estimator on 2D Dataset."""
    results = DerSimonianLaird().fit_dataset(dataset_2d).summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 3)
    assert tau2.shape == (3,)
    assert fe_stats["est"].shape == (2, 3)
    assert fe_stats["se"].shape == (2, 3)
    assert fe_stats["ci_l"].shape == (2, 3)
    assert fe_stats["ci_u"].shape == (2, 3)
    assert fe_stats["z"].shape == (2, 3)
    assert fe_stats["p"].shape == (2, 3)

    # Check output values
    # First and third sets are identical to previous DL test; second set is
    # randomly different.
    assert np.allclose(beta[:, 0], [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2[0], 8.3627, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1070, 0.7664], atol=1e-4)
    assert not np.allclose(tau2[1], 8.3627, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1070, 0.7664], atol=1e-4)
    assert np.allclose(tau2[2], 8.3627, atol=1e-4)


def test_hedges_estimator(dataset):
    """Test Hedges estimator."""
    # ground truth values are from metafor package in R, except that metafor
    # always gives negligibly different values for tau2, likely due to
    # algorithmic differences in the computation.
    est = Hedges().fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1,)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2, 11.3881, atol=1e-4)
    assert np.allclose(fe_stats["se"].ravel(), [3.0479, 1.1335], atol=1e-4)
    # The unweighted fit that produces tau^2 would have given these instead.
    assert not np.allclose(fe_stats["se"].ravel(), [0.8639, 0.3217], atol=1e-4)


def test_2d_hedges(dataset_2d):
    """Test Hedges estimator on 2D Dataset."""
    results = Hedges().fit_dataset(dataset_2d).summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 3)
    assert tau2.shape == (3,)
    assert fe_stats["est"].shape == (2, 3)
    assert fe_stats["se"].shape == (2, 3)
    assert fe_stats["ci_l"].shape == (2, 3)
    assert fe_stats["ci_u"].shape == (2, 3)
    assert fe_stats["z"].shape == (2, 3)
    assert fe_stats["p"].shape == (2, 3)

    # First and third sets are identical to single dim test; second set is
    # randomly different.
    assert np.allclose(beta[:, 0], [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2[0], 11.3881, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1070, 0.7664], atol=1e-4)
    assert not np.allclose(tau2[1], 11.3881, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1066, 0.7704], atol=1e-4)
    assert np.allclose(tau2[2], 11.3881, atol=1e-4)


def test_variance_based_maximum_likelihood_estimator(dataset):
    """Test VarianceBasedLikelihoodEstimator estimator."""
    # ground truth values are from metafor package in R
    est = VarianceBasedLikelihoodEstimator(method="ML").fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2, 7.7649, atol=1e-4)


def test_variance_based_restricted_maximum_likelihood_estimator(dataset):
    """Test VarianceBasedLikelihoodEstimator estimator with REML."""
    # ground truth values are from metafor package in R
    est = VarianceBasedLikelihoodEstimator(method="REML").fit_dataset(dataset)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (2, 1)
    assert fe_stats["se"].shape == (2, 1)
    assert fe_stats["ci_l"].shape == (2, 1)
    assert fe_stats["ci_u"].shape == (2, 1)
    assert fe_stats["z"].shape == (2, 1)
    assert fe_stats["p"].shape == (2, 1)

    # Check output values
    assert np.allclose(beta.ravel(), [-0.1066, 0.7700], atol=1e-4)
    assert np.allclose(tau2, 10.9499, atol=1e-4)


def test_sample_size_based_maximum_likelihood_estimator(dataset_n):
    """Test SampleSizeBasedLikelihoodEstimator estimator."""
    # test values have not been verified for convergence with other packages
    est = SampleSizeBasedLikelihoodEstimator(method="ML").fit_dataset(dataset_n)
    results = est.summary()
    beta = results.fe_params
    sigma2 = results.estimator.params_["sigma2"]
    tau2 = results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (1, 1)
    assert sigma2.shape == (1, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (1, 1)
    assert fe_stats["se"].shape == (1, 1)
    assert fe_stats["ci_l"].shape == (1, 1)
    assert fe_stats["ci_u"].shape == (1, 1)
    assert fe_stats["z"].shape == (1, 1)
    assert fe_stats["p"].shape == (1, 1)

    # Check output values
    assert np.allclose(beta, [-2.0951], atol=1e-4)
    assert np.allclose(sigma2, 12.777, atol=1e-3)
    assert np.allclose(tau2, 2.8268, atol=1e-4)


def test_sample_size_based_restricted_maximum_likelihood_estimator(dataset_n):
    """Test SampleSizeBasedLikelihoodEstimator REML estimator."""
    # test values have not been verified for convergence with other packages
    est = SampleSizeBasedLikelihoodEstimator(method="REML").fit_dataset(dataset_n)
    results = est.summary()
    beta = results.fe_params
    sigma2 = results.estimator.params_["sigma2"]
    tau2 = results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (1, 1)
    assert sigma2.shape == (1, 1)
    assert tau2.shape == (1, 1)
    assert fe_stats["est"].shape == (1, 1)
    assert fe_stats["se"].shape == (1, 1)
    assert fe_stats["ci_l"].shape == (1, 1)
    assert fe_stats["ci_u"].shape == (1, 1)
    assert fe_stats["z"].shape == (1, 1)
    assert fe_stats["p"].shape == (1, 1)

    # Check output values
    assert np.allclose(beta, [-2.1071], atol=1e-4)
    assert np.allclose(sigma2, 13.048, atol=1e-3)
    assert np.allclose(tau2, 3.2177, atol=1e-4)


def test_2d_parallel_datasets(dataset_2d):
    """Test parallel datasets in the second dimension of y and v."""
    est = VarianceBasedLikelihoodEstimator().fit_dataset(dataset_2d)
    results = est.summary()
    beta, tau2 = results.fe_params, results.tau2
    fe_stats = results.get_fe_stats()

    # Check output shapes
    assert beta.shape == (2, 3)
    assert tau2.shape == (1, 3)
    assert fe_stats["est"].shape == (2, 3)
    assert fe_stats["se"].shape == (2, 3)
    assert fe_stats["ci_l"].shape == (2, 3)
    assert fe_stats["ci_u"].shape == (2, 3)
    assert fe_stats["z"].shape == (2, 3)
    assert fe_stats["p"].shape == (2, 3)

    # Check output values
    # First and third sets are identical to single dim test; 2nd is different
    assert np.allclose(beta[:, 0], [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2[0, 0], 7.7649, atol=1e-4)
    assert not np.allclose(beta[:, 1], [-0.1072, 0.7653], atol=1e-4)
    assert not np.allclose(tau2[0, 1], 7.7649, atol=1e-4)
    assert np.allclose(beta[:, 2], [-0.1072, 0.7653], atol=1e-4)
    assert np.allclose(tau2[0, 2], 7.7649, atol=1e-4)


def test_many_parallel_datasets_are_fitted_in_one_search():
    """The likelihood estimators fit every parallel dataset at once, and say nothing.

    They used to loop over the second dimension in Python and warn that doing so
    would be slow. The search is now vectorized over that dimension, so there is
    nothing to warn about -- but each column still has to come back with the
    answer it would have got on its own.
    """
    rng = np.random.RandomState(0)
    y = rng.normal(size=(10, 40))
    v = np.abs(rng.normal(size=(10, 40))) + 0.5
    X = np.ones((10, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        together = VarianceBasedLikelihoodEstimator().fit(y=y, v=v, X=X)

    for column in (0, 17, 39):
        alone = VarianceBasedLikelihoodEstimator().fit(y=y[:, [column]], v=v[:, [column]], X=X)
        assert np.allclose(
            together.params_["tau2"][:, column], alone.params_["tau2"].ravel(), atol=1e-8
        )
        assert np.allclose(
            together.params_["fe_params"][:, column],
            alone.params_["fe_params"].ravel(),
            atol=1e-8,
        )


def test_one_variance_column_applies_to_every_parallel_dataset():
    """A single column of v or n is shared across the datasets, not rejected.

    y may carry many columns against one column of sampling variances or sample
    sizes, which is how a problem with one observed variance per observation
    arrives. The fit has to read that as the same variance for every dataset.
    """
    rng = np.random.RandomState(1)
    y = rng.normal(size=(12, 4))
    v = np.abs(rng.normal(size=(12, 1))) + 0.5
    n = rng.randint(20, 200, size=(12, 1)).astype(float)
    X = np.ones((12, 1))

    shared = VarianceBasedLikelihoodEstimator().fit(y=y, v=v, X=X)
    repeated = VarianceBasedLikelihoodEstimator().fit(y=y, v=np.repeat(v, 4, axis=1), X=X)

    assert shared.params_["tau2"].shape == (1, 4)
    assert np.allclose(shared.params_["tau2"], repeated.params_["tau2"])
    assert np.allclose(shared.params_["fe_params"], repeated.params_["fe_params"])

    shared_n = SampleSizeBasedLikelihoodEstimator().fit(y=y, n=n, X=X)
    repeated_n = SampleSizeBasedLikelihoodEstimator().fit(y=y, n=np.repeat(n, 4, axis=1), X=X)

    assert shared_n.params_["sigma2"].shape == (1, 4)
    assert np.allclose(shared_n.params_["sigma2"], repeated_n.params_["sigma2"])
    assert np.allclose(shared_n.params_["tau2"], repeated_n.params_["tau2"])


@pytest.mark.parametrize("method", ["ML", "REML"])
def test_variance_based_likelihood_lands_on_a_minimum(dataset, method):
    """The reported tau^2 must beat its neighbours, not merely sit close to one.

    The profiled likelihood is searched rather than solved, so what is worth
    pinning is the property the search is there to deliver.
    """
    est = VarianceBasedLikelihoodEstimator(method=method).fit_dataset(dataset)
    y, v, X = dataset.y, dataset.v, dataset.X
    tau2 = est.params_["tau2"].ravel()
    at_estimate = est._nll_func(tau2, y, v, X)

    for candidate in (np.zeros_like(tau2), tau2 / 2, tau2 * 2, tau2 + 1.0, tau2 * 100):
        assert at_estimate <= est._nll_func(candidate, y, v, X) + 1e-9


@pytest.mark.parametrize("method", ["ML", "REML"])
def test_sample_size_based_likelihood_lands_on_a_minimum(dataset_n, method):
    """The reported variance ratio must beat its neighbours and both extremes."""
    est = SampleSizeBasedLikelihoodEstimator(method=method).fit_dataset(dataset_n)
    y, n, X = dataset_n.y, dataset_n.n, dataset_n.X
    tau2 = est.params_["tau2"].ravel()
    ratio = tau2 / (tau2 + est.params_["sigma2"].ravel())
    at_estimate = est._nll_func(ratio, y, n, X)

    candidates = (
        np.zeros_like(ratio),
        np.ones_like(ratio),
        ratio / 2,
        np.minimum(ratio * 2, 1.0),
    )
    for candidate in candidates:
        assert at_estimate <= est._nll_func(candidate, y, n, X) + 1e-9


def test_sample_size_based_likelihood_matches_a_search_over_both_variances(dataset_n):
    """Profiling the scale out must not lose to a direct search over the components.

    The fit searches ``tau^2 / (tau^2 + sigma^2)`` and recovers the scale in
    closed form, in place of a search over sigma^2 and tau^2 themselves. A grid
    over the original two, scored by the likelihood written out in its original
    form, is an independent check that the substitution costs nothing.
    """
    est = SampleSizeBasedLikelihoodEstimator(method="ML").fit_dataset(dataset_n)
    y, n, X = dataset_n.y, dataset_n.n, dataset_n.X

    def joint_nll(sigma2, tau2):
        """Score the ML objective in its original parameters, minimized over beta."""
        v = tau2 + sigma2 / n
        resid = y - X.dot(weighted_least_squares(y, v, X))
        return -0.5 * (np.log(1.0 / v).sum() - (resid**2 / v).sum())

    at_estimate = joint_nll(est.params_["sigma2"].ravel(), est.params_["tau2"].ravel())
    on_grid = min(
        joint_nll(sigma2, tau2)
        for sigma2 in np.linspace(0.5, 40.0, 80)
        for tau2 in np.linspace(0.0, 12.0, 80)
    )

    assert at_estimate <= on_grid + 1e-9


@pytest.mark.parametrize(
    "estimator",
    [WeightedLeastSquares, DerSimonianLaird, Hedges, VarianceBasedLikelihoodEstimator],
    ids=["WLS", "DL", "HE", "ML"],
)
def test_model_based_cov_matches_the_fitted_weights(dataset, estimator):
    """The reported covariance must be (X'WX)^-1 under the coefficients' own weights.

    Every estimator here fits the coefficients with ``1 / (v + tau^2)`` weights, so
    ``(X'WX)^-1`` under those same weights is the only matrix that is their covariance.
    Hedges previously reported the covariance of an unweighted fit instead -- the OLS
    fit it uses internally to obtain tau^2 -- which left the standard errors several
    times too small and unrelated to the coefficients beside them.
    """
    results = estimator().fit_dataset(dataset).summary()
    tau2 = np.ravel(results.tau2)[0]
    w = 1.0 / (dataset.v + tau2)
    expected = np.linalg.pinv(dataset.X.T @ np.diag(w.ravel()) @ dataset.X)

    assert np.allclose(results.fe_se.ravel(), np.sqrt(np.diag(expected)))


# -----------------------------------------------------------------------------
# Dependent estimates: group labels and weighting schemes
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "estimator, scheme",
    [
        (DerSimonianLaird, "collapse"),
        (Hedges, "rescale"),
        (Hedges, "collapse"),
        (VarianceBasedLikelihoodEstimator, "rescale"),
        (VarianceBasedLikelihoodEstimator, "collapse"),
    ],
    ids=["DL-collapse", "HE-rescale", "HE-collapse", "ML-rescale", "ML-collapse"],
)
def test_aggregating_estimators_reject_a_within_group_varying_design(estimator, scheme):
    """The guard belongs to the reduction, not to one scheme's name.

    Every estimator except DerSimonian-Laird reaches tau^2 through the same
    one-row-per-group aggregate under both schemes, so the design restriction
    applies to both. It used to be checked only for "collapse", whose error
    message then recommended "rescale" -- routing users into the unguarded path.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 2)
    n_estimates = groups.size
    y = rng.standard_normal((n_estimates, 1))
    v = np.full((n_estimates, 1), 0.2)
    varying = np.c_[np.ones(n_estimates), np.tile([-1.0, 1.0], 6)]

    with pytest.raises(ValueError, match="constant within each group"):
        estimator(weight_scheme=scheme).fit(y=y, v=v, X=varying, g=groups)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_dersimonian_laird_accepts_a_within_group_varying_design_under_rescale():
    """The remedy the other estimators' error message points at has to work."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 2)
    n_estimates = groups.size
    varying = np.c_[np.ones(n_estimates), np.tile([-1.0, 1.0], 6)]

    fitted = DerSimonianLaird(weight_scheme="rescale").fit(
        y=rng.standard_normal((n_estimates, 1)),
        v=np.full((n_estimates, 1), 0.2),
        X=varying,
        g=groups,
    )

    assert fitted.tau2_model_ == "correlated-effects"
    assert np.isfinite(fitted.params_["tau2"]).all()


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "scheme, expected",
    [("individual", "independent"), ("rescale", "correlated-effects"), ("collapse", "aggregate")],
)
def test_results_statistics_follow_the_model_the_estimator_recorded(scheme, expected):
    """The reduction is declared once and read, not re-derived and hoped to match.

    Whichever way the dependence was modelled, Q has to count independent groups
    and the interval has to describe the same quantity as the estimate it
    accompanies.
    """
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(12), 4)
    n_estimates = groups.size
    y = rng.normal(size=n_estimates) + np.repeat(rng.normal(scale=0.3, size=12), 4)
    dataset = Dataset(y=y, v=np.full(n_estimates, 0.2), g=groups)

    results = DerSimonianLaird(weight_scheme=scheme).fit_dataset(dataset).summary()
    assert results.estimator.tau2_model_ == expected
    assert results._tau2_model() == expected

    _, _, _, analysis_groups = results._analysis_arrays()
    assert (analysis_groups is not None) is (expected == "correlated-effects")

    re_stats = results.get_re_stats()
    tau2 = float(np.ravel(re_stats["tau^2"])[0])
    assert float(np.ravel(re_stats["ci_l"])[0]) <= tau2 <= float(np.ravel(re_stats["ci_u"])[0])

    if scheme != "individual":
        # Q counts the 12 groups, not the 48 rows, however it got there.
        raw = DerSimonianLaird().fit_dataset(dataset).summary().get_heterogeneity_stats()
        assert float(np.ravel(results.get_heterogeneity_stats()["Q"])[0]) < float(
            np.ravel(raw["Q"])[0]
        )


def test_estimators_groups_change_only_inference(grouped_estimator):
    """Group labels must move the covariance but not the point estimates.

    Omitting them must reproduce the previous behaviour exactly.
    """
    estimator, _, build_inputs = grouped_estimator
    kwargs, groups = build_inputs()

    naive = estimator().fit(**kwargs)
    robust = estimator().fit(**kwargs, g=groups)

    assert naive.n_groups_ is None
    assert "n_groups" not in naive.params_
    assert robust.n_groups_ == np.unique(groups).size
    assert np.allclose(naive.params_["fe_params"], robust.params_["fe_params"])
    assert np.allclose(naive.params_["tau2"], robust.params_["tau2"])
    assert not np.allclose(naive.params_["inv_cov"], robust.params_["inv_cov"])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_permutation_test_handles_parallel_datasets(grouped_estimator):
    """tau^2 is reported as (D,) by some estimators and (1, D) by others.

    permutation_test read ``tau2[i]`` as dataset i's value, which is a whole row
    of a two-dimensional array: with one parallel dataset it broadcast by luck,
    and with more it raised before producing any p-value.
    """
    estimator, second_arg, build_inputs = grouped_estimator
    kwargs, groups = build_inputs(n_groups=6, n_per_group=2, n_datasets=3)
    dataset = Dataset(
        y=kwargs["y"],
        X=kwargs["X"],
        g=groups,
        add_intercept=False,
        **{second_arg: kwargs[second_arg]},
    )

    perm = estimator().fit_dataset(dataset).summary().permutation_test(n_perm=16)

    assert perm.perm_p["fe_p"].shape[-1] == 3
    assert np.all(perm.perm_p["fe_p"] > 0)
    assert perm.perm_p["tau2_p"].shape == (3,)


def test_estimators_groups_widen_standard_errors(grouped_estimator):
    """With dependent estimates, robust SEs should exceed model-based ones."""
    estimator, _, build_inputs = grouped_estimator
    kwargs, groups = build_inputs(n_groups=12, n_per_group=4)

    naive = estimator().fit(**kwargs).summary()
    robust = estimator().fit(**kwargs, g=groups).summary()

    assert np.all(robust.fe_se > naive.fe_se)


def test_estimator_rejects_unknown_weight_scheme(grouped_estimator):
    """An unrecognised weighting scheme should fail at construction."""
    with pytest.raises(ValueError, match="Invalid weight_scheme"):
        grouped_estimator[0](weight_scheme="nonsense")


@pytest.mark.parametrize("rho", [-0.5, 1.5])
def test_rho_is_checked_by_the_same_mechanism_as_weight_scheme(grouped_estimator, rho):
    """Both constructor arguments are declared, so both are checked together.

    weight_scheme was validated at construction and rho was not, so an
    out-of-range correlation survived until some later call happened to read it.
    """
    with pytest.raises(ValueError, match="Invalid rho"):
        grouped_estimator[0](weight_scheme="rescale", rho=rho)


def test_rho_warns_when_the_weight_scheme_cannot_use_it(grouped_estimator):
    """Setting rho under the default scheme did nothing, and said nothing."""
    with pytest.warns(UserWarning, match="weight_scheme='individual'"):
        estimator = grouped_estimator[0](rho=0.5)

    # Still honoured as the value to use if the scheme is changed later.
    assert estimator.rho == 0.5


def test_likelihood_estimators_accept_positional_arguments():
    """fit(y, v, X) is the documented call signature and must work."""
    rng = np.random.RandomState(0)
    y = rng.randn(10, 1)
    v = np.abs(rng.randn(10, 1)) + 0.5
    X = np.ones((10, 1))

    positional = VarianceBasedLikelihoodEstimator().fit(y, v, X)
    keyword = VarianceBasedLikelihoodEstimator().fit(y=y, v=v, X=X)

    assert np.allclose(positional.params_["fe_params"], keyword.params_["fe_params"])
    assert np.allclose(positional.params_["tau2"], keyword.params_["tau2"])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_estimators_accept_unsortable_group_labels(weight_scheme):
    """Any hashable label, as encode_groups documents -- not just sortable ones.

    np.unique needs an ordering, so a mix of str and int raised TypeError from
    inside the weight-scheme helpers even though the group primitives handle it.
    """
    rng = np.random.RandomState(0)
    y = rng.randn(12, 1)
    mixed = np.array([1, 1, "b", "b", 2, 2, "d", "d", 3, 3, "f", "f"], dtype=object)

    fitted = DerSimonianLaird(weight_scheme=weight_scheme).fit(
        y=y, v=np.full((12, 1), 0.2), X=np.ones((12, 1)), g=mixed
    )
    assert fitted.n_groups_ == 6

    equivalent = DerSimonianLaird(weight_scheme=weight_scheme).fit(
        y=y, v=np.full((12, 1), 0.2), X=np.ones((12, 1)), g=np.repeat(np.arange(6), 2)
    )
    assert np.allclose(fitted.params_["fe_params"], equivalent.params_["fe_params"])

    # The result statistics have to encode the labels the same way; counting
    # them with np.unique raised TypeError only once the fit was over.
    dataset = Dataset(y=y, v=np.full((12, 1), 0.2), g=mixed)
    results = DerSimonianLaird(weight_scheme=weight_scheme).fit_dataset(dataset).summary()
    assert np.isfinite(np.ravel(results.get_re_stats()["tau^2"])).all()
    assert np.all(results.permutation_test(n_perm=64).perm_p["fe_p"] > 0)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize("n_perm", [100000, 2**8], ids=["more-than-exact", "exactly-exact"])
def test_permutation_flips_whole_groups_and_leaves_the_estimator_alone(weight_scheme, n_perm):
    """Dependent rows are exchangeable only as complete groups.

    Flipping them independently builds a null about half as wide as the truth,
    which showed up as ~25% rejection at a nominal 5%. Separately, refitting
    through the live estimator overwrote params_["dof"] with a
    permutation-shaped array, so fe_dof and to_df broke afterwards.

    ``n_perm == 2**m`` is the boundary: the test counts as exact there, but the
    sign patterns were only built on the strict inequality, so asking for
    exactly the exhaustive number raised UnboundLocalError.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 2)
    dataset = Dataset(y=rng.normal(size=16), v=np.full(16, 0.2), g=groups)
    results = DerSimonianLaird(weight_scheme=weight_scheme).fit_dataset(dataset).summary()

    before = np.ravel(results.fe_dof).copy()
    perm = results.permutation_test(n_perm=n_perm)

    # 8 groups, not 16 rows: 2**8 sign patterns exhaust the null.
    assert perm.n_perm == 2**8
    assert perm.exact
    assert np.allclose(np.ravel(results.fe_dof), before)
    perm.to_df()


def test_cluster_weighting_removes_the_replication_advantage():
    """A group contributing many estimates must not outvote a group with one."""
    n_singletons = 8
    groups = np.array([0] * 6 + list(range(1, n_singletons + 1)))
    n_estimates = groups.size
    y = np.zeros((n_estimates, 1))
    y[:6] = 1.0  # the big group is the only one with a non-zero effect
    v = np.ones((n_estimates, 1))
    dataset = Dataset(y=y, v=v, g=groups)

    individual = WeightedLeastSquares().fit_dataset(dataset).summary()
    clustered = WeightedLeastSquares(weight_scheme="rescale").fit_dataset(dataset).summary()

    # Six of fourteen estimates, but only one of nine groups.
    assert np.isclose(individual.get_fe_stats()["est"].ravel()[0], 6 / 14)
    assert np.isclose(clustered.get_fe_stats()["est"].ravel()[0], 1 / 9)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
@pytest.mark.parametrize(
    "estimator",
    [WeightedLeastSquares, DerSimonianLaird, Hedges, VarianceBasedLikelihoodEstimator],
    ids=["WLS", "DL", "HE", "ML"],
)
def test_variance_estimators_group_mode_matches_explicit_collapse(estimator):
    """Group mode must fit the algebraically equivalent one-row-per-group model."""
    y = np.array([[1.0], [3.0], [6.0], [10.0], [20.0], [22.0]])
    v = np.array([[1.0], [4.0], [2.0], [8.0], [5.0], [3.0]])
    X = np.ones((6, 1))
    groups = np.array([0, 0, 1, 1, 2, 2])
    rho = 0.4
    collapsed_y, collapsed_v, collapsed_X = collapse_groups(y, v, X, groups, rho=rho)

    expected = estimator().fit(y=collapsed_y, v=collapsed_v, X=collapsed_X, g=np.arange(3))
    observed = estimator(weight_scheme="collapse", rho=rho).fit(y=y, v=v, X=X, g=groups)

    assert observed.n_groups_ == 3
    for key in ("fe_params", "tau2", "inv_cov"):
        assert np.allclose(observed.params_[key], expected.params_[key])


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_sample_size_likelihood_group_mode_matches_explicit_collapse():
    """The likelihood receives one effective n per independent group.

    Effective, not raw. Rows in a group share subjects, so n must not be
    counted once per row -- but they are not perfect duplicates either, and
    pinning n at the group's raw value assumes they are. The effective size
    ``s*n / (1 + rho(s-1))`` interpolates between the two.
    """
    y = np.array([[1.0], [3.0], [6.0], [10.0], [20.0], [22.0]])
    n = np.array([[20.0], [20.0], [50.0], [50.0], [100.0], [100.0]])
    X = np.ones((6, 1))
    groups = np.array([0, 0, 1, 1, 2, 2])
    collapsed_y, collapsed_n, collapsed_X = collapse_groups_by_n(y, n, X, groups, rho=DEFAULT_RHO)

    expected = SampleSizeBasedLikelihoodEstimator().fit(
        y=collapsed_y, n=collapsed_n, X=collapsed_X, g=np.arange(3)
    )
    observed = SampleSizeBasedLikelihoodEstimator(weight_scheme="collapse").fit(
        y=y, n=n, X=X, g=groups
    )

    for key in ("fe_params", "sigma2", "tau2", "inv_cov"):
        assert np.allclose(observed.params_[key], expected.params_[key])


def test_collapse_mode_honours_rho_for_sample_sizes():
    """rho=1 is the only value for which a group's raw n is its effective n.

    Rows in a group share subjects, so n must not be counted once per row --
    but they are not perfect duplicates either. For s estimates from the same
    n subjects correlated at rho, Var(ybar) = (sigma^2 / n)(1 + rho(s-1))/s,
    i.e. an effective size s*n / (1 + rho(s-1)) running from n at rho=1 up to
    s*n at rho=0. Pinning it at n biases sigma^2 low by (1 + rho(s-1))/s --
    a factor of five for four uncorrelated estimates per group. Observations
    from one group share subjects but measure different outcomes, so rho is
    well below one and the bias is real.
    """
    rng = np.random.RandomState(0)
    n_groups, group_size = 10, 4
    groups = np.repeat(np.arange(n_groups), group_size)
    n_estimates = groups.size
    n = np.repeat(rng.randint(40, 160, n_groups).astype(float), group_size)[:, None]
    y = rng.randn(n_estimates, 1)
    X = np.ones((n_estimates, 1))

    effective = {}
    for rho in (0.0, 0.5, 1.0):
        _, collapsed_n, _, _ = _collapse_n_inputs(y, n, X, groups, "collapse", rho)
        effective[rho] = collapsed_n

    # Less assumed correlation means more independent information per group.
    assert np.all(effective[0.0] > effective[0.5])
    assert np.all(effective[0.5] > effective[1.0])

    # The endpoints are the two formulas being interpolated between, and the
    # closed form holds in between.
    _, raw_n, _ = collapse_groups_by_n(y, n, X, groups, rho=1.0)
    assert np.allclose(effective[1.0], raw_n)
    assert np.allclose(effective[0.0], group_size * raw_n)
    for rho in (0.0, 0.5, 1.0):
        assert np.allclose(effective[rho], group_size * raw_n / (1.0 + rho * (group_size - 1)))


def test_group_collapse_rejects_a_saturated_collapsed_design(variance_estimator):
    """Collapsing can saturate a design that was identified before it.

    Nine rows and three predictors is unremarkable, but three groups leave
    m == p, where the moment estimators divide by zero and report tau^2 = inf
    with zero standard errors.
    """
    groups = np.repeat([0, 1, 2], 3)
    X = np.c_[np.ones(9), np.repeat([0.0, 1.0, 2.0], 3), np.repeat([0.0, 0.0, 1.0], 3)]
    y = np.random.RandomState(0).randn(9, 1)

    with pytest.raises(ValueError, match="number of groups must exceed"):
        variance_estimator(weight_scheme="collapse").fit(
            y=y, v=np.full((9, 1), 0.1), X=X, g=groups
        )


def test_group_design_check_tolerates_floating_point_noise():
    """A predictor constant in intent may differ in its last bits."""
    groups = np.repeat([0, 1, 2], 3)
    y = np.random.RandomState(0).randn(9, 1)
    v = np.full((9, 1), 0.1)

    X = np.c_[np.ones(9), np.repeat([1.0, 2.0, 3.0], 3)]
    X[1, 1] += 1e-15
    Hedges(weight_scheme="collapse").fit(y=y, v=v, X=X, g=groups)  # must not raise

    # Genuine within-group variation is still rejected.
    X[1, 1] = 99.0
    with pytest.raises(ValueError, match="constant"):
        Hedges(weight_scheme="collapse").fit(y=y, v=v, X=X, g=groups)


def test_variance_components_are_insensitive_to_duplication(variance_estimator):
    """Duplicating a group's estimate must not shrink tau^2.

    Repeated estimates from one group agree with each other by construction. An
    estimator that counts them as independent sees less dispersion than the row
    count implies and shrinks tau^2 toward zero, which sharpens the weights and
    makes downstream inference anti-conservative.
    """
    rng = np.random.default_rng(4)
    n_estimates, n_datasets = 12, 8
    y = rng.standard_normal((n_estimates, n_datasets)) * 2.0
    v = np.full((n_estimates, n_datasets), 0.5)
    groups = np.arange(n_estimates)

    single = variance_estimator(weight_scheme="rescale")
    single.fit_dataset(Dataset(y=y, v=v, g=groups))

    # Group 0 now contributes four identical estimates instead of one.
    dupe_idx = np.r_[np.zeros(4, dtype=int), np.arange(1, n_estimates)]
    duped = variance_estimator(weight_scheme="rescale")
    duped.fit_dataset(Dataset(y=y[dupe_idx], v=v[dupe_idx], g=groups[dupe_idx]))

    assert np.allclose(np.mean(single.summary().tau2), np.mean(duped.summary().tau2), rtol=0.05)


def test_sample_size_identifiability_is_checked_on_the_fitted_values():
    """sigma^2 and tau^2 are identified by the n the likelihood actually sees.

    Under weight_scheme='rescale' those are effective sample sizes, which vary
    with group size even when every raw n is identical.
    """
    sizes = [1] * 6 + [20] * 6
    groups = np.concatenate([[j] * s for j, s in enumerate(sizes)])
    n_estimates = groups.size
    n = np.full((n_estimates, 1), 50.0)  # every raw n identical
    X = np.ones((n_estimates, 1))
    y = 0.5 + np.random.RandomState(2).randn(n_estimates, 1) * 0.4

    fitted = SampleSizeBasedLikelihoodEstimator(weight_scheme="rescale", rho=0.3).fit(
        y=y, n=n, X=X, g=groups
    )
    assert np.isfinite(fitted.params_["sigma2"]).all()

    # Genuinely unidentifiable input is still refused.
    with pytest.raises(ValueError, match="all-equal sample sizes"):
        SampleSizeBasedLikelihoodEstimator().fit(y=y, n=n, X=X)


def test_sample_size_identifiability_is_judged_per_dataset():
    """The spread that separates sigma^2 from tau^2 has to be inside one column.

    Columns that are each constant still differ from one another, so a spread taken
    over the whole array reports variation that no single likelihood can use. Each
    column is its own fit and has to be judged on its own.
    """
    y = np.array([[1.0, 2.0], [3.0, 1.0], [6.0, 4.0], [2.0, 5.0], [4.0, 3.0]])
    X = np.ones((5, 1))

    # Every column holds one constant sample size; only the columns differ.
    with pytest.raises(ValueError, match="2 of 2 parallel datasets"):
        SampleSizeBasedLikelihoodEstimator().fit(y=y, n=np.array([[20.0, 50.0]] * 5), X=X)

    # One constant column beside one that varies: the constant one still counts.
    mixed = np.array([[20.0, 20.0], [20.0, 45.0], [20.0, 80.0], [20.0, 120.0], [20.0, 200.0]])
    with pytest.raises(ValueError, match="1 of 2 parallel datasets"):
        SampleSizeBasedLikelihoodEstimator().fit(y=y, n=mixed, X=X)


def test_near_equal_sample_sizes_warn_per_dataset():
    """One barely-varying column warns even when the others vary plenty."""
    rng = np.random.RandomState(0)
    y = rng.randn(20, 2)
    n = np.column_stack([rng.randint(20, 200, size=20).astype(float), np.full(20, 100.0)])
    n[0, 1] = 101.0

    with pytest.warns(UserWarning, match="1 of 2 parallel datasets"):
        SampleSizeBasedLikelihoodEstimator().fit(y=y, n=n, X=np.ones((20, 1)))


def test_near_equal_sample_sizes_warn_rather_than_abort():
    """``raise Warning`` aborts the fit; this path should only warn."""
    n = np.full((20, 1), 100.0)
    n[0] = 101.0
    y = np.random.RandomState(0).randn(20, 1)

    with pytest.warns(UserWarning, match="too close"):
        SampleSizeBasedLikelihoodEstimator().fit(y=y, n=n, X=np.ones((20, 1)))
