"""Tests for estimators that use stan.

cmdstanpy and CmdStan are optional, so the tests that sample are marked ``stan``
and skip when either is missing. The tests that only exercise the translation
from PyMARE's inputs to Stan's data block are deliberately left unmarked: they
need neither, so they run in the ordinary unit job on every platform, which is
where the defects this file now pins would have been caught years earlier.
"""

import os.path as op
import warnings

import numpy as np
import pytest

from pymare import meta_regression
from pymare.estimators import StanMetaRegression, VarianceBasedLikelihoodEstimator
from pymare.estimators.estimators import _build_stan_data
from pymare.results import BayesianMetaRegressionResults
from pymare.tests.utils import (
    STAN_VALIDATION_CELLS,
    STAN_VALIDATION_THRESHOLDS,
    cmdstan_is_available,
    load_stan_validation,
)

requires_cmdstan = pytest.mark.skipif(
    not cmdstan_is_available(),
    reason="requires cmdstanpy and a CmdStan installation",
)


# -----------------------------------------------------------------------------
# Translation into Stan's data block. No CmdStan needed.
# -----------------------------------------------------------------------------


def test_stan_data_passes_standard_deviations():
    """Stan's normal() takes a scale, so v must be converted before it is passed."""
    v = np.array([0.04, 0.09, 0.16])
    data = _build_stan_data(np.array([1.0, 2.0, 3.0]), v, np.ones((3, 1)))

    np.testing.assert_allclose(data["sigma"], np.sqrt(v))


def test_stan_data_derives_the_tau_prior_scale_from_the_data():
    """The default prior scale is the larger of the two scales the data carry.

    Taking only the sampling standard deviation understates tau whenever the
    between-group spread is the larger of the two, which validation/stan
    measures as credible-interval coverage of 0.83 against a nominal 0.95.
    """
    v = np.array([0.04, 0.09, 0.16])
    spread_dominates = np.array([1.0, 2.0, 3.0])
    data = _build_stan_data(spread_dominates, v, np.ones((3, 1)))

    assert data["tau_prior_scale"] == pytest.approx(np.std(spread_dominates))
    assert data["tau_prior_scale"] > np.sqrt(np.mean(v))

    # ... and the sampling error is the floor when the estimates all coincide,
    # where the spread alone would be zero and so not a usable scale.
    identical = _build_stan_data(np.full(3, 2.0), v, np.ones((3, 1)))

    assert identical["tau_prior_scale"] == pytest.approx(np.sqrt(np.mean(v)))
    assert identical["tau_prior_scale"] > 0

    explicit = _build_stan_data(spread_dominates, v, np.ones((3, 1)), tau_prior_scale=7.0)
    assert explicit["tau_prior_scale"] == 7.0


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param(["a", "a", "b", "b", "c"], id="strings"),
        pytest.param([10, 10, 20, 20, 30], id="non-consecutive ints"),
        pytest.param(np.array([10, 10, 20, 20, 30]), id="ndarray"),
        pytest.param(np.array([[10], [10], [20], [20], [30]]), id="column vector"),
    ],
)
def test_stan_data_encodes_arbitrary_group_labels(groups):
    """Labels of any hashable type become the 1..K codes the Stan program declares.

    The column-vector case is what ``Dataset.g`` holds, and the ndarray cases
    used to raise outright: ``groups or default`` asks a whole array for its
    truth value.
    """
    y = np.arange(5.0)
    data = _build_stan_data(y, np.ones(5), np.ones((5, 1)), groups=groups)

    np.testing.assert_array_equal(data["id"], [1, 1, 2, 2, 3])
    assert data["K"] == 3
    assert data["id"].min() >= 1 and data["id"].max() <= data["K"]


def test_stan_data_treats_each_observation_as_its_own_group_by_default():
    """Without groups, K equals N and every observation gets a distinct code."""
    data = _build_stan_data(np.arange(5.0), np.ones(5), np.ones((5, 1)))

    assert data["K"] == data["N"] == 5
    np.testing.assert_array_equal(data["id"], [1, 2, 3, 4, 5])


def test_stan_data_design_matrix_has_one_row_per_observation():
    """X is per-observation, so its row count tracks N and not K."""
    X = np.column_stack([np.ones(6), np.arange(6.0)])
    data = _build_stan_data(np.arange(6.0), np.ones(6), X, groups=[1, 1, 2, 2, 3, 3])

    assert data["X"].shape == (data["N"], data["C"]) == (6, 2)
    assert data["K"] == 3
    assert data["K"] < data["N"]


def test_stan_data_promotes_a_one_dimensional_design_matrix():
    """A single predictor may be passed as a 1d array."""
    data = _build_stan_data(np.arange(4.0), np.ones(4), np.arange(4.0))

    assert data["X"].shape == (4, 1)
    assert data["C"] == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"v": np.ones(3)}, "one sampling variance per observation", id="short v"),
        pytest.param({"v": np.array([1.0, 0.0, 1.0, 1.0])}, "must all be positive", id="zero v"),
        pytest.param(
            {"v": np.array([1.0, -1.0, 1.0, 1.0])}, "must all be positive", id="negative v"
        ),
        pytest.param({"X": np.ones((3, 1))}, "one row per observation", id="short X"),
    ],
)
def test_stan_data_rejects_inputs_that_disagree_about_n(kwargs, match):
    """Shape and positivity are checked once, at the boundary."""
    call = {"y": np.arange(4.0), "v": np.ones(4), "X": np.ones((4, 1))}
    call.update(kwargs)

    with pytest.raises(ValueError, match=match):
        _build_stan_data(**call)


def test_stan_2d_input_failure(dataset_2d):
    """Run smoke test for StanMetaRegression on 2D data.

    No CmdStan needed: the shape is rejected before the model is compiled.
    """
    with pytest.raises(ValueError) as exc:
        StanMetaRegression().fit_dataset(dataset_2d)
    assert str(exc.value).startswith("The StanMetaRegression")


def test_fit_dataset_forwards_dataset_g(planted_hierarchical_dataset):
    """fit_dataset must route dataset.g into fit()'s groups argument.

    _dataset_attr_map was empty, so the argument fell back to its None default
    and every fit_dataset() call silently modelled each observation as its own
    group. Presetting .model with a stub keeps this test free of CmdStan: fit()
    only compiles when it finds no model.
    """
    dataset, _ = planted_hierarchical_dataset
    est = StanMetaRegression()
    est.model = _StubModel()

    est.fit_dataset(dataset)

    assert est.data["N"] == 90
    assert est.data["K"] == 30
    np.testing.assert_array_equal(est.data["id"], np.repeat(np.arange(1, 31), 3))


def test_fit_warns_about_divergent_transitions():
    """A divergent fit must warn through the warnings module, not only the log.

    CmdStanPy logs its own diagnostic message, but a log record obeys no
    warning filter and cannot be asserted on. Divergences mean the sampler
    could not reach part of the posterior, so a summary that looks ordinary may
    not be.
    """
    est = StanMetaRegression()
    est.model = _StubModel(divergences=np.array([3.0, 1.0]))

    with pytest.warns(UserWarning, match="4 divergent transition"):
        est.fit(np.arange(4.0), np.ones(4), np.ones((4, 1)))


def test_fit_is_quiet_when_there_are_no_divergences():
    """The converse: a clean fit must not cry wolf."""
    est = StanMetaRegression()
    est.model = _StubModel()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.fit(np.arange(4.0), np.ones(4), np.ones((4, 1)))


def test_compile_falls_back_when_the_package_directory_is_read_only(monkeypatch, tmp_path):
    """An unwritable site-packages must not make the estimator unusable.

    CmdStanPy compiles beside the .stan source, which is inside the installed
    package. That directory is read-only in plenty of ordinary installations,
    and the resulting error would otherwise surface from the middle of fit().
    """
    cmdstanpy = pytest.importorskip("cmdstanpy")
    monkeypatch.setenv("HOME", str(tmp_path))
    attempts = []

    def fake_model(stan_file=None, exe_file=None, force_compile=False):
        attempts.append(exe_file)
        if exe_file is None:
            raise PermissionError("read-only file system")
        return "compiled"

    # Stub the CmdStan lookup as well as the compiler, so this exercises the
    # fallback itself rather than requiring a real CmdStan to get that far.
    monkeypatch.setattr(cmdstanpy, "cmdstan_path", lambda: str(tmp_path))
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", fake_model)

    est = StanMetaRegression()
    with pytest.warns(UserWarning, match="not writable"):
        est.compile()

    assert est.model == "compiled"
    assert attempts[0] is None
    assert attempts[1] == op.join(str(tmp_path), ".pymare", "stan", "meta_regression")


class _StubModel:
    """Stand in for a compiled CmdStanModel so fit() can run without CmdStan."""

    def __init__(self, divergences=None):
        self.divergences = np.zeros(2) if divergences is None else divergences

    def sample(self, data=None, **kwargs):
        """Return an object exposing only what fit() reads off a CmdStanMCMC."""
        return _StubFit(self.divergences)


class _StubFit:
    """Stand in for a CmdStanMCMC, exposing only its per-chain divergence counts."""

    def __init__(self, divergences):
        self.divergences = divergences


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwarg", "replacement"),
    [("num_samples", "iter_sampling"), ("num_warmup", "iter_warmup"), ("num_chains", "chains")],
)
def test_pystan_sampling_kwarg_names_are_rejected(kwarg, replacement):
    """Reject PyStan names, which CmdStanPy would report as an unhelpful TypeError."""
    with pytest.raises(TypeError, match=replacement):
        StanMetaRegression(**{kwarg: 100})


@pytest.mark.parametrize("bad", [0.0, -1.0, "wide"])
def test_tau_prior_scale_is_validated_at_construction(bad):
    """A prior scale must be a positive number, and None means 'derive it'."""
    with pytest.raises(ValueError):
        StanMetaRegression(tau_prior_scale=bad)

    assert StanMetaRegression(tau_prior_scale=None).tau_prior_scale is None


def test_summary_before_fit_raises():
    """summary() must not report on an estimator that has not been fitted."""
    with pytest.raises(ValueError, match="hasn't been fitted yet"):
        StanMetaRegression().summary()


# -----------------------------------------------------------------------------
# Results container. No CmdStan needed; ArviZ is enough.
# -----------------------------------------------------------------------------


def test_plot_rejects_unknown_kind():
    """An unknown plot kind raises the documented ValueError.

    The guard used to be unreachable: a two-argument getattr raised
    AttributeError before the None check below it could run.
    """
    pytest.importorskip("arviz")
    results = BayesianMetaRegressionResults.__new__(BayesianMetaRegressionResults)
    results.data = None

    with pytest.raises(ValueError, match="no plotting function"):
        results.plot(kind="not_a_real_plot")


def test_plot_defaults_to_the_summary_variables(monkeypatch):
    """plot() must not default to plotting one panel per group.

    A fitted model carries one theta per group, so plotting everything is
    illegible and, under ArviZ 1.x, exceeds rcParams["plot.max_subplots"] and
    raises. The default therefore matches summary()'s selection.
    """
    az = pytest.importorskip("arviz")
    captured = {}

    def fake_plot_trace(data, var_names=None, **kwargs):
        captured["var_names"] = var_names
        return "figure"

    monkeypatch.setattr(az, "plot_trace", fake_plot_trace)
    results = BayesianMetaRegressionResults.__new__(BayesianMetaRegressionResults)
    results.data = None

    assert results.plot(kind="trace") == "figure"
    assert captured["var_names"] == ["beta", "tau2"]

    results.plot(kind="trace", include_theta=True)
    assert captured["var_names"] == ["beta", "tau2", "theta"]

    # An explicit selection still wins.
    results.plot(kind="trace", var_names=["tau2"])
    assert captured["var_names"] == ["tau2"]


def test_plot_omits_var_names_for_plotters_that_reject_it(monkeypatch):
    """Not every ArviZ plot takes var_names; plot_energy takes only the data."""
    az = pytest.importorskip("arviz")
    captured = {}

    def fake_plot_energy(data, **kwargs):
        captured["kwargs"] = kwargs
        return "figure"

    # plot_energy genuinely has no var_names parameter, and neither does this
    # stub, so the selection above must not be forced onto it.

    monkeypatch.setattr(az, "plot_energy", fake_plot_energy)
    results = BayesianMetaRegressionResults.__new__(BayesianMetaRegressionResults)
    results.data = None

    assert results.plot(kind="energy") == "figure"
    assert "var_names" not in captured["kwargs"]


@pytest.mark.parametrize("bad_ci", [0, 100, -5, 101])
def test_results_reject_an_impossible_credible_interval(bad_ci):
    """The ci argument is a percentage, so it must lie strictly inside (0, 100)."""
    pytest.importorskip("arviz")
    with pytest.raises(ValueError, match="must lie in"):
        BayesianMetaRegressionResults(None, None, ci=bad_ci)


@pytest.mark.parametrize("ci", [50.0, 95.0])
def test_summary_requests_the_configured_credible_interval(ci):
    """The ci argument must reach ArviZ. It was previously stored and never used."""
    az = pytest.importorskip("arviz")
    from pymare.results import _arviz_credible_interval_kwargs

    kwargs = _arviz_credible_interval_kwargs(ci)
    probability = kwargs.get("ci_prob", kwargs.get("hdi_prob"))

    assert probability == pytest.approx(ci / 100.0)
    if int(az.__version__.split(".")[0]) >= 1:
        # ArviZ 1.x defaults to an equal-tailed interval and to stringifying the
        # summary for display; neither is what this container promises.
        assert kwargs["ci_kind"] == "hdi"
        assert kwargs["round_to"] == "none"


# -----------------------------------------------------------------------------
# The recorded simulation results. No CmdStan needed; the numbers are pinned.
# -----------------------------------------------------------------------------


def test_recorded_validation_covers_every_design_cell():
    """The pinned results must describe the grid the harness actually runs.

    Without this, adding a cell to validation/stan/simulate.py and forgetting to
    regenerate would leave the new cell permanently unmeasured while the file
    still looked current.
    """
    recorded = load_stan_validation()
    names = [cell["name"] for cell in recorded["cells"]]

    assert tuple(names) == STAN_VALIDATION_CELLS
    assert len(names) == len(set(names)), "duplicate cells in the recorded results"
    assert recorded["replications"] >= 100


def test_recorded_validation_meets_its_thresholds():
    """Every design cell must clear the coverage floor and the bias ceiling.

    This is what makes the recorded file load-bearing rather than decorative. It
    checks the numbers already measured rather than re-measuring, so it costs
    nothing and runs everywhere; the scheduled Stan validation workflow is what
    re-measures and enforces the same thresholds against a fresh run.

    The thresholds are not decoration either: the first prior scale tried here
    produced coverage of 0.810 in the ``sigma x0.1`` cell, which this floor
    rejects.
    """
    recorded = load_stan_validation()
    floor = STAN_VALIDATION_THRESHOLDS["min_coverage"]
    ceiling = STAN_VALIDATION_THRESHOLDS["max_beta_bias"]

    undercovered = [
        (cell["name"], cell["beta_coverage"])
        for cell in recorded["cells"]
        if cell["beta_coverage"] < floor
    ]
    assert not undercovered, f"cells below {floor:.2f} coverage: {undercovered}"

    biased = [
        (cell["name"], cell["beta_bias"])
        for cell in recorded["cells"]
        if abs(cell["beta_bias"]) > ceiling
    ]
    assert not biased, f"cells with |beta bias| above {ceiling}: {biased}"


# -----------------------------------------------------------------------------
# Sampling. Needs CmdStan.
# -----------------------------------------------------------------------------


@pytest.mark.stan
@requires_cmdstan
def test_recovers_planted_parameters(planted_hierarchical_dataset):
    """The fitted posterior must recover the parameters the data were built from.

    This is the test that distinguishes tau from tau2 and variances from
    standard deviations: the planted tau2 of 0.25 and tau of 0.5 are far enough
    apart that reporting either under the other's name lands outside the
    interval asserted here.
    """
    dataset, truth = planted_hierarchical_dataset
    est = StanMetaRegression(iter_sampling=2000, chains=4, seed=8675309, show_progress=False)
    results = est.fit_dataset(dataset).summary()
    summary = results.summary()

    for i, expected in enumerate(truth["beta"]):
        mean = float(summary.loc[f"beta[{i}]", "mean"])
        sd = float(summary.loc[f"beta[{i}]", "sd"])
        assert abs(mean - expected) < 3 * sd, f"beta[{i}] = {mean}, expected ~{expected}"

    tau2 = float(summary.loc["tau2", "mean"])
    assert 0.15 < tau2 < 0.45, f"tau2 = {tau2}, expected ~{truth['tau2']}"
    assert tau2 < truth["tau"], "tau2 looks like tau, not tau squared"


@pytest.mark.stan
@requires_cmdstan
def test_matches_maximum_likelihood_without_groups(dataset):
    """Ungrouped, the model marginalizes to the one ML already maximizes.

    With every observation in its own group the hierarchical model collapses to
    ``y ~ N(X beta, sqrt(v + tau2))``, which is exactly
    VarianceBasedLikelihoodEstimator's ML likelihood. Under a diffuse prior on
    tau the posterior means must therefore agree with the ML estimates up to
    Monte Carlo error, which pins the Stan program against an implementation
    that shares none of its code.
    """
    est = StanMetaRegression(
        tau_prior_scale=100.0, iter_sampling=4000, chains=4, seed=20250818, show_progress=False
    )
    summary = est.fit_dataset(dataset).summary().summary()

    ml = VarianceBasedLikelihoodEstimator(method="ML").fit_dataset(dataset)
    ml_beta = np.asarray(ml.params_["fe_params"]).ravel()

    for i, expected in enumerate(ml_beta):
        posterior_mean = float(summary.loc[f"beta[{i}]", "mean"])
        assert posterior_mean == pytest.approx(expected, abs=0.15 * max(abs(expected), 1.0))


@pytest.mark.stan
@requires_cmdstan
def test_meta_regression_dispatches_to_stan(planted_hierarchical_dataset):
    """The functional entry point must reach this estimator and its results class.

    pymare.meta_regression(method="stan") is the documented one-call API and has
    its own dispatch table in core.py, which no test previously exercised.
    """
    dataset, _ = planted_hierarchical_dataset

    results = meta_regression(
        y=dataset.y,
        v=dataset.v,
        X=dataset.X[:, 1:],
        g=dataset.g,
        method="stan",
        iter_sampling=500,
        chains=2,
        seed=99,
        show_progress=False,
    )

    assert isinstance(results, BayesianMetaRegressionResults)
    assert list(results.summary().index) == ["beta[0]", "beta[1]", "tau2"]


@pytest.mark.stan
@requires_cmdstan
def test_the_compiled_model_is_reused_across_fits(planted_hierarchical_dataset):
    """compile() once, fit many.

    The class docstring has always promised this, but it could not be done: the
    old compile() read self.data, which only fit() assigned, so calling it
    directly raised AttributeError and every fit recompiled. Under CmdStanPy the
    executable does not depend on the data, so the promise is now keepable.
    """
    dataset, _ = planted_hierarchical_dataset
    est = StanMetaRegression(iter_sampling=200, chains=1, seed=5, show_progress=False)

    est.compile()
    compiled = est.model
    assert compiled is not None

    est.fit_dataset(dataset)
    assert est.model is compiled

    est.fit_dataset(dataset)
    assert est.model is compiled


@pytest.mark.stan
@requires_cmdstan
def test_summary_and_plot_round_trip(planted_hierarchical_dataset):
    """The results container reports the expected rows and plots without error."""
    dataset, _ = planted_hierarchical_dataset
    est = StanMetaRegression(iter_sampling=500, chains=2, seed=1234, show_progress=False)
    results = est.fit_dataset(dataset).summary()

    assert isinstance(results, BayesianMetaRegressionResults)

    without_theta = results.summary()
    assert list(without_theta.index) == ["beta[0]", "beta[1]", "tau2"]

    with_theta = results.summary(include_theta=True)
    assert len(with_theta) == len(without_theta) + 30

    assert results.plot(kind="trace") is not None
