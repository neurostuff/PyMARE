"""Tests for estimators that use stan.

cmdstanpy and CmdStan are optional, so the tests that sample are marked ``stan``
and skip when either is missing. The tests that only exercise the translation
from PyMARE's inputs to Stan's data block are deliberately left unmarked: they
need neither, so they run in the ordinary unit job on every platform, which is
where the defects this file now pins would have been caught years earlier.
"""

import ntpath
import os.path as op
import posixpath
import sys
import warnings

import numpy as np
import pytest

from pymare import meta_regression
from pymare.estimators import StanMetaRegression, VarianceBasedLikelihoodEstimator
from pymare.estimators.estimators import (
    STAN_MODEL_PATH,
    _build_stan_data,
    _import_cmdstanpy,
)
from pymare.results import (
    BayesianMetaRegressionResults,
    _accepts_var_names,
    _arviz_credible_interval_kwargs,
)
from pymare.tests import conftest
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
# Detecting a missing dependency. No CmdStan needed -- the point is its absence.
# -----------------------------------------------------------------------------


def test_import_cmdstanpy_reports_a_missing_package(monkeypatch):
    """The message must name the install command, not just the module."""
    monkeypatch.setitem(sys.modules, "cmdstanpy", None)

    with pytest.raises(ImportError, match=r"pip install pymare\[stan\]"):
        _import_cmdstanpy()


def test_import_cmdstanpy_reports_a_missing_cmdstan(monkeypatch):
    """Installed cmdstanpy with no CmdStan is a different problem with a different fix.

    Reporting them separately matters because `pip install` cannot solve the
    second one: CmdStan is a C++ build, not a Python package.
    """
    cmdstanpy = pytest.importorskip("cmdstanpy")

    def no_cmdstan():
        raise ValueError("No CmdStan directory")

    monkeypatch.setattr(cmdstanpy, "cmdstan_path", no_cmdstan)

    with pytest.raises(ImportError, match="install_cmdstan") as exc:
        _import_cmdstanpy()
    assert "No CmdStan directory" in str(exc.value), "the underlying reason should survive"


def test_cmdstan_is_available_is_false_without_the_package(monkeypatch):
    """The gate must answer False, not raise, when cmdstanpy is absent."""
    monkeypatch.setitem(sys.modules, "cmdstanpy", None)

    assert cmdstan_is_available() is False


def test_cmdstan_is_available_is_false_without_an_installation(monkeypatch):
    """Installing cmdstanpy from PyPI does not install CmdStan, so both are checked.

    Checking only the import would reproduce the original defect in a new
    costume: a gate that reports ready for an environment that can only fail.
    """
    cmdstanpy = pytest.importorskip("cmdstanpy")

    def no_cmdstan():
        raise ValueError("No CmdStan directory")

    monkeypatch.setattr(cmdstanpy, "cmdstan_path", no_cmdstan)

    assert cmdstan_is_available() is False


def test_collection_hook_fails_only_when_cmdstan_is_declared_present(monkeypatch):
    """The skip-versus-fail asymmetry that keeps a green CI log honest."""
    monkeypatch.setattr(conftest, "cmdstan_is_available", lambda: False)

    # Unset: a contributor without CmdStan sees skips, not failures.
    monkeypatch.delenv("PYMARE_REQUIRE_CMDSTAN", raising=False)
    assert conftest.pytest_collection_modifyitems(None, []) is None

    # Set: the environment claims it can run them, so their absence is an error.
    monkeypatch.setenv("PYMARE_REQUIRE_CMDSTAN", "1")
    with pytest.raises(pytest.UsageError, match="install_cmdstan"):
        conftest.pytest_collection_modifyitems(None, [])

    # Set, and genuinely available: nothing to complain about.
    monkeypatch.setattr(conftest, "cmdstan_is_available", lambda: True)
    assert conftest.pytest_collection_modifyitems(None, []) is None


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

    Without the mapping the argument falls back to None and every observation is
    silently modelled as its own group. Presetting .model with a stub keeps this
    free of CmdStan, since fit() only compiles when it finds no model.
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


def test_fake_home_redirects_expanduser_on_every_platform(fake_home):
    """The home-directory redirect must hold under Windows path rules too.

    ntpath is importable everywhere, so this catches on any platform the mistake
    that only shows up on a Windows runner: setting HOME alone leaves
    ntpath.expanduser pointing at the real user profile.
    """
    assert posixpath.expanduser("~") == str(fake_home)
    assert ntpath.expanduser("~") == str(fake_home)
    assert op.expanduser("~") == str(fake_home)


def test_compile_falls_back_when_the_package_directory_is_read_only(monkeypatch, fake_home):
    """An unwritable site-packages must not make the estimator unusable.

    CmdStanPy compiles beside the .stan source, which lives inside the installed
    package, and that directory is read-only in plenty of installations. Note the
    exception: CmdStanPy reports every failed ``make`` as ValueError, whatever
    went wrong, so a handler for PermissionError would never fire.
    """
    cmdstanpy = pytest.importorskip("cmdstanpy")
    compiled_from = []

    def fake_model(stan_file=None, exe_file=None, force_compile=False):
        compiled_from.append(stan_file)
        if stan_file == STAN_MODEL_PATH:
            raise ValueError(f"Failed to compile Stan model '{stan_file}'.")
        return "compiled"

    monkeypatch.setattr(cmdstanpy, "cmdstan_path", lambda: str(fake_home))
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", fake_model)

    est = StanMetaRegression()
    with pytest.warns(UserWarning, match="not writable"):
        est.compile()

    assert est.model == "compiled"
    # The second attempt compiles a *copy*, not the packaged file: exe_file
    # names an executable to reuse rather than a destination to build into.
    assert compiled_from[0] == STAN_MODEL_PATH
    assert compiled_from[1] == op.join(str(fake_home), ".pymare", "stan", "meta_regression.stan")
    assert op.exists(compiled_from[1]), "the fallback must copy the model somewhere writable"
    assert op.getmtime(compiled_from[1]) == op.getmtime(
        STAN_MODEL_PATH
    ), "copy2 preserves the mtime so the cached build is not invalidated every run"


def test_compile_reports_the_original_error_when_the_fallback_also_fails(monkeypatch, fake_home):
    """A broken model must not be reported as a permissions problem.

    Both compiles fail for a model that does not parse, and it is the first
    error that names the real cause.
    """
    cmdstanpy = pytest.importorskip("cmdstanpy")

    def always_fails(stan_file=None, exe_file=None, force_compile=False):
        raise ValueError(f"Syntax error in '{stan_file}'")

    monkeypatch.setattr(cmdstanpy, "cmdstan_path", lambda: str(fake_home))
    monkeypatch.setattr(cmdstanpy, "CmdStanModel", always_fails)

    est = StanMetaRegression()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no misleading "not writable" warning
        with pytest.raises(ValueError, match="Syntax error") as exc:
            est.compile()
    assert STAN_MODEL_PATH in str(exc.value), "the packaged path is the one that failed first"


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


@pytest.mark.parametrize("major", [0, 1])
def test_credible_interval_kwargs_track_the_installed_arviz(monkeypatch, major):
    """Both ArviZ generations must be asked for the same interval.

    1.x renamed hdi_prob to ci_prob, defaults to an equal-tailed interval rather
    than a highest-density one, and formats summaries as strings unless told not
    to. Only one branch is reachable on any given install, so the other is
    exercised by pinning the reported version.
    """
    az = pytest.importorskip("arviz")
    monkeypatch.setattr(az, "__version__", f"{major}.3.0")

    kwargs = _arviz_credible_interval_kwargs(95.0)

    if major >= 1:
        assert kwargs == {"ci_prob": 0.95, "ci_kind": "hdi", "round_to": "none"}
    else:
        assert kwargs == {"hdi_prob": 0.95}


def test_accepts_var_names_handles_an_unreadable_signature():
    """Anything whose signature cannot be read must be treated as not taking var_names.

    inspect.signature raises rather than answering for some objects, and plot()
    calls this before deciding what to pass, so an exception here would surface
    as a broken plot rather than as a missing default.
    """
    pytest.importorskip("arviz")

    assert _accepts_var_names(lambda data, var_names=None: None) is True
    # Reads fine, simply has no such parameter.
    assert _accepts_var_names(lambda data: None) is False
    # Cannot be read at all: inspect raises TypeError for a non-callable.
    assert _accepts_var_names(object()) is False


def test_results_accept_a_converted_object_without_cmdstanpy(monkeypatch):
    """The container must work when cmdstanpy is absent but the data is already converted.

    A caller who has their own InferenceData should not need the sampler
    installed to summarize it, so the import is lazy and its failure is not one.
    """
    pytest.importorskip("arviz")
    monkeypatch.setitem(sys.modules, "cmdstanpy", None)

    results = BayesianMetaRegressionResults("already-converted", None, ci=90.0)

    assert results.data == "already-converted"
    assert results.ci == 90.0


@pytest.mark.parametrize("ci", [50.0, 95.0])
def test_summary_requests_the_configured_credible_interval(ci):
    """The ci argument must reach ArviZ. It was previously stored and never used."""
    az = pytest.importorskip("arviz")
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
    reads the numbers already measured rather than re-measuring, so it costs
    nothing and runs everywhere; the scheduled Stan validation workflow is what
    re-measures and applies the same thresholds to a fresh run.
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


def test_recorded_validation_summarizes_the_worst_coefficient():
    """The reported figures must be the worst coefficient, not an average of them.

    Averaging across coefficients lets a well-estimated intercept mask a badly
    estimated moderator, which is exactly the failure the unbalanced-covariate
    cells exist to detect. The thresholds are therefore applied to the minimum
    coverage and the largest absolute bias across coefficients, and this pins
    that so the summary cannot quietly become a mean.
    """
    recorded = load_stan_validation()

    for cell in recorded["cells"]:
        per_coverage = cell["beta_coverage_per_coefficient"]
        per_bias = cell["beta_bias_per_coefficient"]

        assert len(per_coverage) == len(per_bias) == len(cell["true_beta"])
        assert cell["beta_coverage"] == pytest.approx(min(per_coverage))
        assert cell["beta_bias"] == pytest.approx(max(per_bias, key=abs))


def test_recorded_validation_used_a_fixed_truth():
    """Bias is only meaningful if the coefficients being recovered are held fixed.

    Under a redrawn symmetric truth the signed errors average to zero for any
    estimator at all, including one that always returns zero, so the bias
    ceiling would certify nothing.
    """
    recorded = load_stan_validation()
    truths = {tuple(cell["true_beta"]) for cell in recorded["cells"]}

    # Every cell draws its coefficients from the same fixed vector, truncated to
    # however many predictors that cell uses.
    longest = max(truths, key=len)
    for truth in truths:
        assert truth == longest[: len(truth)]
        assert all(value != 0 for value in truth), "a zero coefficient cannot show bias"


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

    The compiled executable does not depend on the data, so fitting must reuse it
    rather than rebuilding per call.
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


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("v", np.array([1.0, np.nan, 1.0, 1.0])),
        ("v", np.array([1.0, np.inf, 1.0, 1.0])),
        ("y", np.array([1.0, np.nan, 1.0, 1.0])),
        ("X", np.array([[1.0], [np.nan], [1.0], [1.0]])),
    ],
)
def test_stan_data_rejects_non_finite_inputs(field, bad):
    """Reject NaN here, rather than leaving CmdStan to refuse it while reading data.

    NaN fails every comparison, so a positivity check alone lets it through:
    ``np.nan <= 0`` is False. It then flows into sqrt() and into the prior
    scale, and only surfaces when CmdStan refuses to load the data -- an error
    that names a Stan variable rather than the input that caused it.
    """
    call = {"y": np.arange(4.0), "v": np.ones(4), "X": np.ones((4, 1))}
    call[field] = bad

    with pytest.raises(ValueError, match="must all be finite"):
        _build_stan_data(**call)


def test_stan_data_still_rejects_non_positive_variances():
    """The finiteness check must not have displaced the positivity one."""
    with pytest.raises(ValueError, match="must all be positive"):
        _build_stan_data(np.arange(4.0), np.array([1.0, 0.0, 1.0, 1.0]), np.ones((4, 1)))


def test_stan_data_rejects_composite_group_labels():
    """The documented contract is scalar labels, and this is why.

    Numpy reads a sequence of tuples as a 2-dimensional array, so a tuple label
    is not a label at all by the time encode_groups sees it. The docstring says
    scalar rather than hashable for this reason.
    """
    with pytest.raises(ValueError, match="one-dimensional"):
        _build_stan_data(
            np.arange(3.0),
            np.ones(3),
            np.ones((3, 1)),
            groups=[("a", 1), ("a", 1), ("b", 2)],
        )
