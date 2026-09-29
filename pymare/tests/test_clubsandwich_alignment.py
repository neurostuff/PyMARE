"""Alignment between PyMARE and the R package clubSandwich on CR2 and its dof.

:func:`~pymare.stats.cluster_robust_cov` takes the ``CRn`` names from
``clubSandwich`` :footcite:p:`pustejovsky2018small`, and
:func:`~pymare.stats.satterthwaite_dof` implements the degrees of freedom that
package pairs with ``CR2``. This module checks that against the package itself.

It is not a duplicate of :mod:`pymare.tests.test_robumeta_alignment`, which
covers a different question about the same estimator. ``robumeta`` is the
reference for the correlated-effects *working model* -- how weight is spread
across a study's rows, which PyMARE spells ``weight_scheme="rescale"`` -- and it
assumes the sampling variances are constant within a study. ``clubSandwich`` is
the reference for the CR2 residual adjustment itself, under whatever weights,
which is what PyMARE does with ``weight_scheme="individual"`` and group labels.

**What agrees.** When the sampling variances are constant within each cluster,
PyMARE's coefficients, CR2 covariance, Satterthwaite degrees of freedom and
p-values are ``clubSandwich``'s to machine precision, over every model and both
closed-form estimators.

**What does not.** When they vary within a cluster, the standard errors diverge
by up to 1e-2 relative and the degrees of freedom by 4e-3. The cause is exact and
:func:`test_cr2_divergence_is_the_whitening_metric` pins it: both implementations
build the same Bell-McCaffrey matrix :footcite:p:`bell2002bias`
``B_j = W_j^-1 - X_j (X'WX)^-1 X_j'`` and invert its square root, but they do it
in different metrics. ``clubSandwich`` forms ``A_j = Psi_j^(1/2) (Psi_j^(1/2) B_j
Psi_j^(1/2))^(-1/2) Psi_j^(1/2)`` with ``Psi = W^-1`` the assumed target;
PyMARE's ``_cr2_scores`` forms ``A_j = W_j^(1/2) (W_j^(1/2) B_j W_j^(1/2))^(-1/2)
W_j^(1/2)``. A matrix square root does not commute with an asymmetric
congruence, so the two coincide if and only if ``W_j`` is a multiple of the
identity -- that is, when the weights are constant within the cluster. PyMARE's
form is the one :footcite:t:`fisher2015robumeta` give as ``A_j^C``, which its
docstring says, and the correlated-effects model it belongs to has constant
within-study weights by construction. It is not ``clubSandwich``'s ``CR2`` once
they vary, which is worth knowing given where the name came from.

``validation/clubsandwich/README.md`` records the measurements.

References
----------
.. footbibliography::

"""

import json
import os.path as op

import numpy as np
import pytest

from pymare import Dataset
from pymare.estimators import DerSimonianLaird, WeightedLeastSquares
from pymare.tests.utils import get_test_data_path

# Both warnings are expected over this grid and say nothing about alignment.
# "Cluster-robust" fires because ten studies is few for a sandwich, which is the
# point of comparing the small-sample correction at all; the Satterthwaite one
# fires on the three-predictor model, where `between` is constant within a study
# and so is carried by very few clusters. clubSandwich reports the same low
# degrees of freedom, which is what these tests check.
pytestmark = [
    pytest.mark.clubsandwich,
    pytest.mark.filterwarnings("ignore:Cluster-robust"),
    pytest.mark.filterwarnings("ignore:Satterthwaite degrees of freedom below"),
]

with open(op.join(get_test_data_path(), "clubsandwich_reference.json")) as _fobj:
    REFERENCE = json.load(_fobj)

CASES = REFERENCE["cases"]

#: Tolerance where the two implementations compute the same thing. Worst
#: observed 7.1e-15, on a p-value under the three-predictor model; the
#: coefficients and the covariance agree to a few multiples of machine epsilon.
RTOL = 1e-12

#: Absolute floor, for the covariance's off-diagonal entries which pass through
#: zero.
ATOL = 1e-14

#: Moderator columns each model adds beside the intercept, matching ``mods`` in
#: ``validation/clubsandwich/run_clubsandwich.R``.
MODELS = {"intercept": [], "within": ["within"], "both": ["within", "between"]}

#: The estimators compared, in metafor's names. Both reach tau^2 in closed form,
#: so nothing but the covariance sits between PyMARE and clubSandwich.
#: ``weight_scheme="individual"`` throughout: it is the scheme that uses
#: ``1 / (v + tau^2)`` row by row, which is the weight matrix clubSandwich takes
#: from the ``rma.uni`` fit. The other two schemes are robumeta's question.
ESTIMATORS = {
    "FE": lambda: WeightedLeastSquares(tau2=0.0, weight_scheme="individual"),
    "DL": lambda: DerSimonianLaird(weight_scheme="individual"),
}

#: The variance column whose values are constant within each study, and so the
#: condition under which the two CR2 adjustments coincide. Named rather than
#: inferred so that the split between the two tests below is explicit.
CONSTANT_WITHIN_CLUSTER = "var_constant_within_study"

AGREEING_CASES = [case for case in CASES if case["variances"] == CONSTANT_WITHIN_CLUSTER]

DIVERGING_CASES = [case for case in CASES if case["variances"] != CONSTANT_WITHIN_CLUSTER]


def case_id(case):
    """Name a case by the three knobs that distinguish it."""
    columns = "shared" if case["variances"] == CONSTANT_WITHIN_CLUSTER else "varying"
    return f"{columns}-v-{case['model']}-{case['method']}"


def build_dataset(frame, case):
    """Assemble the model PyMARE should fit, with the study labels attached."""
    columns = MODELS[case["model"]]
    return Dataset(
        y=frame["effect"].to_numpy(),
        v=frame[case["variances"]].to_numpy(),
        X=frame[columns].to_numpy() if columns else None,
        add_intercept=True,
        g=frame["study"].to_numpy(),
    )


def fit(frame, case):
    """Fit one case and return its results object."""
    return ESTIMATORS[case["method"]]().fit_dataset(build_dataset(frame, case)).summary()


def inverse_sqrt(matrix):
    """Return the symmetric inverse square root of a positive semidefinite matrix.

    Restricted to the range, as ``clubSandwich``'s ``matrix_power(g, -1/2)`` is:
    a cluster whose rows are fitted away exactly leaves a singular ``B_j``, and
    the pseudo-inverse drops that direction rather than dividing by zero.
    """
    values, vectors = np.linalg.eigh(matrix)
    keep = values > values.max() * 1e-12
    return (vectors[:, keep] * values[keep] ** -0.5) @ vectors[:, keep].T


def cr2_standard_errors(y, v, X, groups, tau2, metric):
    """Compute CR2 standard errors under one of the two whitening metrics.

    Parameters
    ----------
    y, v, X, groups : :obj:`numpy.ndarray`
        One case's data, with the intercept already in ``X``.
    tau2 : :obj:`float`
        The variance component the weights use.
    metric : {"clubSandwich", "pymare"}
        Which congruence to apply to the Bell-McCaffrey matrix before taking its
        inverse square root: the target ``Psi_j^(1/2) = W_j^(-1/2)``, or the
        weights ``W_j^(1/2)``.

    Returns
    -------
    :obj:`numpy.ndarray`
        The square roots of the sandwich's diagonal.

    Notes
    -----
    Written out from the definition rather than called from
    :mod:`pymare.stats`, which is the point: the two forms differ only in the
    line selected by ``metric``, so a test that reproduces each implementation's
    output from this one function has located the difference between them and
    not merely measured it.
    """
    weights = 1.0 / (v + tau2)
    W = np.diag(weights)
    bread = np.linalg.inv(X.T @ W @ X)
    resid = y - X @ (bread @ X.T @ W @ y)

    meat = np.zeros((X.shape[1], X.shape[1]))
    for label in np.unique(groups):
        rows = np.flatnonzero(groups == label)
        X_j = X[rows]
        root = np.diag(np.sqrt(weights[rows]))
        inverse_root = np.diag(1.0 / np.sqrt(weights[rows]))
        bell_mccaffrey = np.diag(1.0 / weights[rows]) - X_j @ bread @ X_j.T
        if metric == "clubSandwich":
            adjustment = (
                inverse_root
                @ inverse_sqrt(inverse_root @ bell_mccaffrey @ inverse_root)
                @ inverse_root
            )
            score = X_j.T @ np.diag(weights[rows]) @ adjustment @ resid[rows]
        else:
            adjustment = inverse_sqrt(root @ bell_mccaffrey @ root)
            score = X_j.T @ root @ adjustment @ root @ resid[rows]
        meat += np.outer(score, score)

    return np.sqrt(np.diag(bread @ meat @ bread))


def design_arrays(frame, case):
    """Return ``(y, v, X, groups)`` for one case, with the intercept in ``X``."""
    y = frame["effect"].to_numpy()
    columns = MODELS[case["model"]]
    X = np.column_stack([np.ones_like(y)] + [frame[name].to_numpy() for name in columns])
    return y, frame[case["variances"]].to_numpy(), X, frame["study"].to_numpy()


@pytest.mark.parametrize("case", AGREEING_CASES, ids=[case_id(case) for case in AGREEING_CASES])
def test_cr2_matches_clubsandwich_with_constant_within_cluster_weights(case, clubsandwich_dataset):
    """With constant within-cluster weights, everything reported must match.

    The whole inference path, not only the standard errors: the coefficients, the
    full CR2 covariance including its off-diagonal entries, the Satterthwaite
    degrees of freedom, and the p-values that follow from the two together. A
    failure here is a failure of PyMARE's cluster-robust estimator, since there
    is no approximation on either side -- tau^2 is closed form for both
    estimators compared, so the only thing between the two implementations is
    the sandwich.
    """
    results = fit(clubsandwich_dataset, case)
    stats = results.get_fe_stats()
    n_preds = 1 + len(MODELS[case["model"]])

    assert np.allclose(np.ravel(results.tau2), case["tau2"], rtol=RTOL, atol=ATOL)
    assert np.allclose(np.ravel(stats["est"]), case["beta"], rtol=RTOL)
    assert np.allclose(np.ravel(stats["se"]), case["se"], rtol=RTOL)
    assert np.allclose(np.ravel(results.fe_dof), case["dof"], rtol=RTOL)
    assert np.allclose(np.ravel(stats["p"]), case["pval"], rtol=RTOL)

    covariance = np.asarray(results.estimator.params_["inv_cov"]).reshape(n_preds, n_preds)
    assert np.allclose(
        covariance, np.reshape(case["cov"], (n_preds, n_preds)), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("case", CASES, ids=[case_id(case) for case in CASES])
def test_coefficients_and_tau2_match_clubsandwich_everywhere(case, clubsandwich_dataset):
    """The point estimates must match whether or not the weights vary.

    CR2 changes the covariance and nothing else, so the coefficients and tau^2
    have to agree on every case in the grid. Separating this from the test above
    is what makes the divergence attributable: if the coefficients also moved,
    the standard errors would be disagreeing for a reason that had nothing to do
    with the residual adjustment.
    """
    results = fit(clubsandwich_dataset, case)
    assert np.allclose(np.ravel(results.tau2), case["tau2"], rtol=RTOL, atol=ATOL)
    assert np.allclose(np.ravel(results.get_fe_stats()["est"]), case["beta"], rtol=RTOL)


@pytest.mark.parametrize("case", CASES, ids=[case_id(case) for case in CASES])
def test_cr2_divergence_is_the_whitening_metric(case, clubsandwich_dataset):
    """Locate the CR2 divergence in the congruence each implementation applies.

    Three assertions, which together say exactly where the two part company and
    leave nothing to a tolerance:

    1.  The ``clubSandwich`` metric, written out from the definition in
        :func:`cr2_standard_errors`, reproduces the pinned standard errors on
        every case in the grid.
    2.  The PyMARE metric, from the same function, reproduces what PyMARE
        reports on every case in the grid.
    3.  The two metrics give the same answer when the within-cluster weights are
        constant, and different answers when they are not.

    So the divergence is not a bug in either sandwich, a different target
    covariance, or a different set of clusters. It is the choice of metric in
    which the Bell-McCaffrey matrix's inverse square root is taken, and it is
    invisible until the sampling variances vary inside a cluster.
    """
    y, v, X, groups = design_arrays(clubsandwich_dataset, case)
    tau2 = float(np.ravel(fit(clubsandwich_dataset, case).tau2)[0])

    from_clubsandwich = cr2_standard_errors(y, v, X, groups, tau2, "clubSandwich")
    from_pymare = cr2_standard_errors(y, v, X, groups, tau2, "pymare")
    reported = np.ravel(fit(clubsandwich_dataset, case).get_fe_stats()["se"])

    assert np.allclose(from_clubsandwich, case["se"], rtol=RTOL)
    assert np.allclose(from_pymare, reported, rtol=RTOL)

    if case["variances"] == CONSTANT_WITHIN_CLUSTER:
        assert np.allclose(from_pymare, from_clubsandwich, rtol=RTOL)
    else:
        assert not np.allclose(from_pymare, from_clubsandwich, rtol=1e-6)


def test_reference_covers_every_combination(clubsandwich_dataset):
    """The pinned grid must stay the full grid, on the data it claims.

    Without this the alignment check could quietly shrink to the cases that
    happen to pass -- and in particular to the constant-variance column, which
    is the half that agrees.
    """
    assert {(case["variances"], case["model"], case["method"]) for case in CASES} == {
        (variances, model, method)
        for variances in ("var_constant_within_study", "var_within_study")
        for model in MODELS
        for method in ESTIMATORS
    }

    n_groups = clubsandwich_dataset["study"].nunique()
    for case in CASES:
        n_preds = 1 + len(MODELS[case["model"]])
        assert case["n_groups"] == n_groups, case_id(case)
        assert len(case["beta"]) == n_preds, case_id(case)
        assert len(case["cov"]) == n_preds**2, case_id(case)
        # The Satterthwaite degrees of freedom cannot exceed the number of
        # clusters, and fall well below it when a predictor is thinly supported.
        assert all(0 < dof <= n_groups for dof in case["dof"]), case_id(case)


def test_the_two_variance_columns_differ_in_the_way_the_split_assumes(clubsandwich_dataset):
    """One variance column must be constant within study and the other not.

    The split between the agreeing and diverging halves of this module rests on
    that property of the input, so it is asserted rather than assumed: a CSV
    edited to make both columns constant would turn
    :func:`test_cr2_divergence_is_the_whitening_metric`'s third assertion into a
    statement about nothing.
    """
    grouped = clubsandwich_dataset.groupby("study")
    assert (grouped["var_constant_within_study"].nunique() == 1).all()
    assert (grouped["var_within_study"].nunique() > 1).any()
