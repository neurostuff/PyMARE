"""Alignment between PyMARE's permutation test and metafor's ``permutest``.

:meth:`~pymare.results.MetaRegressionResults.permutation_test` and
``metafor::permutest`` do the same thing: enumerate or sample a permutation set,
refit the model on each member, and report the proportion of permuted statistics
at least as extreme as the observed one. Only the exact mode can be compared --
the approximate one draws from each package's own generator -- so the reference
is ``permutest(exact = TRUE)`` on the designs small enough to enumerate, and
PyMARE is asked for the same enumeration.

The two agree exactly, in all ten cases and on both coefficients of the
moderator models. They did not when this module was written, for two reasons
that it still isolates rather than merely asserting the totals:

-   :func:`test_metafor_permutest_is_reproduced_by_the_z_statistic` counts the
    permutations independently of the production path, on ``|beta / se|`` and
    with the observed statistic read out of the identity permutation in the same
    batch. It is what established that the permutation sets and the batched
    refits were already right, and it remains the check that says *why* the
    totals agree rather than only that they do.
-   :func:`test_permutation_p_value_matches_metafor` is what PyMARE reports.

The two defects it used to record, both now fixed in
:meth:`~pymare.results.MetaRegressionResults.permutation_test`:

**The statistic.** PyMARE counted ``|beta|`` where ``permutest`` counts
``|beta / se|``. A permutation test needs a statistic whose null distribution
does not move with the parameters being permuted away, and ``|beta|`` does:
refitting a permuted dataset re-estimates tau^2, which changes the weights and
so the standard error. The two coincide only where the standard error is
invariant under the permutation -- a fixed-effects, intercept-only model under
sign flipping -- which is why four of the ten cases agreed anyway, and why
``unequal_k5`` under ``DL`` came out at 0.5625 against ``permutest``'s 0.5.

**The tie.** The observed statistic is computed by a different code path from
the permuted ones, and the two could disagree by a unit in the last place, so an
exactly inclusive comparison dropped the identity permutation -- the one that
reproduces the observed data, and must therefore count -- along with its
sign-flipped mirror. That understated the p-value by ``2 / 2**K``: 0.033203125
against ``permutest``'s 0.03515625 on ``extreme_k10``. The comparison now allows
the same square-root-of-epsilon slack ``permutest`` does.

"""

import copy
import itertools
import json
import math
import os.path as op

import numpy as np
import pytest

from pymare import Dataset
from pymare.estimators import DerSimonianLaird, WeightedLeastSquares
from pymare.tests.utils import get_test_data_path

pytestmark = pytest.mark.metafor

with open(op.join(get_test_data_path(), "metafor_permutest_reference.json")) as _fobj:
    REFERENCE = json.load(_fobj)

CASES = REFERENCE["cases"]

#: A permutation p-value is a count over a known denominator, so agreement is
#: exact or it is not agreement. This is here only to absorb the division.
RTOL = 1e-12

#: Moderator columns each model adds beside the intercept, matching ``mods`` in
#: ``validation/metafor/run_permutest.R``.
MODELS = {"intercept": [], "one": ["mod1"]}

#: The estimators compared. ``small_sample_correction="wald"`` because
#: ``permutest`` was asked with ``test="z"``: the correction rescales the
#: standard error, and comparing a corrected statistic against an uncorrected
#: reference would confound the two things this module is trying to separate.
ESTIMATORS = {
    "FE": lambda: WeightedLeastSquares(tau2=0.0, small_sample_correction="wald"),
    "DL": lambda: DerSimonianLaird(small_sample_correction="wald"),
}


def case_id(case):
    """Name a case by the three knobs that distinguish it."""
    return f"{case['design']}-{case['model']}-{case['method']}"


def design_arrays(frame, case):
    """Return ``(y, v, moderators)`` for one case, without the intercept."""
    rows = frame[frame["case"] == case["design"]]
    columns = MODELS[case["model"]]
    moderators = rows[columns].to_numpy() if columns else None
    return rows["y"].to_numpy(), rows["v"].to_numpy(), moderators


def enumerate_permutations(case, n_obs):
    """Return the permuted-column indices, and which column is the observed data.

    The observed statistic is read out of the enumeration rather than computed
    separately, which is what makes the tie exact: the identity permutation *is*
    the observed dataset, so its statistic is the observed statistic to the last
    bit, however the batched refit happens to round.

    Returns
    -------
    :obj:`tuple` of (:obj:`numpy.ndarray`, :obj:`numpy.ndarray`, :obj:`int`)
        Sign multipliers of shape ``(K, n_perm)`` (all ones for a model with
        moderators), row indices of shape ``(K, n_perm)``, and the column
        holding the identity permutation.
    """
    if MODELS[case["model"]]:
        orders = list(itertools.permutations(range(n_obs)))
        rows = np.array(orders).T
        signs = np.ones_like(rows)
        identity = orders.index(tuple(range(n_obs)))
    else:
        signs = np.array(list(itertools.product([-1, 1], repeat=n_obs))).T
        rows = np.repeat(np.arange(n_obs)[:, None], signs.shape[1], axis=1)
        identity = int(np.flatnonzero((signs == 1).all(axis=0))[0])
    return signs, rows, identity


def permuted_statistics(case, frame):
    """Refit every permutation of one case and return ``|beta|`` and ``|beta / se|``.

    One batched call per case: PyMARE's closed-form estimators accept a column
    per dataset, which is the same vectorization
    :meth:`~pymare.results.MetaRegressionResults.permutation_test` uses
    internally, so this exercises the estimator on exactly the inputs the
    production path would hand it.
    """
    y, v, moderators = design_arrays(frame, case)
    n_obs = y.shape[0]
    signs, rows, identity = enumerate_permutations(case, n_obs)

    design = np.column_stack([np.ones(n_obs)] + ([moderators] if moderators is not None else []))
    params = (
        copy.copy(ESTIMATORS[case["method"]]()).fit(y=y[rows] * signs, v=v[rows], X=design).params_
    )
    beta = np.atleast_2d(params["fe_params"])
    cov = np.asarray(params["inv_cov"])
    se = np.sqrt(np.stack([cov[i, i, :] for i in range(beta.shape[0])]))
    return np.abs(beta), np.abs(beta / se), identity


@pytest.mark.parametrize("case", CASES, ids=[case_id(case) for case in CASES])
def test_metafor_permutest_is_reproduced_by_the_z_statistic(case, metafor_dataset):
    """Counting the same permutations on ``|z|`` must reproduce metafor exactly.

    This is the load-bearing test of the module. It uses PyMARE's own estimator,
    PyMARE's own enumeration of the permutation set, and the same inclusive
    comparison PyMARE uses -- changing only the statistic counted, from the
    coefficient to the coefficient over its standard error. That it then matches
    ``permutest`` in every case, to the last bit of a rational number, is what
    establishes that the rest of PyMARE's permutation machinery is right and
    that :func:`test_permutation_p_value_matches_metafor` fails for the two
    reasons its marker names and not for some third one.

    The observed statistic is taken from the identity permutation inside the
    same batch, so the permutation that reproduces the data ties with it
    exactly. metafor gets the same effect by comparing against
    ``|zval| - sqrt(eps)``.
    """
    _, z_statistic, identity = permuted_statistics(case, metafor_dataset)
    observed = z_statistic[:, [identity]]
    p_values = (z_statistic >= observed).mean(axis=1)
    assert z_statistic.shape[1] == case["n_perm"]
    assert np.allclose(p_values, case["pval"], rtol=RTOL)


@pytest.mark.filterwarnings("ignore:Cluster-robust")
def test_permutation_p_value_matches_metafor(metafor_dataset):
    """The exact permutation p-values PyMARE reports must be metafor's.

    Asserted over the whole grid in one test rather than parametrized, so that a
    failure reports every case that moved rather than the first. This was a
    strict xfail until the two defects in the module docstring were fixed.
    """
    mismatched = []
    for case in CASES:
        y, v, moderators = design_arrays(metafor_dataset, case)
        dataset = Dataset(y=y, v=v, X=moderators, add_intercept=True)
        results = ESTIMATORS[case["method"]]().fit_dataset(dataset).summary()
        permuted = results.permutation_test(n_perm=int(case["n_perm"]))
        assert permuted.exact and permuted.n_perm == case["n_perm"], case_id(case)

        reported = np.ravel(permuted.perm_p["fe_p"])
        if not np.allclose(reported, case["pval"], rtol=RTOL):
            mismatched.append((case_id(case), list(reported), case["pval"]))

    assert not mismatched, mismatched


def test_reference_covers_the_enumerable_designs(metafor_dataset):
    """The pinned grid must be the designs small enough to enumerate exactly.

    ``2**K`` sign flips for the intercept-only models and ``K!`` orderings for
    the moderator ones, which is why the grid stops where it does: the recorded
    ``n_perm`` is asserted against the design's own size so that a reference
    regenerated against a different CSV, or with ``exact`` dropped, cannot pass
    for the pinned one.
    """
    assert {(case["design"], case["model"], case["method"]) for case in CASES} == {
        (design, "intercept", method)
        for design in ("equal_k5", "unequal_k5", "extreme_k10")
        for method in ("FE", "DL")
    } | {
        (design, "one", method) for design in ("equal_k5", "unequal_k5") for method in ("FE", "DL")
    }

    for case in CASES:
        n_obs = (metafor_dataset["case"] == case["design"]).sum()
        expected = 2 ** n_obs if not MODELS[case["model"]] else math.factorial(n_obs)
        assert case["n_perm"] == expected, case_id(case)
        assert len(case["pval"]) == 1 + len(MODELS[case["model"]]), case_id(case)
