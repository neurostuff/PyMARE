"""Alignment between PyMARE and the R package metafor on the Knapp-Hartung adjustment.

PyMARE's ``test`` parameter implements the adjustment
:footcite:p:`knapp2003improved` in the form ``metafor``'s ``rma.uni`` applies it,
so this module checks that the two still agree over the whole inference path --
coefficients, standard errors, degrees of freedom, p-values and interval bounds --
and not only over the scale factor.

Two tests, answering two questions:

-   :func:`test_adjustment_matches_metafor` supplies PyMARE with ``metafor``'s own
    tau^2, which removes the tau^2 estimators from the comparison. A failure there
    is a failure of the adjustment. This is the one that covers all 180 cases,
    including every ``method`` whose tau^2 PyMARE reaches by a different route.
-   :func:`test_estimator_matches_metafor` runs end to end for the two estimators
    that reach tau^2 in closed form, so nothing but the adjustment sits between
    the two implementations.

Whether PyMARE's ML, REML and Hedges tau^2 match ``metafor``'s is a separate
question that predates this work -- they differ by up to the tau^2 search
tolerance, and Hedges by a definitional choice. ``validation/metafor/README.md``
records the measurements; comparing them here would mix an optimizer's tolerance
into a check on a closed-form scale factor.

The reference values are pinned in ``data/metafor_reference.json`` because metafor
is an R package and cannot be a test dependency. The pin is kept honest by
regenerating it -- see :func:`pymare.tests.utils.load_metafor_reference`.

References
----------
.. footbibliography::

"""

import numpy as np
import pytest

from pymare import Dataset
from pymare.estimators import DerSimonianLaird, WeightedLeastSquares
from pymare.tests.utils import load_metafor_reference

pytestmark = pytest.mark.metafor

REFERENCE = load_metafor_reference()

#: Tolerances for a comparison that should be exact. The two implementations do
#: the same arithmetic in a different order, so the coefficients, standard errors
#: and p-values agree to a few multiples of machine epsilon. The absolute floor is
#: set by the interval bounds, worst observed 6e-11: those multiply the standard
#: error by a t quantile, and ``scipy.stats.t.ppf`` and R's ``qt`` do not return
#: bit-identical quantiles.
RTOL = 1e-11
ATOL = 1e-9

#: Moderator columns each model adds beside the intercept, matching the ``mods``
#: list in ``validation/metafor/run_metafor.R``. metafor puts the intercept first
#: and so does :class:`~pymare.core.Dataset`, so the coefficient vectors line up
#: position by position.
MODELS = {"intercept": [], "one": ["mod1"], "two": ["mod1", "mod2"]}

#: metafor's ``test`` spellings mapped onto PyMARE's ``small_sample_correction``
#: values. The pinned file keeps metafor's vocabulary because it records what
#: metafor was *asked*; the translation belongs here, in the open, rather than
#: being hidden by regenerating the reference under PyMARE's names.
CORRECTIONS = {
    "z": "wald",
    "knha": "knapp-hartung",
    "adhoc": "knapp-hartung-conservative",
}

#: metafor ``method`` values whose tau^2 PyMARE reaches in closed form, and which
#: can therefore be compared end to end at ``RTOL``. ``"FE"`` is the
#: fixed-effects model, which PyMARE spells as a known tau^2 of zero.
CLOSED_FORM = {
    "FE": lambda correction: WeightedLeastSquares(tau2=0.0, small_sample_correction=correction),
    "DL": lambda correction: DerSimonianLaird(small_sample_correction=correction),
}

EXACT_CASES = [case for case in REFERENCE["cases"] if case["method"] in CLOSED_FORM]


def case_id(case):
    """Name a case by the four knobs that distinguish it."""
    return f"{case['design']}-{case['model']}-{case['method']}-{case['test']}"


def build_dataset(frame, case):
    """Select one design's rows and assemble the model PyMARE should fit."""
    rows = frame[frame["case"] == case["design"]]
    columns = MODELS[case["model"]]
    return Dataset(
        y=rows["y"].to_numpy(),
        v=rows["v"].to_numpy(),
        X=rows[columns].to_numpy() if columns else None,
        add_intercept=True,
    )


def assert_matches(results, case):
    """Compare every reported fixed-effect statistic against the reference.

    Notes
    -----
    The p-value is compared only under the two corrected options. Under
    ``"wald"`` PyMARE computes it as ``1 - |0.5 - Phi(z)| * 2``, which cancels
    catastrophically in the far tail and disagrees with metafor's
    ``2 * pnorm(-|z|)`` by an arbitrarily large *relative* amount once the p-value
    drops below about 1e-15. That predates this branch and is not what this module
    measures; the adjustment's own path goes through ``2 * t.sf(...)``, which does
    not cancel and agrees to 3e-15.
    """
    stats = results.get_fe_stats()
    for key, expected in (
        ("est", "beta"),
        ("se", "se"),
        ("ci_l", "ci_lb"),
        ("ci_u", "ci_ub"),
    ):
        assert np.allclose(np.ravel(stats[key]), case[expected], rtol=RTOL, atol=ATOL), key
    if case["test"] != "z":
        assert np.allclose(np.ravel(stats["p"]), case["pval"], rtol=RTOL, atol=ATOL)

    if case["dof"] is None:
        assert results.fe_dof is None
    else:
        assert np.all(results.fe_dof == case["dof"])


@pytest.mark.parametrize(
    "case", REFERENCE["cases"], ids=[case_id(case) for case in REFERENCE["cases"]]
)
def test_adjustment_matches_metafor(case, metafor_dataset):
    """The adjustment must be metafor's function of the same tau^2."""
    estimator = WeightedLeastSquares(
        tau2=float(case["tau2"]), small_sample_correction=CORRECTIONS[case["test"]]
    )
    assert_matches(estimator.fit_dataset(build_dataset(metafor_dataset, case)).summary(), case)


@pytest.mark.parametrize("case", EXACT_CASES, ids=[case_id(case) for case in EXACT_CASES])
def test_estimator_matches_metafor(case, metafor_dataset):
    """The closed-form estimators must match end to end, tau^2 included."""
    results = (
        CLOSED_FORM[case["method"]](CORRECTIONS[case["test"]])
        .fit_dataset(build_dataset(metafor_dataset, case))
        .summary()
    )
    assert np.allclose(np.ravel(results.tau2), case["tau2"], rtol=RTOL, atol=ATOL)
    assert_matches(results, case)


def test_correction_map_covers_metafor_and_pymare_alike():
    """The translation must be total in both directions.

    A value added to either vocabulary and not to the other would otherwise be
    silently untested: an unmapped metafor spelling raises a KeyError only for the
    cases that use it, and an unmapped PyMARE value simply never gets compared.
    """
    from pymare.estimators.estimators import SMALL_SAMPLE_CORRECTIONS

    assert set(CORRECTIONS) == {c["test"] for c in REFERENCE["cases"]}
    assert set(CORRECTIONS.values()) == set(SMALL_SAMPLE_CORRECTIONS)


def test_reference_covers_every_combination():
    """The pinned grid must stay the full grid, with K - P degrees of freedom.

    Two guards in one, because both defend the same thing -- that the pinned file
    still says what the tests above assume. Without the first, the alignment check
    could quietly shrink to the cases that happen to pass; without the second, a
    regenerated file reporting some other quantity under ``ddf`` would be adopted
    silently.
    """
    designs = {"equal_k5", "unequal_k5", "extreme_k10", "moderate_k20"}
    methods = {"FE", "DL", "HE", "ML", "REML"}
    expected = {
        (design, model, method, test)
        for design in designs
        for model in MODELS
        for method in methods
        for test in ("z", "knha", "adhoc")
    }
    assert {
        (c["design"], c["model"], c["method"], c["test"]) for c in REFERENCE["cases"]
    } == expected

    sizes = {"equal_k5": 5, "unequal_k5": 5, "extreme_k10": 10, "moderate_k20": 20}
    for case in REFERENCE["cases"]:
        n_preds = 1 + len(MODELS[case["model"]])
        expected_dof = None if case["test"] == "z" else sizes[case["design"]] - n_preds
        assert case["dof"] == expected_dof, case_id(case)
