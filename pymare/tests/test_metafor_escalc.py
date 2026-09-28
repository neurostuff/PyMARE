"""Alignment between PyMARE's effect-size converters and metafor's ``escalc``.

:class:`~pymare.effectsize.OneSampleEffectSizeConverter` and
:class:`~pymare.effectsize.TwoSampleEffectSizeConverter` turn study-level summary
statistics into an estimate and its sampling variance, which is what
``metafor::escalc`` does. This module compares the two, measure by measure, and
is deliberately explicit about which of the three relationships each pair is in:

**The same closed form.** The raw mean and its variance, the raw correlation, the
Fisher z-transformed correlation and its variance, the raw mean difference, and
the pooled standard deviation. These are compared at :data:`RTOL_EXACT` and a
failure means one of the two is wrong.

**The same quantity, one of them approximated.** PyMARE corrects a standardized
mean for bias with ``1 - 3/(4m - 1)``; metafor uses the exact
``c(m) = gamma(m/2) / (sqrt(m/2) gamma((m-1)/2))``. The approximation is
second-order accurate in ``m``, so rather than pinning a tolerance these tests
assert the error stays under :data:`BIAS_CORRECTION_BOUND` divided by ``m**2``,
which is a statement about the approximation and not about either
implementation's current output.

**Different formulas for the same thing.** metafor takes the sampling variance of
a correlation to be ``(1 - r**2)**2 / (n - 1)`` and PyMARE takes it to be
``(1 - r**2) / (n - 2)``; metafor's standardized-mean variances are large-sample
approximations where PyMARE's are the exact noncentral-t expressions. These
cannot be verified against each other, so
:func:`test_raw_correlation_variance_is_a_different_formula` and
``validation/metafor/README.md`` record what each one is instead.

Four comparisons are marked :func:`pytest.mark.xfail` with ``strict=True``,
because measuring these turned up defects rather than divergences: three
sampling-variance expressions in ``pymare/effectsize/expressions.json`` have
misplaced parentheses or a sign error, one of which is the subject of
https://github.com/neurostuff/PyMARE/pull/144. Each marker names the expression
and what it should be. ``strict=True`` is the point: when one is corrected the
test passes, pytest reports XPASS as a failure, and the marker has to be removed
in the same change.

"""

import json
import os.path as op

import numpy as np
import pytest

from pymare.effectsize import OneSampleEffectSizeConverter, TwoSampleEffectSizeConverter
from pymare.tests.utils import get_test_data_path

pytestmark = pytest.mark.metafor

with open(op.join(get_test_data_path(), "metafor_escalc_reference.json")) as _fobj:
    REFERENCE = json.load(_fobj)

CASES = REFERENCE["cases"]

#: Tolerance for the measures both implementations reach by the same closed
#: form. Observed exact on every row of the input grid; this allows for the
#: reassociation a compiler or a different order of operations can introduce.
RTOL_EXACT = 1e-13

#: Absolute floor for the rows where the true value is zero -- the grid includes
#: a zero-effect design so that the bias-corrected measures are checked where
#: the correction has nothing to scale.
ATOL_EXACT = 1e-15

#: Numerator of the bound on PyMARE's bias-correction approximation. The error
#: of ``1 - 3/(4m - 1)`` against the exact ``c(m)`` falls as ``m**-2``, with a
#: constant measured at 0.043 over the grid -- worst 2.7e-3 at ``m = 4``, the
#: smallest this grid goes, and 2.0e-7 at ``m = 398``. Rounded up to 0.05, which
#: leaves the bound tight enough that a first-order error would break it.
BIAS_CORRECTION_BOUND = 0.05

#: Bound on the relative difference between PyMARE's and metafor's *corrected*
#: two-sample standardized-mean-difference variance, as a multiple of
#: ``1 / (n1 + n2)``. The two use different approximations of the same quantity
#: -- PyMARE ``d**2 / (2 (n1 + n2 - 2))`` scaled by ``j**2``, metafor
#: ``g**2 / (2 (n1 + n2))`` -- which differ at order ``1 / N``. Worst observed
#: 0.72 of this bound, at ``n1 = n2 = 5``.
SMD_VARIANCE_BOUND = 2.0

#: Factor within which PyMARE's exact standardized-mean variances must sit
#: relative to metafor's large-sample approximations. Not a tolerance: the two
#: are different expressions and disagree by 2.5x at ``n = 5``, where the
#: ``(n - 1) / (n - 3)`` inflation in the exact form is largest. It is a bound
#: loose enough to be satisfied by any implementation of the right quantity and
#: tight enough to catch the sign error the markers below describe, which puts
#: PyMARE out by one to two orders of magnitude.
SAME_ORDER_FACTOR = 4.0


def case_ids():
    """Name each row of the input grid."""
    return [case["case"] for case in CASES]


@pytest.fixture(scope="module")
def escalc_inputs():
    """Load the summary statistics the escalc reference values were computed on."""
    import pandas as pd

    return pd.read_csv(op.join(get_test_data_path(), "metafor_escalc_inputs.csv"))


@pytest.fixture(scope="module")
def one_sample(escalc_inputs):
    """Build a converter over the single-group columns of the input grid."""
    return OneSampleEffectSizeConverter(
        m=escalc_inputs["m"].to_numpy(),
        sd=escalc_inputs["sd"].to_numpy(),
        n=escalc_inputs["n"].to_numpy(),
    )


@pytest.fixture(scope="module")
def correlations(escalc_inputs):
    """Build a converter over the correlation columns of the input grid.

    Separate from :func:`one_sample` because the correlation measures are solved
    from ``r`` and ``n`` while the mean measures are solved from ``m``, ``sd``
    and ``n``; one converter holding all five would let the solver reach a
    measure by a path the documented one does not offer.
    """
    return OneSampleEffectSizeConverter(
        r=escalc_inputs["r"].to_numpy(), n=escalc_inputs["n"].to_numpy()
    )


@pytest.fixture(scope="module")
def two_sample(escalc_inputs):
    """Build a converter over the two-group columns of the input grid."""
    return TwoSampleEffectSizeConverter(
        m1=escalc_inputs["m1"].to_numpy(),
        m2=escalc_inputs["m2"].to_numpy(),
        sd1=escalc_inputs["sd1"].to_numpy(),
        sd2=escalc_inputs["sd2"].to_numpy(),
        n1=escalc_inputs["n1"].to_numpy(),
        n2=escalc_inputs["n2"].to_numpy(),
    )


def measure(converter, name):
    """Return one measure and its variance as a pair of 1-D arrays."""
    dataset = converter.to_dataset(measure=name)
    return np.ravel(dataset.y), np.ravel(dataset.v)


def expected(field, key):
    """Collect one escalc column across the grid, in input order."""
    return np.array([case[field][key] for case in CASES], dtype=float)


def reference(field):
    """Collect one scalar reference column across the grid, in input order."""
    return np.array([case[field] for case in CASES], dtype=float)


def assert_exact(got, want, label):
    """Compare a whole column at :data:`RTOL_EXACT`, naming the rows that miss."""
    close = np.isclose(got, want, rtol=RTOL_EXACT, atol=ATOL_EXACT)
    missed = [
        f"{case['case']}: {g!r} != {w!r}"
        for case, g, w, ok in zip(CASES, got, want, close)
        if not ok
    ]
    assert not missed, f"{label}: " + "; ".join(missed)


# -----------------------------------------------------------------------------
# The same closed form on both sides.
# -----------------------------------------------------------------------------


def test_raw_mean_matches_metafor(one_sample):
    """``RM`` must be ``escalc(measure="MN")``, estimate and variance alike."""
    y, v = measure(one_sample, "RM")
    assert_exact(y, expected("MN", "yi"), "RM estimate")
    assert_exact(v, expected("MN", "vi"), "RM variance")


def test_raw_correlation_matches_metafor(correlations):
    """``R``'s estimate must be ``escalc(measure="COR")``'s.

    The estimate only. The two variances are different formulas, which
    :func:`test_raw_correlation_variance_is_a_different_formula` records.
    """
    y, _ = measure(correlations, "R")
    assert_exact(y, expected("COR", "yi"), "R estimate")


def test_fisher_z_correlation_matches_metafor(correlations):
    """``ZR`` must be ``escalc(measure="ZCOR")``, estimate and variance alike.

    The one transformed correlation measure where PyMARE and metafor agree on
    both halves: ``atanh(r)`` and ``1 / (n - 3)``.
    """
    y, v = measure(correlations, "ZR")
    assert_exact(y, expected("ZCOR", "yi"), "ZR estimate")
    assert_exact(v, expected("ZCOR", "vi"), "ZR variance")


def test_raw_mean_difference_matches_metafor(two_sample):
    """``RMD``'s estimate must be ``escalc(measure="MD")``'s."""
    y, _ = measure(two_sample, "RMD")
    assert_exact(y, expected("MD", "yi"), "RMD estimate")


def test_pooled_standard_deviation_matches_metafor(two_sample, escalc_inputs):
    """The pooled SD must be the one metafor divides by.

    ``escalc`` does not report ``sdpi``, but it is recoverable from what it does
    report: the bias-corrected estimate divided by the exact correction factor is
    the raw Cohen's d, and the raw mean difference over that is the pooled SD.
    Checking it separately means a failure in
    :func:`test_standardized_mean_difference_matches_metafor` can be read as
    being about the correction factor rather than about the pooling.
    """
    # The zero-effect row has m1 == m2, so d is zero and the quotient is not
    # defined. Its pooled SD is covered by every other row's.
    varies = escalc_inputs["m1"].to_numpy() != escalc_inputs["m2"].to_numpy()
    metafor_d = expected("SMD", "yi") / reference("cm_two_sample")
    metafor_sdp = expected("MD", "yi")[varies] / metafor_d[varies]
    got = np.ravel(two_sample.get("sdp"))[varies]
    assert np.allclose(got, metafor_sdp, rtol=RTOL_EXACT)


# -----------------------------------------------------------------------------
# The same quantity, PyMARE approximating metafor's exact correction factor.
# -----------------------------------------------------------------------------


def bias_correction_bound(dof):
    """Return the allowed relative error of the correction factor at ``m = dof``."""
    return BIAS_CORRECTION_BOUND / np.asarray(dof, dtype=float) ** 2


def test_bias_correction_approximates_the_exact_factor(one_sample, two_sample, escalc_inputs):
    """``1 - 3/(4m - 1)`` must approximate ``c(m)`` to second order.

    Checked directly on the factor rather than only through the measures that
    use it, so that a change to the approximation is attributed here rather than
    showing up as a drifting tolerance on two other tests. Both converters
    expose the factor as ``j``; the degrees of freedom are ``n - 1`` for a single
    group and ``n1 + n2 - 2`` for two.
    """
    single_dof = escalc_inputs["n"].to_numpy() - 1
    pair_dof = escalc_inputs["n1"].to_numpy() + escalc_inputs["n2"].to_numpy() - 2
    for converter, dof, exact in (
        (one_sample, single_dof, reference("cm_one_sample")),
        (two_sample, pair_dof, reference("cm_two_sample")),
    ):
        approximate = np.ravel(converter.get("j"))
        error = np.abs(approximate - exact) / exact
        assert np.all(error <= bias_correction_bound(dof)), list(
            zip(case_ids(), error, bias_correction_bound(dof))
        )


def test_standardized_mean_matches_metafor(one_sample, escalc_inputs):
    """``SM`` must be metafor's single-group standardized mean.

    metafor has no single-group standardized mean, so the reference is
    ``escalc(measure="SMCC")`` with the second measurement set to zero and
    uncorrelated with the first, which reduces algebraically to ``m / sd`` with
    the exact correction applied on ``n - 1`` degrees of freedom. See the header
    of ``validation/metafor/run_escalc.R``.

    The estimate only, and only to the bias-correction bound: the two agree
    exactly on ``m / sd`` and differ only in the factor multiplying it.
    """
    y, _ = measure(one_sample, "SM")
    want = expected("SMCC", "yi")
    error = np.abs(y - want) / np.where(want == 0, 1.0, np.abs(want))
    assert np.all(error <= bias_correction_bound(escalc_inputs["n"].to_numpy() - 1))


def test_standardized_mean_difference_matches_metafor(two_sample, escalc_inputs):
    """``SMD``'s estimate must be ``escalc(measure="SMD")``'s.

    To the bias-correction bound, for the same reason as
    :func:`test_standardized_mean_matches_metafor`: the raw ``d`` and the pooled
    SD agree exactly, as the two tests above establish, so the whole of the
    difference here is the correction factor.
    """
    y, _ = measure(two_sample, "SMD")
    want = expected("SMD", "yi")
    error = np.abs(y - want) / np.where(want == 0, 1.0, np.abs(want))
    dof = escalc_inputs["n1"].to_numpy() + escalc_inputs["n2"].to_numpy() - 2
    assert np.all(error <= bias_correction_bound(dof))


# -----------------------------------------------------------------------------
# Different formulas for the same thing, recorded rather than compared.
# -----------------------------------------------------------------------------


def test_raw_correlation_variance_is_a_different_formula(correlations, escalc_inputs):
    """Record that ``R``'s variance is not metafor's, and which is which.

    metafor uses the asymptotic sampling variance of a correlation,
    ``(1 - r**2)**2 / (n - 1)``. PyMARE uses ``(1 - r**2) / (n - 2)``, which is
    the squared standard error of ``r`` under the null hypothesis of no
    correlation rather than its sampling variance at the observed value. The two
    diverge without limit as ``|r|`` approaches one -- 51x apart at ``r = 0.99``
    on this grid -- so no tolerance relates them.

    This asserts that each side is the formula named above and nothing more. It
    exists so that the divergence cannot quietly change shape: if either
    expression were replaced, this test would say so rather than a tolerance
    somewhere else drifting.
    """
    _, v = measure(correlations, "R")
    r = escalc_inputs["r"].to_numpy()
    n = escalc_inputs["n"].to_numpy()
    assert np.allclose(v, (1 - r**2) / (n - 2), rtol=RTOL_EXACT)
    assert np.allclose(expected("COR", "vi"), (1 - r**2) ** 2 / (n - 1), rtol=RTOL_EXACT)


def test_standardized_mean_variances_are_exact_not_asymptotic(one_sample, escalc_inputs):
    """Record that the single-group standardized-mean variances are not metafor's.

    metafor reports ``1 / n + y**2 / (2n)``, the large-sample approximation.
    PyMARE's expressions are the exact noncentral-t variance,
    ``(n - 1)/(n - 3) (1/n + d**2) - d**2 / c**2``, which is the better quantity
    but not the same one: the two are 2.5x apart at ``n = 5``. So the single-
    group standardized-mean variance has no usable metafor reference, and
    :func:`test_standardized_mean_variance_is_the_same_order_as_metafor` is the
    most that can be asserted across the two.

    This test pins metafor's side of that statement, which is the half that can
    be checked exactly, so that the factor-of-four bound in the other test is
    known to be comparing against the asymptotic formula and not against
    something else metafor might report in a future release.
    """
    n = escalc_inputs["n"].to_numpy()
    y = expected("SMCC", "yi")
    assert np.allclose(expected("SMCC", "vi"), 1 / n + y**2 / (2 * n), rtol=RTOL_EXACT)


# -----------------------------------------------------------------------------
# Defects, recorded as strict xfails until the expressions are corrected.
# -----------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v_rmd in pymare/effectsize/expressions.json reads "
        "'v_rmd - (sd1**2 / n1) + (sd2**2 / n2)', which solves to "
        "sd1**2/n1 - sd2**2/n2 rather than the sum. The variance comes out "
        "negative whenever the second group is the more variable one, and "
        "exactly zero for two equally sized, equally variable groups -- which "
        "gives that study infinite weight. The fix is to parenthesize the "
        "denominator, as PR #144 did for the two-sample Cohen's d"
    ),
)
def test_raw_mean_difference_variance_matches_metafor(two_sample):
    """``RMD``'s variance must be ``escalc(measure="MD")``'s, exactly.

    ``sd1**2 / n1 + sd2**2 / n2`` on both sides, with nothing approximated, so
    this is an exact comparison once the expression is corrected: it was
    verified to agree on every row of the grid to zero relative error with the
    parentheses in place.
    """
    _, v = measure(two_sample, "RMD")
    assert_exact(v, expected("MD", "vi"), "RMD variance")


def test_standardized_mean_difference_variance_matches_metafor(two_sample, escalc_inputs):
    """``SMD``'s variance must agree with metafor's to order ``1 / N``.

    Not exactly: PyMARE scales ``d**2 / (2 (n1 + n2 - 2))`` by ``j**2`` and
    metafor adds ``g**2 / (2 (n1 + n2))``, two approximations of the same
    variance that differ at order ``1 / N``. :data:`SMD_VARIANCE_BOUND` is that
    order, measured at 0.72 of the bound in the worst cell of the grid.

    This was a strict xfail until
    https://github.com/neurostuff/PyMARE/pull/144 corrected the expression:
    it used to read ``d**2 / 2 * (n1 + n2 - 2)``, multiplying the
    squared-effect term by the residual degrees of freedom instead of dividing
    by twice them, which put the variance out by up to three orders of
    magnitude and made it *grow* with the sample size.
    """
    _, v = measure(two_sample, "SMD")
    want = expected("SMD", "vi")
    total = escalc_inputs["n1"].to_numpy() + escalc_inputs["n2"].to_numpy()
    error = np.abs(v - want) / want
    assert np.all(error <= SMD_VARIANCE_BOUND / total), list(zip(case_ids(), error))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v_sm in pymare/effectsize/expressions.json reads "
        "'v_sm - ((n - 1)/(n - 3)) * j**2 * (1 / n + d**2) - d**2', which "
        "solves to A + d**2 where the noncentral-t variance is A - d**2. The "
        "reported variance is two orders of magnitude too large for any "
        "appreciable effect, and grows with the effect instead of being "
        "dominated by 1/n"
    ),
)
def test_standardized_mean_variance_is_the_same_order_as_metafor(one_sample):
    """``SM``'s variance must be within a factor of metafor's approximation.

    A factor and not a tolerance, for the reason
    :func:`test_standardized_mean_variances_are_exact_not_asymptotic` gives: the
    exact and asymptotic expressions are 2.5x apart at ``n = 5``. Anything
    outside :data:`SAME_ORDER_FACTOR` is not two approximations of one variance
    disagreeing, and that is what this is here to catch.
    """
    _, v = measure(one_sample, "SM")
    ratio = v / expected("SMCC", "vi")
    assert np.all(ratio <= SAME_ORDER_FACTOR), list(zip(case_ids(), ratio))
    assert np.all(ratio >= 1 / SAME_ORDER_FACTOR), list(zip(case_ids(), ratio))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "v_d (one-sample) in pymare/effectsize/expressions.json reads "
        "'v_d - ((n - 1)/(n - 3)) * (1 / n + d**2) - d**2 / j**2 * n', so the "
        "last term is added and scaled by n where the noncentral-t variance "
        "subtracts d**2 / j**2. The reported variance therefore grows with the "
        "sample size, which is backwards"
    ),
)
def test_one_sample_d_variance_is_the_same_order_as_metafor(one_sample):
    """``D``'s variance must be within a factor of metafor's approximation.

    metafor reports no uncorrected d, so its variance is recovered by undoing
    the exact correction: ``Var(d) = Var(g) / c(m)**2``. Bounded by a factor for
    the same reason as :func:`test_standardized_mean_variance_is_the_same_order_as_metafor`.
    """
    _, v = measure(one_sample, "D")
    metafor_variance = expected("SMCC", "vi") / reference("cm_one_sample") ** 2
    ratio = v / metafor_variance
    assert np.all(ratio <= SAME_ORDER_FACTOR), list(zip(case_ids(), ratio))
    assert np.all(ratio >= 1 / SAME_ORDER_FACTOR), list(zip(case_ids(), ratio))


# -----------------------------------------------------------------------------
# Guards on the reference itself.
# -----------------------------------------------------------------------------


def test_reference_covers_every_input_row(escalc_inputs):
    """The pinned cases must be the input grid, in order and complete.

    Every test above lines a PyMARE column up against a reference column by
    position, so a reference that dropped or reordered a row would compare the
    wrong pairs rather than fail. The designs are named here as well, so that
    dropping one from the CSV is a failure and not a quieter check.
    """
    assert case_ids() == list(escalc_inputs["case"])
    assert case_ids() == [
        "balanced_small",
        "balanced_large",
        "unbalanced",
        "very_unbalanced",
        "zero_effect",
        "large_effect",
        "heteroscedastic",
        "tiny",
    ]
    for case in CASES:
        for field in ("MN", "COR", "ZCOR", "SMCC", "MD", "SMD"):
            assert set(case[field]) == {"yi", "vi"}, (case["case"], field)
            assert all(isinstance(value, (int, float)) for value in case[field].values())
        for field in ("cm_one_sample", "cm_two_sample"):
            assert 0 < case[field] < 1, (case["case"], field)
