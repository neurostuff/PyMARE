"""Measure the Type I error and CI coverage of PyMARE's small-sample corrections.

The evidence for turning the Knapp-Hartung adjustment on by default. Mirrors
``validation/stan``: the reference here is the data-generating process rather
than another package, and what is pinned is the claim rather than the numbers,
because the numbers are Monte Carlo estimates.

Run from the repository root::

    python validation/knapp_hartung/simulate.py --check

Exits non-zero if any design cell misses :data:`THRESHOLDS`, and writes the
whole grid to ``--out`` for inspection. Self-contained: nothing in ``pymare``
imports from here, and no measurement is pinned into the test suite -- these are
Monte Carlo estimates, and re-measuring is cheap enough to be the honest way to
check the claim.

Replicates go on PyMARE's parallel-dataset axis, so one fit covers a whole cell's
Monte Carlo. That is what makes a 144-cell grid affordable at all.
"""

import argparse
import json
import os.path as op
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, op.join(op.dirname(op.abspath(__file__)), "..", ".."))

from pymare import Dataset  # noqa: E402
from pymare.estimators import (  # noqa: E402
    DerSimonianLaird,
    VarianceBasedLikelihoodEstimator,
)

#: Nominal level every rate below is compared against.
ALPHA = 0.05

#: What this grid has to measure for ``"knapp-hartung"`` to be a defensible
#: default.
#: Thresholds rather than pinned values, because these are Monte Carlo estimates:
#: a correct implementation produces different numbers every run.
#:
#: ``max_knha_excess_over_wald`` is the one that decides the default -- in no cell
#: may ``"knapp-hartung"`` sit further from nominal than ``"wald"`` by more than
#: this. The worst
#: measured excess is 0.0012, below the 0.0022 Monte Carlo standard error at
#: 10,000 replications, so the ceiling is about twice that error rather than the
#: measured worst case. ``well_conditioned_bounds`` is the claim the docstrings
#: make, over cells with at least ``well_conditioned_min_obs`` observations.
#: ``max_conservative_floor`` pins the measurement that keeps
#: ``"knapp-hartung-conservative"`` an option rather than the default: some cell
#: has to collapse below it.
THRESHOLDS = {
    "max_knha_excess_over_wald": 0.005,
    "well_conditioned_bounds": (0.03, 0.08),
    "well_conditioned_min_obs": 20,
    "max_conservative_floor": 0.02,
}

#: Fewest replications at which ``--check`` is allowed to pass. Below this the
#: Monte Carlo standard error at ``ALPHA`` exceeds 0.007, which is larger than
#: ``max_knha_excess_over_wald``, so the check would be measuring noise.
MIN_REPLICATIONS_TO_CHECK = 2000

#: Ratio of the largest sampling variance to the smallest, within a cell. 1 is
#: the equal-precision case the method was derived for. 1e4 is there because
#: :footcite:t:`inthout2014hartung` and :footcite:t:`roever2015hartung` both
#: report the adjustment exceeding its nominal level for few observations of very
#: unequal precision, so it is the cell most likely to embarrass the default.
RATIOS = (1.0, 1e1, 1e2, 1e4)

#: Observation counts. 5 is small enough that the uncorrected Wald test is
#: visibly wrong; 40 is large enough that it is nearly right, which is what makes
#: the "costs nothing when unnecessary" half of the argument measurable.
OBSERVATION_COUNTS = (5, 10, 20, 40)

#: Heterogeneity as a multiple of the mean sampling variance. 0.0 is the cell
#: where tau^2 is truly zero, so it measures what the correction costs when there
#: is no tau^2 uncertainty for it to account for.
TAU2_MULTIPLES = (0.0, 0.5, 2.0)

#: Predictor counts, including the intercept. p > 1 tests a moderator
#: coefficient, which is the meta-regression case rather than the overall-effect
#: case the earliest papers treat.
PREDICTOR_COUNTS = (1, 2, 3)

CORRECTIONS = ("wald", "knapp-hartung", "knapp-hartung-conservative")

#: The two tau^2 estimators measured. Both, because the scale factor is a
#: function of the fitted weights and so of tau^2, and a result that held for
#: only one of a moment estimator and a likelihood estimator would not support a
#: default that applies to both.
ESTIMATORS = {
    "DL": lambda correction: DerSimonianLaird(small_sample_correction=correction),
    "REML": lambda correction: VarianceBasedLikelihoodEstimator(
        method="REML", small_sample_correction=correction
    ),
}


def cell_name(n_obs, n_preds, ratio, tau2_multiple):
    """Name a design cell by the four knobs that distinguish it."""
    return f"K={n_obs} P={n_preds} ratio={ratio:g} tau2={tau2_multiple:g}"


def design(rng, n_obs, n_preds, ratio):
    """Return the fixed variance column and design matrix for one cell.

    ``v`` is log-spaced between the endpoints, so the ratio is exactly ``ratio``
    rather than whatever the draws happened to span, and its mean is held at 1 so
    that ``tau2_multiple`` means the same thing at every ratio.
    """
    v = np.exp(np.linspace(0.0, np.log(ratio), n_obs))
    v = v / v.mean()
    columns = [np.ones(n_obs)] + [rng.normal(size=n_obs) for _ in range(n_preds - 1)]
    return v[:, None], np.column_stack(columns)


def run_cell(rng, n_obs, n_preds, ratio, tau2_multiple, n_reps):
    """Return the rejection rate and CI coverage for every estimator x test.

    Notes
    -----
    The coefficient under test has a true value of exactly zero, so the rejection
    rate is a Type I error rate and the nominal coverage of zero is 1 - ALPHA.
    With a moderator in the model the *intercept* carries a real effect, so the
    null being tested is "this moderator does nothing" rather than "nothing is
    going on" -- which is the null a meta-regression actually poses.

    The random effect is drawn per observation, not per replicate. One shift
    applied to every row of a replicate is a displaced mean rather than
    heterogeneity: it makes the null false, and it drives every tau^2 estimator to
    zero so that DL and REML return identical rates.
    """
    v, X = design(rng, n_obs, n_preds, ratio)
    tau2 = tau2_multiple * float(v.mean())

    beta_true = np.zeros(n_preds)
    tested = 0 if n_preds == 1 else n_preds - 1
    if n_preds > 1:
        beta_true[0] = 0.3

    y = X @ beta_true[:, None] + rng.normal(scale=np.sqrt(v), size=(n_obs, n_reps))
    if tau2 > 0:
        y = y + rng.normal(scale=np.sqrt(tau2), size=(n_obs, n_reps))

    dataset = Dataset(y=y, v=np.repeat(v, n_reps, axis=1), X=X, add_intercept=False)

    rates = {}
    for est_name, build in ESTIMATORS.items():
        for correction in CORRECTIONS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = build(correction).fit_dataset(dataset).summary()
                stats = results.get_fe_stats()
            p_values = stats["p"][tested]
            lower, upper = stats["ci_l"][tested], stats["ci_u"][tested]
            defined = np.isfinite(p_values)
            rates[f"{est_name}|{correction}"] = {
                "reject": float(np.mean(p_values[defined] < ALPHA)),
                "coverage": float(np.mean(((lower <= 0.0) & (0.0 <= upper))[defined])),
                "undefined": float(np.mean(~defined)),
            }
    return rates


def threshold_failures(results):
    """Return the ways ``results`` misses :data:`THRESHOLDS`, as messages."""
    ceiling = THRESHOLDS["max_knha_excess_over_wald"]
    low, high = THRESHOLDS["well_conditioned_bounds"]
    min_obs = THRESHOLDS["well_conditioned_min_obs"]

    problems = []
    conservative = []
    for cell in results:
        for key, rate in cell["rates"].items():
            estimator, correction = key.split("|")
            if correction == "knapp-hartung-conservative":
                conservative.append(rate["reject"])
                continue
            if correction != "knapp-hartung":
                continue
            plain = cell["rates"][f"{estimator}|wald"]["reject"]
            excess = abs(rate["reject"] - ALPHA) - abs(plain - ALPHA)
            if excess > ceiling:
                problems.append(
                    f"{cell['name']} {estimator}: knapp-hartung {rate['reject']:.4f} is "
                    f"{excess:.4f} further from {ALPHA} than wald {plain:.4f}, over {ceiling}"
                )
            if cell["n_obs"] >= min_obs and not low <= rate["reject"] <= high:
                problems.append(
                    f"{cell['name']} {estimator}: knapp-hartung {rate['reject']:.4f} outside "
                    f"[{low}, {high}] with K >= {min_obs}"
                )

    if conservative and min(conservative) > THRESHOLDS["max_conservative_floor"]:
        problems.append(
            f"the conservative variant never fell below "
            f"{THRESHOLDS['max_conservative_floor']} (lowest {min(conservative):.4f}); the "
            "measurement that rules it out as the default is gone"
        )
    return problems


def main():
    """Run the grid, record it, and optionally check it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replications",
        type=int,
        default=10000,
        help="Monte Carlo replications per cell (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260819, help="seed for the whole grid (default: 20260819)"
    )
    parser.add_argument(
        "--out",
        default=op.join(op.dirname(op.abspath(__file__)), "measured.json"),
        help="where to write the full grid (default: alongside this script)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any cell misses THRESHOLDS",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    started = time.time()
    results = []
    for n_obs in OBSERVATION_COUNTS:
        for n_preds in PREDICTOR_COUNTS:
            if n_obs - n_preds < 1:
                continue
            for ratio in RATIOS:
                for multiple in TAU2_MULTIPLES:
                    rates = run_cell(rng, n_obs, n_preds, ratio, multiple, args.replications)
                    results.append(
                        {
                            "name": cell_name(n_obs, n_preds, ratio, multiple),
                            "n_obs": n_obs,
                            "n_preds": n_preds,
                            "ratio": ratio,
                            "tau2_multiple": multiple,
                            "rates": rates,
                        }
                    )
                    print(
                        f"{results[-1]['name']:<34}"
                        + "  ".join(f"{key}={rates[key]['reject']:.4f}" for key in sorted(rates)),
                        flush=True,
                    )

    payload = {
        "alpha": ALPHA,
        "replications": args.replications,
        "seed": args.seed,
        "monte_carlo_se": float(np.sqrt(ALPHA * (1 - ALPHA) / args.replications)),
        "thresholds": THRESHOLDS,
        "elapsed_seconds": round(time.time() - started, 1),
        "cells": results,
    }
    with open(args.out, "w") as fobj:
        json.dump(payload, fobj, indent=2)
        fobj.write("\n")

    print(f"\nwrote {op.normpath(args.out)} in {payload['elapsed_seconds']}s")

    if not args.check:
        return 0

    # A short run can clear the excess ceiling by luck, since that ceiling is
    # smaller than the Monte Carlo standard error at a few hundred replications.
    if args.replications < MIN_REPLICATIONS_TO_CHECK:
        print(
            f"\n--check needs at least {MIN_REPLICATIONS_TO_CHECK} replications to mean "
            f"anything; got {args.replications}."
        )
        return 1

    problems = threshold_failures(results)
    if problems:
        print("\nTHRESHOLDS NOT MET:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"all {len(results)} cells meet the thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
