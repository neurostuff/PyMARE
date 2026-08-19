"""Measure the Stan estimator's bias and credible-interval coverage.

Not run in CI. The fast tests in ``pymare/tests/test_stan_estimators.py`` check
that the estimator recovers one planted configuration; this checks that it does
so across the grid of designs where meta-regression estimators are known to
break, and reports coverage, which is the property the model's correctness
actually rests on and which no single fit can establish.

Run with::

    make validate_stan

or equivalently::

    python validation/stan/simulate.py --check

which writes ``pymare/tests/data/stan_validation.json`` and exits non-zero if any
design cell misses ``pymare.tests.utils.STAN_VALIDATION_THRESHOLDS``. Two tests
in ``pymare/tests/test_stan_estimators.py`` hold that recorded file to the same
thresholds on every run, so the record cannot go stale unnoticed, and the
"Validate the Stan model" workflow re-measures on a schedule.

This directory's README.md explains the arrangement and records what was
measured.
"""

import argparse
import collections
import json
import logging
import os
import os.path as op
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cmdstanpy
import numpy as np

sys.path.insert(0, op.join(op.dirname(op.abspath(__file__)), "..", ".."))

from pymare import Dataset  # noqa: E402
from pymare.estimators import StanMetaRegression  # noqa: E402
from pymare.tests.utils import STAN_VALIDATION_THRESHOLDS  # noqa: E402

# CmdStanPy narrates every chain through its own handler; at a few thousand fits
# that is the only thing on screen.
cmdstanpy.disable_logging()
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

#: One entry per design cell, each varying a single factor away from BASE, so a
#: failure is attributable to that factor. A full factorial would cost several
#: times the fits and still need reading this way.
CELLS = [
    # Number of groups: the axis the prior on tau is most sensitive to.
    {"name": "groups=5", "n_groups": 5},
    {"name": "groups=20", "n_groups": 20},
    {"name": "groups=50", "n_groups": 50},
    # Between-group variance, including the boundary at zero where the funnel
    # geometry is worst.
    {"name": "tau2=0", "tau2": 0.0},
    {"name": "tau2=0.1", "tau2": 0.1},
    {"name": "tau2=1", "tau2": 1.0},
    # Group structure.
    {"name": "singletons", "group_size": 1},
    {"name": "unequal groups", "group_size": "unequal"},
    # Scale of the variance column. A scale-dependent bug reads as fine at one
    # scale, which is why a fixed prior scale would not survive this row.
    {"name": "sigma x0.1", "sigma_scale": 0.1},
    {"name": "sigma x10", "sigma_scale": 10.0},
    # Predictor count.
    {"name": "1 predictor", "n_predictors": 1},
    {"name": "3 predictors", "n_predictors": 3},
    # The cell usually omitted: a group-level moderator carried by only a
    # handful of groups, which is where robust variance estimators break down.
    {"name": "unbalanced covariate", "unbalanced": True},
    {"name": "unbalanced covariate, tau2=1", "unbalanced": True, "tau2": 1.0},
]

#: Fewest replications at which ``--check`` is allowed to pass. Matches the
#: floor the pinned results are held to in ``test_stan_estimators.py``.
MIN_REPLICATIONS_TO_CHECK = 100

BASE = {
    "n_groups": 20,
    "group_size": 3,
    "tau2": 0.1,
    "sigma_scale": 1.0,
    "n_predictors": 2,
    "unbalanced": False,
}


#: True coefficients, held fixed across replications. Redrawing them from a
#: symmetric distribution makes bias uninformative: an estimator that always
#: returned zero would have errors of -beta, averaging to zero over replications,
#: and would clear any bias threshold.
TRUE_BETA = np.array([0.5, -0.8, 0.3])


def simulate(rng, n_groups, group_size, tau2, sigma_scale, n_predictors, unbalanced):
    """Draw one dataset from the model the Stan program encodes."""
    if group_size == "unequal":
        sizes = rng.integers(1, 6, size=n_groups)
    else:
        sizes = np.full(n_groups, group_size)
    groups = np.repeat(np.arange(n_groups), sizes)
    n_observations = groups.size

    if unbalanced:
        # A group-level moderator switched on for only three groups.
        carriers = rng.choice(n_groups, size=min(3, n_groups), replace=False)
        moderators = np.isin(groups, carriers).astype(float)[:, None]
        for _ in range(n_predictors - 2):
            moderators = np.column_stack([moderators, rng.normal(size=n_observations)])
    else:
        moderators = rng.normal(size=(n_observations, max(n_predictors - 1, 0)))

    X = (
        np.column_stack([np.ones(n_observations), moderators])
        if moderators.size
        else np.ones((n_observations, 1))
    )
    beta = TRUE_BETA[: X.shape[1]]

    theta = rng.normal(0, np.sqrt(tau2), size=n_groups)
    sigma = rng.uniform(0.1, 0.4, size=n_observations) * sigma_scale
    y = X @ beta + theta[groups] + rng.normal(0, sigma)

    predictors = X[:, 1:] if X.shape[1] > 1 else None
    dataset = Dataset(y=y, v=sigma**2, X=predictors, g=groups)
    return dataset, beta, tau2


def run_cell(cell, replications, seed):
    """Fit one design cell `replications` times and accumulate the summaries."""
    config = dict(BASE)
    config.update({k: v for k, v in cell.items() if k != "name"})
    rng = np.random.default_rng(seed)

    # Per coefficient, not pooled: in the unbalanced cells the sparse moderator
    # is the coefficient at risk, and intercept coverage would mask it.
    beta_errors = collections.defaultdict(list)
    covered = collections.defaultdict(list)
    tau2_errors, tau2_truth, divergent = [], [], 0

    for replication in range(replications):
        dataset, beta, tau2 = simulate(rng, **config)
        est = StanMetaRegression(
            iter_sampling=1000,
            iter_warmup=1000,
            chains=2,
            seed=int(rng.integers(1, 2**31 - 1)),
            show_progress=False,
        )
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = est.fit_dataset(dataset).summary().summary()
        divergent += any("divergent" in str(w.message) for w in caught)

        for i, true_value in enumerate(beta):
            row = summary.loc[f"beta[{i}]"]
            beta_errors[i].append(float(row["mean"]) - true_value)
            lower, upper = _interval(row)
            covered[i].append(bool(lower <= true_value <= upper))

        tau2_errors.append(float(summary.loc["tau2", "mean"]) - tau2)
        tau2_truth.append(tau2)

    print(f"  done: {cell['name']}", flush=True)
    return {
        "name": cell["name"],
        "config": {k: str(v) for k, v in config.items()},
        "replications": replications,
        "true_beta": [float(b) for b in TRUE_BETA[: len(_coefficients(covered))]],
        "beta_bias_per_coefficient": _per_coefficient(beta_errors),
        "beta_coverage_per_coefficient": _per_coefficient(covered),
        # The figures the thresholds are applied to summarize the *worst*
        # coefficient rather than the average of them.
        "beta_bias": max(_per_coefficient(beta_errors), key=abs),
        "beta_coverage": min(_per_coefficient(covered)),
        "coverage_se": float(
            np.sqrt(
                min(_per_coefficient(covered))
                * (1 - min(_per_coefficient(covered)))
                / len(covered[_coefficients(covered)[0]])
            )
        ),
        "tau2_bias": float(np.mean(tau2_errors)),
        "tau2_truth": float(np.mean(tau2_truth)),
        "fits_with_divergences": int(divergent),
    }


def _coefficients(per_coefficient):
    """Return the coefficient indices present, in order."""
    return sorted(per_coefficient)


def _per_coefficient(per_coefficient):
    """Reduce a per-coefficient mapping of samples to a list of means."""
    return [float(np.mean(per_coefficient[i])) for i in _coefficients(per_coefficient)]


def _interval(row):
    """Pull the credible interval out of a summary row, across ArviZ versions."""
    # ArviZ 0.x names them hdi_2.5%/hdi_97.5%; 1.x names them hdi95_lb/hdi95_ub.
    lower = [c for c in row.index if c.startswith("hdi") and ("lb" in c or c.endswith("2.5%"))]
    upper = [c for c in row.index if c.startswith("hdi") and ("ub" in c or c.endswith("97.5%"))]
    return float(row[lower[0]]), float(row[upper[0]])


def main():
    """Run the grid, write the results, and optionally enforce the thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="cells to fit concurrently; each fit already uses one core per chain",
    )
    parser.add_argument(
        "--out",
        default=op.join(
            op.dirname(op.abspath(__file__)),
            "..",
            "..",
            "pymare",
            "tests",
            "data",
            "stan_validation.json",
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any cell misses STAN_VALIDATION_THRESHOLDS",
    )
    args = parser.parse_args()

    # Compile before forking. CmdStanPy builds in place, so parallel make
    # invocations on the same source collide: on a cold cache one worker
    # reliably fails before any cell runs. One compile up front makes every
    # worker a cache hit.
    print("compiling the model", flush=True)
    StanMetaRegression().compile()

    started = time.time()
    if args.jobs == 1:
        results = []
        for index, cell in enumerate(CELLS):
            print(f"[{index + 1}/{len(CELLS)}] {cell['name']}", flush=True)
            results.append(run_cell(cell, args.replications, args.seed + index))
    else:
        # Cells are independent and each seeded from its own index, so running
        # them concurrently changes nothing about the numbers -- only the
        # wall-clock. Each fit already uses 2 cores for its 2 chains.
        print(f"running {len(CELLS)} cells across {args.jobs} processes", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(run_cell, cell, args.replications, args.seed + index)
                for index, cell in enumerate(CELLS)
            ]
            results = [future.result() for future in futures]

    payload = {
        "replications": args.replications,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - started, 1),
        "thresholds": STAN_VALIDATION_THRESHOLDS,
        "cells": results,
    }
    with open(args.out, "w") as fobj:
        json.dump(payload, fobj, indent=2)

    print(f"\n{'cell':<32}{'worst bias':>11}{'worst cov':>10}{'tau2 bias':>11}{'diverg.':>9}")
    for cell in results:
        print(
            f"{cell['name']:<32}{cell['beta_bias']:>11.4f}"
            f"{cell['beta_coverage']:>10.3f}{cell['tau2_bias']:>11.4f}"
            f"{cell['fits_with_divergences']:>9d}"
        )
    print(f"\nwrote {op.normpath(args.out)} in {payload['elapsed_seconds']}s")

    if not args.check:
        return 0

    # A short run can clear the coverage floor by luck: at 10 replications the
    # estimate moves in steps of 0.05 with a standard error of 0.07.
    if args.replications < MIN_REPLICATIONS_TO_CHECK:
        print(
            f"\n--check needs at least {MIN_REPLICATIONS_TO_CHECK} replications to mean "
            f"anything; got {args.replications}."
        )
        return 1

    # The same thresholds the pinned results are held to, applied to this run.
    floor = STAN_VALIDATION_THRESHOLDS["min_coverage"]
    ceiling = STAN_VALIDATION_THRESHOLDS["max_beta_bias"]
    failures = [
        f"{cell['name']}: coverage {cell['beta_coverage']:.3f} below {floor:.2f}"
        for cell in results
        if cell["beta_coverage"] < floor
    ] + [
        f"{cell['name']}: |beta bias| {abs(cell['beta_bias']):.4f} above {ceiling}"
        for cell in results
        if abs(cell["beta_bias"]) > ceiling
    ]
    if failures:
        print("\nTHRESHOLDS NOT MET:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"all {len(results)} cells meet the thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
