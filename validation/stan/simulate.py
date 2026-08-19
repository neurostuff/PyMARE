"""Measure the Stan estimator's bias and credible-interval coverage.

Not run in CI. The fast tests in ``pymare/tests/test_stan_estimators.py`` check
that the estimator recovers one planted configuration; this checks that it does
so across the grid of designs where meta-regression estimators are known to
break, and reports coverage, which is the property the model's correctness
actually rests on and which no single fit can establish.

Run with::

    python validation/stan/simulate.py --replications 100 --out results.json

The measured output is recorded in this directory's README.md.
"""

import argparse
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

# CmdStanPy narrates every chain through its own handler; at a few thousand fits
# that is the only thing on screen.
cmdstanpy.disable_logging()
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

#: One entry per design cell. Each varies a single factor away from the base
#: configuration, which is the arrangement that attributes a failure to a
#: factor; a full factorial over five factors would cost 5x the fits and still
#: need this reading to interpret.
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

BASE = {
    "n_groups": 20,
    "group_size": 3,
    "tau2": 0.1,
    "sigma_scale": 1.0,
    "n_predictors": 2,
    "unbalanced": False,
}


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
    beta = rng.normal(size=X.shape[1])

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

    beta_errors, covered, tau2_errors, tau2_truth, divergent = [], [], [], [], 0

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
            beta_errors.append(float(row["mean"]) - true_value)
            lower, upper = _interval(row)
            covered.append(bool(lower <= true_value <= upper))

        tau2_errors.append(float(summary.loc["tau2", "mean"]) - tau2)
        tau2_truth.append(tau2)

    print(f"  done: {cell['name']}", flush=True)
    return {
        "name": cell["name"],
        "config": {k: str(v) for k, v in config.items()},
        "replications": replications,
        "beta_bias": float(np.mean(beta_errors)),
        "beta_rmse": float(np.sqrt(np.mean(np.square(beta_errors)))),
        "beta_coverage": float(np.mean(covered)),
        "coverage_se": float(np.sqrt(np.mean(covered) * (1 - np.mean(covered)) / len(covered))),
        "tau2_bias": float(np.mean(tau2_errors)),
        "tau2_truth": float(np.mean(tau2_truth)),
        "fits_with_divergences": int(divergent),
    }


def _interval(row):
    """Pull the credible interval out of a summary row, across ArviZ versions."""
    # ArviZ 0.x names them hdi_2.5%/hdi_97.5%; 1.x names them hdi95_lb/hdi95_ub.
    lower = [c for c in row.index if c.startswith("hdi") and ("lb" in c or c.endswith("2.5%"))]
    upper = [c for c in row.index if c.startswith("hdi") and ("ub" in c or c.endswith("97.5%"))]
    return float(row[lower[0]]), float(row[upper[0]])


def main():
    """Run the grid and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="cells to fit concurrently; each fit already uses one core per chain",
    )
    parser.add_argument("--out", default=op.join(op.dirname(op.abspath(__file__)), "results.json"))
    args = parser.parse_args()

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
        "cells": results,
    }
    with open(args.out, "w") as fobj:
        json.dump(payload, fobj, indent=2)

    print(f"\n{'cell':<32}{'beta bias':>11}{'coverage':>10}{'tau2 bias':>11}{'diverg.':>9}")
    for cell in results:
        print(
            f"{cell['name']:<32}{cell['beta_bias']:>11.4f}"
            f"{cell['beta_coverage']:>10.3f}{cell['tau2_bias']:>11.4f}"
            f"{cell['fits_with_divergences']:>9d}"
        )
    print(f"\nwrote {args.out} in {payload['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
