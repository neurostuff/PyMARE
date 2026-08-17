"""Benchmark cluster-robust variance estimation against the model-based path.

The sandwich does work comparable to the weighted least squares solve itself:
it forms the per-study scores, sums them within groups, and contracts two
``p x p`` matrices per dataset. Measured overhead is therefore a constant
factor of roughly 1-1.5x on top of a naive fit, not a rounding error. What
matters is that the factor stays *constant*. This script guards against:

* an accidental per-dataset (per-voxel) Python loop, which would make the cost
  scale with ``d`` instead of vectorizing over it;
* a bad ``einsum`` contraction path, which showed up as overhead growing with
  the number of groups (over 10x at m=100) before the group sum was rewritten
  with ``add.reduceat``;
* materializing anything of size ``(d, k, k)``, which blows up memory.

Run with::

    python benchmarks/bench_cluster_robust.py
"""

import time

import numpy as np

from pymare.estimators import DerSimonianLaird, WeightedLeastSquares

# (n_studies, n_datasets) combinations, sized like real IBMA problems.
SHAPES = [
    (10, 10_000),
    (50, 10_000),
    (50, 100_000),
    (200, 100_000),
    (200, 200_000),
]
STUDIES_PER_GROUP = 2
N_REPEATS = 3


def make_data(n_studies, n_datasets, seed=0):
    """Build a one-sample design with dependent estimates."""
    rng = np.random.RandomState(seed)
    y = rng.standard_normal((n_studies, n_datasets))
    v = np.abs(rng.standard_normal((n_studies, n_datasets))) + 0.5
    X = np.ones((n_studies, 1))
    groups = np.repeat(np.arange(n_studies // STUDIES_PER_GROUP), STUDIES_PER_GROUP)
    return y, v, X, groups


def timeit(func, n_repeats=N_REPEATS):
    """Return the best wall-clock time over several repeats."""
    best = float("inf")
    for _ in range(n_repeats):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    """Time each estimator with and without group labels."""
    estimators = [
        ("WeightedLeastSquares", WeightedLeastSquares),
        ("DerSimonianLaird", DerSimonianLaird),
    ]

    header = (
        f"{'estimator':22s} {'k':>5s} {'d':>8s} {'m':>5s} "
        f"{'naive(s)':>9s} {'robust(s)':>10s} {'overhead':>9s}"
    )
    print(header)
    print("-" * len(header))

    for name, estimator in estimators:
        for n_studies, n_datasets in SHAPES:
            y, v, X, groups = make_data(n_studies, n_datasets)
            n_groups = np.unique(groups).size

            naive = timeit(lambda: estimator().fit(y=y, v=v, X=X))
            robust = timeit(lambda: estimator().fit(y=y, v=v, X=X, g=groups))

            overhead = 100.0 * (robust / naive - 1.0)
            print(
                f"{name:22s} {n_studies:5d} {n_datasets:8d} {n_groups:5d} "
                f"{naive:9.3f} {robust:10.3f} {overhead:+8.1f}%"
            )


if __name__ == "__main__":
    main()
