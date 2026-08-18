"""Benchmark cluster-robust variance estimation against the model-based path.

Passing ``g`` still costs several times the naive fit, and the fit is not
where the time goes. At k=200, m=100 (groups of two), d=50,000 the split is:

===================================== =========
weighted least squares solve            0.19 s
CR0 sandwich                            0.17 s
CR2 sandwich (the default)              1.08 s
Satterthwaite degrees of freedom        1.16 s
===================================== =========

Both dominant terms are per-group work that cannot be folded into the same
contraction as the fit. CR2 inflates each group's residuals by
``(I_j - H_j)^{-1/2}``, and the degrees of freedom need the same adjustment,
so both pay for it once per group per dataset. CR0 needs neither, which is why
it stays the cheapest -- it is also anti-conservative, so the cost buys
something.

``H_j`` has rank at most ``p``, so neither term forms the ``n_j x n_j``
eigendecomposition its definition suggests; both reduce to a ``p x p`` problem
(see ``_cr2_low_rank_factors``). That is worth 5-11x for intercept-only models
and grows with group size, which is why the two columns above no longer
dominate the way they did.

Two structural discounts remain, on top of that reduction. At the same shape,
the degrees of freedom cost:

===================================== =========
groups of two (the table above)         1.20 s
singleton groups                        0.75 s
one shared ``v`` column                 0.02 s
===================================== =========

Singleton groups make ``(I_j - H_j)`` a scalar, so the adjustment is a plain
reciprocal square root. Identical weight columns make the adjustment identical
across parallel datasets, so it is solved once and broadcast -- the weights and
the design determine it, never ``y``.

What this script guards against is the *scaling*, not the constant:

* an accidental per-dataset Python loop, which would make the cost scale with
  ``d`` instead of vectorizing over it;
* a bad ``einsum`` contraction path, which showed up as overhead growing with
  the number of groups (over 10x at m=100) before the group sum was rewritten
  with ``add.reduceat``;
* per-group Python overhead in the CR2 and dof paths, which is why groups of
  equal size are batched into one call rather than looped over;
* materializing anything of size ``(d, k, k)``, which blows up memory.

So the number to watch is not the overhead column itself but whether it stays
flat as ``d`` grows at fixed ``k`` and ``m``.

Run with::

    python benchmarks/bench_cluster_robust.py
"""

import time

import numpy as np

from pymare.estimators import DerSimonianLaird, WeightedLeastSquares
from pymare.stats import cluster_robust_cov, satterthwaite_dof, weighted_least_squares

# (n_estimates, n_datasets) combinations, sized like large parallel
# meta-analyses, where many datasets are analyzed side by side.
SHAPES = [
    (10, 10_000),
    (50, 10_000),
    (50, 100_000),
    (200, 100_000),
    (200, 200_000),
]
ESTIMATES_PER_GROUP = 2
N_REPEATS = 3


def make_data(n_estimates, n_datasets, seed=0):
    """Build a one-sample design with dependent estimates."""
    rng = np.random.RandomState(seed)
    y = rng.standard_normal((n_estimates, n_datasets))
    v = np.abs(rng.standard_normal((n_estimates, n_datasets))) + 0.5
    X = np.ones((n_estimates, 1))
    groups = np.repeat(np.arange(n_estimates // ESTIMATES_PER_GROUP), ESTIMATES_PER_GROUP)
    return y, v, X, groups


def timeit(func, n_repeats=N_REPEATS):
    """Return the best wall-clock time over several repeats."""
    best = float("inf")
    for _ in range(n_repeats):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def breakdown():
    """Time the sandwich and the degrees of freedom separately.

    The lumped overhead below says how much ``g`` costs; this says where it
    goes, so the claims in the module docstring stay checkable.
    """
    header = (
        f"{'k':>5s} {'d':>8s} {'m':>5s} {'fit(s)':>8s} "
        f"{'CR0(s)':>8s} {'CR2(s)':>8s} {'dof(s)':>8s}"
    )
    print("component breakdown (WeightedLeastSquares inputs)")
    print(header)
    print("-" * len(header))

    for n_estimates, n_datasets in SHAPES:
        y, v, X, groups = make_data(n_estimates, n_datasets)
        n_groups = np.unique(groups).size
        beta, model_cov = weighted_least_squares(y, v, X, return_cov=True)

        fit = timeit(lambda: weighted_least_squares(y, v, X, return_cov=True))
        cr0 = timeit(
            lambda: cluster_robust_cov(
                y, v, X, beta, groups, model_cov=model_cov, method="CR0", small_sample=False
            )
        )
        cr2 = timeit(
            lambda: cluster_robust_cov(y, v, X, beta, groups, model_cov=model_cov, method="CR2")
        )
        dof = timeit(lambda: satterthwaite_dof(X, 1.0 / v, groups, model_cov=model_cov))
        print(
            f"{n_estimates:5d} {n_datasets:8d} {n_groups:5d} {fit:8.3f} "
            f"{cr0:8.3f} {cr2:8.3f} {dof:8.3f}"
        )


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
        for n_estimates, n_datasets in SHAPES:
            y, v, X, groups = make_data(n_estimates, n_datasets)
            n_groups = np.unique(groups).size

            naive = timeit(lambda: estimator().fit(y=y, v=v, X=X))
            robust = timeit(lambda: estimator().fit(y=y, v=v, X=X, g=groups))

            # Reported as a multiple, not a percentage: the overhead is a
            # factor of tens, and "+4000%" is harder to read than "40x".
            print(
                f"{name:22s} {n_estimates:5d} {n_datasets:8d} {n_groups:5d} "
                f"{naive:9.3f} {robust:10.3f} {robust / naive:8.1f}x"
            )

    print()
    breakdown()


if __name__ == "__main__":
    main()
