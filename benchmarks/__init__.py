"""Benchmarks for PyMARE, run by asv.

The suite is what the ``Benchmark`` workflow compares between a pull request and
its base, so its job is to cover the functions a change is likely to slow down
and to stay fast enough to run twice on every pull request. Sizes are chosen for
that: large enough that the vectorized paths dominate the Python overhead, small
enough that the whole suite times in seconds.

``bench_cluster_robust.py`` is the exception -- it is a standalone report, not an
asv benchmark, and is described in its own docstring.
"""
