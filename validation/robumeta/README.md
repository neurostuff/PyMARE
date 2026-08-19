# robumeta reference values

PyMARE's correlated-effects working model (`weight_scheme="rescale"`) implements
the estimator of Hedges, Tipton & Johnson (2010), in the form given as equations
(7)-(9) of the robumeta paper (Fisher & Tipton, arXiv:1503.02220). This directory
regenerates the reference values that `pymare/tests/test_robumeta_alignment.py`
pins.

```bash
validation/robumeta/regenerate.sh
```

That rewrites `pymare/tests/data/robumeta_reference.json` in place, from a Docker
image with pinned R and robumeta versions. Unchanged numbers produce an
unchanged file, so the diff is the answer:

```bash
make check_robumeta_alignment
```

robumeta is an R package and cannot be a test dependency, so the numbers are
pinned rather than recomputed on every test run. What keeps a pinned file from
becoming a stale one is the `Check robumeta alignment` workflow, which runs the
script above on every pull request and fails on any difference. Rerun it and
commit the result when you change the estimator on purpose.

## What agrees

PyMARE and robumeta agree to ~1e-14 on tau^2, the coefficients, the
cluster-robust standard errors and the Satterthwaite degrees of freedom, for
every model, every rho, and both variance columns -- one constant within a
study, one varying sharply within it.

That covers the whole inference path, not just the variance component: the
coefficients come from the correlated-effects weights of equation (7), the
standard errors from the CR2 sandwich, and the degrees of freedom from the
Satterthwaite approximation.

## What is compared

Twenty-four cases, the full grid of:

| Knob | Values |
| --- | --- |
| model | `effect ~ 1`, `effect ~ within`, `effect ~ within + between` |
| rho | 0.0, 0.4, 0.8, 1.0 |
| variances | `var_constant_within_study`, `var_within_study` |

`test_reference_covers_every_combination` asserts that grid, so the alignment
check cannot quietly shrink to the cases that happen to pass.

robumeta's other options have no PyMARE counterpart to compare against:
`modelweights = "HIER"` is a different working model, and `small = FALSE` drops
the small-sample corrections that PyMARE's estimators always apply. Those are
where the two implementations stop overlapping, not where they disagree.
