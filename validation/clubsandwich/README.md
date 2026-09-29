# clubSandwich reference values

PyMARE's cluster-robust covariance takes the `CRn` names from the R package
[clubSandwich](https://cran.r-project.org/package=clubSandwich)
(Pustejovsky & Tipton, 2018, *Journal of Business & Economic Statistics* 36(4),
672-683), and `pymare.stats.satterthwaite_dof` implements the degrees of freedom
that package pairs with `CR2`. This directory regenerates the reference values
`pymare/tests/test_clubsandwich_alignment.py` pins.

```bash
make check_clubsandwich_alignment   # needs Docker; rewrites the pinned file in place
make test_clubsandwich              # check PyMARE against the pinned values
```

Regeneration goes through a Docker image with pinned R, metafor and clubSandwich
versions. metafor is in the image because clubSandwich has no model of its own:
`vcovCR` and `coef_test` take a fitted `rma.uni` and replace its covariance, so
the fit has to come from metafor. Mirrors `validation/robumeta` and
`validation/metafor` otherwise, including the numeric rather than byte
comparison the workflow makes -- see `validation/compare_reference.py`.

## Why this is not the robumeta check again

`validation/robumeta` and this directory answer different questions about the
same estimator, which is why both exist.

| | robumeta | clubSandwich |
| --- | --- | --- |
| Reference for | the correlated-effects **working model** -- how weight is spread across a study's rows | the **CR2 residual adjustment** and its Satterthwaite degrees of freedom |
| PyMARE spelling | `weight_scheme="rescale"` | `weight_scheme="individual"` with group labels |
| Assumes | sampling variances constant within a study | nothing about them |

robumeta cannot distinguish the two CR2 adjustments below, because its model has
constant within-study weights by construction. That is exactly the condition
under which they coincide.

## What is compared

12 cases, the full grid of variance column x model x tau^2 estimator, on the
same `pymare/tests/data/robumeta_correlated_effects.csv` the robumeta check
uses. `test_reference_covers_every_combination` asserts that grid.

| Knob | Values |
| --- | --- |
| variance column | `var_constant_within_study`, `var_within_study` |
| model | `effect ~ 1`, `effect ~ within`, `effect ~ within + between` |
| tau^2 estimator | `FE`, `DL` |

Only the two estimators that reach tau^2 in closed form, so that nothing but the
sandwich sits between the two implementations. The CSV's two variance columns are
the point of the grid: one is constant inside every study and the other is not,
and `test_the_two_variance_columns_differ_in_the_way_the_split_assumes` asserts
that property of the input rather than trusting the column names.

Everything both implementations report is compared -- coefficients, the full CR2
covariance including its off-diagonal entries, the Satterthwaite degrees of
freedom, the standard errors and the p-values that follow from them.

## What agrees

**With the sampling variances constant within each cluster, everything agrees to
machine precision, in all six cases:**

| Quantity | Worst relative deviation |
| --- | --- |
| coefficients | 2.9e-15 |
| CR2 covariance | 5.1e-15 |
| standard errors | 1.8e-15 |
| Satterthwaite dof | 2.4e-15 |
| p-values | 5.4e-15 |

The coefficients and tau^2 agree in all twelve cases, varying variances
included, to 3.5e-15. CR2 changes the covariance and nothing else, so that is
what makes the divergence below attributable to the sandwich.

## What does not, and why

**With the sampling variances varying inside a cluster, the covariance
diverges:**

| Quantity | Worst relative deviation |
| --- | --- |
| CR2 covariance | 5.8e-2 |
| standard errors | 1.0e-2 |
| Satterthwaite dof | 3.9e-3 |
| p-values | 3.4e-2 |

The cause is exact, and `test_cr2_divergence_is_the_whitening_metric` pins it by
writing both forms out from their definitions and showing each reproduces its
own implementation's output. Both build the same Bell-McCaffrey matrix
(Bell & McCaffrey, 2002, *Survey Methodology* 28(2), 169-181)

```
B_j = W_j^-1 - X_j (X'WX)^-1 X_j'
```

and take its inverse square root, but in different metrics. With `Psi = W^-1`
the assumed target covariance, clubSandwich forms

```
A_j = Psi_j^(1/2) (Psi_j^(1/2) B_j Psi_j^(1/2))^(-1/2) Psi_j^(1/2)
```

and PyMARE's `_cr2_scores` forms

```
A_j = W_j^(1/2) (W_j^(1/2) B_j W_j^(1/2))^(-1/2) W_j^(1/2)
```

A matrix square root does not commute with an asymmetric congruence, so the two
are equal if and only if `W_j` is a multiple of the identity -- that is, when the
weights, and hence the sampling variances, are constant within the cluster.

## Why PyMARE keeps its form

Neither implementation is approximating the other, and the difference is not a
defect in either. Bell and McCaffrey define `A_j` by the condition that the
adjusted residuals carry the working-model covariance,

```
A_j B_j A_j' = Psi_j
```

and **both forms satisfy it exactly** -- measured at 1.7e-15 (clubSandwich) and
1.3e-15 (PyMARE) on the varying-variance column of this directory's grid.
Feeding each through to the sandwich, both give `E[V_R] = (X'WX)^-1` to every
digit under the working model, so both are *exactly unbiased* in the sense CR2
exists to provide.

The condition does not determine `A_j` uniquely: it constrains it only through
`A_j B_j A_j'`. clubSandwich closes that freedom by requiring `A_j` symmetric;
PyMARE's is symmetric in the whitened metric instead. On the same grid,
`max |A - A'|` is 2.8e-17 for clubSandwich's and 9.1e-2 for PyMARE's. That is
the entire difference between them.

**What PyMARE's choice buys is an algorithm.** In the whitened metric the matrix
whose inverse square root is needed is

```
W_j^(1/2) B_j W_j^(1/2) = I - X~_j M X~_j'
```

the identity minus a rank-`p` term, so its spectrum collapses to `p` non-unit
eigenvalues *whatever the group size* and `pymare.stats._cr2_low_rank_factors`
can take the inverse square root in `p x p` work. clubSandwich's matrix is
`Psi_j^2` minus a rank-`p` term, and because that diagonal part is not a multiple
of the identity its spectrum does not collapse -- measured on random designs with
`p = 2`, PyMARE's matrix has 2 non-repeated eigenvalues at every group size while
clubSandwich's has `n_j`:

| group size | eigenvalues off the repeated one, PyMARE | clubSandwich |
| --- | --- | --- |
| 6 | 2 | 6 |
| 40 | 2 | 40 |
| 200 | 2 | 200 |

So the symmetric form needs the full `n_j x n_j` eigendecomposition. Timed
against the `p x p` one it replaces:

| `n_j` | full `n_j x n_j` | `p x p` | ratio |
| --- | --- | --- | --- |
| 40 | 146 us | 7.7 us | 19x |
| 200 | 3,108 us | 7.5 us | 416x |
| 800 | 52,293 us | 8.9 us | 5,901x |

The square root not commuting with an asymmetric congruence is at once why the
two forms differ at all and why only one of them factors.

PyMARE's form is also the one Fisher & Tipton (2015, arXiv:1503.02220) give as
`A_j^C`, which `pymare.stats._cr2_scores` says in its docstring, and the
correlated-effects model it belongs to has constant within-study weights by
construction -- which is why `validation/robumeta` cannot tell the two apart.

### What is genuinely against it

The published small-sample simulation evidence (Tipton 2015; Imbens & Kolesar
2016; Pustejovsky & Tipton 2018) is for the symmetric form. Exact unbiasedness
holds for both *under the working model*; how the two behave when that model is
wrong is studied for one of them and not the other. A user comparing against
`clubSandwich` or `metafor::robust(..., clubSandwich = TRUE)` will see different
standard errors whenever sampling variances vary inside a cluster, which is the
common case rather than the corner one.

`method="CR2"` is kept, because it is a CR2 by the defining condition and
because of the complexity argument above. What was missing was saying so: the
`method` parameter and `_cr2_scores` now both record that this is the
whitened-metric solution, that it coincides with clubSandwich exactly when
within-cluster weights are constant, and that it differs otherwise.

## What is not compared

`CR0`, which PyMARE also offers. clubSandwich's `CR0` is the unadjusted
sandwich, and PyMARE's applies an `m / (m - p)` scaling that
`cluster_robust_cov`'s docstring already records as being neither clubSandwich's
`CR1` nor Stata's `CR1S` -- it is the original adjustment of Hedges, Tipton &
Johnson (2010), kept for reproducing analyses that predate the leverage-based
corrections. There is nothing in clubSandwich to compare it against.

`CR1`, `CR3` and `CR4` have no PyMARE counterpart.

`weight_scheme="rescale"` and `"collapse"` change the weights away from
`1 / (v + tau^2)`, which is the weight matrix clubSandwich reads off the
`rma.uni` fit. `validation/robumeta` is the reference for those.
