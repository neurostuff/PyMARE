# metafor reference values

[metafor](https://cran.r-project.org/package=metafor) (Viechtbauer, 2010,
*Journal of Statistical Software* 36(3), 1-48) is the reference implementation
for most of what PyMARE computes. This directory regenerates the values three
test modules pin against it:

| Reference file | metafor call | Test module |
| --- | --- | --- |
| `metafor_reference.json` | `rma.uni`, `confint` | `test_metafor_alignment.py`, `test_metafor_random_effects.py` |
| `metafor_escalc_reference.json` | `escalc` | `test_metafor_escalc.py` |
| `metafor_permutest_reference.json` | `permutest(exact = TRUE)` | `test_metafor_permutest.py` |

```bash
make check_metafor_alignment     # needs Docker; rewrites the pinned files in place
make test_metafor                # check PyMARE against the pinned values
```

Regeneration goes through a Docker image with pinned R and metafor versions.
metafor is an R package and cannot be a test dependency, which is why the numbers
are pinned rather than recomputed on every test run. The `Check alignment with R
packages` workflow regenerates them on every pull request and fails if they move,
which is what keeps a pinned file from quietly becoming a stale one. The
comparison is numeric, through the shared `validation/compare_reference.py`: the
numbers are written at full double precision and R reaches them through linear
algebra whose last bits depend on which BLAS kernel its image picks for the CPU
it runs on, so regenerating an unchanged tree on a different machine moves
numbers by up to 9.5e-14 relative on the inference path, and 3.3e-9 on an `ML`
tau^2 where an optimizer stops a step earlier or later.

The pinned files record metafor's own vocabulary -- its `test=` spellings, its
`measure=` names -- because they record what metafor was *asked*. Each test
module translates to PyMARE's names in the open rather than hiding the mapping in
the generator.

---

# `rma.uni`

## What agrees

**The Knapp-Hartung adjustment agrees to machine precision, in all 180 cases:**
1.8e-15 on the coefficients, 5.6e-16 on the standard errors, 3.0e-15 on the
p-values, and 5.9e-11 absolute on the interval bounds -- the last bounded not by
the adjustment but by `scipy.stats.t.ppf` and R's `qt` disagreeing in their final
bits. The degrees of freedom agree exactly, being a count.

That comparison supplies PyMARE with metafor's own tau^2, which is what isolates
the adjustment from the tau^2 estimators. `FE` and `DL` also agree end to end,
tau^2 included, because both reach it in closed form.

**Everything else `rma.uni` reports and PyMARE also computes agrees too**, over
the 60 distinct design x model x estimator cells:

| Quantity | PyMARE | Worst relative deviation |
| --- | --- | --- |
| `QE`, `QEp` | `get_heterogeneity_stats()["Q"]`, `["p(Q)"]` | 2.9e-14, 1.5e-13 |
| `QEp` in logs | `["logp(Q)"]` | 4.7e-14 |
| `I2`, `H2` for `FE` and `DL` | `["I^2"]`, `["H"]` | 4.9e-14, 1.5e-14 |
| `confint` tau^2 bounds | `get_re_stats()["ci_l"]`, `["ci_u"]` | 1.3e-13 |
| `HE` tau^2, intercept-only models | `Hedges().fit(...)` | 2.2e-16 |
| `ML`, `REML` tau^2 | `VarianceBasedLikelihoodEstimator` | 2.7e-5 |

Two notes on that table.

`I2` and `H2` are floored by PyMARE and not by metafor -- at 0 and 1
respectively, as Higgins & Thompson (2002) define them -- so the comparison
applies the floors to metafor's values rather than dropping the cells where they
bite. metafor can print an `H2` below one.

The `confint` reference is generated with `control = list(tol = 1e-12)`. Its
default is `uniroot`'s `.Machine$double.eps^0.25`, about 1.2e-4 relative, which
is far coarser than the bound itself; pinning the default would check PyMARE
against metafor's display precision instead of against the bound it solves for.
Finding this is also what turned up the defect below.

## What this check found

`pymare.stats.q_profile` computed both tau^2 bounds by minimizing
`(Q(tau^2) - crit)**2` with `scipy.optimize.minimize`. Squaring turns a
transversal crossing into a tangential minimum, so the gradient the minimizer
follows vanishes as the critical value is approached and it stops early in the
flat upper tail. The upper bound was out by up to 3.6e-2 relative. It now solves
for the root with `scipy.optimize.brentq` and agrees with metafor to 1.3e-13.

Both tests that pinned the old value asserted it to two decimals, which is why
nothing failed.

## What does not agree, and why

Three divergences, all in tau^2 or in quantities derived from it, and all
visible with no correction applied -- so none of them is caused by the
adjustment. They are why `test_metafor_alignment` compares `ML`, `REML` and `HE`
only through metafor's own tau^2: folding an optimizer's tolerance into a check
on a closed-form scale factor would blunt it.
`test_metafor_random_effects` covers each of them directly instead, by asserting
the *cause* rather than the size of the gap.

| Divergence | Size | Cause |
| --- | --- | --- |
| `I2`, `H2` for `HE`, `ML`, `REML` | unbounded | PyMARE always reports the Q-based Higgins-Thompson pair. metafor reports that pair only for `FE` and `DL`, where it coincides with `tau^2 / (tau^2 + v_t)`, and switches to the tau^2-based pair otherwise -- so metafor's `I2` depends on which tau^2 estimator was asked for and PyMARE's does not. Both are defensible; they are not the same number. |
| `HE` tau^2, models with moderators | up to 0.14 relative | metafor subtracts `tr(PV) / (K - P)`, PyMARE subtracts `sum(v) / K`. With an intercept as the only predictor `P = I - J/K`, the trace is `sum(v)(K - 1)/K`, and the two are algebraically the same; with a moderator they are not. So the divergence is specific to meta-regression. A previous version of this README recorded it as general. |
| `ML`, `REML` tau^2 | ~3e-5 relative | PyMARE profiles tau^2 at `xtol=1e-6`; metafor runs its own optimizer to its own tolerance. |
| `ML` on `extreme_k10` with one moderator | 0 vs 0.011 | The two searches land on opposite sides of the tau^2 = 0 boundary, where a profile likelihood is flattest because the weights are most unequal. metafor is the one that stops at zero. A previous version of this README had the direction backwards. |

## What is compared

180 cases, the full grid of design x model x tau^2 estimator x `test`.
`test_reference_covers_every_combination` asserts that grid, so the check cannot
quietly shrink to the cases that happen to pass. The heterogeneity statistics and
the tau^2 interval do not depend on `test`, which
`test_heterogeneity_does_not_depend_on_the_correction` asserts from the reference
before the other tests drop to the 60 distinct cells.

| Knob | Values |
| --- | --- |
| design | `equal_k5`, `unequal_k5`, `extreme_k10`, `moderate_k20` |
| model | `y ~ 1`, `y ~ mod1`, `y ~ mod1 + mod2` |
| tau^2 estimator | `FE`, `DL`, `HE`, `ML`, `REML` |
| `test` (metafor's spellings) | `z`, `knha`, `adhoc` |

The designs are in `metafor_small_sample.csv`, chosen to bracket the condition
that decides whether the adjustment behaves -- how unequal the weights are, and
how few observations there are:

| Design | K | max(v) / min(v) |
| --- | --- | --- |
| `equal_k5` | 5 | 1.5 |
| `unequal_k5` | 5 | 250 |
| `extreme_k10` | 10 | 10,000 |
| `moderate_k20` | 20 | 30 |

`extreme_k10` is there because IntHout, Ioannidis & Borm (2014) and Röver, Knapp
& Friede (2015) both report the adjustment exceeding its nominal level for few
observations of very unequal precision. Comparing against metafor in that cell
checks that PyMARE reproduces the reference implementation there too, including
its `test="adhoc"` remedy, which PyMARE spells
`"knapp-hartung-conservative"`. Whether the adjustment is *worth having on* is a
different question, measured in `validation/knapp_hartung`.

## What is not compared

`rma.uni`'s omnibus test of moderators (`QM`, and its shift from chi-square to F
under `test="knha"`) has no PyMARE counterpart -- PyMARE reports per-coefficient
statistics and has no joint test. `test="t"`, the t reference without the
covariance scaling, is not exposed by PyMARE either: nothing recommends it as a
default and it is not needed to reproduce `knha`.

`rma.uni`'s prediction interval (`predict`) has no PyMARE counterpart. Neither do
its other tau^2 estimators (`EB`, `PM`, `SJ`, `GENQ`) or its other confidence
interval methods for tau^2 (`confint(type = "PL")`), and PyMARE's
`SampleSizeBasedLikelihoodEstimator` has no metafor counterpart, since metafor
has no estimator that takes sample sizes in place of variances.

---

# `escalc`

`test_metafor_escalc.py` compares PyMARE's effect-size converters against
`escalc` on an eight-row grid of summary statistics in
`metafor_escalc_inputs.csv` -- balanced and unbalanced groups, n from 5 to 400,
zero to very large effects, correlations from 0 to 0.99.

Each comparison is in one of three relationships, and the test module is
explicit about which.

## The same closed form

Exact, to 1e-13:

| PyMARE | `escalc` measure | Compared |
| --- | --- | --- |
| `RM` | `MN` | estimate and variance |
| `R` | `COR` | estimate |
| `ZR` | `ZCOR` | estimate and variance |
| `RMD` | `MD` | estimate |
| `sdp` | the pooled SD `escalc` divides by, recovered as `MD$yi / (SMD$yi / c(m))` | value |

## The same quantity, one side approximated

PyMARE corrects a standardized mean for bias with `1 - 3/(4m - 1)`; metafor uses
the exact `c(m) = gamma(m/2) / (sqrt(m/2) gamma((m-1)/2))`. Rather than pin a
tolerance, the tests bound the error at `0.05 / m**2` -- the approximation's
actual second order, measured at `0.043 / m**2` over the grid, worst 2.7e-3 at
`m = 4` and 2.0e-7 at `m = 398`. A first-order error would break that bound.
This covers the `SM` and `SMD` estimates and the factor itself.

metafor has no single-group standardized mean, so `SM`'s reference is
`escalc(measure = "SMCC")` with the second measurement set to zero and
uncorrelated with the first. SMCC's change-score SD is then `sd1` and its
numerator is `m1`, so the measure reduces algebraically to `m / sd` with the
exact correction on `n - 1` degrees of freedom -- which is a reference and not an
approximation. The header of `run_escalc.R` spells this out.

## Different formulas for the same thing

Two, neither of which can be verified against the other. Each is recorded by a
test asserting which formula each side uses, so the divergence cannot change
shape unnoticed:

| Quantity | metafor | PyMARE |
| --- | --- | --- |
| raw correlation variance | `(1 - r**2)**2 / (n - 1)`, the asymptotic sampling variance | `(1 - r**2) / (n - 2)`, the squared standard error under the null of no correlation. 51x apart at `r = 0.99` |
| single-group standardized-mean variances | `1/n + y**2 / (2n)`, the large-sample approximation | the exact noncentral-t expressions, which are the better quantity. 2.5x apart at `n = 5` |

The second is why `test_standardized_mean_variance_is_the_same_order_as_metafor`
bounds by a factor of four rather than a tolerance.

## What this check found

Three defects in `pymare/effectsize/expressions.json`, beyond the one
[PR #144](https://github.com/neurostuff/PyMARE/pull/144) fixes, all of the same
kind: a missing pair of parentheses changing what the expression solves to.

| Expression | Reads | Solves to | Should be | Effect |
| --- | --- | --- | --- | --- |
| `v_rmd` | `v_rmd - (sd1**2 / n1) + (sd2**2 / n2)` | `sd1**2/n1 - sd2**2/n2` | the sum | negative variance whenever the second group is the more variable one, and exactly zero for two equally sized equally variable groups -- which gives that study infinite weight |
| `v_sm` | `... * (1/n + d**2) - d**2` | `A + d**2` | `A - d**2` | the single-group Hedges' g variance is 9x to 110x too large, and grows with the effect instead of being dominated by `1/n` |
| `v_d` (one-sample) | `... - d**2 / j**2 * n` | `A + n d**2 / j**2` | `A - d**2 / j**2` | the variance grows with the sample size |

Each is recorded as an `xfail(strict=True)` naming the expression and what it
should be, alongside the two-sample `v_d` that PR #144 fixes. `strict` is the
point: correcting an expression turns the test green, pytest reports XPASS as a
failure, and the marker has to go in the same change. All four were confirmed to
flip to XPASS under the corresponding one-line fix, and under all four together
the rest of the suite still passes -- so nothing currently pins the wrong values.

## What is not compared

`escalc`'s binary-outcome measures (`RR`, `OR`, `RD`, `PETO`, ...), its
proportion and rate measures, and its other standardized measures (`SMDH`,
`SMD1`, `SMCR`, `ROM`, ...) have no PyMARE counterpart. PyMARE's two-sample `D`
and one-sample `D` have no `escalc` counterpart either, every standardized
measure metafor offers being bias-corrected; they are recoverable as `yi / c(m)`,
which is how the one-sample `D` variance above is compared at all.

---

# `permutest`

`test_metafor_permutest.py` compares PyMARE's permutation test against
`permutest(exact = TRUE)` on ten cases: the designs small enough to enumerate --
`2**K` sign flips for the intercept-only models up to K = 10, and `K!` orderings
for the moderator ones at K = 5. Only the exact mode, since the approximate one
draws from each package's own generator.

## What agrees

Counting the same permutations of the same PyMARE fits on `|beta / se|` instead
of `|beta|`, and reading the observed statistic out of the identity permutation
in the same batch, reproduces `permutest` **exactly in all ten cases**, both
coefficients of the moderator models included. So the permutation sets agree, the
batched refits agree, and the inclusive comparison agrees.

## What does not, and why

What PyMARE actually reports does not, for two reasons.

**The statistic.** `permutest` counts `|beta / se|`; PyMARE counts `|beta|`. A
permutation test needs a statistic whose null distribution does not move with
what is being permuted away, and `|beta|` does: refitting a permuted dataset
re-estimates tau^2, which changes the weights and so the standard error. The two
coincide only where the standard error is invariant under the permutation -- a
fixed-effects intercept-only model under sign flipping -- which is why four of
the ten cases agree anyway. On `unequal_k5` under `DL`, PyMARE reports 0.5625
against `permutest`'s 0.5.

**The tie.** The observed estimate is computed by a different code path from the
permuted ones and the two differ by one unit in the last place, so the inclusive
comparison can drop the identity permutation -- the one that reproduces the
observed data and must therefore count -- along with its sign-flipped mirror.
That understates the p-value by `2 / 2**K` whenever it bites: 0.033203125 against
`permutest`'s 0.03515625 on `extreme_k10`. metafor avoids this by comparing
against `|zval| - sqrt(eps)`.

Both are recorded by one strict `xfail` covering the whole grid rather than a
parametrized one, because the second cause is a last-place rounding difference
that need not reproduce on every platform in the test matrix while the first is
structural.

## What is not compared

`permutest`'s `permci = TRUE`, which inverts the permutation test for a
confidence interval, has no PyMARE counterpart. Neither does its omnibus
`QM` permutation p-value, for the same reason `rma.uni`'s `QM` is not compared.
PyMARE's tau^2 permutation p-value (`perm_p["tau2_p"]`) has no `permutest`
counterpart.
