# Validating the Stan meta-regression model

What `pymare/estimators/stan/meta_regression.stan` claims, and what it was
measured to do. This mirrors `validation/robumeta`, except that the reference
here is the data-generating process rather than another package: the model is
fitted to data simulated from itself, and checked for whether it recovers what
was planted and whether its credible intervals cover at their nominal rate.

Not run in CI. The fast tests in `pymare/tests/test_stan_estimators.py` check one
planted configuration; this checks the grid.

## The model

```
y_i     ~ normal(x_i' beta + theta_{g(i)}, sigma_i)   i = 1..N
theta_g ~ normal(0, tau)                              g = 1..K
tau     ~ normal(0, tau_prior_scale)  truncated at 0
beta    ~ (improper uniform)
```

`sigma_i = sqrt(v_i)` is the known sampling **standard deviation**; `tau2 =
tau^2` is the reported between-group variance. `theta` is non-centered
(`theta = tau * theta_raw`, `theta_raw ~ std_normal()`).

This is the Stan User's Guide random-effects meta-analysis model
([Measurement Error and Meta-Analysis][sug]) with that guide's stated extension
to observation-level predictors. The half-normal prior on `tau` follows Stan's
current [prior choice recommendations][priors], which supersede the `cauchy(0, 5)`
still shown in the guide.

[sug]: https://mc-stan.org/docs/stan-users-guide/measurement-error.html
[priors]: https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations

## Reproducing

```bash
pip install -e .[stan]
python -m cmdstanpy.install_cmdstan
python validation/stan/simulate.py --replications 100
```

About 10 minutes on 8 cores. Results are written to `results.json`, which is
committed, so a change to the model can be diffed against it.

## What was measured

100 replications per cell, seed 20260818, CmdStan 2.36.0, 2 chains x 1000 draws.
Each cell varies one factor away from a base of 20 groups of 3, `tau2 = 0.1`, 2
predictors, sampling SDs drawn from `uniform(0.1, 0.4)`.

Coverage is the fraction of 95% credible intervals for `beta` containing the
planted value; nominal is 0.950 and the Monte Carlo standard error is about
0.015 to 0.030. The two coverage columns are the two candidate defaults for
`tau_prior_scale` (see below).

| cell | coverage, `sqrt(mean(v))` | coverage, `max(std(y), sqrt(mean(v)))` | tau2 bias, old | tau2 bias, new | true tau2 | beta bias | fits with divergences |
| --- | --- | --- | --- | --- | --- | --- | --- |
| groups=5 | 0.910 | **0.925** | -0.002 | +0.114 | 0.10 | -0.0021 | 63 |
| groups=20 | 0.940 | **0.950** | +0.004 | +0.017 | 0.10 | -0.0029 | 0 |
| groups=50 | 0.960 | **0.955** | +0.006 | +0.010 | 0.10 | -0.0022 | 0 |
| tau2=0 | 0.970 | **0.965** | +0.004 | +0.004 | 0.00 | +0.0002 | 11 |
| tau2=0.1 | 0.930 | **0.940** | +0.007 | +0.020 | 0.10 | -0.0065 | 0 |
| tau2=1 | 0.880 | **0.935** | -0.334 | +0.085 | 1.00 | +0.0151 | 0 |
| singletons | 0.955 | **0.960** | -0.006 | +0.016 | 0.10 | +0.0068 | 3 |
| unequal groups | 0.930 | **0.935** | +0.009 | +0.023 | 0.10 | -0.0001 | 0 |
| sigma x0.1 | 0.810 | **0.925** | -0.070 | +0.016 | 0.10 | +0.0014 | 0 |
| sigma x10 | 0.935 | **0.955** | +0.379 | +0.386 | 0.10 | -0.0188 | 9 |
| 1 predictor | 0.950 | **0.950** | +0.002 | +0.010 | 0.10 | +0.0089 | 0 |
| 3 predictors | 0.943 | **0.940** | +0.004 | +0.018 | 0.10 | +0.0000 | 0 |
| unbalanced covariate | 0.965 | **0.965** | +0.008 | +0.020 | 0.10 | +0.0080 | 0 |
| unbalanced covariate, tau2=1 | 0.870 | **0.945** | -0.300 | +0.153 | 1.00 | -0.0081 | 0 |

`beta` is unbiased throughout: the largest bias in any cell is 0.019, against
coefficients of order 1.

## The choice of `tau_prior_scale`, decided by measurement

The first default tried was `sqrt(mean(v))`, the typical sampling standard
deviation. The grid rejected it. Coverage fell to **0.810** when the sampling
SDs were small relative to the between-group spread (`sigma x0.1`), to 0.880 at
`tau2=1`, and to 0.870 in the unbalanced-covariate cell at `tau2=1`. In each,
`tau2` was badly *under*-estimated: -70%, -33% and -30% respectively.

The cause is that `sqrt(mean(v))` measures sampling noise, which is not the
quantity `tau` describes. When heterogeneity is much larger than sampling error
the prior is far too tight, `tau` is shrunk toward zero, the uncertainty it
contributes to `beta` is understated, and the intervals are too narrow.

A direct comparison on the failing cells (60 replications each):

| cell | `sqrt(mean(v))` | `std(y)` | `max` of both |
| --- | --- | --- | --- |
| base tau2=0.1 | 0.883 | 0.950 | **0.950** |
| sigma x0.1 | 0.833 | 0.917 | **0.958** |
| sigma x10 | 0.975 | 0.950 | **0.967** |
| tau2=1 | 0.950 | 0.908 | **0.967** |
| tau2=0 | 0.975 | 0.975 | **0.967** |

`max(std(y), sqrt(mean(v)))` is the only candidate at or above nominal
everywhere, and it is what the estimator now uses. The reasoning the numbers
support:

- **The errors are asymmetric.** A scale that is too small costs *coverage*,
  which is a correctness failure. A scale that is too large costs only
  precision in `tau2` while coverage stays at nominal — visible in `sigma x10`,
  where `tau2` is inflated by 0.39 but coverage is 0.955. A default should
  therefore err large.
- **Both terms are needed.** `tau` is the standard deviation of the group means,
  so it cannot plausibly exceed the spread of the estimates, which makes
  `std(y)` the natural scale. But `std(y)` alone is zero when every estimate
  coincides, and zero is not a usable scale; `sqrt(mean(v))` is the floor that
  prevents it.

## Known limits

- **Five groups is not enough to identify `tau`.** At `groups=5`, 63 of 100 fits
  reported divergent transitions and `tau2` was over-estimated by 0.11 against a
  truth of 0.10. The wider prior made this worse than the rejected default did
  (20 fits), which is the honest cost of the change: with five groups the data
  say very little about the group-level variance, so the posterior follows
  whatever the prior says. The estimator warns on divergences, so this surfaces
  rather than passing silently. Prefer a non-Bayesian estimator, or supply
  `tau_prior_scale` from external knowledge, when groups are this few.
- **`tau` is not identified when sampling error dwarfs it.** In `sigma x10` the
  sampling SDs are 1 to 4 while `tau` is 0.32, and `tau2` is over-estimated by
  0.39 under every candidate prior. Nothing in the data distinguishes a small
  `tau` from zero at that noise level. Coverage for `beta` is unaffected.
- Coverage is measured for `beta` only. `tau2` is reported as bias, not
  coverage, because its posterior is strongly skewed at small `K`.

## stanc pedantic mode

`stanc --warn-pedantic` reports exactly two warnings, both expected:

```
Warning: The parameter tau has no priors. This means either no prior is
    provided, or the prior(s) depend on data variables. In the later case,
    this may be a false positive.
Warning: The parameter beta has no priors. ...
```

The first is the false positive its own text describes: `tau` does have a prior,
but its scale is a data variable, which the check cannot see through. The second
is accurate and deliberate — `beta` keeps Stan's implicit improper uniform prior,
which is what makes the posterior means agree with maximum likelihood. That
agreement is pinned by `test_matches_maximum_likelihood_without_groups`.

Because one of the two warnings is a true positive by design, the CI step that
runs pedantic mode reports rather than gates.
