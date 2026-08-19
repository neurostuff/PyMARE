# Validating the Stan meta-regression model

What `pymare/estimators/stan/meta_regression.stan` claims, and what it was
measured to do. This mirrors `validation/robumeta`, except that the reference
here is the data-generating process rather than another package: the model is
fitted to data simulated from itself, and checked for whether it recovers what
was planted and whether its credible intervals cover at their nominal rate.

The fast tests in `pymare/tests/test_stan_estimators.py` check one planted
configuration; this checks the grid. It is not run per pull request, but it is
not detached from CI either — see "How this is wired into CI" below.

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

## How this is wired into CI

Three layers, mirroring the robumeta alignment in `validation/robumeta`:

1. **`make validate_stan`** re-measures the grid and fails if any design cell
   misses `pymare.tests.utils.STAN_VALIDATION_THRESHOLDS`. It writes the results
   to `pymare/tests/data/stan_validation.json`.
2. **`test_recorded_validation_meets_its_thresholds`** and
   **`test_recorded_validation_covers_every_design_cell`** hold that recorded
   file to the same thresholds and to the expected list of cells. They read the
   file rather than re-measuring, so they cost nothing and run on every pull
   request on every platform. This is what stops the record from going stale
   unnoticed — a pinned file nothing reads is decoration.
3. **The `Validate the Stan model` workflow** re-measures on a schedule, on
   pushes to master touching the model, and on demand.

The grid is about 1400 fits and takes ten minutes, which is why it is not part
of `test_stan` and not run per pull request.

**Why thresholds rather than pinned values.** The robumeta reference is
deterministic, so its workflow can require the file not to move. These numbers
are Monte Carlo estimates with a standard error of 0.015 to 0.030, so a correct
model produces different numbers every run and an exact pin would fail
constantly. What is pinned instead is the claim the file exists to support:
worst-coefficient coverage at or above 0.85 and |beta bias| at or below 0.10 in every cell. That
floor is set from measurement: the correct model's tightest cell reads 0.900 and
the rejected prior reads 0.710 and 0.830, so 0.85 sits between them — about two
standard errors below the worst honest result and clear of the failures. A
minimum over coefficients is biased downward, which is why the floor is not
nearer the nominal 0.95.
`--check` refuses to certify a run of fewer than 100 replications, so a short run
cannot clear the floor by luck.

## Reproducing

```bash
pip install -e .[stan]
python -m cmdstanpy.install_cmdstan
make validate_stan          # or: python validation/stan/simulate.py --check
```

About 10 minutes on 8 cores.

## What was measured

100 replications per cell, seed 20260818, CmdStan 2.36.0, 2 chains x 1000 draws.
Each cell varies one factor away from a base of 20 groups of 3, `tau2 = 0.1`, 2
predictors, sampling SDs drawn from `uniform(0.1, 0.4)`.

The true coefficients are **fixed** at `[0.5, -0.8, 0.3]`, truncated to the
number of predictors, rather than redrawn each replication. That matters: with a
symmetric redrawn truth the signed errors average to zero for any estimator at
all, so the bias threshold would certify nothing — an estimator that always
returned zero cleared it about 85% of the time.

Coverage is reported **per coefficient**, and the threshold is applied to the
**worst** of them rather than the average. Averaging lets a well-estimated
intercept mask a badly estimated moderator, which is the failure the unbalanced
cells exist to detect — and it did mask it: the rejected prior scale below reads
0.810 pooled but 0.710 on its worst coefficient.

The two coverage columns are the two candidate defaults for `tau_prior_scale`,
both measured under this per-coefficient metric.

| cell | worst cov, `sqrt(mean(v))` | worst cov, `max(std(y), sqrt(mean(v)))` | per coefficient | tau2 bias, old | tau2 bias, new | true tau2 | worst beta bias | divergent fits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groups=5 | 0.910 | **0.930** | 0.93, 0.93 | +0.000 | +0.122 | 0.10 | +0.0092 | 54 |
| groups=20 | 0.900 | **0.900** | 0.91, 0.90 | -0.002 | +0.010 | 0.10 | -0.0057 | 0 |
| groups=50 | 0.960 | **0.960** | 0.99, 0.96 | +0.004 | +0.008 | 0.10 | -0.0047 | 0 |
| tau2=0 | 0.960 | **0.960** | 0.97, 0.96 | +0.004 | +0.004 | 0.00 | +0.0048 | 14 |
| tau2=0.1 | 0.930 | **0.930** | 0.94, 0.93 | +0.003 | +0.018 | 0.10 | -0.0036 | 0 |
| tau2=1 | 0.910 | **0.960** | 0.96, 0.97 | -0.285 | +0.185 | 1.00 | +0.0103 | 0 |
| singletons | 0.940 | **0.950** | 0.98, 0.95 | -0.003 | +0.021 | 0.10 | -0.0136 | 1 |
| unequal groups | 0.930 | **0.940** | 0.94, 0.99 | +0.008 | +0.024 | 0.10 | -0.0042 | 0 |
| sigma x0.1 | 0.710 | **0.940** | 0.94, 0.95 | -0.069 | +0.021 | 0.10 | +0.0027 | 0 |
| sigma x10 | 0.940 | **0.940** | 0.98, 0.94 | +0.416 | +0.421 | 0.10 | -0.0450 | 6 |
| 1 predictor | 0.940 | **0.940** | 0.94 | +0.000 | +0.008 | 0.10 | -0.0045 | 0 |
| 3 predictors | 0.920 | **0.920** | 0.94, 0.92, 0.99 | +0.005 | +0.021 | 0.10 | -0.0099 | 0 |
| unbalanced covariate | 0.960 | **0.950** | 0.97, 0.95 | +0.008 | +0.020 | 0.10 | -0.0086 | 0 |
| unbalanced covariate, tau2=1 | 0.830 | **0.930** | 0.98, 0.93 | -0.346 | +0.049 | 1.00 | -0.0242 | 0 |

`beta` is unbiased throughout: the largest bias on any coefficient in any cell
is 0.045, against coefficients of order 1.

## The choice of `tau_prior_scale`, decided by measurement

The first default tried was `sqrt(mean(v))`, the typical sampling standard
deviation. The grid rejected it. Worst-coefficient coverage fell to **0.710**
when the sampling SDs were small relative to the between-group spread
(`sigma x0.1`) and to **0.830** in the unbalanced-covariate cell at `tau2=1`. In
each, `tau2` was badly *under*-estimated: -69% and -35% respectively.

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
