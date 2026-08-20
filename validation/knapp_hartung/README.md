# Validating the Knapp-Hartung default

Why `small_sample_correction="knapp-hartung"` is on by default, measured rather than argued.

```bash
make validate_knapp_hartung      # ~20 min; exits non-zero if the claim fails
```

Self-contained: nothing in `pymare` imports from here and no measurement is pinned
into the test suite. These are Monte Carlo estimates, so re-measuring is the
honest check and a pinned number would move on every run. `validation/metafor`
answers the different question of whether PyMARE computes the *same* adjustment
metafor does -- it does, to machine precision, which says nothing about whether
the adjustment is worth having on.

## The grid

144 cells, 10,000 replications each, so the Monte Carlo standard error at
alpha = 0.05 is 0.0022. Replications go on PyMARE's parallel-dataset axis, which
is what makes the grid affordable: one fit covers a whole cell.

| Knob | Values | Why |
| --- | --- | --- |
| observations `K` | 5, 10, 20, 40 | 5 is where the uncorrected test is visibly wrong; 40 is where it is nearly right |
| predictors `P` | 1, 2, 3 | `P > 1` tests a *moderator* coefficient, the meta-regression case |
| `max(v)/min(v)` | 1, 10, 100, 10,000 | the known failure mode is few observations of very unequal precision |
| tau^2 | 0, 0.5, 2 x mean(v) | tau^2 = 0 is where the Wald test is provably correct, so it measures the cost |

Crossed with all three `small_sample_correction` values and with two tau^2 estimators
(DerSimonian-Laird and REML), because a result holding for a moment estimator but
not a likelihood estimator would not support a default applying to both.

## What was measured

Median Type I error at alpha = 0.05, DerSimonian-Laird, by observation count and
by weight ratio:

| `K` | `wald` | `knapp-hartung` | `conservative` |
| --- | --- | --- | --- |
| 5 | 0.098 | **0.051** | 0.005 |
| 10 | 0.083 | **0.053** | 0.043 |
| 20 | 0.065 | **0.051** | 0.046 |
| 40 | 0.058 | **0.051** | 0.047 |

| `max(v)/min(v)` | `wald` | `knapp-hartung` | `conservative` |
| --- | --- | --- | --- |
| 1 | 0.058 | **0.050** | 0.036 |
| 10 | 0.064 | **0.052** | 0.039 |
| 100 | 0.071 | **0.054** | 0.044 |
| 10,000 | 0.089 | **0.057** | 0.048 |

95% interval coverage, median over all cells and both estimators: 0.936 for
`wald`, **0.949** for `knapp-hartung`, 0.957 for the conservative variant.

### The result that decides the default

**`knapp-hartung` is at least as close to nominal as `wald` in every one of the
144 cells**,
for both estimators, to within Monte Carlo noise: the worst excess measured is
+0.0012 against a standard error of 0.0022. There is no cell in this grid where
turning the correction on costs anything.

That includes the cell where the Wald test is provably right -- variances known,
tau^2 truly zero. There `wald` measures 0.038 (conservative, because tau^2 is
*estimated* and lands at zero much of the time) and `knapp-hartung` 0.049. The adjustment
is not merely cheap there; it is closer to nominal. Against that, at `K = 5` the
uncorrected test rejects at 0.098 on median and 0.310 in the worst cell of the
grid. Near-zero cost when unnecessary, large gain when necessary.

### Where it still does not hold

The correction halves the error in the worst corner but does not fix it:

| cell | `wald` | `knapp-hartung` | `conservative` |
| --- | --- | --- | --- |
| `K=5 P=3 ratio=100 tau2=2` | 0.308 | 0.203 | 0.142 |
| `K=5 P=3 ratio=10000 tau2=0.5` | 0.291 | 0.168 | 0.122 |
| `K=5 P=1 ratio=10000 tau2=0.5` | 0.238 | 0.122 | 0.105 |

Five observations and three predictors leaves two residual degrees of freedom,
with one observation carrying most of the weight. No reference distribution
rescues that, and the worst case improves monotonically with `K - P`: the maximum
over cells falls from 0.203 at `K - P = 2` to 0.058 at `K - P = 37`. This is the
failure mode IntHout, Ioannidis & Borm (2014) and Röver, Knapp & Friede (2015)
report, and the reason the conservative variant is offered.

### Why the conservative variant is offered rather than defaulted to

Flooring the scale factor at 1 helps in exactly those anti-conservative cells and
overcorrects badly everywhere else: median 0.005 at `K = 5` -- a hundredth of
nominal -- below 0.02 in a quarter of all cells, minimum 0.000. The right choice
when precisions are very unequal and observations few; the wrong default.

## What `--check` enforces

Three thresholds, each the reason for one decision, in `THRESHOLDS` at the top of
`simulate.py`.

| Threshold | The decision it defends |
| --- | --- |
| `max_knha_excess_over_wald` | the default. No cell may pay more than 0.005 for having the correction on. |
| `well_conditioned_bounds` | the docstring's claim, for `K >= 20`: `knapp-hartung` inside [0.03, 0.08] at every weight ratio. |
| `max_conservative_floor` | why the conservative variant is not the default. Some cell must still collapse below 0.02. |

## One honest limit

The errors are normal and the sampling variances are treated as known and exact,
which is the model Knapp and Hartung assume. Nothing here speaks to skewed effect
sizes, to sampling variances estimated with their own error, or to
`SampleSizeBasedLikelihoodEstimator`, whose `v = sigma^2 / n` is built from an
estimated `sigma^2` rather than supplied. That estimator gets the same default on
the same reasoning, without the same measurement behind it.
