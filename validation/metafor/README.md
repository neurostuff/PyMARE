# metafor reference values

PyMARE's `small_sample_correction` parameter implements the Knapp-Hartung adjustment (Knapp &
Hartung, 2003, *Statistics in Medicine* 22(17), 2693-2710) in the form `metafor`'s
`rma.uni` applies it, including the modification metafor spells `test="adhoc"` and PyMARE
`"knapp-hartung-conservative"`. This
directory regenerates the reference values `pymare/tests/test_metafor_alignment.py`
pins.

```bash
make check_metafor_alignment     # needs Docker; rewrites the pinned file in place
make test_metafor                # check PyMARE against the pinned values
```

Regeneration goes through a Docker image with pinned R and metafor versions, so
unchanged numbers produce an unchanged file and the diff is the answer. metafor is
an R package and cannot be a test dependency, which is why the numbers are pinned
rather than recomputed on every test run. Mirrors `validation/robumeta`, except
that there is no scheduled workflow re-verifying the pin against metafor; adding
one means copying `robumeta-alignment.yml` and generalising the numeric comparison
in `validation/robumeta/compare_reference.py`, which is a follow-up rather than
part of this change.

The pinned file records metafor's own `test=` spellings, because it records what
metafor was asked; `CORRECTIONS` in the alignment module translates them to
PyMARE's `small_sample_correction` values.

## What agrees

**The adjustment itself agrees to machine precision, in all 180 cases:** 1.8e-15
on the coefficients, 5.6e-16 on the standard errors, 3.0e-15 on the p-values, and
5.9e-11 absolute on the interval bounds -- the last bounded not by the adjustment
but by `scipy.stats.t.ppf` and R's `qt` disagreeing in their final bits. The
degrees of freedom agree exactly, being a count.

That comparison supplies PyMARE with metafor's own tau^2, which is what isolates
the adjustment from the tau^2 estimators. `FE` and `DL` also agree end to end,
tau^2 included, because both reach it in closed form.

## What does not, and why

Three divergences, all in tau^2 and all visible with no correction applied, so none of them is
caused by the adjustment. They are why the alignment tests compare `ML`, `REML`
and `HE` only through metafor's own tau^2 -- folding an optimizer's tolerance into
a check on a closed-form scale factor would blunt it.

| Divergence | Size | Cause |
| --- | --- | --- |
| `ML`, `REML` tau^2 | ~3e-5 relative | PyMARE profiles tau^2 at `xtol=1e-6`; metafor runs its own optimizer to its own tolerance |
| `ML` on `extreme_k10` with one moderator | 0 vs 0.011 | the two optimizers land on opposite sides of the tau^2 = 0 boundary, where a profile likelihood is flattest because the weights are most unequal |
| `HE` tau^2 | up to 0.08 absolute | PyMARE and metafor differ on whether the mean sampling variance is taken over raw or weighted rows; predates this work by years, and `test_hedges_estimator` has recorded it since PyMARE's Hedges estimator was written |

## What is compared

180 cases, the full grid of design x model x tau^2 estimator x `test`.
`test_reference_covers_every_combination` asserts that grid, so the check cannot
quietly shrink to the cases that happen to pass.

| Knob | Values |
| --- | --- |
| design | `equal_k5`, `unequal_k5`, `extreme_k10`, `moderate_k20` |
| model | `y ~ 1`, `y ~ mod1`, `y ~ mod1 + mod2` |
| tau^2 estimator | `FE`, `DL`, `HE`, `ML`, `REML` |
| `test` (metafor's spellings) | `z`, `knha`, `adhoc` |

The designs are in `pymare/tests/data/metafor_small_sample.csv`, chosen to bracket
the condition that decides whether the adjustment behaves -- how unequal the
weights are, and how few observations there are:

| Design | K | max(v) / min(v) |
| --- | --- | --- |
| `equal_k5` | 5 | 1.5 |
| `unequal_k5` | 5 | 250 |
| `extreme_k10` | 10 | 10,000 |
| `moderate_k20` | 20 | 30 |

`extreme_k10` is there because IntHout, Ioannidis & Borm (2014) and Röver, Knapp &
Friede (2015) both report the adjustment exceeding its nominal level for few
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
