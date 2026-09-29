# Reference values for PyMARE's effect-size converters.
#
# Writes pymare/tests/data/metafor_escalc_reference.json, which
# pymare/tests/test_metafor_escalc.py reads. Run it through the harness in this
# directory rather than directly, so the R and metafor versions are the pinned
# ones:
#
#     validation/metafor/regenerate.sh
#
# metafor is not a test dependency, so the numbers are pinned rather than
# recomputed on every test run. The alignment workflow regenerates them and
# fails on any difference beyond the tolerances in
# validation/compare_reference.py, which is what keeps the pin honest.
#
# The output is written by hand rather than with jsonlite so the formatting is
# byte-stable: every number goes through "%.17g", which round-trips a double
# exactly.
#
# Which escalc measure corresponds to which PyMARE measure is not always
# obvious, so each one is named here with the PyMARE spelling it is the
# reference for:
#
#   escalc measure   PyMARE measure                      converter
#   --------------   ---------------------------------   ----------
#   MN               RM   (raw mean)                     one-sample
#   COR              R    (raw correlation)              one-sample
#   ZCOR             ZR   (Fisher z correlation)         one-sample
#   SMCC, see below  SM   (standardized mean)            one-sample
#   MD               RMD  (raw mean difference)          two-sample
#   SMD              SMD  (standardized mean difference) two-sample
#
# metafor has no single-group standardized mean, so SM is referenced against
# measure="SMCC" -- the standardized mean *change* -- with the second
# measurement set to zero and uncorrelated with the first. SMCC's change score
# standard deviation is then sd1 and its numerator is m1, so the measure reduces
# algebraically to the single-group m / sd with metafor's exact bias correction
# applied on n - 1 degrees of freedom. That is precisely the quantity PyMARE
# calls SM, which is why the substitution is a reference and not an
# approximation.
#
# PyMARE's "D" (two-sample) and one-sample Cohen's d have no escalc counterpart:
# every standardized measure metafor offers is bias-corrected. They are
# recoverable as yi / cm, and the exact correction factors are recorded below
# for that reason as well as to bound PyMARE's approximation of them.
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "/data/metafor_escalc_inputs.csv"
out_path <- if (length(args) >= 2) args[[2]] else "/data/metafor_escalc_reference.json"

d <- read.csv(csv_path)

# metafor's exact bias correction, c(m) = gamma(m/2) / (sqrt(m/2) gamma((m-1)/2)),
# through lgamma so that it does not overflow at the large sample sizes in the
# input. PyMARE approximates this with 1 - 3/(4m - 1); recording the exact value
# is what lets the alignment test bound that approximation rather than merely
# observe it.
cm <- function(m) exp(lgamma(m / 2) - log(sqrt(m / 2)) - lgamma((m - 1) / 2))

one <- escalc(measure = "MN", mi = d$m, sdi = d$sd, ni = d$n)
cor_raw <- escalc(measure = "COR", ri = d$r, ni = d$n)
cor_z <- escalc(measure = "ZCOR", ri = d$r, ni = d$n)

# The SMCC reduction described in the header comment. m2i and sd2i are zero and
# ri is zero, so sddi = sqrt(sd1^2 + 0 - 0) = sd1.
std_mean <- escalc(
  measure = "SMCC",
  m1i = d$m, m2i = rep(0, nrow(d)),
  sd1i = d$sd, sd2i = rep(0, nrow(d)),
  ni = d$n, ri = rep(0, nrow(d))
)

mean_diff <- escalc(
  measure = "MD",
  m1i = d$m1, m2i = d$m2, sd1i = d$sd1, sd2i = d$sd2, n1i = d$n1, n2i = d$n2
)
std_mean_diff <- escalc(
  measure = "SMD",
  m1i = d$m1, m2i = d$m2, sd1i = d$sd1, sd2i = d$sd2, n1i = d$n1, n2i = d$n2
)

scalar_json <- function(x) {
  if (length(x) == 0 || is.null(x) || all(is.na(x))) "null" else sprintf("%.17g", x[[1]])
}

lines <- c(
  "{",
  '  "source": {',
  sprintf('    "data": "%s",', basename(csv_path)),
  '    "call": "escalc(measure = <measure>, ...)",',
  paste0(
    '    "sm_call": "escalc(measure = \\"SMCC\\", m1i = m, m2i = 0, sd1i = sd, ',
    'sd2i = 0, ni = n, ri = 0)",'
  ),
  sprintf('    "metafor_version": "%s",', as.character(packageVersion("metafor"))),
  sprintf('    "r_version": "%s"', paste(R.version$major, R.version$minor, sep = ".")),
  "  },",
  '  "cases": ['
)

for (i in seq_len(nrow(d))) {
  lines <- c(
    lines,
    sprintf('    {"case": "%s",', d$case[[i]]),
    sprintf('     "MN": {"yi": %s, "vi": %s},', scalar_json(one$yi[i]), scalar_json(one$vi[i])),
    sprintf(
      '     "COR": {"yi": %s, "vi": %s},',
      scalar_json(cor_raw$yi[i]), scalar_json(cor_raw$vi[i])
    ),
    sprintf(
      '     "ZCOR": {"yi": %s, "vi": %s},',
      scalar_json(cor_z$yi[i]), scalar_json(cor_z$vi[i])
    ),
    sprintf(
      '     "SMCC": {"yi": %s, "vi": %s},',
      scalar_json(std_mean$yi[i]), scalar_json(std_mean$vi[i])
    ),
    sprintf(
      '     "MD": {"yi": %s, "vi": %s},',
      scalar_json(mean_diff$yi[i]), scalar_json(mean_diff$vi[i])
    ),
    sprintf(
      '     "SMD": {"yi": %s, "vi": %s},',
      scalar_json(std_mean_diff$yi[i]), scalar_json(std_mean_diff$vi[i])
    ),
    # The exact correction factors, on the degrees of freedom each measure uses:
    # n - 1 for the single-group standardized mean, n1 + n2 - 2 for the
    # two-group one.
    sprintf('     "cm_one_sample": %s,', scalar_json(cm(d$n[[i]] - 1))),
    sprintf(
      '     "cm_two_sample": %s}%s',
      scalar_json(cm(d$n1[[i]] + d$n2[[i]] - 2)),
      if (i < nrow(d)) "," else ""
    )
  )
}

lines <- c(lines, "  ]", "}")
writeLines(lines, out_path)
cat(sprintf("wrote %d cases to %s\n", nrow(d), out_path))
