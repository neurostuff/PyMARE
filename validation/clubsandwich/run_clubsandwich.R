# Reference values for PyMARE's cluster-robust covariance and its degrees of freedom.
#
# Writes pymare/tests/data/clubsandwich_reference.json, which
# pymare/tests/test_clubsandwich_alignment.py reads. Run it through the harness
# in this directory rather than directly, so the R, metafor and clubSandwich
# versions are the pinned ones:
#
#     validation/clubsandwich/regenerate.sh
#
# clubSandwich is not a test dependency, so the numbers are pinned rather than
# recomputed on every test run. The alignment workflow regenerates them and
# fails on any difference beyond the tolerances in
# validation/compare_reference.py, which is what keeps the pin honest.
#
# Why clubSandwich and not robumeta, which validation/robumeta already covers:
# the two answer different questions about the same estimator. robumeta is the
# reference for the correlated-effects *working model* -- how weight is spread
# across a study's rows -- which PyMARE spells weight_scheme="rescale", and it
# assumes the sampling variances are constant within a study. clubSandwich is
# the reference for the CR2 *residual adjustment* and the Satterthwaite degrees
# of freedom it needs, under whatever weights, and is the implementation the
# CRn names in pymare.stats.cluster_robust_cov are taken from.
#
# The input is the robumeta CSV because it carries two variance columns for the
# same effects: one constant within each study and one varying. That pair is
# what separates the case where PyMARE and clubSandwich agree exactly from the
# case where they do not -- see the README in this directory.
#
# The output is written by hand rather than with jsonlite so the formatting is
# byte-stable: every number goes through "%.17g", which round-trips a double
# exactly.
library(metafor)
library(clubSandwich)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "/data/robumeta_correlated_effects.csv"
out_path <- if (length(args) >= 2) args[[2]] else "/data/clubsandwich_reference.json"

d <- read.csv(csv_path)

# The moderator columns each model adds beside the intercept, named rather than
# passed as a formula so the column order is explicit: coef_test reports the
# intercept first and so does pymare.core.Dataset, so the vectors line up
# position by position. The same three models validation/robumeta uses.
mods <- list(intercept = character(0), within = "within", both = c("within", "between"))

# Only the estimators whose tau^2 PyMARE reaches in closed form, so that nothing
# but the covariance sits between the two implementations. "FE" is the
# fixed-effects model, which PyMARE spells WeightedLeastSquares(tau2=0).
methods <- c("FE", "DL")

variance_columns <- c("var_constant_within_study", "var_within_study")

cases <- expand.grid(
  variances = variance_columns, model = names(mods), method = methods,
  stringsAsFactors = FALSE
)

vector_json <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

scalar_json <- function(x) {
  if (length(x) == 0 || is.null(x) || all(is.na(x))) "null" else sprintf("%.17g", x[[1]])
}

lines <- c(
  "{",
  '  "source": {',
  sprintf('    "data": "%s",', basename(csv_path)),
  paste0(
    '    "call": "coef_test(rma.uni(effect, <variances>, mods = <model>, ',
    'method = <method>), vcov = \\"CR2\\", cluster = study)",'
  ),
  sprintf('    "metafor_version": "%s",', as.character(packageVersion("metafor"))),
  sprintf(
    '    "clubSandwich_version": "%s",',
    as.character(packageVersion("clubSandwich"))
  ),
  sprintf('    "r_version": "%s"', paste(R.version$major, R.version$minor, sep = ".")),
  "  },",
  '  "cases": ['
)

for (i in seq_len(nrow(cases))) {
  case <- cases[i, ]
  columns <- mods[[case$model]]

  # rma.uni rejects a NULL passed through a variable, so the intercept-only
  # model has to omit the argument rather than pass nothing to it.
  fit <- suppressWarnings(if (length(columns) == 0) {
    rma.uni(yi = d$effect, vi = d[[case$variances]], method = case$method)
  } else {
    rma.uni(
      yi = d$effect, vi = d[[case$variances]],
      mods = as.matrix(d[, columns, drop = FALSE]), method = case$method
    )
  })

  # vcov = "CR2" and the Satterthwaite degrees of freedom together, which is
  # what PyMARE reports when group labels are supplied: coef_test's p_Satt is
  # the two-sided t p-value on df_Satt, and its SE is the square root of the
  # CR2 covariance's diagonal.
  test <- coef_test(fit, vcov = "CR2", cluster = d$study)

  # The full CR2 covariance, not only its diagonal: PyMARE returns a matrix and
  # the off-diagonal entries are what the Satterthwaite degrees of freedom of a
  # linear combination would rest on, so pinning them costs nothing and closes
  # off a way for the two to agree on every standard error while disagreeing
  # about the covariance.
  covariance <- as.matrix(vcovCR(fit, cluster = d$study, type = "CR2"))

  lines <- c(
    lines,
    sprintf(
      '    {"variances": "%s", "model": "%s", "method": "%s",',
      case$variances, case$model, case$method
    ),
    sprintf('     "tau2": %s,', scalar_json(fit$tau2)),
    sprintf('     "beta": [%s],', vector_json(test$beta)),
    sprintf('     "se": [%s],', vector_json(test$SE)),
    sprintf('     "dof": [%s],', vector_json(test$df_Satt)),
    sprintf('     "pval": [%s],', vector_json(test$p_Satt)),
    # Row-major, which is how numpy.reshape will read it back.
    sprintf('     "cov": [%s],', vector_json(as.vector(t(covariance)))),
    sprintf(
      '     "n_groups": %.17g}%s',
      length(unique(d$study)),
      if (i < nrow(cases)) "," else ""
    )
  )
}

lines <- c(lines, "  ]", "}")
writeLines(lines, out_path)
cat(sprintf("wrote %d cases to %s\n", nrow(cases), out_path))
