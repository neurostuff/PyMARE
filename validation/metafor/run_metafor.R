# Reference values for PyMARE's Knapp-Hartung adjustment.
#
# Writes pymare/tests/data/metafor_reference.json, which
# pymare/tests/test_metafor_alignment.py reads. Run it through the harness in
# this directory rather than directly, so the R and metafor versions are the
# pinned ones:
#
#     validation/metafor/regenerate.sh
#
# metafor is not a test dependency, so the numbers are pinned rather than
# recomputed on every test run. The alignment workflow regenerates them and
# fails on any difference, which is what keeps the pin honest.
#
# The output is written by hand rather than with jsonlite so the formatting is
# byte-stable: every number goes through "%.17g", which round-trips a double
# exactly, so regenerating an unchanged reference produces an unchanged file and
# "git diff --exit-code" means what it says.
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "/data/metafor_small_sample.csv"
out_path <- if (length(args) >= 2) args[[2]] else "/data/metafor_reference.json"

d <- read.csv(csv_path)

# The four designs bracket the condition the literature says decides whether the
# adjustment behaves: how unequal the weights are, and how few observations
# there are. See the CSV's own header row for the variance ranges.
design_names <- unique(d$case)

# The moderator columns each model adds beside the intercept. Named rather than
# passed as a formula so the column order is explicit: metafor puts the intercept
# first, and so does pymare.core.Dataset, so the coefficient vectors line up
# position by position and can be compared without matching names.
mods <- list(intercept = character(0), one = "mod1", two = c("mod1", "mod2"))

# tau^2 estimators, in metafor's names. "FE" is the fixed-effects model, which
# PyMARE spells WeightedLeastSquares(tau2=0); the rest all estimate tau^2.
methods <- c("FE", "DL", "HE", "ML", "REML")

tests <- c("z", "knha", "adhoc")

cases <- expand.grid(
  design = design_names, model = names(mods), method = methods, test = tests,
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
    '    "call": "rma.uni(y, v, mods = <model>, data = <design>, ',
    'method = <method>, test = <test>)",'
  ),
  sprintf('    "metafor_version": "%s",', as.character(packageVersion("metafor"))),
  sprintf('    "r_version": "%s"', paste(R.version$major, R.version$minor, sep = ".")),
  "  },",
  '  "cases": ['
)

for (i in seq_len(nrow(cases))) {
  case <- cases[i, ]
  sub <- d[d$case == case$design, ]

  # metafor warns that the Knapp-Hartung method is not meant for a
  # fixed-effects model, and PyMARE's WeightedLeastSquares defaults to test="z"
  # for the same reason. The combination is still recorded, because a user who
  # asks for it explicitly should get metafor's answer.
  columns <- mods[[case$model]]
  # rma.uni rejects a NULL passed through a variable, so the intercept-only
  # model has to omit the argument rather than pass nothing to it.
  fit <- suppressWarnings(if (length(columns) == 0) {
    rma.uni(
      yi = sub$y, vi = sub$v,
      method = case$method, test = case$test
    )
  } else {
    rma.uni(
      yi = sub$y, vi = sub$v, mods = as.matrix(sub[, columns, drop = FALSE]),
      method = case$method, test = case$test
    )
  })

  lines <- c(
    lines,
    sprintf(
      '    {"design": "%s", "model": "%s", "method": "%s", "test": "%s",',
      case$design, case$model, case$method, case$test
    ),
    sprintf('     "tau2": %s,', scalar_json(fit$tau2)),
    sprintf('     "beta": [%s],', vector_json(as.vector(fit$beta))),
    sprintf('     "se": [%s],', vector_json(fit$se)),
    sprintf('     "pval": [%s],', vector_json(fit$pval)),
    sprintf('     "ci_lb": [%s],', vector_json(fit$ci.lb)),
    sprintf('     "ci_ub": [%s],', vector_json(fit$ci.ub)),
    sprintf(
      '     "dof": %s}%s',
      scalar_json(fit$ddf),
      if (i < nrow(cases)) "," else ""
    )
  )
}

lines <- c(lines, "  ]", "}")
writeLines(lines, out_path)
cat(sprintf("wrote %d cases to %s\n", nrow(cases), out_path))
