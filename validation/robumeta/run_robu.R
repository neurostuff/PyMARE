# Reference values for PyMARE's correlated-effects implementation.
#
# Writes pymare/tests/data/robumeta_reference.json, which
# pymare/tests/test_robumeta_alignment.py reads. Run it through the harness in
# this directory rather than directly, so the R and robumeta versions are the
# pinned ones:
#
#     validation/robumeta/regenerate.sh
#
# robumeta is not a test dependency, so the numbers are pinned rather than
# recomputed on every test run. The alignment workflow regenerates them and
# fails on any difference, which is what keeps the pin honest.
#
# The output is written by hand rather than with jsonlite so the formatting is
# byte-stable: every number goes through "%.17g", which round-trips a double
# exactly, so regenerating an unchanged reference produces an unchanged file
# and "git diff --exit-code" means what it says.
library(robumeta)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "/data/robumeta_correlated_effects.csv"
out_path <- if (length(args) >= 2) args[[2]] else "/data/robumeta_reference.json"

d <- read.csv(csv_path)
forms <- list(intercept = effect ~ 1, within = effect ~ within, both = effect ~ within + between)
variance_columns <- c("var_constant_within_study", "var_within_study")
rhos <- c(0.0, 0.4, 0.8, 1.0)

# Every combination of the three knobs, in a fixed order so the file is stable.
cases <- expand.grid(
  model = names(forms), rho = rhos, variances = variance_columns,
  stringsAsFactors = FALSE
)

vector_json <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

lines <- c(
  "{",
  '  "source": {',
  sprintf('    "data": "%s",', basename(csv_path)),
  paste0(
    '    "call": "robu(<model>, data, studynum = study, var.eff.size = <variances>, ',
    'modelweights = \\"CORR\\", rho = <rho>, small = TRUE)",'
  ),
  sprintf('    "robumeta_version": "%s",', as.character(packageVersion("robumeta"))),
  sprintf('    "r_version": "%s"', paste(R.version$major, R.version$minor, sep = ".")),
  "  },",
  '  "cases": ['
)

for (i in seq_len(nrow(cases))) {
  case <- cases[i, ]
  d$v <- d[[case$variances]]
  fit <- robu(forms[[case$model]],
    data = d, studynum = study, var.eff.size = v,
    modelweights = "CORR", rho = case$rho, small = TRUE
  )
  lines <- c(
    lines,
    sprintf(
      '    {"model": "%s", "rho": %.1f, "variances": "%s",',
      case$model, case$rho, case$variances
    ),
    sprintf('     "tau2": %.17g,', fit$mod_info$tau.sq[[1]]),
    sprintf('     "beta": [%s],', vector_json(fit$reg_table$b.r)),
    sprintf('     "se": [%s],', vector_json(fit$reg_table$SE)),
    sprintf(
      '     "dof": [%s]}%s',
      vector_json(fit$reg_table$dfs),
      if (i < nrow(cases)) "," else ""
    )
  )
}

lines <- c(lines, "  ]", "}")
writeLines(lines, out_path)
cat(sprintf("wrote %d cases to %s\n", nrow(cases), out_path))
