# Reference values for PyMARE's permutation test.
#
# Writes pymare/tests/data/metafor_permutest_reference.json, which
# pymare/tests/test_metafor_permutest.py reads. Run it through the harness in
# this directory rather than directly, so the R and metafor versions are the
# pinned ones:
#
#     validation/metafor/regenerate.sh
#
# Only exact tests are recorded. permutest's approximate mode draws random
# permutations, so its p-values are not reproducible across runs and could not
# be pinned; with exact = TRUE it enumerates the whole permutation set and the
# p-value is a property of the data. That is also the only mode in which a
# comparison against PyMARE means anything, since PyMARE draws its own
# permutations from its own generator.
#
# Which permutation set gets enumerated depends on the model, in both
# implementations:
#
#   - Intercept-only: the 2^K assignments of sign to the K estimates.
#   - With moderators: the K! orderings. metafor permutes the rows of the
#     moderator matrix and PyMARE permutes the (y, v) pairs, which enumerate
#     the same set -- one is the other under the inverse permutation, and the
#     whole set is covered either way.
#
# So the grid stops at K = 10 for the intercept-only models and K = 5 for the
# moderator ones: 10! refits would make regenerating this file take longer than
# everything else in validation/ put together, for no coverage that 5! does not
# already give.
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "/data/metafor_small_sample.csv"
out_path <- if (length(args) >= 2) args[[2]] else "/data/metafor_permutest_reference.json"

d <- read.csv(csv_path)

# test = "z" throughout. permutest refers the permuted statistics to their own
# distribution rather than to a reference one, so the small-sample correction
# has nothing to do here, and PyMARE's permutation_test likewise reports a
# p-value that does not depend on it.
cases <- rbind(
  expand.grid(
    design = c("equal_k5", "unequal_k5", "extreme_k10"), model = "intercept",
    method = c("FE", "DL"), stringsAsFactors = FALSE
  ),
  expand.grid(
    design = c("equal_k5", "unequal_k5"), model = "one",
    method = c("FE", "DL"), stringsAsFactors = FALSE
  )
)

mods <- list(intercept = character(0), one = "mod1")

vector_json <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

lines <- c(
  "{",
  '  "source": {',
  sprintf('    "data": "%s",', basename(csv_path)),
  '    "call": "permutest(rma.uni(...), exact = TRUE)",',
  sprintf('    "metafor_version": "%s",', as.character(packageVersion("metafor"))),
  sprintf('    "r_version": "%s"', paste(R.version$major, R.version$minor, sep = ".")),
  "  },",
  '  "cases": ['
)

for (i in seq_len(nrow(cases))) {
  case <- cases[i, ]
  sub <- d[d$case == case$design, ]
  columns <- mods[[case$model]]

  fit <- suppressWarnings(if (length(columns) == 0) {
    rma.uni(yi = sub$y, vi = sub$v, method = case$method, test = "z")
  } else {
    rma.uni(
      yi = sub$y, vi = sub$v, mods = as.matrix(sub[, columns, drop = FALSE]),
      method = case$method, test = "z"
    )
  })
  perm <- suppressWarnings(permutest(fit, exact = TRUE, progbar = FALSE))

  # The size of the permutation set, recorded so the alignment test can assert
  # that PyMARE enumerated the same one rather than falling back to sampling.
  n_perm <- if (length(columns) == 0) 2^nrow(sub) else factorial(nrow(sub))

  lines <- c(
    lines,
    sprintf(
      '    {"design": "%s", "model": "%s", "method": "%s",',
      case$design, case$model, case$method
    ),
    sprintf('     "pval": [%s],', vector_json(perm$pval)),
    sprintf('     "zval": [%s],', vector_json(fit$zval)),
    sprintf('     "n_perm": %.17g}%s', n_perm, if (i < nrow(cases)) "," else "")
  )
}

lines <- c(lines, "  ]", "}")
writeLines(lines, out_path)
cat(sprintf("wrote %d cases to %s\n", nrow(cases), out_path))
