.PHONY: all_tests benchmark check_metafor_alignment check_robumeta_alignment help
.PHONY: install_cmdstan lint test_metafor test_robumeta test_stan unittest
.PHONY: validate_knapp_hartung validate_stan

# --cov-append matches what CI does, so a local run of two targets in a row
# reports their combined coverage rather than only the last one's.
PYTEST_COV := --cov-append --cov-report=xml --cov=pymare

all_tests: lint unittest test_stan test_robumeta test_metafor test_clubsandwich

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint                       to run flake8 over pymare and the benchmarks"
	@echo "  unittest                   to run every test except the Stan sampling ones"
	@echo "  install_cmdstan            to install the CmdStan that test_stan needs"
	@echo "  test_stan                  to run the Stan sampling tests (needs the stan extra and CmdStan)"
	@echo "  test_robumeta              to run the robumeta alignment tests"
	@echo "  test_metafor               to run the metafor alignment tests"
	@echo "  test_clubsandwich          to run the clubSandwich alignment tests"
	@echo "  check_robumeta_alignment   to regenerate the robumeta reference values (needs Docker)"
	@echo "  check_metafor_alignment    to regenerate the metafor reference values (needs Docker)"
	@echo "  check_clubsandwich_alignment  to regenerate the clubSandwich reference values (needs Docker)"
	@echo "  validate_stan              to re-measure the Stan model's bias and coverage (~10 min)"
	@echo "  validate_knapp_hartung     to re-measure the small-sample corrections (~20 min)"
	@echo "  benchmark                  to run the asv suite once in the current environment"
	@echo "  all_tests                  to run lint and every test target"

lint:
	@flake8 pymare benchmarks

unittest:
	@python -m pytest -m "not stan" $(PYTEST_COV)

# CmdStan is a C++ build rather than a Python package, so `pip install -e .[stan]`
# gets cmdstanpy but not the CmdStan it drives. This is the missing second step.
install_cmdstan:
	@python -m cmdstanpy.install_cmdstan

# Skips rather than fails when CmdStan is absent. CI sets PYMARE_REQUIRE_CMDSTAN=1
# so that a job which is supposed to have it goes red instead of quietly empty.
test_stan:
	@python -m pytest -m "stan" $(PYTEST_COV)

test_robumeta:
	@python -m pytest -m "robumeta" $(PYTEST_COV)

test_metafor:
	@python -m pytest -m "metafor" $(PYTEST_COV)

test_clubsandwich:
	@python -m pytest -m "clubsandwich" $(PYTEST_COV)

# Re-measures the Type I error of the small-sample corrections and fails if any cell
# misses the thresholds the default rests on. Not wired into CI and nothing is
# pinned from it: these are Monte Carlo estimates, so re-measuring is the honest
# check. Slow -- about twenty minutes for 144 cells at 10,000 replications.
validate_knapp_hartung:
	@python validation/knapp_hartung/simulate.py --check

# What the "Validate the Stan model" workflow runs. Regenerates
# pymare/tests/data/stan_validation.json and fails if any design cell misses the
# thresholds in pymare.tests.utils.STAN_VALIDATION_THRESHOLDS. Slow -- about ten
# minutes -- which is why it is not part of unittest or test_stan.
validate_stan:
	@python validation/stan/simulate.py --check

# What the alignment workflows run. Needs Docker, because the reference values
# come from R.
#
# Each target regenerates its package's reference files in place and then
# compares them against the pinned copies from git. Numerically, through
# validation/compare_reference.py, rather than with `git diff --exit-code`,
# which the first two of these used to use: the numbers are written at full
# double precision and R reaches them through linear algebra whose last bits
# depend on which BLAS kernel its image picks for the CPU it runs on, so two
# machines with identical R and package versions produce files that differ in
# the 16th digit. A byte comparison reads that as drift.
#
# Note that these rewrite the working tree, so the comparison is against HEAD
# rather than against the files on disk.
COMPARE_REFERENCE = validation/compare_reference.py

define compare_against_head
	pinned="$$(mktemp)"; git show HEAD:pymare/tests/data/$(1) > "$$pinned"; \
		python $(COMPARE_REFERENCE) "$$pinned" pymare/tests/data/$(1); \
		status=$$?; rm -f "$$pinned"; exit $$status

endef

check_robumeta_alignment:
	@validation/robumeta/regenerate.sh
	@$(call compare_against_head,robumeta_reference.json)

check_metafor_alignment:
	@validation/metafor/regenerate.sh
	@$(foreach reference,metafor_reference.json metafor_escalc_reference.json \
		metafor_permutest_reference.json,$(call compare_against_head,$(reference)))

check_clubsandwich_alignment:
	@validation/clubsandwich/regenerate.sh
	@$(call compare_against_head,clubsandwich_reference.json)

# A smoke test of the benchmark suite, not a measurement: --quick takes one
# sample per benchmark. The Benchmark workflow is what measures, by timing a
# pull request against its base.
benchmark:
	@asv run --quick --python=same
