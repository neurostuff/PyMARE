.PHONY: all_tests benchmark check_robumeta_alignment help install_cmdstan lint test_robumeta
.PHONY: test_stan unittest validate_stan

# --cov-append matches what CI does, so a local run of two targets in a row
# reports their combined coverage rather than only the last one's.
PYTEST_COV := --cov-append --cov-report=xml --cov=pymare

all_tests: lint unittest test_stan test_robumeta

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint                       to run flake8 over pymare and the benchmarks"
	@echo "  unittest                   to run every test except the Stan sampling ones"
	@echo "  install_cmdstan            to install the CmdStan that test_stan needs"
	@echo "  test_stan                  to run the Stan sampling tests (needs the stan extra and CmdStan)"
	@echo "  test_robumeta              to run the robumeta alignment tests"
	@echo "  check_robumeta_alignment   to regenerate the robumeta reference values (needs Docker)"
	@echo "  validate_stan              to re-measure the Stan model's bias and coverage (~10 min)"
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

# What the "Validate the Stan model" workflow runs. Regenerates
# pymare/tests/data/stan_validation.json and fails if any design cell misses the
# thresholds in pymare.tests.utils.STAN_VALIDATION_THRESHOLDS. Slow -- about ten
# minutes -- which is why it is not part of unittest or test_stan.
validate_stan:
	@python validation/stan/simulate.py --check

# What the "Check robumeta alignment" workflow runs. Needs Docker, because the
# reference values come from R.
check_robumeta_alignment:
	@validation/robumeta/regenerate.sh
	@git diff --exit-code -- pymare/tests/data/robumeta_reference.json \
		&& echo "The pinned robumeta reference values still match robumeta."

# A smoke test of the benchmark suite, not a measurement: --quick takes one
# sample per benchmark. The Benchmark workflow is what measures, by timing a
# pull request against its base.
benchmark:
	@asv run --quick --python=same
