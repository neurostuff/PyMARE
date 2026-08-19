.PHONY: all_tests benchmark check_robumeta_alignment help lint test_robumeta test_stan unittest

# --cov-append matches what CI does, so a local run of two targets in a row
# reports their combined coverage rather than only the last one's.
PYTEST_COV := --cov-append --cov-report=xml --cov=pymare

all_tests: lint unittest test_stan test_robumeta

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint                       to run flake8 over pymare and the benchmarks"
	@echo "  unittest                   to run every test except the Stan ones"
	@echo "  test_stan                  to run the Stan estimator tests (needs the stan extra)"
	@echo "  test_robumeta              to run the robumeta alignment tests"
	@echo "  check_robumeta_alignment   to regenerate the robumeta reference values (needs Docker)"
	@echo "  benchmark                  to run the asv suite once in the current environment"
	@echo "  all_tests                  to run lint and every test target"

lint:
	@flake8 pymare benchmarks

unittest:
	@python -m pytest -m "not stan" $(PYTEST_COV)

test_stan:
	@python -m pytest -m "stan" $(PYTEST_COV)

test_robumeta:
	@python -m pytest -m "robumeta" $(PYTEST_COV)

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
