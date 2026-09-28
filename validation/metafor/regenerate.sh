#!/usr/bin/env bash
# Regenerate the metafor reference values PyMARE's alignment tests read.
#
# Run from the repository root:
#
#     validation/metafor/regenerate.sh
#
# Rewrites all three files in place:
#
#     pymare/tests/data/metafor_reference.json           rma.uni
#     pymare/tests/data/metafor_escalc_reference.json    escalc
#     pymare/tests/data/metafor_permutest_reference.json permutest
#
# The alignment workflow runs this script and then compares each result
# numerically against the pinned file with validation/compare_reference.py, so
# CI and a local run cannot drift apart. Numerically rather than with
# `git diff --exit-code`, which this script used to rely on: the numbers are
# written at full double precision and R reaches them through linear algebra
# whose last bits depend on which BLAS kernel its image picks for the CPU it
# runs on, so two machines with identical R and metafor versions produce files
# that differ in the 16th digit.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${repo_root}/pymare/tests/data"

docker build --quiet -t pymare-metafor "${repo_root}/validation/metafor" >/dev/null

run() {
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -v "${data_dir}:/data" \
        pymare-metafor "$@"
}

run /opt/run_metafor.R /data/metafor_small_sample.csv /data/metafor_reference.json
run /opt/run_escalc.R /data/metafor_escalc_inputs.csv /data/metafor_escalc_reference.json
run /opt/run_permutest.R /data/metafor_small_sample.csv /data/metafor_permutest_reference.json
