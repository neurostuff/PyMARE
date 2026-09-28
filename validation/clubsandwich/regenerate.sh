#!/usr/bin/env bash
# Regenerate the clubSandwich reference values PyMARE's alignment test reads.
#
# Run from the repository root:
#
#     validation/clubsandwich/regenerate.sh
#
# Rewrites pymare/tests/data/clubsandwich_reference.json in place. The
# alignment workflow runs this script and then compares the result numerically
# against the pinned file, so CI and a local run cannot drift apart.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${repo_root}/pymare/tests/data"

docker build --quiet -t pymare-clubsandwich "${repo_root}/validation/clubsandwich" >/dev/null
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${data_dir}:/data" \
    pymare-clubsandwich \
    /data/robumeta_correlated_effects.csv /data/clubsandwich_reference.json
