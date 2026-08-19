#!/usr/bin/env bash
# Regenerate the robumeta reference values PyMARE's alignment test reads.
#
# Run from the repository root:
#
#     validation/robumeta/regenerate.sh
#
# Rewrites pymare/tests/data/robumeta_reference.json in place. If the file
# changes, either robumeta or PyMARE's reference moved, and the diff says
# which numbers. The alignment workflow runs this script and fails on a
# non-empty diff, so CI and a local run cannot drift apart.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${repo_root}/pymare/tests/data"

docker build --quiet -t pymare-robumeta "${repo_root}/validation/robumeta" >/dev/null
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${data_dir}:/data" \
    pymare-robumeta \
    /data/robumeta_correlated_effects.csv /data/robumeta_reference.json
