#!/usr/bin/env bash
# Regenerate the metafor reference values PyMARE's alignment test reads.
#
# Run from the repository root:
#
#     validation/metafor/regenerate.sh
#
# Rewrites pymare/tests/data/metafor_reference.json in place. If the file
# changes, either metafor or PyMARE's reference moved, and the diff says which
# numbers. The alignment workflow runs this script and fails on a non-empty
# diff, so CI and a local run cannot drift apart.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${repo_root}/pymare/tests/data"

docker build --quiet -t pymare-metafor "${repo_root}/validation/metafor" >/dev/null
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${data_dir}:/data" \
    pymare-metafor \
    /data/metafor_small_sample.csv /data/metafor_reference.json
