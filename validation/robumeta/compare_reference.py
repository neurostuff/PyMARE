#!/usr/bin/env python
"""Compare a regenerated robumeta reference file against the pinned one.

Usage
-----
::

    validation/robumeta/compare_reference.py PINNED REGENERATED

Exits non-zero, naming the values that moved, if the two disagree by more than
:data:`RTOL`.

Why not ``git diff``
--------------------
The reference values are written at full double precision, and ``robu()`` reaches
them through linear algebra whose last bits depend on which BLAS kernel R's image
picks for the CPU it runs on. Two GitHub runners therefore produce files that
differ in the 16th significant digit -- on one observed run, 84 of 168 values, by
at most 2.6e-16 absolute -- with the R and robumeta versions identical. A byte
comparison reads that as drift and fails a tree that never touched the harness.
"""

import json
import sys

import numpy as np

#: Relative tolerance for the numbers. Set below the ``RTOL`` that
#: ``pymare/tests/test_robumeta_alignment.py`` holds PyMARE to, so that a pin this
#: check accepts is still good to the precision that test relies on, and far above
#: the 4e-14 that runner-to-runner rounding has been seen to produce.
RTOL = 1e-11

#: Absolute floor, for any value near zero where a relative tolerance says little.
ATOL = 1e-12


def numbers(document):
    """Return the file's numbers, labelled by the case and quantity they belong to.

    Parameters
    ----------
    document : :obj:`dict`
        A parsed reference file.

    Returns
    -------
    :obj:`dict`
        Label to array. Comparing two of these compares the values, the number of
        them, and which cases are present, all at once.
    """
    return {
        f"{case['model']} rho={case['rho']} {case['variances']} {key}": np.atleast_1d(case[key])
        for case in document["cases"]
        for key in ("tau2", "beta", "se", "dof")
    }


def main(argv):
    """Compare the two files named on the command line."""
    if len(argv) != 3:
        print(__doc__)
        return 2

    pinned, regenerated = (json.load(open(path)) for path in argv[1:3])
    problems = []

    # Exactly, because this is where a rotted image shows up: it records the R and
    # robumeta versions the numbers came from, and those either match or the pin
    # is stale rather than wobbly.
    if pinned["source"] != regenerated["source"]:
        problems.append(f"source block moved: {pinned['source']} -> {regenerated['source']}")

    old, new = numbers(pinned), numbers(regenerated)
    if old.keys() != new.keys():
        problems.append(f"cases moved: {sorted(old.keys() ^ new.keys())}")
    for label in sorted(old.keys() & new.keys()):
        if old[label].shape != new[label].shape:
            problems.append(f"{label}: {old[label].size} values -> {new[label].size}")
        elif not np.allclose(old[label], new[label], rtol=RTOL, atol=ATOL):
            problems.append(f"{label}: {old[label].tolist()} -> {new[label].tolist()}")

    if not problems:
        print(f"The pinned robumeta reference values still match robumeta, to {RTOL:g}.")
        return 0

    print("## robumeta reference values moved")
    print()
    print(
        "`validation/robumeta/regenerate.sh` produced numbers more than"
        f" {RTOL:g} relative away from those pinned in"
        " `pymare/tests/data/robumeta_reference.json`. Either the pinned image no"
        " longer computes what it did, or it is no longer the image the pin came from."
    )
    print()
    for problem in problems:
        print(f"- {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
