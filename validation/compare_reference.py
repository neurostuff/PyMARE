#!/usr/bin/env python
"""Compare a regenerated R reference file against the pinned one.

Usage
-----
::

    validation/compare_reference.py PINNED REGENERATED

Exits non-zero, naming the values that moved, if the two disagree by more than
:data:`RTOL`.

Works on any of the reference files under ``pymare/tests/data`` that
``validation/*/run_*.R`` writes, without being told which: every one of them is
a ``{"source": {...}, "cases": [...]}`` document whose cases hold numbers, lists
of numbers, nested objects of them, or the strings that name the case. See
:func:`numbers`.

Why not ``git diff``
--------------------
The reference values are written at full double precision, and R reaches them
through linear algebra whose last bits depend on which BLAS kernel its image
picks for the CPU it runs on. Two GitHub runners therefore produce files that
differ in the 16th significant digit -- on one observed run of the robumeta
reference, 84 of 168 values, by at most 2.6e-16 absolute -- with the R and
package versions identical. A byte comparison reads that as drift and fails a
tree that never touched the harness.

This generalizes the per-package script ``validation/robumeta`` used to carry,
which the metafor README named as a follow-up. That was worth doing rather than
copying, because the metafor reference has grown past what a byte comparison can
police: regenerating it under the same R and metafor versions on a different BLAS
moves numbers by up to 9.5e-14 relative on the inference path and 3.3e-9 on an ML
tau^2, the latter being an optimizer landing a step earlier or later.
"""

import json
import sys

import numpy as np

#: Relative tolerance for the numbers. Set at or below the tolerance the
#: alignment tests hold PyMARE to for the same quantities, so that a pin this
#: check accepts is still good to the precision those tests rely on.
RTOL = 1e-11

#: Absolute floor, for any value near zero where a relative tolerance says
#: little.
ATOL = 1e-12

#: Keys whose values are searched by their own rules rather than compared as
#: numbers: strings naming the case, which :func:`numbers` folds into the label
#: instead. Anything not listed here and not a number is a mistake in the
#: generator, and :func:`numbers` raises rather than skipping it -- a reference
#: file that stopped recording a quantity would otherwise pass this check by
#: having nothing left to compare.
LABEL_KEYS = ("case", "design", "model", "method", "test", "variances", "rho")

#: Quantities allowed to move further than :data:`RTOL`, with the tolerance each
#: gets instead and why. Keyed by the trailing component of the label, so it
#: applies to that quantity in every case of every file.
LOOSE = {
    # Both metafor and PyMARE reach an ML or REML tau^2 by numerical search, and
    # a different BLAS moves the objective enough for the search to stop a step
    # earlier or later. Observed 3.3e-9 relative between two runs that agreed on
    # every closed-form quantity to 1e-14. Still far tighter than the 1e-4 that
    # pymare/tests/test_metafor_random_effects.py holds a profiled tau^2 to.
    "tau2": 1e-7,
}


def numbers(document, path=()):
    """Return the document's numbers, labelled by where each one came from.

    Parameters
    ----------
    document : :obj:`dict` or :obj:`list` or :obj:`float`
        A parsed reference file, or any part of one.
    path : :obj:`tuple` of :obj:`str`, optional
        The labels of the enclosing cases and keys, used to build the label.

    Returns
    -------
    :obj:`dict`
        Label to array. Comparing two of these compares the values, the number
        of them, and which cases are present, all at once.

    Raises
    ------
    TypeError
        If a value is neither a number, nor a list of them, nor a nested object
        of them, nor one of :data:`LABEL_KEYS`. Raising rather than skipping is
        deliberate: a quantity this function silently ignored would be a
        quantity the check does not police.
    """
    if isinstance(document, dict):
        label = tuple(str(document[key]) for key in LABEL_KEYS if key in document)
        collected = {}
        for key, value in document.items():
            if key in LABEL_KEYS:
                continue
            collected.update(numbers(value, path + label + (key,)))
        return collected

    if isinstance(document, list) and document and isinstance(document[0], dict):
        return {
            label: value for entry in document for label, value in numbers(entry, path).items()
        }

    if document is None:
        # A quantity the R side reported as NA, such as the degrees of freedom
        # of an uncorrected fit. Compared as a value so that becoming a number,
        # or a number becoming null, is a difference rather than a silent skip.
        return {" ".join(path): np.array([np.nan])}

    try:
        values = np.atleast_1d(np.asarray(document, dtype=float))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{' '.join(path)}: not numeric ({document!r})") from error
    return {" ".join(path): values}


def tolerance(label):
    """Return the relative tolerance for one label, per :data:`LOOSE`."""
    return LOOSE.get(label.rsplit(" ", 1)[-1], RTOL)


def compare(pinned, regenerated):
    """Return a list of human-readable differences, empty when the two agree."""
    problems = []

    # Exactly, because this is where a rotted image shows up: the source block
    # records the R and package versions the numbers came from, and those either
    # match or the pin is stale rather than wobbly.
    if pinned["source"] != regenerated["source"]:
        problems.append(f"source block moved: {pinned['source']} -> {regenerated['source']}")

    old, new = numbers(pinned["cases"]), numbers(regenerated["cases"])
    if old.keys() != new.keys():
        problems.append(f"cases moved: {sorted(old.keys() ^ new.keys())}")

    for label in sorted(old.keys() & new.keys()):
        rtol = tolerance(label)
        if old[label].shape != new[label].shape:
            problems.append(f"{label}: {old[label].size} values -> {new[label].size}")
        elif not np.allclose(old[label], new[label], rtol=rtol, atol=ATOL, equal_nan=True):
            problems.append(
                f"{label} (rtol {rtol:g}): {old[label].tolist()} -> {new[label].tolist()}"
            )

    return problems


def main(argv):
    """Compare the two files named on the command line."""
    if len(argv) != 3:
        print(__doc__)
        return 2

    # Named after the regenerated file rather than the pinned one, because the
    # pinned copy is usually a temporary file `git show` was piped into.
    name = argv[2].rsplit("/", 1)[-1]
    with open(argv[1]) as fobj:
        pinned = json.load(fobj)
    with open(argv[2]) as fobj:
        regenerated = json.load(fobj)

    problems = compare(pinned, regenerated)
    if not problems:
        print(f"The pinned values in {name} still match R, to {RTOL:g}.")
        return 0

    print(f"## Reference values moved in {name}")
    print()
    print(
        "Regenerating produced numbers further from those pinned in"
        f" `pymare/tests/data/{name}` than the tolerances in"
        " `validation/compare_reference.py` allow. Either the pinned image no"
        " longer computes what it did, or it is no longer the image the pin came"
        " from."
    )
    print()
    for problem in problems:
        print(f"- {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
