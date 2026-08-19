"""Utilities for the PyMARE test suite.

Helpers that are not fixtures and not tests. Fixtures belong in ``conftest.py``,
so this holds the plumbing that fixtures and parametrization both need -- most
importantly the reference values that cannot be recomputed at test time because
they come from another implementation.
"""

import json
import os.path as op


def get_test_data_path():
    """Return the path to the test data directory.

    Returns
    -------
    :obj:`str`
        Absolute path to ``pymare/tests/data``.
    """
    return op.abspath(op.join(op.dirname(__file__), "data"))


def load_robumeta_reference():
    """Load the reference values the R package robumeta produced.

    Returns
    -------
    :obj:`dict`
        The parsed contents of ``data/robumeta_reference.json``: a ``"source"``
        record naming the R and robumeta versions that produced the numbers, and
        a ``"cases"`` list holding one entry per (model, rho, variance column)
        combination.

    Notes
    -----
    robumeta is not a test dependency, so these numbers are pinned rather than
    recomputed on every run. ``validation/robumeta/regenerate.sh`` rewrites the
    file from the pinned R image, and the ``Check robumeta alignment`` workflow
    runs that script on every pull request and fails if anything moved -- which
    is what keeps a pinned file from quietly becoming a stale one.
    """
    with open(op.join(get_test_data_path(), "robumeta_reference.json")) as fobj:
        return json.load(fobj)
