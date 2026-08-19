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


def cmdstan_is_available():
    """Report whether the Stan estimator can actually be run here.

    Returns
    -------
    :obj:`bool`
        True when ``cmdstanpy`` imports *and* it can find a CmdStan
        installation.

    Notes
    -----
    Both halves matter. ``cmdstanpy`` installs cleanly from PyPI without
    CmdStan, which is a C++ build rather than a Python package, so an import
    check alone would report an environment as ready when it can only fail.

    The failure this replaced was subtler still: the previous gate probed
    ``find_spec("pystan")``, but PyStan 3 is distributed as ``pystan`` and
    imported as ``stan``, so the probe was unsatisfiable and the estimator's
    only real test skipped everywhere, including in the CI job that existed to
    run it. A skip reads as a pass in a CI log, which is why the
    ``pytest_collection_modifyitems`` hook in ``conftest.py`` consults this
    function and fails the run wherever ``PYMARE_REQUIRE_CMDSTAN`` says Stan is
    expected.
    """
    try:
        import cmdstanpy
    except ImportError:
        return False

    try:
        cmdstanpy.cmdstan_path()
    except Exception:
        return False

    return True
